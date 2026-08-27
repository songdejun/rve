"""
RVQ-based KV Cache with gradual precision degradation
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from vector_quantize_pytorch import ResidualVQ


class RVQKVCache:
    """
    KV Cache with RVQ-based gradual eviction

    Instead of deleting entire tokens, we evict residual layers progressively
    to achieve memory reduction while maintaining positional encoding continuity.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_residual_layers: int = 8,
        codebook_size: int = 256,
        device: str = "cuda"
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_residual_layers = n_residual_layers
        self.device = device

        # RVQ for key and value separately
        hidden_dim = num_heads * head_dim
        self.key_rvq = ResidualVQ(
            dim=hidden_dim,
            num_quantizers=n_residual_layers,
            codebook_size=codebook_size,
        ).to(device)

        self.value_rvq = ResidualVQ(
            dim=hidden_dim,
            num_quantizers=n_residual_layers,
            codebook_size=codebook_size,
        ).to(device)

        # Storage: {layer_idx: (keys, values, active_layers_mask)}
        self.cache = {}

        # Track how many residual layers each token position uses
        # token_id -> number of active layers (8 = full precision, 4 = half evicted)
        self.active_layers = {}

        self.trained = False

    def train_codebooks(self, sample_keys: torch.Tensor, sample_values: torch.Tensor):
        """
        Train RVQ codebooks on sample KV tensors

        Args:
            sample_keys: [batch, seq_len, num_heads, head_dim]
            sample_values: [batch, seq_len, num_heads, head_dim]
        """
        # Flatten to [batch * seq_len, num_heads * head_dim]
        batch, seq_len, num_heads, head_dim = sample_keys.shape
        keys_flat = sample_keys.reshape(-1, num_heads * head_dim)
        values_flat = sample_values.reshape(-1, num_heads * head_dim)

        print(f"Training RVQ codebooks on {keys_flat.shape[0]} samples...")

        # Train key RVQ
        with torch.no_grad():
            _, _, _ = self.key_rvq(keys_flat)

        # Train value RVQ
        with torch.no_grad():
            _, _, _ = self.value_rvq(values_flat)

        self.trained = True
        print("RVQ codebooks trained!")

    def quantize_and_store(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        start_pos: int = 0
    ):
        """
        Quantize KV tensors with RVQ and store

        Args:
            layer_idx: transformer layer index
            keys: [batch, seq_len, num_heads, head_dim]
            values: [batch, seq_len, num_heads, head_dim]
            start_pos: starting position in sequence
        """
        assert self.trained, "Must train codebooks first!"

        batch, seq_len, num_heads, head_dim = keys.shape

        # Flatten for RVQ
        keys_flat = keys.reshape(-1, num_heads * head_dim)
        values_flat = values.reshape(-1, num_heads * head_dim)

        # Quantize with RVQ - get indices for all residual layers
        _, key_indices, _ = self.key_rvq(keys_flat)
        _, value_indices, _ = self.value_rvq(values_flat)

        # key_indices: [batch*seq_len, n_residual_layers]
        # value_indices: [batch*seq_len, n_residual_layers]

        # Store indices instead of full vectors
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {
                'key_indices': [],
                'value_indices': [],
            }

        self.cache[layer_idx]['key_indices'].append(key_indices)
        self.cache[layer_idx]['value_indices'].append(value_indices)

        # Initialize all tokens as full precision (all layers active)
        for pos in range(start_pos, start_pos + seq_len):
            self.active_layers[pos] = self.n_residual_layers

    def evict_residual_layers(self, token_positions: List[int], keep_layers: int = 4):
        """
        Evict residual layers for specified token positions

        Args:
            token_positions: list of token positions to evict from
            keep_layers: how many residual layers to keep (e.g., 4 out of 8)
        """
        for pos in token_positions:
            if pos in self.active_layers:
                self.active_layers[pos] = min(keep_layers, self.active_layers[pos])

        print(f"Evicted layers for {len(token_positions)} tokens, keeping {keep_layers} layers")

    def reconstruct_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct KV tensors from RVQ indices, respecting active layers

        Returns:
            keys: [batch, seq_len, num_heads, head_dim]
            values: [batch, seq_len, num_heads, head_dim]
        """
        if layer_idx not in self.cache:
            return None, None

        # Concatenate all stored indices
        key_indices = torch.cat(self.cache[layer_idx]['key_indices'], dim=0)
        value_indices = torch.cat(self.cache[layer_idx]['value_indices'], dim=0)

        seq_len = key_indices.shape[0]

        # Reconstruct with partial residual layers based on active_layers
        keys_list = []
        values_list = []

        for pos in range(seq_len):
            active = self.active_layers.get(pos, self.n_residual_layers)

            # Only use first 'active' layers for reconstruction
            k_idx = key_indices[pos:pos+1, :active]  # [1, active_layers]
            v_idx = value_indices[pos:pos+1, :active]

            # Reconstruct using RVQ codebooks
            # This is simplified - actual implementation depends on library API
            k_reconstructed = self._reconstruct_from_indices(self.key_rvq, k_idx)
            v_reconstructed = self._reconstruct_from_indices(self.value_rvq, v_idx)

            keys_list.append(k_reconstructed)
            values_list.append(v_reconstructed)

        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)

        # Reshape back to [batch, seq_len, num_heads, head_dim]
        keys = keys.reshape(1, seq_len, self.num_heads, self.head_dim)
        values = values.reshape(1, seq_len, self.num_heads, self.head_dim)

        return keys, values

    def _reconstruct_from_indices(self, rvq_model: ResidualVQ, indices: torch.Tensor) -> torch.Tensor:
        """
        Helper to reconstruct vector from RVQ indices

        Args:
            rvq_model: the RVQ model (key or value)
            indices: [batch, n_layers] quantized indices

        Returns:
            reconstructed: [batch, dim]
        """
        # Get codebooks from RVQ
        # vector-quantize-pytorch stores codebooks in rvq_model.layers
        reconstructed = torch.zeros(indices.shape[0], self.num_heads * self.head_dim, device=self.device)

        for layer_idx in range(indices.shape[1]):
            # Get the codebook for this residual layer
            codebook = rvq_model.layers[layer_idx].codebook  # [codebook_size, dim]
            idx = indices[:, layer_idx]  # [batch]

            # Look up and accumulate residuals
            residual = codebook[idx]
            reconstructed = reconstructed + residual

        return reconstructed

    def get_memory_usage(self) -> dict:
        """Calculate current memory usage"""
        total_tokens = len(self.active_layers)
        total_layers = sum(self.active_layers.values())
        avg_layers = total_layers / total_tokens if total_tokens > 0 else 0

        # Memory per token = num_layers * (log2(codebook_size) bits per layer)
        # Compared to full precision: num_heads * head_dim * 2 (for FP16) bytes

        return {
            'total_tokens': total_tokens,
            'average_layers_per_token': avg_layers,
            'compression_ratio': self.n_residual_layers / avg_layers if avg_layers > 0 else 1.0
        }
