"""
Baseline: Simple token eviction (H2O-style)
"""
import torch
from typing import Optional, Tuple, Dict, List


class TokenEvictionCache:
    """
    Baseline: Binary token eviction

    Evicts entire tokens based on importance scores (attention-based or recency)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_tokens: int = 2048,
        eviction_ratio: float = 0.5,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_tokens = max_tokens
        self.eviction_ratio = eviction_ratio
        self.device = device

        # Cache: layer_idx -> (keys, values)
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.current_length = 0

        # Track which positions are kept
        self.kept_positions: List[int] = []

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ):
        """
        Update cache and evict if needed

        Args:
            layer_idx: transformer layer index
            keys: [batch, seq_len, num_heads, head_dim]
            values: [batch, seq_len, num_heads, head_dim]
            attention_scores: [batch, num_heads, seq_len] importance scores
        """
        if layer_idx not in self.cache:
            self.cache[layer_idx] = (keys, values)
            self.current_length = keys.shape[1]
        else:
            # Concatenate with existing cache
            old_k, old_v = self.cache[layer_idx]
            new_k = torch.cat([old_k, keys], dim=1)
            new_v = torch.cat([old_v, values], dim=1)

            self.cache[layer_idx] = (new_k, new_v)
            self.current_length = new_k.shape[1]

        # Evict if exceeds max
        if self.current_length > self.max_tokens:
            self._evict_tokens(attention_scores)

    def _evict_tokens(self, attention_scores: Optional[torch.Tensor] = None):
        """
        Evict tokens to bring cache size down

        Strategy: Keep recent tokens + high attention tokens
        """
        target_length = int(self.max_tokens * (1 - self.eviction_ratio))
        tokens_to_keep = target_length

        if attention_scores is not None:
            # H2O-style: keep high attention tokens
            # attention_scores: [batch, num_heads, seq_len]
            importance = attention_scores.mean(dim=(0, 1))  # [seq_len]

            # Keep top-k important tokens + recent tokens
            recent_window = min(tokens_to_keep // 2, 128)
            num_important = tokens_to_keep - recent_window

            # Keep recent tokens
            recent_positions = list(range(self.current_length - recent_window, self.current_length))

            # Keep important tokens from earlier positions
            early_scores = importance[:-recent_window]
            _, important_indices = torch.topk(early_scores, min(num_important, len(early_scores)))
            important_positions = important_indices.tolist()

            # Combine
            self.kept_positions = sorted(set(important_positions + recent_positions))
        else:
            # Fallback: just keep recent tokens
            self.kept_positions = list(range(self.current_length - tokens_to_keep, self.current_length))

        # Actually evict from all layers
        for layer_idx in self.cache:
            keys, values = self.cache[layer_idx]

            # Select kept positions
            kept_keys = keys[:, self.kept_positions, :, :]
            kept_values = values[:, self.kept_positions, :, :]

            self.cache[layer_idx] = (kept_keys, kept_values)

        self.current_length = len(self.kept_positions)
        print(f"Evicted to {self.current_length} tokens")

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV for a layer"""
        return self.cache.get(layer_idx, (None, None))

    def get_memory_usage(self) -> dict:
        """Calculate memory usage"""
        # FP16: 2 bytes per element
        bytes_per_token = self.num_heads * self.head_dim * 2 * 2  # *2 for K and V
        total_bytes = self.current_length * bytes_per_token * self.num_layers

        return {
            'total_tokens': self.current_length,
            'total_bytes': total_bytes,
            'mb': total_bytes / 1024 / 1024
        }


class NoEvictionCache:
    """
    Baseline: No eviction, just standard KV cache
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, device: str = "cuda"):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.current_length = 0

    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        if layer_idx not in self.cache:
            self.cache[layer_idx] = (keys, values)
            self.current_length = keys.shape[1]
        else:
            old_k, old_v = self.cache[layer_idx]
            new_k = torch.cat([old_k, keys], dim=1)
            new_v = torch.cat([old_v, values], dim=1)
            self.cache[layer_idx] = (new_k, new_v)
            self.current_length = new_k.shape[1]

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.cache.get(layer_idx, (None, None))

    def get_memory_usage(self) -> dict:
        bytes_per_token = self.num_heads * self.head_dim * 2 * 2
        total_bytes = self.current_length * bytes_per_token * self.num_layers

        return {
            'total_tokens': self.current_length,
            'total_bytes': total_bytes,
            'mb': total_bytes / 1024 / 1024
        }
