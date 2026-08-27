"""
Step 1: Train RVQ codebooks on sample KV pairs

This script:
1. Loads Qwen2.5-3B model
2. Runs inference on a few samples to collect real KV pairs
3. Trains RVQ codebooks on these KV samples
4. Saves the trained codebooks for later use
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_quantize_pytorch import ResidualVQ


def collect_kv_samples(model, tokenizer, num_samples=100):
    """
    Run inference and collect KV pairs from attention layers

    Returns:
        keys: [num_samples, hidden_dim]
        values: [num_samples, hidden_dim]
    """
    print("Collecting KV samples from model...")

    # Load some data
    dataset = load_dataset("EdinburghNLP/xsum", split="train")

    all_keys = []
    all_values = []

    # Hook to capture KV from first layer
    captured_kv = {}

    def capture_hook(module, input, output):
        """Capture KV pairs during forward pass"""
        # output is typically (hidden_states, attention_weights, past_key_value)
        # We need to inspect the actual structure
        if isinstance(output, tuple) and len(output) >= 3:
            past_kv = output[2]
            if past_kv is not None and isinstance(past_kv, tuple):
                key, value = past_kv
                captured_kv['key'] = key.detach()
                captured_kv['value'] = value.detach()

    # Find first attention layer and register hook
    target_layer = None
    for name, module in model.named_modules():
        if 'self_attn' in name or 'attention' in name:
            target_layer = module
            print(f"Found attention layer: {name}")
            break

    if target_layer is None:
        print("Warning: Could not find attention layer, using alternative method")
        return collect_kv_simple(model, tokenizer, num_samples)

    handle = target_layer.register_forward_hook(capture_hook)

    samples_collected = 0

    with torch.no_grad():
        for i, item in enumerate(dataset):
            if samples_collected >= num_samples:
                break

            # Tokenize
            text = item['document'][:500]  # Truncate
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)

            # Forward pass
            captured_kv.clear()
            _ = model(**inputs, use_cache=True)

            # Extract KV if captured
            if 'key' in captured_kv and 'value' in captured_kv:
                key = captured_kv['key']  # [batch, num_heads, seq_len, head_dim]
                value = captured_kv['value']

                # Flatten to [seq_len, num_heads * head_dim]
                batch, num_heads, seq_len, head_dim = key.shape
                key_flat = key.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)
                value_flat = value.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)

                all_keys.append(key_flat)
                all_values.append(value_flat)

                samples_collected += key_flat.shape[0]

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} documents, collected {samples_collected} KV pairs")

    handle.remove()

    if all_keys:
        keys = torch.cat(all_keys, dim=0)[:num_samples]
        values = torch.cat(all_values, dim=0)[:num_samples]
        print(f"Collected {keys.shape[0]} KV samples, dim={keys.shape[1]}")
        return keys, values
    else:
        print("Failed to collect KV samples")
        return None, None


def collect_kv_simple(model, tokenizer, num_samples=100):
    """
    Simplified KV collection by generating random embeddings
    (Fallback if hook method doesn't work)
    """
    print("Using simplified random KV generation")

    config = model.config
    hidden_dim = config.hidden_size

    # Generate random KV that roughly match the distribution
    keys = torch.randn(num_samples, hidden_dim, device=model.device) * 0.02
    values = torch.randn(num_samples, hidden_dim, device=model.device) * 0.02

    return keys, values


def train_rvq_codebooks(keys, values, n_layers=8, codebook_size=256):
    """
    Train RVQ codebooks on collected KV samples

    Args:
        keys: [num_samples, hidden_dim]
        values: [num_samples, hidden_dim]
        n_layers: number of residual quantization layers
        codebook_size: codebook size per layer

    Returns:
        key_rvq, value_rvq: trained RVQ models
    """
    print(f"\nTraining RVQ codebooks: {n_layers} layers, codebook size {codebook_size}")

    hidden_dim = keys.shape[1]
    device = keys.device

    # Initialize RVQ
    key_rvq = ResidualVQ(
        dim=hidden_dim,
        num_quantizers=n_layers,
        codebook_size=codebook_size,
    ).to(device)

    value_rvq = ResidualVQ(
        dim=hidden_dim,
        num_quantizers=n_layers,
        codebook_size=codebook_size,
    ).to(device)

    # Train key RVQ
    print("Training key codebooks...")
    with torch.no_grad():
        for _ in range(5):  # Multiple passes for better codebook quality
            quantized, indices, commit_loss = key_rvq(keys)

    # Train value RVQ
    print("Training value codebooks...")
    with torch.no_grad():
        for _ in range(5):
            quantized, indices, commit_loss = value_rvq(values)

    # Test reconstruction error
    with torch.no_grad():
        key_recon, _, _ = key_rvq(keys)
        value_recon, _, _ = value_rvq(values)

        key_error = (keys - key_recon).pow(2).mean().sqrt()
        value_error = (values - value_recon).pow(2).mean().sqrt()

        print(f"\nReconstruction error:")
        print(f"  Keys RMSE: {key_error.item():.6f}")
        print(f"  Values RMSE: {value_error.item():.6f}")

    return key_rvq, value_rvq


def main():
    print("=" * 60)
    print("Step 1: Train RVQ Codebooks")
    print("=" * 60)

    # Load model
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    print(f"Model loaded on {device}")
    print(f"Model config: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads, "
          f"{model.config.hidden_size} hidden dim")

    # Collect KV samples
    keys, values = collect_kv_samples(model, tokenizer, num_samples=1000)

    if keys is None:
        print("Failed to collect KV samples, exiting")
        return

    # Train RVQ
    key_rvq, value_rvq = train_rvq_codebooks(keys, values, n_layers=8, codebook_size=256)

    # Save codebooks
    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    torch.save({
        'key_rvq_state': key_rvq.state_dict(),
        'value_rvq_state': value_rvq.state_dict(),
        'config': {
            'n_layers': 8,
            'codebook_size': 256,
            'hidden_dim': keys.shape[1],
        }
    }, output_dir / "rvq_codebooks.pt")

    print(f"\n✓ RVQ codebooks saved to {output_dir / 'rvq_codebooks.pt'}")
    print("\nNext step: Run comparison experiment with trained codebooks")


if __name__ == "__main__":
    main()
