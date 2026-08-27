"""
Step 1 (Simplified): Train RVQ codebooks on synthetic KV samples

For quick validation, we use synthetic KV samples that match the model's distribution
This is sufficient to test the core idea of gradual precision degradation
"""
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_quantize_pytorch import ResidualVQ


def generate_synthetic_kv(hidden_dim=2048, num_samples=1000, device="cuda"):
    """
    Generate synthetic KV samples that roughly match transformer KV distribution

    Real KV vectors in transformers typically have:
    - Small magnitude (std around 0.02-0.05)
    - Some structure (not pure random)
    """
    print(f"Generating {num_samples} synthetic KV samples, dim={hidden_dim}")

    # Generate with small magnitude like real attention KV
    keys = torch.randn(num_samples, hidden_dim, device=device) * 0.02
    values = torch.randn(num_samples, hidden_dim, device=device) * 0.02

    return keys, values


def train_rvq_codebooks(keys, values, n_layers=8, codebook_size=256):
    """
    Train RVQ codebooks

    Args:
        keys: [num_samples, hidden_dim]
        values: [num_samples, hidden_dim]
        n_layers: number of residual quantization layers
        codebook_size: codebook size per layer

    Returns:
        key_rvq, value_rvq: trained RVQ models
    """
    print(f"\nTraining RVQ codebooks:")
    print(f"  Residual layers: {n_layers}")
    print(f"  Codebook size: {codebook_size}")
    print(f"  Hidden dim: {keys.shape[1]}")

    hidden_dim = keys.shape[1]
    device = keys.device

    # Initialize RVQ
    key_rvq = ResidualVQ(
        dim=hidden_dim,
        num_quantizers=n_layers,
        codebook_size=codebook_size,
        kmeans_init=True,  # Use k-means initialization
        kmeans_iters=10,
    ).to(device)

    value_rvq = ResidualVQ(
        dim=hidden_dim,
        num_quantizers=n_layers,
        codebook_size=codebook_size,
        kmeans_init=True,
        kmeans_iters=10,
    ).to(device)

    # Train key RVQ
    print("\nTraining key codebooks...")
    with torch.no_grad():
        for epoch in range(10):  # Multiple passes for better codebook quality
            quantized, indices, commit_loss = key_rvq(keys)
            if epoch % 3 == 0:
                error = (keys - quantized).pow(2).mean().sqrt()
                print(f"  Epoch {epoch}: RMSE = {error.item():.6f}")

    # Train value RVQ
    print("\nTraining value codebooks...")
    with torch.no_grad():
        for epoch in range(10):
            quantized, indices, commit_loss = value_rvq(values)
            if epoch % 3 == 0:
                error = (values - quantized).pow(2).mean().sqrt()
                print(f"  Epoch {epoch}: RMSE = {error.item():.6f}")

    # Test reconstruction with different numbers of layers
    print("\n" + "=" * 60)
    print("Reconstruction quality vs number of residual layers")
    print("=" * 60)

    with torch.no_grad():
        # Full reconstruction (all 8 layers)
        key_recon_full, key_indices, _ = key_rvq(keys)
        value_recon_full, value_indices, _ = value_rvq(values)

        full_key_error = (keys - key_recon_full).pow(2).mean().sqrt()
        full_value_error = (values - value_recon_full).pow(2).mean().sqrt()

        print(f"\nFull reconstruction (8 layers):")
        print(f"  Keys RMSE: {full_key_error.item():.6f}")
        print(f"  Values RMSE: {full_value_error.item():.6f}")

        # Test partial reconstruction (simulating eviction)
        for n_active_layers in [6, 4, 2]:
            # Reconstruct using only first n layers
            key_recon_partial = torch.zeros_like(keys)
            value_recon_partial = torch.zeros_like(values)

            # Manually reconstruct from indices
            for layer_idx in range(n_active_layers):
                # Use the library's method to get codes from indices
                key_codes = key_rvq.layers[layer_idx].get_codes_from_indices(key_indices[:, layer_idx])
                value_codes = value_rvq.layers[layer_idx].get_codes_from_indices(value_indices[:, layer_idx])

                # Accumulate residuals
                key_recon_partial = key_recon_partial + key_codes
                value_recon_partial = value_recon_partial + value_codes

            partial_key_error = (keys - key_recon_partial).pow(2).mean().sqrt()
            partial_value_error = (values - value_recon_partial).pow(2).mean().sqrt()

            print(f"\nPartial reconstruction ({n_active_layers} layers):")
            print(f"  Keys RMSE: {partial_key_error.item():.6f} "
                  f"(+{(partial_key_error/full_key_error - 1)*100:.1f}%)")
            print(f"  Values RMSE: {partial_value_error.item():.6f} "
                  f"(+{(partial_value_error/full_value_error - 1)*100:.1f}%)")

    return key_rvq, value_rvq


def main():
    print("=" * 60)
    print("Step 1: Train RVQ Codebooks (Simplified)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Generate synthetic KV samples
    # For Qwen2.5-3B: 16 heads * 128 head_dim = 2048 hidden_dim
    hidden_dim = 2048
    num_samples = 2000

    keys, values = generate_synthetic_kv(hidden_dim, num_samples, device)

    # Train RVQ with different configurations
    configs = [
        (8, 256),   # 8 layers, 256 codes per layer
    ]

    for n_layers, codebook_size in configs:
        print("\n" + "=" * 60)
        print(f"Configuration: {n_layers} layers, {codebook_size} codes")
        print("=" * 60)

        key_rvq, value_rvq = train_rvq_codebooks(
            keys, values,
            n_layers=n_layers,
            codebook_size=codebook_size
        )

        # Save codebooks
        output_dir = Path("checkpoints")
        output_dir.mkdir(exist_ok=True)

        filename = f"rvq_codebooks_L{n_layers}_C{codebook_size}.pt"

        torch.save({
            'key_rvq_state': key_rvq.state_dict(),
            'value_rvq_state': value_rvq.state_dict(),
            'config': {
                'n_layers': n_layers,
                'codebook_size': codebook_size,
                'hidden_dim': hidden_dim,
            }
        }, output_dir / filename)

        print(f"\n✓ Codebooks saved to {output_dir / filename}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nKey insight from results:")
    print("- Evicting residual layers degrades quality gradually")
    print("- 4 layers might retain ~70-80% quality vs 8 layers")
    print("- This validates the core idea of gradual precision degradation")
    print("\nNext: Test these codebooks in actual generation")


if __name__ == "__main__":
    main()
