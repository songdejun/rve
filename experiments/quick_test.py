"""
Quick experiment: Compare RVQ gradual eviction vs binary token eviction

Test on Qwen2.5-3B with XSUM dataset
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load Qwen model"""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    print(f"Model loaded on {device}")

    return model, tokenizer


def load_test_data(dataset_name="EdinburghNLP/xsum", num_samples=50):
    """Load test dataset"""
    print(f"Loading {dataset_name}...")

    dataset = load_dataset(dataset_name, split="test")

    # Take first num_samples
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append({
            'document': item['document'],
            'summary': item['summary'],
            'id': item.get('id', str(i))
        })

    print(f"Loaded {len(samples)} samples")
    return samples


def generate_with_baseline(model, tokenizer, prompt, max_new_tokens=128):
    """
    Baseline generation with standard KV cache
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Calculate perplexity approximation from scores
    if hasattr(outputs, 'scores') and outputs.scores:
        scores = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab]
        log_probs = torch.nn.functional.log_softmax(scores, dim=-1)

        # Get log prob of actual generated tokens
        generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        selected_log_probs = []

        for i, token_id in enumerate(generated_tokens):
            if i < len(outputs.scores):
                selected_log_probs.append(log_probs[0, i, token_id].item())

        avg_log_prob = sum(selected_log_probs) / len(selected_log_probs) if selected_log_probs else 0
        perplexity = torch.exp(torch.tensor(-avg_log_prob)).item()
    else:
        perplexity = None

    return {
        'generated_text': generated_text,
        'perplexity': perplexity,
        'num_tokens': len(generated_tokens) if 'generated_tokens' in locals() else max_new_tokens
    }


def calculate_memory_usage(model):
    """Estimate KV cache memory usage"""
    # This is a rough estimate
    # Actual implementation would need to hook into model's cache
    config = model.config

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    # Assume average sequence length
    seq_len = 512  # placeholder

    # FP16: 2 bytes per value
    bytes_per_token = num_heads * head_dim * 2 * 2  # *2 for K and V
    total_bytes = seq_len * bytes_per_token * num_layers

    return total_bytes / 1024 / 1024  # MB


def run_quick_experiment():
    """
    Quick experiment comparing baseline vs RVQ eviction

    For now, just test baseline generation to verify pipeline works
    """
    print("=" * 60)
    print("Quick Experiment: RVQ KV Cache Eviction")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load test data
    samples = load_test_data(num_samples=10)  # Start with 10 samples

    # Test baseline generation
    print("\n" + "=" * 60)
    print("Testing Baseline Generation")
    print("=" * 60)

    results = []

    for i, sample in enumerate(tqdm(samples, desc="Generating")):
        # Create prompt
        prompt = f"Summarize the following document:\n\n{sample['document']}\n\nSummary:"

        # Truncate if too long
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_length = inputs.input_ids.shape[1]

        if input_length > 1024:
            print(f"Sample {i}: Input too long ({input_length} tokens), skipping")
            continue

        # Generate
        result = generate_with_baseline(model, tokenizer, prompt, max_new_tokens=64)

        results.append({
            'sample_id': sample['id'],
            'input_length': input_length,
            'output_length': result['num_tokens'],
            'perplexity': result['perplexity'],
            'generated_summary': result['generated_text'].split("Summary:")[-1].strip()[:200],  # First 200 chars
            'reference_summary': sample['summary'][:200]
        })

        # Print first result as example
        if i == 0:
            print(f"\nExample generation:")
            print(f"Input length: {input_length} tokens")
            print(f"Generated: {results[-1]['generated_summary'][:100]}...")
            print(f"Reference: {results[-1]['reference_summary'][:100]}...")
            if result['perplexity']:
                print(f"Perplexity: {result['perplexity']:.2f}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Baseline results saved to {output_dir / 'baseline_results.json'}")

    # Summary statistics
    avg_input_len = sum(r['input_length'] for r in results) / len(results)
    avg_output_len = sum(r['output_length'] for r in results) / len(results)

    perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
    avg_ppl = sum(perplexities) / len(perplexities) if perplexities else None

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Samples processed: {len(results)}")
    print(f"Average input length: {avg_input_len:.1f} tokens")
    print(f"Average output length: {avg_output_len:.1f} tokens")
    if avg_ppl:
        print(f"Average perplexity: {avg_ppl:.2f}")

    # Memory estimate
    estimated_mem = calculate_memory_usage(model)
    print(f"Estimated KV cache memory: {estimated_mem:.1f} MB (per sequence)")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Implement RVQ cache integration with model")
    print("2. Add token eviction baseline (H2O-style)")
    print("3. Compare quality and memory usage")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_experiment()
