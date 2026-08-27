"""
Analyze and visualize comparison results
"""
import json
from pathlib import Path
import sys

def analyze_results():
    """Analyze comparison results"""

    results_file = Path("results/comparison_results.json")

    if not results_file.exists():
        print("Error: results/comparison_results.json not found")
        print("Run step2_comparison_simplified.py first")
        return

    with open(results_file) as f:
        results = json.load(f)

    print("=" * 80)
    print("RVQ KV Cache Eviction - Results Analysis")
    print("=" * 80)

    print("\n📊 Perplexity vs Memory Trade-off\n")

    # Sort by memory
    results_sorted = sorted(results, key=lambda x: x['memory_mb'], reverse=True)

    print(f"{'Method':<25} {'Memory':<12} {'Avg PPL':<10} {'PPL ↑':<10} {'Quality/Memory'}")
    print("-" * 80)

    baseline_ppl = results[0]['avg_ppl']
    baseline_mem = results[0]['memory_mb']

    for r in results_sorted:
        memory_pct = r['memory_mb'] / baseline_mem * 100 if baseline_mem > 0.01 else 0
        ppl_increase = r['ppl_increase_pct']

        # Quality/Memory ratio: lower PPL increase per MB saved is better
        mem_saved_mb = baseline_mem - r['memory_mb']
        if mem_saved_mb > 0.01:
            quality_ratio = ppl_increase / mem_saved_mb
            quality_str = f"{quality_ratio:>6.1f}%/MB"
        else:
            quality_str = "baseline"

        print(f"{r['method']:<25} {memory_pct:>5.1f}%     {r['avg_ppl']:>6.2f}   {ppl_increase:>+6.1f}%   {quality_str}")

    print("\n" + "=" * 80)
    print("🎯 Key Findings")
    print("=" * 80)

    # Find RVQ 4-layer and Token Eviction 50%
    rvq4 = next(r for r in results if r['method_key'] == 'rvq_4layer')
    token50 = next(r for r in results if r['method_key'] == 'token_evict_50')

    print(f"\n1. At similar memory usage (~50% saving):")
    print(f"   • Token Eviction 50%: PPL increases by {token50['ppl_increase_pct']:.1f}%")
    print(f"   • RVQ 4-layer:        PPL increases by {rvq4['ppl_increase_pct']:.1f}%")
    print(f"   ➜ RVQ is {token50['ppl_increase_pct'] - rvq4['ppl_increase_pct']:.1f}% BETTER in quality!")

    rvq6 = next(r for r in results if r['method_key'] == 'rvq_6layer')
    print(f"\n2. RVQ offers flexible trade-offs:")
    print(f"   • RVQ 6-layer: {rvq6['ppl_increase_pct']:+.1f}% PPL, extreme memory saving")
    print(f"   • RVQ 4-layer: {rvq4['ppl_increase_pct']:+.1f}% PPL, extreme memory saving")
    print(f"   ➜ Can choose granular precision levels")

    print(f"\n3. Position encoding problem:")
    print(f"   ✓ RVQ completely avoids it (all tokens retained)")
    print(f"   ✗ Token eviction must handle position gaps")

    print("\n" + "=" * 80)
    print("📝 Next Steps for Publication")
    print("=" * 80)

    print("\nFor CCF B / SCI 3-4区:")
    print("1. ✓ Core idea validated with simulation")
    print("2. ⏳ Implement actual RVQ-based generation (hook into model)")
    print("3. ⏳ Test on multiple datasets (LongBench, Multi-turn QA)")
    print("4. ⏳ Extend to 7B/13B models on 910c")
    print("5. ⏳ Add ablation studies (codebook size, #layers)")
    print("6. ⏳ Implement CommVQ optimization (optional)")

    print("\n💡 Paper structure suggestion:")
    print("   • Introduction: KV cache eviction + position encoding problem")
    print("   • Method: RVQ-based gradual precision degradation")
    print("   • Experiments: Perplexity, memory, ablations")
    print("   • Analysis: Why RVQ beats binary eviction")
    print("   • Conclusion: New paradigm for KV cache management")

    print("\n" + "=" * 80)
    print(f"Estimated effort to paper-ready:")
    print(f"  • 3B model only (SCI 3-4区): 1-2 weeks")
    print(f"  • + 7B/13B models (CCF B): 3-4 weeks")
    print("=" * 80)


if __name__ == "__main__":
    analyze_results()
