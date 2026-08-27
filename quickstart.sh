#!/bin/bash
# Quick start guide for RVQ KV Cache experiments

echo "RVQ KV Cache Eviction - Quick Start"
echo "===================================="
echo ""

echo "Current status:"
echo "✓ Environment setup complete"
echo "✓ Model downloaded (Qwen2.5-3B-Instruct)"
echo "✓ Datasets ready (XSUM, CNN/DailyMail)"
echo "✓ Baseline test passed"
echo ""

echo "Next steps:"
echo ""
echo "Step 1: Train RVQ codebooks"
echo "  Run: .venv/bin/python experiments/step1_train_codebooks.py"
echo "  Time: ~5-10 minutes"
echo "  Output: checkpoints/rvq_codebooks.pt"
echo ""

echo "Step 2: Run comparison experiment (coming next)"
echo "  Compare: Baseline vs H2O vs RVQ eviction"
echo "  Metrics: Perplexity, Memory usage, Speed"
echo ""

echo "Step 3: Analyze results"
echo "  Generate plots and tables"
echo ""

echo "===================================="
echo "To start: .venv/bin/python experiments/step1_train_codebooks.py"
