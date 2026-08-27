"""
RVQ KV Cache Eviction Experiment

Gradual precision degradation via residual quantization
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

__all__ = ["quick_test"]
