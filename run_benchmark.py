#!/usr/bin/env python3
"""
AI Benchmark Suite - Main Entry Point

Simple wrapper script to run the benchmark suite.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from benchmark_runner import main

if __name__ == "__main__":
    main()