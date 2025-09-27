#!/usr/bin/env python3
"""
Unified AI Benchmarking Suite Runner

Orchestrates multiple evaluation harnesses (BigCode, LM-Eval) with unified
configuration and results processing.
"""

import sys
import os
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class BenchmarkResult:
    """Unified result structure across harnesses"""
    harness: str
    task: str
    model: str
    score: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_time: float

class Harness(Enum):
    """Available evaluation harnesses"""
    BIGCODE = "bigcode"
    LM_EVAL = "lm_eval"
    CUSTOM = "custom"

class UnifiedRunner:
    """Main orchestrator for the benchmarking suite"""

    def __init__(self, config_dir: Path = None):
        self.project_root = PROJECT_ROOT
        self.config_dir = config_dir or self.project_root / "config"
        self.harnesses_dir = self.project_root / "harnesses"
        self.results_dir = self.project_root / "results"

        # Load configurations
        self.models_config = self._load_config("models.yaml")
        self.suite_config = self._load_config("suite_definitions.yaml")
        self.harness_config = self._load_config("harness_mappings.yaml")

    def _load_config(self, filename: str) -> Dict:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            return {}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_harness_for_task(self, task: str) -> Harness:
        """Determine which harness to use for a given task"""
        mappings = self.harness_config.get('task_mappings', {})
        harness_name = mappings.get(task, 'bigcode')  # Default to bigcode

        try:
            return Harness(harness_name)
        except ValueError:
            return Harness.BIGCODE

    def _run_bigcode_harness(self, task: str, model: str, **kwargs) -> BenchmarkResult:
        """Run BigCode evaluation harness"""
        harness_dir = self.harnesses_dir / "bigcode-evaluation-harness"
        venv_python = harness_dir / "venv" / "bin" / "python"

        if not venv_python.exists():
            raise RuntimeError(f"BigCode harness not set up. Run setup first.")

        # Build command
        cmd = [
            str(venv_python), "main.py",
            "--model", model,
            "--tasks", task,
            "--allow_code_execution",
            "--save_generations",
            "--generation_only" if kwargs.get('generation_only', False) else "",
        ]

        # Add optional parameters
        if limit := kwargs.get('limit'):
            cmd.extend(["--limit", str(limit)])
        if n_samples := kwargs.get('n_samples'):
            cmd.extend(["--n_samples", str(n_samples)])
        if temperature := kwargs.get('temperature'):
            cmd.extend(["--temperature", str(temperature)])

        # Remove empty strings
        cmd = [c for c in cmd if c]

        # Execute
        result = subprocess.run(cmd, cwd=harness_dir, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"BigCode harness failed: {result.stderr}")

        # Parse results (simplified - would need proper JSON parsing)
        return BenchmarkResult(
            harness="bigcode",
            task=task,
            model=model,
            score=0.0,  # Would parse from actual output
            metrics={},
            metadata={"stdout": result.stdout, "stderr": result.stderr},
            execution_time=0.0
        )

    def _run_lm_eval_harness(self, task: str, model: str, **kwargs) -> BenchmarkResult:
        """Run LM-Eval harness"""
        harness_dir = self.harnesses_dir / "lm-evaluation-harness"

        # Build command - this would need proper LM-eval command structure
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model}",
            "--tasks", task,
            "--output_path", str(self.results_dir / "language_tasks"),
        ]

        # Add optional parameters
        if batch_size := kwargs.get('batch_size'):
            cmd.extend(["--batch_size", str(batch_size)])

        # Execute
        result = subprocess.run(cmd, cwd=harness_dir, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"LM-Eval harness failed: {result.stderr}")

        # Parse results (simplified)
        return BenchmarkResult(
            harness="lm_eval",
            task=task,
            model=model,
            score=0.0,  # Would parse from actual output
            metrics={},
            metadata={"stdout": result.stdout, "stderr": result.stderr},
            execution_time=0.0
        )

    def run_benchmark(self, task: str, model: str, **kwargs) -> BenchmarkResult:
        """Run a single benchmark task"""
        harness = self._get_harness_for_task(task)

        print(f"Running {task} on {model} using {harness.value} harness...")

        if harness == Harness.BIGCODE:
            return self._run_bigcode_harness(task, model, **kwargs)
        elif harness == Harness.LM_EVAL:
            return self._run_lm_eval_harness(task, model, **kwargs)
        else:
            raise ValueError(f"Unsupported harness: {harness}")

    def run_suite(self, suite_name: str, models: List[str], **kwargs) -> List[BenchmarkResult]:
        """Run a predefined benchmark suite"""
        suite_def = self.suite_config.get('suites', {}).get(suite_name)
        if not suite_def:
            raise ValueError(f"Unknown suite: {suite_name}")

        results = []
        tasks = suite_def.get('tasks', [])

        for model in models:
            for task in tasks:
                try:
                    result = self.run_benchmark(task, model, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Error running {task} on {model}: {e}")

        return results

    def setup_harnesses(self):
        """Set up evaluation harnesses"""
        print("Setting up BigCode harness...")
        self._setup_bigcode()

        print("Setting up LM-Eval harness...")
        self._setup_lm_eval()

    def _setup_bigcode(self):
        """Set up BigCode evaluation harness"""
        harness_dir = self.harnesses_dir / "bigcode-evaluation-harness"
        venv_dir = harness_dir / "venv"

        if venv_dir.exists():
            print("BigCode harness already set up")
            return

        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=harness_dir, check=True)

        # Install dependencies
        pip_path = venv_dir / "bin" / "pip"
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_path), "install", "-e", "."], cwd=harness_dir, check=True)

        print("✅ BigCode harness setup complete")

    def _setup_lm_eval(self):
        """Set up LM-Eval harness"""
        harness_dir = self.harnesses_dir / "lm-evaluation-harness"

        # Check if already installed
        try:
            import lm_eval
            print("✅ LM-Eval harness already available")
            return
        except ImportError:
            pass

        # Install LM-Eval harness
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                      cwd=harness_dir, check=True)

        print("✅ LM-Eval harness setup complete")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Unified AI Benchmarking Suite")
    parser.add_argument("--setup", action="store_true", help="Set up harnesses")
    parser.add_argument("--task", help="Single task to run")
    parser.add_argument("--suite", help="Benchmark suite to run")
    parser.add_argument("--model", help="Model to evaluate")
    parser.add_argument("--models", nargs="+", help="Multiple models to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of problems")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")

    args = parser.parse_args()

    runner = UnifiedRunner()

    if args.setup:
        runner.setup_harnesses()
        return

    if not (args.task or args.suite):
        parser.error("Must specify either --task or --suite")

    models = args.models or ([args.model] if args.model else [])
    if not models:
        parser.error("Must specify --model or --models")

    kwargs = {
        'limit': args.limit,
        'n_samples': args.n_samples,
        'temperature': args.temperature,
    }

    if args.task:
        for model in models:
            result = runner.run_benchmark(args.task, model, **kwargs)
            print(f"Result: {result}")

    elif args.suite:
        results = runner.run_suite(args.suite, models, **kwargs)
        print(f"Suite completed. {len(results)} results generated.")

if __name__ == "__main__":
    main()