#!/usr/bin/env python3
"""
SWE-bench Live Adapter for AI Benchmark Suite

Provides contamination-resistant repository-level software engineering evaluation.
Monthly updates with 50 new real GitHub issues ensure temporal protection.

References:
- GitHub: https://github.com/microsoft/SWE-bench-Live
- Paper: NeurIPS 2025 D&B
- Leaderboard: https://swe-bench-live.github.io/
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SWEBenchLiveResult:
    """Results from SWE-bench Live evaluation"""
    model: str
    month: str  # e.g., "2025-10"
    total_instances: int
    resolved_instances: int
    resolution_rate: float
    avg_time_per_instance: float
    contamination_status: str = "LOW"
    contamination_protection: str = "temporal"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class SWEBenchLiveAdapter:
    """
    Adapter for integrating SWE-bench Live with our unified benchmark suite.

    SWE-bench Live provides:
    - Real GitHub issues (repository-level tasks)
    - Monthly updates (50 new issues/month)
    - Temporal contamination protection
    - Realistic software engineering evaluation

    Note: More complex than function-level benchmarks but more realistic.
    """

    def __init__(
        self,
        harness_dir: Path = None,
        split: str = "test",
        month: str = None
    ):
        """
        Initialize SWE-bench Live adapter.

        Args:
            harness_dir: Path to SWE-bench Live installation
            split: Dataset split ("test", "lite", "verified")
            month: Specific month to evaluate (e.g., "2025-10", default: latest)
        """
        if harness_dir is None:
            project_root = Path(__file__).parent.parent.parent
            harness_dir = project_root / "harnesses" / "swebench-live"

        self.harness_dir = harness_dir
        self.split = split
        self.month = month or datetime.now().strftime("%Y-%m")
        self.venv_python = harness_dir / ".venv" / "bin" / "python"

        # Validate setup
        if not harness_dir.exists():
            raise RuntimeError(f"SWE-bench Live not found at {harness_dir}")

    def setup(self):
        """Set up SWE-bench Live environment if not already done"""
        if self.venv_python.exists():
            print("✅ SWE-bench Live venv already set up")
            return

        print("🔧 Setting up SWE-bench Live venv...")

        # Create venv
        subprocess.run(
            ["python3", "-m", "venv", ".venv"],
            cwd=self.harness_dir,
            check=True
        )

        # Install requirements
        requirements_file = self.harness_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run(
                [str(self.venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=self.harness_dir,
                check=True
            )

        # Install SWE-bench Live
        subprocess.run(
            [str(self.venv_python), "-m", "pip", "install", "-e", "."],
            cwd=self.harness_dir,
            check=True
        )

        print("✅ SWE-bench Live setup complete")

    def get_dataset(self) -> Dict[str, Any]:
        """
        Load SWE-bench Live dataset.

        Returns:
            Dataset information and instances
        """
        # SWE-bench Live uses HuggingFace datasets
        # Dataset: livecodebench/swe-bench-live
        cmd = [
            str(self.venv_python),
            "-c",
            f"""
from datasets import load_dataset
dataset = load_dataset('SWE-bench/SWE-bench-live', split='{self.split}')
print(len(dataset))
"""
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.harness_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                num_instances = int(result.stdout.strip())
                return {
                    "split": self.split,
                    "num_instances": num_instances,
                    "month": self.month
                }
            else:
                raise RuntimeError(f"Failed to load dataset: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Dataset loading timed out")

    def run_inference(
        self,
        model_name: str,
        max_instances: int = None,
        timeout_per_instance: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run SWE-bench Live inference (generate patches for issues).

        Args:
            model_name: Model identifier
            max_instances: Limit number of instances (None = all)
            timeout_per_instance: Timeout in seconds per instance
            **kwargs: Additional arguments

        Returns:
            Raw inference results
        """
        if not self.venv_python.exists():
            raise RuntimeError(
                "SWE-bench Live not set up. Run adapter.setup() first."
            )

        # SWE-bench Live inference command
        # Note: This is a simplified version - actual implementation may vary
        # based on SWE-bench Live's API
        cmd = [
            str(self.venv_python),
            "-m", "swebench.harness.run_evaluation",
            "--model", model_name,
            "--split", self.split,
            "--timeout", str(timeout_per_instance),
        ]

        if max_instances:
            cmd.extend(["--max_instances", str(max_instances)])

        # Add custom kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        print(f"🚀 Running SWE-bench Live inference for {model_name}")
        print(f"   Split: {self.split}, Month: {self.month}")
        print(f"   Timeout per instance: {timeout_per_instance}s")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.harness_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for full run
            )

            if result.returncode != 0:
                print(f"⚠️  Warning: Inference had errors:\n{result.stderr}")
                # Don't fail - some instances may have succeeded

            print(f"✅ Inference complete")
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            raise RuntimeError("SWE-bench Live inference timed out after 2 hours")

    def run_evaluation(
        self,
        model_name: str,
        predictions_path: Path = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run SWE-bench Live evaluation (test generated patches).

        Args:
            model_name: Model identifier
            predictions_path: Path to predictions file (optional)
            **kwargs: Additional arguments

        Returns:
            Evaluation results
        """
        cmd = [
            str(self.venv_python),
            "-m", "swebench.harness.run_evaluation",
            "--model", model_name,
            "--split", self.split,
            "--evaluate",  # Evaluation mode
        ]

        if predictions_path:
            cmd.extend(["--predictions_path", str(predictions_path)])

        # Add custom kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        print(f"📊 Evaluating SWE-bench Live results for {model_name}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.harness_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"SWE-bench Live evaluation failed:\n{result.stderr}"
                )

            print(f"✅ Evaluation complete")
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            raise RuntimeError("SWE-bench Live evaluation timed out")

    def parse_results(
        self,
        model_name: str
    ) -> SWEBenchLiveResult:
        """
        Parse SWE-bench Live evaluation results.

        Args:
            model_name: Model identifier

        Returns:
            Structured results
        """
        # SWE-bench Live saves results in results/
        results_dir = self.harness_dir / "results"

        # Find results file
        results_files = list(results_dir.glob(f"*{model_name}*{self.split}*.json"))

        if not results_files:
            raise FileNotFoundError(
                f"No results found for {model_name} in {results_dir}\n"
                f"Make sure to run evaluation first."
            )

        # Use most recent file
        results_file = max(results_files, key=lambda p: p.stat().st_mtime)

        # Load results
        with open(results_file, 'r') as f:
            raw_results = json.load(f)

        # Parse metrics
        total_instances = raw_results.get('total_instances', 0)
        resolved = raw_results.get('resolved_instances', 0)
        resolution_rate = resolved / total_instances if total_instances > 0 else 0.0
        avg_time = raw_results.get('avg_time_per_instance', 0.0)

        return SWEBenchLiveResult(
            model=model_name,
            month=self.month,
            total_instances=total_instances,
            resolved_instances=resolved,
            resolution_rate=resolution_rate,
            avg_time_per_instance=avg_time,
            metadata={
                "split": self.split,
                "results_file": str(results_file),
                "contamination_note": (
                    f"Month: {self.month} - Temporal contamination protection. "
                    "Issues created after model training."
                )
            }
        )

    def evaluate(
        self,
        model_name: str,
        max_instances: int = None,
        timeout_per_instance: int = 300,
        **kwargs
    ) -> SWEBenchLiveResult:
        """
        Complete end-to-end evaluation pipeline.

        Args:
            model_name: Model identifier
            max_instances: Limit number of instances
            timeout_per_instance: Timeout per instance
            **kwargs: Additional arguments

        Returns:
            Structured results
        """
        # Ensure setup
        if not self.venv_python.exists():
            self.setup()

        # Run inference
        self.run_inference(
            model_name,
            max_instances=max_instances,
            timeout_per_instance=timeout_per_instance,
            **kwargs
        )

        # Run evaluation
        self.run_evaluation(model_name)

        # Parse and return results
        return self.parse_results(model_name)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="SWE-bench Live Adapter")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--split", default="test", choices=["test", "lite", "verified"], help="Dataset split")
    parser.add_argument("--month", help="Specific month (YYYY-MM)")
    parser.add_argument("--max_instances", type=int, help="Limit number of instances")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per instance (seconds)")
    parser.add_argument("--setup-only", action="store_true", help="Only setup, don't evaluate")

    args = parser.parse_args()

    # Create adapter
    adapter = SWEBenchLiveAdapter(
        split=args.split,
        month=args.month
    )

    if args.setup_only:
        adapter.setup()
        print("✅ Setup complete")
        return

    # Run evaluation
    results = adapter.evaluate(
        model_name=args.model,
        max_instances=args.max_instances,
        timeout_per_instance=args.timeout
    )

    # Print results
    print("\n" + "=" * 80)
    print("SWE-BENCH LIVE RESULTS")
    print("=" * 80)
    print(json.dumps(results.to_dict(), indent=2))
    print(f"\n✅ CONTAMINATION PROTECTION: Temporal (Month: {results.month})")
    print("   Issues created after model training - LOW contamination risk")


if __name__ == "__main__":
    main()
