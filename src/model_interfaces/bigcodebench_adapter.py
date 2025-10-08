#!/usr/bin/env python3
"""
BigCodeBench Adapter for AI Benchmark Suite

Provides enhanced code generation evaluation with complex function calls
and realistic programming scenarios. Better quality than HumanEval but
still has moderate contamination risk (public benchmark).

References:
- GitHub: https://github.com/bigcode-project/bigcodebench
- Paper: ICLR'25 (Accepted)
- Leaderboard: https://bigcode-bench.github.io/
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
class BigCodeBenchResult:
    """Results from BigCodeBench evaluation"""
    model: str
    subset: str  # "full" or "hard"
    total_problems: int
    pass_at_1: float
    pass_at_10: Optional[float] = None
    problems_solved: int = 0
    problems_attempted: int = 0
    execution_time: float = 0.0
    contamination_warning: str = "MODERATE - Public benchmark"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BigCodeBenchAdapter:
    """
    Adapter for integrating BigCodeBench with our unified benchmark suite.

    BigCodeBench is the "next generation of HumanEval" with:
    - Complex function calls
    - Realistic code patterns
    - Enhanced test cases (EvalPlus methodology)

    Note: Still a public benchmark, so moderate contamination risk.
    Use alongside LiveCodeBench for cross-validation.
    """

    def __init__(
        self,
        harness_dir: Path = None,
        subset: str = "full"
    ):
        """
        Initialize BigCodeBench adapter.

        Args:
            harness_dir: Path to BigCodeBench installation
            subset: "full" or "hard" (hard = more challenging problems)
        """
        if harness_dir is None:
            project_root = Path(__file__).parent.parent.parent
            harness_dir = project_root / "harnesses" / "bigcodebench"

        self.harness_dir = harness_dir
        self.subset = subset
        self.venv_python = harness_dir / ".venv" / "bin" / "python"

        # Validate setup
        if not harness_dir.exists():
            raise RuntimeError(f"BigCodeBench not found at {harness_dir}")

    def setup(self):
        """Set up BigCodeBench environment if not already done"""
        if self.venv_python.exists():
            print("✅ BigCodeBench venv already set up")
            return

        print("🔧 Setting up BigCodeBench venv...")

        # Create venv
        subprocess.run(
            ["python3", "-m", "venv", ".venv"],
            cwd=self.harness_dir,
            check=True
        )

        # Install BigCodeBench
        subprocess.run(
            [str(self.venv_python), "-m", "pip", "install", "-e", "."],
            cwd=self.harness_dir,
            check=True
        )

        print("✅ BigCodeBench setup complete")

    def run_inference(
        self,
        model_name: str,
        n_samples: int = 1,
        temperature: float = 0.0,
        backend: str = "openai",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run BigCodeBench inference (code generation).

        Args:
            model_name: Model identifier
            n_samples: Number of samples per problem (default: 1)
            temperature: Sampling temperature
            backend: API backend ("openai", "anthropic", "vllm", etc.)
            **kwargs: Additional arguments

        Returns:
            Raw inference results
        """
        if not self.venv_python.exists():
            raise RuntimeError(
                "BigCodeBench not set up. Run adapter.setup() first."
            )

        # BigCodeBench uses generate command
        cmd = [
            str(self.venv_python),
            "-m", "bigcodebench.generate",
            "--model", model_name,
            "--subset", self.subset,
            "--n_samples", str(n_samples),
            "--temperature", str(temperature),
            "--backend", backend,
        ]

        # Add custom kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        print(f"🚀 Running BigCodeBench inference for {model_name}")
        print(f"   Subset: {self.subset}, Samples: {n_samples}, Backend: {backend}")

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
                    f"BigCodeBench inference failed:\n{result.stderr}"
                )

            print(f"✅ Inference complete")
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            raise RuntimeError("BigCodeBench inference timed out after 1 hour")

    def run_evaluation(
        self,
        model_name: str,
        split: str = "complete",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run BigCodeBench evaluation (execute and score generated code).

        Args:
            model_name: Model identifier (must match inference model)
            split: Test split ("complete" or "instruct")
            **kwargs: Additional arguments

        Returns:
            Evaluation results
        """
        # BigCodeBench uses evaluate command
        cmd = [
            str(self.venv_python),
            "-m", "bigcodebench.evaluate",
            "--model", model_name,
            "--subset", self.subset,
            "--split", split,
        ]

        # Add custom kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        print(f"📊 Evaluating BigCodeBench results for {model_name}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.harness_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"BigCodeBench evaluation failed:\n{result.stderr}"
                )

            print(f"✅ Evaluation complete")
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            raise RuntimeError("BigCodeBench evaluation timed out after 30 min")

    def parse_results(
        self,
        model_name: str
    ) -> BigCodeBenchResult:
        """
        Parse BigCodeBench evaluation results.

        Args:
            model_name: Model identifier

        Returns:
            Structured results
        """
        # BigCodeBench saves results in eval_results/
        results_dir = self.harness_dir / "eval_results"

        # Find results file (pattern: {model_name}_{subset}_*.json)
        results_files = list(results_dir.glob(f"*{model_name}*{self.subset}*.json"))

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
        pass_at_1 = raw_results.get('pass@1', 0.0)
        pass_at_10 = raw_results.get('pass@10', None)
        total_problems = raw_results.get('total', 0)
        solved = int(pass_at_1 * total_problems) if total_problems > 0 else 0

        return BigCodeBenchResult(
            model=model_name,
            subset=self.subset,
            total_problems=total_problems,
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            problems_solved=solved,
            problems_attempted=total_problems,
            metadata={
                "backend": raw_results.get('backend', 'unknown'),
                "n_samples": raw_results.get('n_samples', 1),
                "temperature": raw_results.get('temperature', 0.0),
                "results_file": str(results_file),
                "contamination_note": (
                    "Public benchmark - moderate contamination risk. "
                    "Use for cross-validation with LiveCodeBench."
                )
            }
        )

    def evaluate(
        self,
        model_name: str,
        n_samples: int = 1,
        temperature: float = 0.0,
        backend: str = "openai",
        split: str = "complete",
        **kwargs
    ) -> BigCodeBenchResult:
        """
        Complete end-to-end evaluation pipeline.

        Args:
            model_name: Model identifier
            n_samples: Number of samples per problem
            temperature: Sampling temperature
            backend: API backend
            split: Test split
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
            n_samples=n_samples,
            temperature=temperature,
            backend=backend,
            **kwargs
        )

        # Run evaluation
        self.run_evaluation(model_name, split=split)

        # Parse and return results
        return self.parse_results(model_name)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="BigCodeBench Adapter")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--subset", default="full", choices=["full", "hard"], help="Problem subset")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--backend", default="openai", help="API backend")
    parser.add_argument("--split", default="complete", help="Test split")
    parser.add_argument("--setup-only", action="store_true", help="Only setup, don't evaluate")

    args = parser.parse_args()

    # Create adapter
    adapter = BigCodeBenchAdapter(subset=args.subset)

    if args.setup_only:
        adapter.setup()
        print("✅ Setup complete")
        return

    # Run evaluation
    results = adapter.evaluate(
        model_name=args.model,
        n_samples=args.n_samples,
        temperature=args.temperature,
        backend=args.backend,
        split=args.split
    )

    # Print results
    print("\n" + "=" * 80)
    print("BIGCODEBENCH RESULTS")
    print("=" * 80)
    print(json.dumps(results.to_dict(), indent=2))
    print("\n⚠️  CONTAMINATION WARNING: Public benchmark - moderate risk")
    print("   Use alongside LiveCodeBench for cross-validation")


if __name__ == "__main__":
    main()
