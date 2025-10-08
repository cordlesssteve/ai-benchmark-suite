#!/usr/bin/env python3
"""
LiveCodeBench Adapter for AI Benchmark Suite

Provides contamination-resistant code generation evaluation through temporal filtering.
Problems are annotated with release dates, allowing evaluation only on problems
released after model training cutoffs.

References:
- Paper: https://arxiv.org/abs/2403.07974
- GitHub: https://github.com/LiveCodeBench/LiveCodeBench
- Leaderboard: https://livecodebench.github.io/leaderboard.html
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Model training cutoffs (add more as needed)
MODEL_CUTOFFS = {
    "gpt-4-turbo": "2023-12-31",
    "claude-3-opus": "2023-08-31",
    "claude-3-sonnet": "2023-08-31",
    "qwen2.5-coder:3b": "2024-09-30",
    "qwen2.5-coder:7b": "2024-09-30",
    "deepseek-coder:6.7b": "2024-08-31",
    "deepseek-coder:33b": "2024-08-31",
    "codellama:7b": "2024-01-31",
    "codellama:13b": "2024-01-31",
    "phi3:latest": "2024-04-30",
    "phi3.5:latest": "2024-08-31",
    "default": "2024-01-01"  # Conservative default
}


@dataclass
class LiveCodeBenchResult:
    """Results from LiveCodeBench evaluation"""
    model: str
    release_version: str
    scenario: str
    total_problems: int
    clean_problems: int
    contaminated_problems: int
    training_cutoff: str
    contamination_status: str
    pass_at_1: float
    pass_at_10: Optional[float] = None
    problems_solved: int = 0
    problems_attempted: int = 0
    execution_time: float = 0.0
    contamination_protection: str = "temporal"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class LiveCodeBenchAdapter:
    """
    Adapter for integrating LiveCodeBench with our unified benchmark suite.

    Provides temporal contamination protection by filtering problems based on
    release dates relative to model training cutoffs.
    """

    def __init__(
        self,
        harness_dir: Path = None,
        release_version: str = "release_v6",
        scenario: str = "codegeneration",
        model_cutoffs: Dict[str, str] = None
    ):
        """
        Initialize LiveCodeBench adapter.

        Args:
            harness_dir: Path to LiveCodeBench installation
            release_version: Dataset version (release_v1 through release_v6)
            scenario: Evaluation scenario (codegeneration, execution, test_generation)
            model_cutoffs: Custom training cutoff dates (default: MODEL_CUTOFFS)
        """
        if harness_dir is None:
            project_root = Path(__file__).parent.parent.parent
            harness_dir = project_root / "harnesses" / "livecodebench"

        self.harness_dir = harness_dir
        self.venv_python = harness_dir / ".venv" / "bin" / "python"
        self.release_version = release_version
        self.scenario = scenario
        self.model_cutoffs = model_cutoffs or MODEL_CUTOFFS

        # Validate setup
        if not harness_dir.exists():
            raise RuntimeError(f"LiveCodeBench not found at {harness_dir}")

        if not self.venv_python.exists():
            raise RuntimeError(
                f"LiveCodeBench venv not set up. "
                f"Run: cd {harness_dir} && uv venv --python 3.11 && "
                f"source .venv/bin/activate && uv pip install -e ."
            )

    def get_model_cutoff(self, model_name: str, override_cutoff: str = None) -> str:
        """
        Get training cutoff date for a model.

        Args:
            model_name: Model identifier
            override_cutoff: Manual cutoff override

        Returns:
            Cutoff date in YYYY-MM-DD format
        """
        if override_cutoff:
            return override_cutoff

        # Try exact match
        if model_name in self.model_cutoffs:
            return self.model_cutoffs[model_name]

        # Try base model name (remove size/version suffix)
        base_name = model_name.split(':')[0]
        if base_name in self.model_cutoffs:
            return self.model_cutoffs[base_name]

        # Default cutoff
        return self.model_cutoffs["default"]

    def run_inference(
        self,
        model_name: str,
        n_samples: int = 10,
        temperature: float = 0.2,
        cutoff_date: str = None,
        use_cache: bool = True,
        continue_existing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run LiveCodeBench inference for a model.

        Args:
            model_name: Model identifier (e.g., "qwen2.5-coder:3b")
            n_samples: Number of samples to generate per problem
            temperature: Sampling temperature
            cutoff_date: Training cutoff override
            use_cache: Cache generated outputs
            continue_existing: Resume from existing results
            **kwargs: Additional arguments for lcb_runner

        Returns:
            Raw results from LiveCodeBench
        """
        cutoff = self.get_model_cutoff(model_name, cutoff_date)

        # Build command
        cmd = [
            str(self.venv_python),
            "-m", "lcb_runner.runner.main",
            "--model", model_name,
            "--scenario", self.scenario,
            "--release_version", self.release_version,
            "--n_samples", str(n_samples),
            "--temperature", str(temperature),
        ]

        if use_cache:
            cmd.append("--use_cache")

        if continue_existing:
            cmd.append("--continue_existing")

        # Add custom kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        # Run inference
        print(f"🚀 Running LiveCodeBench inference for {model_name}")
        print(f"   Release: {self.release_version}, Scenario: {self.scenario}")
        print(f"   Cutoff: {cutoff}, Samples: {n_samples}")

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
                    f"LiveCodeBench inference failed:\n{result.stderr}"
                )

            print(f"✅ Inference complete")
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            raise RuntimeError("LiveCodeBench inference timed out after 1 hour")

    def run_evaluation(
        self,
        model_name: str,
        cutoff_date: str = None,
        **inference_kwargs
    ) -> Dict[str, Any]:
        """
        Run complete evaluation (inference + scoring).

        Args:
            model_name: Model identifier
            cutoff_date: Training cutoff override
            **inference_kwargs: Arguments for run_inference

        Returns:
            Evaluation results
        """
        # Run inference first
        self.run_inference(model_name, cutoff_date=cutoff_date, **inference_kwargs)

        # Run evaluation
        cmd = [
            str(self.venv_python),
            "-m", "lcb_runner.runner.main",
            "--model", model_name,
            "--scenario", self.scenario,
            "--release_version", self.release_version,
            "--evaluate",  # Evaluation mode
        ]

        print(f"📊 Evaluating results for {model_name}")

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
                    f"LiveCodeBench evaluation failed:\n{result.stderr}"
                )

            print(f"✅ Evaluation complete")
            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            raise RuntimeError("LiveCodeBench evaluation timed out after 30 min")

    def parse_results(
        self,
        model_name: str,
        cutoff_date: str = None
    ) -> LiveCodeBenchResult:
        """
        Parse evaluation results and apply temporal filtering.

        Args:
            model_name: Model identifier
            cutoff_date: Training cutoff override

        Returns:
            Structured results with contamination analysis
        """
        cutoff = self.get_model_cutoff(model_name, cutoff_date)

        # Find results file
        # LiveCodeBench saves results in: output/{scenario}/{model_name}/...
        results_dir = self.harness_dir / "output" / self.scenario / model_name
        results_file = results_dir / f"{self.release_version}_results.json"

        if not results_file.exists():
            raise FileNotFoundError(
                f"Results file not found: {results_file}\n"
                f"Make sure to run evaluation first."
            )

        # Load results
        with open(results_file, 'r') as f:
            raw_results = json.load(f)

        # Apply temporal filtering
        total_problems = len(raw_results.get('problems', []))
        clean_problems = []
        contaminated_problems = []

        for problem in raw_results.get('problems', []):
            release_date = problem.get('release_date', '1970-01-01')
            if release_date > cutoff:
                clean_problems.append(problem)
            else:
                contaminated_problems.append(problem)

        # Calculate metrics on clean subset
        clean_pass_1 = sum(
            1 for p in clean_problems
            if p.get('passed', False)
        ) / len(clean_problems) if clean_problems else 0.0

        # Contamination status
        contamination_risk = "LOW" if len(clean_problems) > 0 else "UNKNOWN"

        return LiveCodeBenchResult(
            model=model_name,
            release_version=self.release_version,
            scenario=self.scenario,
            total_problems=total_problems,
            clean_problems=len(clean_problems),
            contaminated_problems=len(contaminated_problems),
            training_cutoff=cutoff,
            contamination_status=contamination_risk,
            pass_at_1=clean_pass_1,
            problems_solved=sum(1 for p in clean_problems if p.get('passed', False)),
            problems_attempted=len(clean_problems),
            metadata={
                "full_set_pass_1": raw_results.get('pass@1', 0.0),
                "benchmark_release_range": raw_results.get('release_range', 'unknown'),
                "note": "clean_subset evaluated for contamination-free results"
            }
        )

    def evaluate(
        self,
        model_name: str,
        cutoff_date: str = None,
        n_samples: int = 10,
        temperature: float = 0.2,
        **kwargs
    ) -> LiveCodeBenchResult:
        """
        Complete end-to-end evaluation pipeline.

        Args:
            model_name: Model identifier
            cutoff_date: Training cutoff override
            n_samples: Number of samples per problem
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Structured results with contamination analysis
        """
        # Run inference
        self.run_inference(
            model_name,
            n_samples=n_samples,
            temperature=temperature,
            cutoff_date=cutoff_date,
            **kwargs
        )

        # Run evaluation
        self.run_evaluation(model_name, cutoff_date=cutoff_date)

        # Parse and return results
        return self.parse_results(model_name, cutoff_date)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="LiveCodeBench Adapter")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--release_version", default="release_v6", help="Dataset version")
    parser.add_argument("--scenario", default="codegeneration", help="Scenario")
    parser.add_argument("--cutoff_date", help="Training cutoff override")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")

    args = parser.parse_args()

    # Create adapter
    adapter = LiveCodeBenchAdapter(
        release_version=args.release_version,
        scenario=args.scenario
    )

    # Run evaluation
    results = adapter.evaluate(
        model_name=args.model,
        cutoff_date=args.cutoff_date,
        n_samples=args.n_samples,
        temperature=args.temperature
    )

    # Print results
    print("\n" + "=" * 80)
    print("LIVECODEBENCH RESULTS")
    print("=" * 80)
    print(json.dumps(results.to_dict(), indent=2))


if __name__ == "__main__":
    main()
