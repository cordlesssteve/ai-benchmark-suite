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

        print(f"Running {task} on {model} using {harness.value} harness with model interface...")

        # Import model interfaces
        from model_interfaces.ollama_interface import OllamaInterface
        from model_interfaces.real_bigcode_adapter import RealBigCodeAdapter
        from model_interfaces.safe_bigcode_adapter import SafeBigCodeAdapter
        from model_interfaces.real_lm_eval_adapter import RealLMEvalAdapter

        # Create model interface based on model configuration
        model_config = self._get_model_config(model)

        if model_config.get('type') == 'ollama' or model in self.models_config.get('ollama_models', {}):
            model_interface = OllamaInterface(model)

            if not model_interface.is_available():
                raise RuntimeError(f"Ollama server not available or model {model} not found")

            # Route to appropriate harness with model interface
            if harness == Harness.BIGCODE:
                # Use safe adapter for Sprint 1.1+ with security isolation
                use_safe_mode = kwargs.get('safe_mode', True)  # Default to safe mode

                if use_safe_mode:
                    # Import and create safety config
                    from model_interfaces.safe_bigcode_adapter import SafeExecutionConfig

                    safety_config_dict = kwargs.get('safety_config', {})
                    safety_config = SafeExecutionConfig(
                        max_execution_time=safety_config_dict.get('max_execution_time', 300),
                        max_memory_mb=safety_config_dict.get('max_memory_mb', 2048),
                        max_file_size_mb=safety_config_dict.get('max_file_size_mb', 100),
                        max_processes=safety_config_dict.get('max_processes', 10),
                        cleanup_on_error=safety_config_dict.get('cleanup_on_error', True)
                    )

                    adapter = SafeBigCodeAdapter(self.project_root, safety_config)
                    print("🔒 Using SafeBigCodeAdapter with security isolation")
                    print(f"   - Max execution time: {safety_config.max_execution_time}s")
                    print(f"   - Max memory: {safety_config.max_memory_mb}MB")
                else:
                    adapter = RealBigCodeAdapter(self.project_root)
                    print("⚠️ Using RealBigCodeAdapter without safety measures")

                result = adapter.run_evaluation(task, model, model_interface, **kwargs)

                if result.success:
                    # Extract primary metric from BigCode results
                    primary_score = 0.0
                    if result.metrics:
                        # Try common BigCode metric names
                        for metric_name in ['pass@1', 'pass@k', 'accuracy', 'score']:
                            if metric_name in result.metrics:
                                primary_score = result.metrics[metric_name]
                                break

                    return BenchmarkResult(
                        harness="real_bigcode",
                        task=task,
                        model=model,
                        score=primary_score,
                        metrics=result.metrics,
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "real_bigcode_harness"
                        },
                        execution_time=result.execution_time
                    )
                else:
                    return BenchmarkResult(
                        harness="real_bigcode_failed",
                        task=task,
                        model=model,
                        score=0.0,
                        metrics={"error": result.error_message},
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "real_bigcode_harness",
                            "error": result.error_message
                        },
                        execution_time=result.execution_time
                    )
            elif harness == Harness.LM_EVAL:
                adapter = RealLMEvalAdapter(self.project_root)
                result = adapter.run_evaluation(task, model, model_interface, **kwargs)

                if result.success:
                    # Extract primary metric from LM-Eval results
                    primary_score = 0.0
                    if result.metrics:
                        # Try common metric names
                        for metric_name in ['accuracy', 'acc', 'exact_match', 'score']:
                            if metric_name in result.metrics:
                                if isinstance(result.metrics[metric_name], dict):
                                    primary_score = result.metrics[metric_name].get('mean', 0.0)
                                else:
                                    primary_score = result.metrics[metric_name]
                                break

                    return BenchmarkResult(
                        harness="real_lm_eval",
                        task=task,
                        model=model,
                        score=primary_score,
                        metrics=result.metrics,
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "real_lm_eval_harness"
                        },
                        execution_time=result.execution_time
                    )
                else:
                    return BenchmarkResult(
                        harness="real_lm_eval_failed",
                        task=task,
                        model=model,
                        score=0.0,
                        metrics={"error": result.error_message},
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "real_lm_eval_harness",
                            "error": result.error_message
                        },
                        execution_time=result.execution_time
                    )
            else:
                # Fallback to direct interface for other harnesses
                test_prompt = "def fibonacci(n):\n    # Complete this function\n"
                interface_result = model_interface.generate(test_prompt, **kwargs)

                if not interface_result.success:
                    raise RuntimeError(f"Model generation failed: {interface_result.error_message}")

                return BenchmarkResult(
                    harness="ollama_direct",
                    task=task,
                    model=model,
                    score=1.0 if len(interface_result.text) > 0 else 0.0,
                    metrics={"generation_length": len(interface_result.text)},
                    metadata={"prompt": test_prompt, "response": interface_result.text},
                    execution_time=interface_result.execution_time
                )
        else:
            raise ValueError(f"Unsupported model type for model: {model}")

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        # Check in ollama_models
        if model in self.models_config.get('ollama_models', {}):
            config = self.models_config['ollama_models'][model].copy()
            config['type'] = 'ollama'
            return config

        # Check in local_models
        if model in self.models_config.get('local_models', {}):
            config = self.models_config['local_models'][model].copy()
            config['type'] = 'huggingface'
            return config

        # Check in api_models
        if model in self.models_config.get('api_models', {}):
            config = self.models_config['api_models'][model].copy()
            config['type'] = 'api'
            return config

        # Default fallback - assume ollama if not found
        return {'type': 'ollama', 'model_name': model}

    def run_suite(self, suite_name: str, models: List[str], **kwargs) -> List[BenchmarkResult]:
        """Run a predefined benchmark suite"""
        suite_def = self.suite_config.get('suites', {}).get(suite_name)
        if not suite_def:
            raise ValueError(f"Unknown suite: {suite_name}")

        results = []
        tasks = suite_def.get('tasks', [])

        print(f"\n🚀 Running {suite_name} suite with {len(models)} model(s) and {len(tasks)} task(s)")
        print("=" * 80)

        for model in models:
            print(f"\n📊 Model: {model}")
            print("-" * 40)

            model_results = []
            total_score = 0.0
            successful_tasks = 0

            for task in tasks:
                try:
                    print(f"  ⏳ {task}...", end=" ", flush=True)
                    result = self.run_benchmark(task, model, **kwargs)
                    results.append(result)
                    model_results.append(result)

                    total_score += result.score
                    successful_tasks += 1

                    print(f"✅ Score: {result.score:.3f} ({result.execution_time:.1f}s)")

                except Exception as e:
                    print(f"❌ Failed: {e}")
                    # Add failed result for completeness
                    failed_result = BenchmarkResult(
                        harness="failed",
                        task=task,
                        model=model,
                        score=0.0,
                        metrics={"error": str(e)},
                        metadata={"error": str(e)},
                        execution_time=0.0
                    )
                    results.append(failed_result)
                    model_results.append(failed_result)

            # Print model summary
            if successful_tasks > 0:
                avg_score = total_score / successful_tasks
                print(f"\n  📈 {model} Summary: {successful_tasks}/{len(tasks)} tasks completed, avg score: {avg_score:.3f}")
            else:
                print(f"\n  💀 {model} Summary: No tasks completed successfully")

        # Print overall suite summary
        self._print_suite_summary(suite_name, results)

        # Save results to JSON if requested
        if kwargs.get('save_results', True):
            self._save_results_json(suite_name, results)

        return results

    def _print_suite_summary(self, suite_name: str, results: List[BenchmarkResult]):
        """Print detailed suite summary with metrics"""
        print(f"\n📊 {suite_name.upper()} SUITE SUMMARY")
        print("=" * 80)

        # Group results by model and task
        by_model = {}
        by_task = {}

        for result in results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)

            if result.task not in by_task:
                by_task[result.task] = []
            by_task[result.task].append(result)

        # Model performance comparison
        print("\n🏆 MODEL PERFORMANCE:")
        for model, model_results in by_model.items():
            successful = [r for r in model_results if r.score > 0]
            if successful:
                avg_score = sum(r.score for r in successful) / len(successful)
                avg_time = sum(r.execution_time for r in successful) / len(successful)
                print(f"  {model:15s} | {len(successful):2d}/{len(model_results):2d} tasks | avg: {avg_score:.3f} | time: {avg_time:.1f}s")
            else:
                print(f"  {model:15s} | {0:2d}/{len(model_results):2d} tasks | avg: 0.000 | time: 0.0s")

        # Task difficulty analysis
        print("\n📋 TASK ANALYSIS:")
        for task, task_results in by_task.items():
            successful = [r for r in task_results if r.score > 0]
            if successful:
                avg_score = sum(r.score for r in successful) / len(successful)
                success_rate = len(successful) / len(task_results)
                print(f"  {task:15s} | {success_rate:.1%} success rate | avg score: {avg_score:.3f}")
            else:
                print(f"  {task:15s} | {0:.1%} success rate | avg score: 0.000")

        # Harness utilization
        print("\n🔧 HARNESS UTILIZATION:")
        harness_counts = {}
        for result in results:
            harness = result.harness
            if harness not in harness_counts:
                harness_counts[harness] = 0
            harness_counts[harness] += 1

        for harness, count in harness_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {harness:20s} | {count:3d} tasks ({percentage:.1f}%)")

        print("\n" + "=" * 80)

    def _save_results_json(self, suite_name: str, results: List[BenchmarkResult]):
        """Save results to JSON file for programmatic analysis"""
        from datetime import datetime
        import json

        # Create results directory if it doesn't exist
        results_dir = self.project_root / "results"
        results_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_name}_{timestamp}.json"
        filepath = results_dir / filename

        # Convert results to JSON-serializable format
        results_data = {
            "suite_name": suite_name,
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(results),
            "results": []
        }

        for result in results:
            results_data["results"].append({
                "harness": result.harness,
                "task": result.task,
                "model": result.model,
                "score": result.score,
                "metrics": result.metrics,
                "metadata": result.metadata,
                "execution_time": result.execution_time
            })

        # Calculate summary statistics
        by_model = {}
        for result in results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)

        summary = {}
        for model, model_results in by_model.items():
            successful = [r for r in model_results if r.score > 0]
            summary[model] = {
                "total_tasks": len(model_results),
                "successful_tasks": len(successful),
                "average_score": sum(r.score for r in successful) / len(successful) if successful else 0.0,
                "average_time": sum(r.execution_time for r in successful) / len(successful) if successful else 0.0,
                "success_rate": len(successful) / len(model_results) if model_results else 0.0
            }

        results_data["summary"] = summary

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"📁 Results saved to: {filepath}")

        return filepath

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
        venv_dir = harness_dir / "venv"

        # Check if venv exists and lm_eval is installed
        if venv_dir.exists():
            venv_python = venv_dir / "bin" / "python"
            try:
                result = subprocess.run([str(venv_python), "-c", "import lm_eval"],
                                      capture_output=True, check=True)
                print("✅ LM-Eval harness already available")
                return
            except subprocess.CalledProcessError:
                pass

        # Create virtual environment for LM-Eval
        if not venv_dir.exists():
            subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=harness_dir, check=True)

        # Install dependencies
        venv_python = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"

        # Upgrade pip
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)

        # Install LM-Eval harness in editable mode
        subprocess.run([str(pip_path), "install", "-e", "."], cwd=harness_dir, check=True)

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
    parser.add_argument("--safe-mode", action="store_true", default=True, help="Use safe execution mode (default: True)")
    parser.add_argument("--unsafe-mode", action="store_true", help="Disable safety measures (dangerous)")
    parser.add_argument("--max-execution-time", type=int, default=300, help="Max execution time per evaluation (seconds)")
    parser.add_argument("--max-memory-mb", type=int, default=2048, help="Max memory usage (MB)")
    parser.add_argument("--max-problems", type=int, default=5, help="Max problems for safety (1-5)")

    # Sprint 1.2: Enhanced testing options
    parser.add_argument("--enhanced-testing", action="store_true", help="Enable enhanced test execution with function extraction")
    parser.add_argument("--test-timeout", type=float, default=3.0, help="Per-test timeout (seconds)")
    parser.add_argument("--max-test-workers", type=int, default=4, help="Max parallel test workers")
    parser.add_argument("--no-parallel-testing", action="store_true", help="Disable parallel test execution")
    parser.add_argument("--no-detailed-reports", action="store_true", help="Disable detailed test reports")

    # Sprint 2.0: Container isolation options
    parser.add_argument("--container-isolation", action="store_true", help="Use Docker containers for maximum isolation")
    parser.add_argument("--container-image", default="python:3.11-slim", help="Docker image for container execution")
    parser.add_argument("--container-memory", default="512m", help="Container memory limit")
    parser.add_argument("--container-cpu", default="0.5", help="Container CPU limit (cores)")
    parser.add_argument("--container-network", action="store_true", help="Enable network access in containers (default: disabled)")

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

    # Determine safety mode
    safe_mode = args.safe_mode and not args.unsafe_mode
    if args.unsafe_mode:
        print("⚠️ WARNING: Running in UNSAFE mode - security measures disabled!")

    # Safety configuration
    safety_config = {
        'max_execution_time': args.max_execution_time,
        'max_memory_mb': args.max_memory_mb,
        'max_file_size_mb': 100,
        'max_processes': 10,
        'cleanup_on_error': True,
        # Sprint 1.2: Enhanced testing configuration
        'test_timeout': args.test_timeout,
        'max_test_workers': args.max_test_workers,
        'enable_function_extraction': True,
        'enable_detailed_reporting': not args.no_detailed_reports,
        'enable_parallel_testing': not args.no_parallel_testing,
        # Sprint 2.0: Container isolation configuration
        'use_container_isolation': args.container_isolation,
        'container_image': args.container_image,
        'container_memory_limit': args.container_memory,
        'container_cpu_limit': args.container_cpu,
        'container_timeout': args.max_execution_time,
        'container_network_isolation': not args.container_network
    }

    kwargs = {
        'limit': min(args.limit or 1, args.max_problems),  # Cap for safety
        'n_samples': args.n_samples,
        'temperature': args.temperature,
        'safe_mode': safe_mode,
        'safety_config': safety_config,
        'enhanced_testing': args.enhanced_testing,  # Sprint 1.2
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