#!/usr/bin/env python3
"""
Enhanced Unified AI Benchmarking Suite Runner with Advanced Prompting

Integrates advanced prompting strategies for optimal performance with
conversational models on code completion tasks.
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
class EnhancedBenchmarkResult:
    """Enhanced result structure with prompting metadata"""
    harness: str
    task: str
    model: str
    score: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_time: float
    prompting_strategy: Optional[str] = None
    is_conversational: Optional[bool] = None

class Harness(Enum):
    """Available evaluation harnesses"""
    BIGCODE = "bigcode"
    LM_EVAL = "lm_eval"
    CUSTOM = "custom"

class EnhancedUnifiedRunner:
    """Enhanced orchestrator with advanced prompting capabilities"""

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

    def run_benchmark(self, task: str, model: str, **kwargs) -> EnhancedBenchmarkResult:
        """Run a single benchmark task with enhanced prompting"""
        harness = self._get_harness_for_task(task)

        print(f"Running {task} on {model} using {harness.value} harness with advanced prompting...")

        # Import enhanced model interfaces
        from model_interfaces.enhanced_ollama_interface import EnhancedOllamaInterface, PromptingStrategy
        from model_interfaces.real_bigcode_adapter import RealBigCodeAdapter
        from model_interfaces.safe_bigcode_adapter import SafeBigCodeAdapter
        from model_interfaces.real_lm_eval_adapter import RealLMEvalAdapter

        # Create enhanced model interface based on model configuration
        model_config = self._get_model_config(model)

        if model_config.get('type') == 'ollama' or model in self.models_config.get('ollama_models', {}):
            # Use enhanced Ollama interface with advanced prompting
            model_interface = EnhancedOllamaInterface(model)

            if not model_interface.is_available():
                raise RuntimeError(f"Ollama server not available or model {model} not found")

            # Get prompting strategy from kwargs or use auto-best
            prompting_strategy = kwargs.get('prompting_strategy', 'auto_best')

            if prompting_strategy == 'auto_best':
                strategy = None  # Will use auto-best fallback
                print(f"🧠 Using auto-best prompting strategy for {model}")
            else:
                try:
                    strategy = PromptingStrategy(prompting_strategy)
                    print(f"🧠 Using {prompting_strategy} prompting strategy")
                except ValueError:
                    strategy = None
                    print(f"⚠️ Unknown strategy '{prompting_strategy}', using auto-best")

            # Route to appropriate harness with enhanced model interface
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
                    print("🔒 Using SafeBigCodeAdapter with security isolation + advanced prompting")
                else:
                    adapter = RealBigCodeAdapter(self.project_root)
                    print("⚠️ Using RealBigCodeAdapter without safety measures + advanced prompting")

                # Create enhanced generation function that uses advanced prompting
                def enhanced_generate(prompt, **gen_kwargs):
                    """Enhanced generation function with advanced prompting"""
                    enhanced_response = model_interface.generate(prompt, strategy=strategy, **gen_kwargs)

                    # Convert to standard response format for adapter compatibility
                    from model_interfaces.ollama_interface import OllamaResponse
                    return OllamaResponse(
                        text=enhanced_response.text,
                        execution_time=enhanced_response.execution_time,
                        success=enhanced_response.success,
                        error_message=enhanced_response.error_message
                    )

                # Monkey patch the model interface for adapter compatibility
                model_interface.generate_original = model_interface.generate
                model_interface.generate = enhanced_generate

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

                    # Get prompting metadata from the last generation
                    last_response = model_interface.generate_original("test", strategy=strategy)

                    return EnhancedBenchmarkResult(
                        harness="enhanced_bigcode",
                        task=task,
                        model=model,
                        score=primary_score,
                        metrics=result.metrics,
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "enhanced_bigcode_harness",
                            "prompting_enabled": True
                        },
                        execution_time=result.execution_time,
                        prompting_strategy=last_response.prompting_strategy,
                        is_conversational=last_response.is_conversational
                    )
                else:
                    return EnhancedBenchmarkResult(
                        harness="enhanced_bigcode_failed",
                        task=task,
                        model=model,
                        score=0.0,
                        metrics={"error": result.error_message},
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "enhanced_bigcode_harness",
                            "error": result.error_message,
                            "prompting_enabled": True
                        },
                        execution_time=result.execution_time,
                        prompting_strategy=prompting_strategy if strategy else "auto_best",
                        is_conversational=None
                    )

            elif harness == Harness.LM_EVAL:
                adapter = RealLMEvalAdapter(self.project_root)

                # Create enhanced generation function for LM-Eval
                def enhanced_generate(prompt, **gen_kwargs):
                    enhanced_response = model_interface.generate(prompt, strategy=strategy, **gen_kwargs)

                    from model_interfaces.ollama_interface import OllamaResponse
                    return OllamaResponse(
                        text=enhanced_response.text,
                        execution_time=enhanced_response.execution_time,
                        success=enhanced_response.success,
                        error_message=enhanced_response.error_message
                    )

                model_interface.generate_original = model_interface.generate
                model_interface.generate = enhanced_generate

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

                    last_response = model_interface.generate_original("test", strategy=strategy)

                    return EnhancedBenchmarkResult(
                        harness="enhanced_lm_eval",
                        task=task,
                        model=model,
                        score=primary_score,
                        metrics=result.metrics,
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "enhanced_lm_eval_harness",
                            "prompting_enabled": True
                        },
                        execution_time=result.execution_time,
                        prompting_strategy=last_response.prompting_strategy,
                        is_conversational=last_response.is_conversational
                    )
                else:
                    return EnhancedBenchmarkResult(
                        harness="enhanced_lm_eval_failed",
                        task=task,
                        model=model,
                        score=0.0,
                        metrics={"error": result.error_message},
                        metadata={
                            "raw_output": result.raw_output,
                            "evaluation_type": "enhanced_lm_eval_harness",
                            "error": result.error_message,
                            "prompting_enabled": True
                        },
                        execution_time=result.execution_time,
                        prompting_strategy=prompting_strategy if strategy else "auto_best",
                        is_conversational=None
                    )
            else:
                # Direct interface test with advanced prompting
                test_prompt = "def fibonacci(n):\n    # Complete this function\n"
                enhanced_response = model_interface.generate(test_prompt, strategy=strategy, **kwargs)

                if not enhanced_response.success:
                    raise RuntimeError(f"Model generation failed: {enhanced_response.error_message}")

                return EnhancedBenchmarkResult(
                    harness="enhanced_ollama_direct",
                    task=task,
                    model=model,
                    score=1.0 if len(enhanced_response.text) > 0 and not enhanced_response.is_conversational else 0.0,
                    metrics={
                        "generation_length": len(enhanced_response.text),
                        "is_conversational": enhanced_response.is_conversational
                    },
                    metadata={
                        "prompt": test_prompt,
                        "response": enhanced_response.text,
                        "raw_response": enhanced_response.raw_response,
                        "prompting_enabled": True
                    },
                    execution_time=enhanced_response.execution_time,
                    prompting_strategy=enhanced_response.prompting_strategy,
                    is_conversational=enhanced_response.is_conversational
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

    def run_suite(self, suite_name: str, models: List[str], **kwargs) -> List[EnhancedBenchmarkResult]:
        """Run a predefined benchmark suite with enhanced prompting"""
        suite_def = self.suite_config.get('suites', {}).get(suite_name)
        if not suite_def:
            raise ValueError(f"Unknown suite: {suite_name}")

        results = []
        tasks = suite_def.get('tasks', [])

        print(f"\n🚀 Running {suite_name} suite with ENHANCED PROMPTING")
        print(f"📊 Models: {len(models)}, Tasks: {len(tasks)}")
        print("🧠 Advanced prompting strategies enabled for optimal performance")
        print("=" * 80)

        for model in models:
            print(f"\n📊 Model: {model}")
            print("-" * 40)

            model_results = []
            total_score = 0.0
            successful_tasks = 0
            conversational_responses = 0

            for task in tasks:
                try:
                    print(f"  ⏳ {task}...", end=" ", flush=True)
                    result = self.run_benchmark(task, model, **kwargs)
                    results.append(result)
                    model_results.append(result)

                    total_score += result.score
                    successful_tasks += 1

                    # Track conversational responses
                    if result.is_conversational:
                        conversational_responses += 1

                    conv_indicator = "🗣️" if result.is_conversational else "🤖"
                    strategy_info = f"({result.prompting_strategy})" if result.prompting_strategy else ""

                    print(f"✅ {conv_indicator} Score: {result.score:.3f} {strategy_info} ({result.execution_time:.1f}s)")

                except Exception as e:
                    print(f"❌ Failed: {e}")
                    # Add failed result for completeness
                    failed_result = EnhancedBenchmarkResult(
                        harness="failed",
                        task=task,
                        model=model,
                        score=0.0,
                        metrics={"error": str(e)},
                        metadata={"error": str(e), "prompting_enabled": True},
                        execution_time=0.0,
                        prompting_strategy="unknown",
                        is_conversational=None
                    )
                    results.append(failed_result)
                    model_results.append(failed_result)

            # Print enhanced model summary
            if successful_tasks > 0:
                avg_score = total_score / successful_tasks
                conv_rate = (conversational_responses / successful_tasks) * 100
                print(f"\n  📈 {model} Enhanced Summary:")
                print(f"     Tasks: {successful_tasks}/{len(tasks)} completed")
                print(f"     Avg Score: {avg_score:.3f}")
                print(f"     Conversational Rate: {conv_rate:.1f}% (lower is better)")
            else:
                print(f"\n  💀 {model} Summary: No tasks completed successfully")

        # Print enhanced suite summary
        self._print_enhanced_suite_summary(suite_name, results)

        # Save enhanced results to JSON
        if kwargs.get('save_results', True):
            self._save_enhanced_results_json(suite_name, results)

        return results

    def _print_enhanced_suite_summary(self, suite_name: str, results: List[EnhancedBenchmarkResult]):
        """Print enhanced suite summary with prompting analytics"""
        print(f"\n📊 {suite_name.upper()} ENHANCED SUITE SUMMARY")
        print("=" * 80)

        # Group results by model and task
        by_model = {}
        by_task = {}
        by_strategy = {}

        for result in results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)

            if result.task not in by_task:
                by_task[result.task] = []
            by_task[result.task].append(result)

            if result.prompting_strategy:
                if result.prompting_strategy not in by_strategy:
                    by_strategy[result.prompting_strategy] = []
                by_strategy[result.prompting_strategy].append(result)

        # Enhanced model performance comparison
        print("\n🏆 ENHANCED MODEL PERFORMANCE:")
        for model, model_results in by_model.items():
            successful = [r for r in model_results if r.score > 0]
            conversational = [r for r in successful if r.is_conversational]

            if successful:
                avg_score = sum(r.score for r in successful) / len(successful)
                avg_time = sum(r.execution_time for r in successful) / len(successful)
                conv_rate = (len(conversational) / len(successful)) * 100
                print(f"  {model:15s} | {len(successful):2d}/{len(model_results):2d} tasks | avg: {avg_score:.3f} | conv: {conv_rate:4.1f}% | time: {avg_time:.1f}s")
            else:
                print(f"  {model:15s} | {0:2d}/{len(model_results):2d} tasks | avg: 0.000 | conv: -.--% | time: 0.0s")

        # Prompting strategy effectiveness
        if by_strategy:
            print("\n🧠 PROMPTING STRATEGY EFFECTIVENESS:")
            for strategy, strategy_results in by_strategy.items():
                successful = [r for r in strategy_results if r.score > 0]
                conversational = [r for r in successful if r.is_conversational]

                if successful:
                    avg_score = sum(r.score for r in successful) / len(successful)
                    conv_rate = (len(conversational) / len(successful)) * 100
                    usage_count = len(strategy_results)
                    print(f"  {strategy:15s} | {usage_count:3d} uses | avg: {avg_score:.3f} | conv: {conv_rate:4.1f}%")

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

        print("\n" + "=" * 80)

    def _save_enhanced_results_json(self, suite_name: str, results: List[EnhancedBenchmarkResult]):
        """Save enhanced results with prompting metadata to JSON file"""
        from datetime import datetime
        import json

        # Create results directory if it doesn't exist
        results_dir = self.project_root / "results"
        results_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_name}_enhanced_{timestamp}.json"
        filepath = results_dir / filename

        # Convert results to JSON-serializable format
        results_data = {
            "suite_name": suite_name,
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(results),
            "enhanced_prompting_enabled": True,
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
                "execution_time": result.execution_time,
                "prompting_strategy": result.prompting_strategy,
                "is_conversational": result.is_conversational
            })

        # Calculate enhanced summary statistics
        by_model = {}
        strategy_stats = {}

        for result in results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)

            if result.prompting_strategy:
                if result.prompting_strategy not in strategy_stats:
                    strategy_stats[result.prompting_strategy] = {"total": 0, "successful": 0, "conversational": 0}

                strategy_stats[result.prompting_strategy]["total"] += 1
                if result.score > 0:
                    strategy_stats[result.prompting_strategy]["successful"] += 1
                if result.is_conversational:
                    strategy_stats[result.prompting_strategy]["conversational"] += 1

        summary = {}
        for model, model_results in by_model.items():
            successful = [r for r in model_results if r.score > 0]
            conversational = [r for r in successful if r.is_conversational]

            summary[model] = {
                "total_tasks": len(model_results),
                "successful_tasks": len(successful),
                "average_score": sum(r.score for r in successful) / len(successful) if successful else 0.0,
                "average_time": sum(r.execution_time for r in successful) / len(successful) if successful else 0.0,
                "success_rate": len(successful) / len(model_results) if model_results else 0.0,
                "conversational_rate": len(conversational) / len(successful) if successful else 0.0
            }

        results_data["summary"] = summary
        results_data["strategy_stats"] = strategy_stats

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"📁 Enhanced results saved to: {filepath}")

        return filepath

def main():
    """Enhanced command-line interface with prompting options"""
    parser = argparse.ArgumentParser(description="Enhanced Unified AI Benchmarking Suite with Advanced Prompting")
    parser.add_argument("--task", help="Single task to run")
    parser.add_argument("--suite", help="Benchmark suite to run")
    parser.add_argument("--model", help="Model to evaluate")
    parser.add_argument("--models", nargs="+", help="Multiple models to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of problems")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")

    # Enhanced prompting options
    parser.add_argument("--prompting-strategy",
                       choices=["auto_best", "code_engine", "silent_generator", "deterministic",
                               "negative_prompt", "format_constraint", "role_based"],
                       default="auto_best",
                       help="Prompting strategy to use (default: auto_best)")

    # Standard options
    parser.add_argument("--safe-mode", action="store_true", default=True, help="Use safe execution mode (default: True)")
    parser.add_argument("--unsafe-mode", action="store_true", help="Disable safety measures (dangerous)")
    parser.add_argument("--max-execution-time", type=int, default=300, help="Max execution time per evaluation (seconds)")
    parser.add_argument("--max-memory-mb", type=int, default=2048, help="Max memory usage (MB)")
    parser.add_argument("--max-problems", type=int, default=5, help="Max problems for safety (1-5)")

    args = parser.parse_args()

    runner = EnhancedUnifiedRunner()

    if not (args.task or args.suite):
        parser.error("Must specify either --task or --suite")

    models = args.models or ([args.model] if args.model else [])
    if not models:
        parser.error("Must specify --model or --models")

    # Safety configuration
    safe_mode = args.safe_mode and not args.unsafe_mode
    if args.unsafe_mode:
        print("⚠️ WARNING: Running in UNSAFE mode - security measures disabled!")

    safety_config = {
        'max_execution_time': args.max_execution_time,
        'max_memory_mb': args.max_memory_mb,
        'max_file_size_mb': 100,
        'max_processes': 10,
        'cleanup_on_error': True,
    }

    kwargs = {
        'limit': min(args.limit or 1, args.max_problems),
        'n_samples': args.n_samples,
        'temperature': args.temperature,
        'safe_mode': safe_mode,
        'safety_config': safety_config,
        'prompting_strategy': args.prompting_strategy,  # Enhanced prompting
    }

    print(f"🧠 Enhanced Prompting Strategy: {args.prompting_strategy}")
    print("=" * 60)

    if args.task:
        for model in models:
            result = runner.run_benchmark(args.task, model, **kwargs)
            print(f"Enhanced Result: {result}")

    elif args.suite:
        results = runner.run_suite(args.suite, models, **kwargs)
        print(f"Enhanced suite completed. {len(results)} results generated.")

if __name__ == "__main__":
    main()