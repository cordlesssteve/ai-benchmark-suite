#!/usr/bin/env python3
"""
Adaptive Ollama CLI
Production-ready command-line interface for the Phase 2 adaptive prompting system.

Usage:
    python adaptive_ollama_cli.py generate "def fibonacci(n):" --model qwen2.5-coder:3b
    python adaptive_ollama_cli.py benchmark --model phi3:latest --problems 10
    python adaptive_ollama_cli.py status --model deepseek-coder:6.7b
"""

import argparse
import json
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_interfaces.adaptive_ollama_interface import AdaptiveOllamaInterface
from prompting.bandit_strategy_selector import PromptingStrategy


class AdaptiveOllamaCLI:
    """Production CLI for adaptive Ollama interface"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize CLI with optional configuration file"""
        self.config = self._load_config(config_path)
        self.interfaces: Dict[str, AdaptiveOllamaInterface] = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 120,
                "quality_threshold": 0.6
            },
            "bandit": {
                "exploration_alpha": 2.0,
                "target_exploration_rate": 0.3,
                "target_diversity": 0.6
            },
            "logging": {
                "level": "INFO",
                "file": None
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)

            # Merge configs
            default_config.update(user_config)

        return default_config

    def _get_interface(self, model_name: str) -> AdaptiveOllamaInterface:
        """Get or create interface for model"""
        if model_name not in self.interfaces:
            self.interfaces[model_name] = AdaptiveOllamaInterface(
                base_url=self.config["ollama"]["base_url"],
                model_name=model_name,
                quality_threshold=self.config["ollama"]["quality_threshold"],
                exploration_alpha=self.config["bandit"]["exploration_alpha"]
            )

        return self.interfaces[model_name]

    def generate(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate single completion with adaptive strategy selection"""
        print(f"🧠 Generating with adaptive strategy selection...")
        print(f"📋 Model: {model}")
        print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        try:
            interface = self._get_interface(model)

            # Check model availability
            if not interface.is_available():
                return {
                    "error": f"Model {model} not available. Is Ollama running?",
                    "suggestion": f"Run: ollama pull {model}"
                }

            # Generate with adaptation
            start_time = time.time()
            response = interface.generate_adaptive(prompt, **kwargs)
            total_time = time.time() - start_time

            # Get adaptation insights
            analytics = interface.get_adaptation_analytics()

            result = {
                "model": model,
                "prompt": prompt,
                "completion": response.text,
                "raw_completion": response.raw_text,
                "execution_time": total_time,
                "http_success": response.http_success,
                "content_quality_success": response.content_quality_success,
                "overall_success": response.overall_success,
                "quality_score": response.quality_score,
                "selected_strategy": response.selected_strategy.value,
                "strategy_confidence": response.strategy_confidence,
                "predicted_reward": response.predicted_reward,
                "is_exploration": response.is_exploration,
                "adaptation_analytics": analytics
            }

            # Display results
            print(f"\n✅ Generation completed in {total_time:.2f}s")
            print(f"🎯 Strategy: {response.selected_strategy.value} (confidence: {response.strategy_confidence:.3f})")
            print(f"📊 Quality Score: {response.quality_score:.3f}")
            print(f"🔍 Exploration: {'Yes' if response.is_exploration else 'No'}")
            print(f"\n📄 Completion:")
            print("-" * 50)
            print(response.text)
            print("-" * 50)

            return result

        except Exception as e:
            error_result = {
                "error": str(e),
                "model": model,
                "prompt": prompt
            }
            print(f"❌ Error: {e}")
            return error_result

    def benchmark(self, model: str, problems: int = 5, dataset: str = "simple") -> Dict[str, Any]:
        """Run benchmark with adaptive strategies"""
        print(f"🏃 Running adaptive benchmark...")
        print(f"📋 Model: {model}")
        print(f"🔢 Problems: {problems}")
        print(f"📊 Dataset: {dataset}")

        # Simple test problems
        test_problems = [
            {
                "id": "add_numbers",
                "prompt": "def add_numbers(a, b):\n    \"\"\"Add two numbers and return the result\"\"\"\n    return",
                "expected_pattern": "a + b"
            },
            {
                "id": "is_even",
                "prompt": "def is_even(n):\n    \"\"\"Check if a number is even\"\"\"\n    return",
                "expected_pattern": "% 2 == 0"
            },
            {
                "id": "max_of_three",
                "prompt": "def max_of_three(a, b, c):\n    \"\"\"Return the maximum of three numbers\"\"\"\n    return",
                "expected_pattern": "max("
            },
            {
                "id": "reverse_string",
                "prompt": "def reverse_string(s):\n    \"\"\"Reverse a string\"\"\"\n    return",
                "expected_pattern": "[::-1]"
            },
            {
                "id": "count_vowels",
                "prompt": "def count_vowels(text):\n    \"\"\"Count vowels in a string\"\"\"\n    count = 0\n    for char in text.lower():\n        if char in",
                "expected_pattern": "aeiou"
            }
        ]

        interface = self._get_interface(model)
        results = []

        for i, problem in enumerate(test_problems[:problems]):
            print(f"\n📝 Problem {i+1}/{problems}: {problem['id']}")

            try:
                response = interface.generate_adaptive(problem["prompt"])

                # Simple pattern matching for correctness
                correct = problem["expected_pattern"] in response.text.lower()

                result = {
                    "problem_id": problem["id"],
                    "prompt": problem["prompt"],
                    "completion": response.text,
                    "correct": correct,
                    "quality_score": response.quality_score,
                    "strategy": response.selected_strategy.value,
                    "confidence": response.strategy_confidence,
                    "execution_time": response.execution_time
                }

                results.append(result)

                print(f"   ✅ Strategy: {response.selected_strategy.value}")
                print(f"   📊 Quality: {response.quality_score:.3f}")
                print(f"   🎯 Correct: {'Yes' if correct else 'No'}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "problem_id": problem["id"],
                    "error": str(e)
                })

        # Calculate statistics
        successful = [r for r in results if 'error' not in r]
        correct = [r for r in successful if r['correct']]

        summary = {
            "model": model,
            "total_problems": problems,
            "successful_completions": len(successful),
            "correct_solutions": len(correct),
            "accuracy": len(correct) / len(successful) if successful else 0,
            "avg_quality_score": sum(r['quality_score'] for r in successful) / len(successful) if successful else 0,
            "avg_execution_time": sum(r['execution_time'] for r in successful) / len(successful) if successful else 0,
            "strategy_distribution": {},
            "results": results
        }

        # Strategy distribution
        strategies = [r['strategy'] for r in successful]
        for strategy in set(strategies):
            summary["strategy_distribution"][strategy] = strategies.count(strategy)

        print(f"\n📈 BENCHMARK RESULTS:")
        print(f"   Accuracy: {summary['accuracy']:.1%}")
        print(f"   Avg Quality: {summary['avg_quality_score']:.3f}")
        print(f"   Avg Time: {summary['avg_execution_time']:.2f}s")
        print(f"   Strategy Distribution: {summary['strategy_distribution']}")

        return summary

    def status(self, model: str) -> Dict[str, Any]:
        """Show model and bandit status"""
        print(f"🔍 Checking status for {model}...")

        interface = self._get_interface(model)

        # Model availability
        available = interface.is_available()
        model_info = interface.get_model_info() if available else {}

        # Bandit statistics
        bandit_stats = interface.bandit_selector.get_exploration_stats()
        performance = interface.bandit_selector.get_strategy_performance()

        # Analytics
        analytics = interface.get_adaptation_analytics()

        status_info = {
            "model": model,
            "available": available,
            "model_info": model_info,
            "bandit_stats": bandit_stats,
            "strategy_performance": performance,
            "adaptation_analytics": analytics
        }

        print(f"📋 Model: {model}")
        print(f"✅ Available: {available}")
        print(f"🎯 Total Requests: {analytics.get('total_requests', 0)}")
        print(f"🔍 Exploration Rate: {bandit_stats.get('exploration_rate', 0):.2f}")
        print(f"🌟 Strategy Diversity: {bandit_stats.get('recent_diversity', 0):.2f}")

        print(f"\n📊 Strategy Performance:")
        for strategy, perf in performance.items():
            if perf['total_trials'] > 0:
                print(f"   {strategy}: {perf['total_trials']} trials, "
                      f"quality={perf['mean_reward']:.3f}, "
                      f"success={perf['success_rate']:.1%}")

        return status_info


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Adaptive Ollama CLI")
    parser.add_argument("--config", help="Configuration file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate single completion")
    gen_parser.add_argument("prompt", help="Prompt to complete")
    gen_parser.add_argument("--model", required=True, help="Ollama model name")
    gen_parser.add_argument("--output", help="Output file for results")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--model", required=True, help="Ollama model name")
    bench_parser.add_argument("--problems", type=int, default=5, help="Number of problems")
    bench_parser.add_argument("--dataset", default="simple", help="Dataset to use")
    bench_parser.add_argument("--output", help="Output file for results")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show model status")
    status_parser.add_argument("--model", required=True, help="Ollama model name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI instance
    cli = AdaptiveOllamaCLI(config_path=args.config)

    # Execute command
    try:
        if args.command == "generate":
            result = cli.generate(args.prompt, args.model)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)

        elif args.command == "benchmark":
            result = cli.benchmark(args.model, args.problems, args.dataset)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)

        elif args.command == "status":
            result = cli.status(args.model)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()