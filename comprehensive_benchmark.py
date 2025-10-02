#!/usr/bin/env python3
"""
Comprehensive Model Benchmarking Suite
Runs all available models through the complete test suite systematically.
"""

import sys
import time
import json
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_interfaces.adaptive_ollama_interface import AdaptiveOllamaInterface
from model_interfaces.adaptive_benchmark_adapter import AdaptiveBenchmarkAdapter


class ComprehensiveBenchmark:
    """Complete benchmarking suite for all models and test types"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get all available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                return models
            else:
                return []
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            return []

    def run_simple_test_suite(self, model_name: str) -> Dict[str, Any]:
        """Run basic coding problems test suite"""
        print(f"📝 Running Simple Test Suite for {model_name}")

        test_problems = [
            {
                "id": "add_numbers",
                "prompt": "def add_numbers(a, b):\n    \"\"\"Add two numbers and return the result\"\"\"\n    return",
                "expected_patterns": ["a + b", "a+b"],
                "category": "arithmetic"
            },
            {
                "id": "is_even",
                "prompt": "def is_even(n):\n    \"\"\"Check if a number is even\"\"\"\n    return",
                "expected_patterns": ["% 2 == 0", "%2==0", "n%2==0"],
                "category": "logic"
            },
            {
                "id": "fibonacci",
                "prompt": "def fibonacci(n):\n    \"\"\"Calculate nth fibonacci number\"\"\"\n    if n <= 1:\n        return",
                "expected_patterns": ["n", "1", "return n"],
                "category": "algorithms"
            },
            {
                "id": "max_of_three",
                "prompt": "def max_of_three(a, b, c):\n    \"\"\"Return the maximum of three numbers\"\"\"\n    return",
                "expected_patterns": ["max(", "max ("],
                "category": "comparison"
            },
            {
                "id": "reverse_string",
                "prompt": "def reverse_string(s):\n    \"\"\"Reverse a string\"\"\"\n    return",
                "expected_patterns": ["[::-1]", "s[::-1]"],
                "category": "string_manipulation"
            },
            {
                "id": "factorial",
                "prompt": "def factorial(n):\n    \"\"\"Calculate factorial of n\"\"\"\n    if n <= 1:\n        return",
                "expected_patterns": ["1", "return 1"],
                "category": "algorithms"
            },
            {
                "id": "count_vowels",
                "prompt": "def count_vowels(text):\n    \"\"\"Count vowels in text\"\"\"\n    vowels = 'aeiou'\n    count = 0\n    for char in text.lower():\n        if char in",
                "expected_patterns": ["vowels", "aeiou"],
                "category": "string_manipulation"
            },
            {
                "id": "sum_list",
                "prompt": "def sum_list(numbers):\n    \"\"\"Sum all numbers in a list\"\"\"\n    total = 0\n    for num in numbers:\n        total +=",
                "expected_patterns": ["num", " num"],
                "category": "iteration"
            },
            {
                "id": "find_max",
                "prompt": "def find_max(numbers):\n    \"\"\"Find maximum number in list\"\"\"\n    if not numbers:\n        return None\n    max_num = numbers[0]\n    for num in numbers[1:]:\n        if num >",
                "expected_patterns": ["max_num", "max_num:"],
                "category": "algorithms"
            },
            {
                "id": "calculator_class",
                "prompt": "class Calculator:\n    \"\"\"Simple calculator\"\"\"\n    def __init__(self):\n        self.",
                "expected_patterns": ["result", "value", "total"],
                "category": "classes"
            }
        ]

        try:
            adapter = AdaptiveBenchmarkAdapter(model_name, base_url=self.base_url)

            if not adapter.is_available():
                return {"error": f"Model {model_name} not available"}

            results = []
            start_time = time.time()

            for problem in test_problems:
                print(f"   Testing: {problem['id']}")

                try:
                    problem_start = time.time()
                    response = adapter.generate_with_details(problem["prompt"])
                    problem_time = time.time() - problem_start

                    # Check if completion matches expected patterns
                    completion = response.get("text", "").lower()
                    correct = any(pattern.lower() in completion for pattern in problem["expected_patterns"])

                    result = {
                        "problem_id": problem["id"],
                        "category": problem["category"],
                        "prompt": problem["prompt"],
                        "completion": response.get("text", ""),
                        "correct": correct,
                        "quality_score": response.get("quality_score", 0.0),
                        "strategy": response.get("selected_strategy", "unknown"),
                        "confidence": response.get("strategy_confidence", 0.0),
                        "execution_time": problem_time,
                        "success": response.get("success", False),
                        "expected_patterns": problem["expected_patterns"]
                    }
                    results.append(result)

                except Exception as e:
                    print(f"   ❌ Problem {problem['id']} failed: {e}")
                    results.append({
                        "problem_id": problem["id"],
                        "error": str(e),
                        "category": problem["category"]
                    })

            total_time = time.time() - start_time

            # Calculate statistics
            successful = [r for r in results if "error" not in r]
            correct = [r for r in successful if r.get("correct", False)]

            # Get adaptation stats
            adaptation_stats = adapter.get_adaptation_stats()

            summary = {
                "model": model_name,
                "test_type": "simple_suite",
                "timestamp": self.session_timestamp,
                "total_problems": len(test_problems),
                "successful_completions": len(successful),
                "correct_solutions": len(correct),
                "accuracy": len(correct) / len(successful) if successful else 0,
                "avg_quality_score": sum(r.get("quality_score", 0) for r in successful) / len(successful) if successful else 0,
                "avg_execution_time": sum(r.get("execution_time", 0) for r in successful) / len(successful) if successful else 0,
                "total_execution_time": total_time,
                "strategy_distribution": {},
                "category_performance": {},
                "adaptation_stats": adaptation_stats,
                "detailed_results": results
            }

            # Strategy distribution
            strategies = [r.get("strategy", "unknown") for r in successful]
            for strategy in set(strategies):
                summary["strategy_distribution"][strategy] = strategies.count(strategy)

            # Category performance
            categories = {}
            for result in successful:
                cat = result.get("category", "unknown")
                if cat not in categories:
                    categories[cat] = {"total": 0, "correct": 0}
                categories[cat]["total"] += 1
                if result.get("correct", False):
                    categories[cat]["correct"] += 1

            for cat, stats in categories.items():
                summary["category_performance"][cat] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "problems": stats["total"]
                }

            print(f"   ✅ Completed: {len(correct)}/{len(successful)} correct ({len(correct)/len(successful)*100:.1f}%)")

            return summary

        except Exception as e:
            return {"error": f"Simple test suite failed: {e}", "model": model_name}

    def run_humaneval_subset(self, model_name: str, num_problems: int = 20) -> Dict[str, Any]:
        """Run subset of HumanEval-style problems"""
        print(f"🧑‍💻 Running HumanEval Subset for {model_name} ({num_problems} problems)")

        humaneval_problems = [
            {
                "id": "HE_001",
                "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "expected_patterns": ["abs(", "for", "range", "len"],
                "difficulty": "medium"
            },
            {
                "id": "HE_002",
                "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "expected_patterns": ["replace", "split", "append", "result"],
                "difficulty": "hard"
            },
            {
                "id": "HE_003",
                "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
                "expected_patterns": ["int(", "number -", "% 1"],
                "difficulty": "easy"
            },
            {
                "id": "HE_004",
                "prompt": "def below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
                "expected_patterns": ["balance", "for", "if", "< 0"],
                "difficulty": "easy"
            },
            {
                "id": "HE_005",
                "prompt": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
                "expected_patterns": ["sum(", "len(", "abs(", "mean"],
                "difficulty": "medium"
            }
        ]

        # Add more problems up to num_problems
        while len(humaneval_problems) < num_problems and len(humaneval_problems) < 20:
            humaneval_problems.extend([
                {
                    "id": f"HE_{len(humaneval_problems)+1:03d}",
                    "prompt": f"def problem_{len(humaneval_problems)+1}(x):\n    \"\"\"Generated problem {len(humaneval_problems)+1}\"\"\"\n    return",
                    "expected_patterns": ["x", "return"],
                    "difficulty": "easy"
                }
            ])

        selected_problems = humaneval_problems[:num_problems]

        try:
            adapter = AdaptiveBenchmarkAdapter(model_name, base_url=self.base_url)

            results = []
            start_time = time.time()

            for problem in selected_problems:
                print(f"   Testing: {problem['id']}")

                try:
                    response = adapter.generate_with_details(problem["prompt"])
                    completion = response.get("text", "").lower()
                    correct = any(pattern.lower() in completion for pattern in problem["expected_patterns"])

                    results.append({
                        "problem_id": problem["id"],
                        "difficulty": problem["difficulty"],
                        "completion": response.get("text", ""),
                        "correct": correct,
                        "quality_score": response.get("quality_score", 0.0),
                        "strategy": response.get("selected_strategy", "unknown"),
                        "execution_time": response.get("execution_time", 0.0)
                    })

                except Exception as e:
                    results.append({"problem_id": problem["id"], "error": str(e)})

            successful = [r for r in results if "error" not in r]
            correct = [r for r in successful if r.get("correct", False)]

            return {
                "model": model_name,
                "test_type": "humaneval_subset",
                "problems_tested": len(selected_problems),
                "successful": len(successful),
                "correct": len(correct),
                "accuracy": len(correct) / len(successful) if successful else 0,
                "avg_quality": sum(r.get("quality_score", 0) for r in successful) / len(successful) if successful else 0,
                "results": results
            }

        except Exception as e:
            return {"error": f"HumanEval subset failed: {e}", "model": model_name}

    def run_domain_specific_tests(self, model_name: str) -> Dict[str, Any]:
        """Run domain-specific coding tests"""
        print(f"🌐 Running Domain-Specific Tests for {model_name}")

        domain_tests = {
            "web_development": [
                {
                    "prompt": "const App = () => {\n  return (\n    <div>\n      <h1>Hello World</h1>\n      ",
                    "expected": ["</div>", "jsx", "react"],
                    "domain": "web"
                },
                {
                    "prompt": "function fetchUserData(userId) {\n  return fetch(`/api/users/${userId}`)\n    .then(response => response.",
                    "expected": ["json()", ".json", "json"],
                    "domain": "web"
                }
            ],
            "data_science": [
                {
                    "prompt": "import pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})\nresult = df.",
                    "expected": ["groupby", "mean", "sum", "apply"],
                    "domain": "data_science"
                },
                {
                    "prompt": "import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\nplt.plot(x, y)\nplt.",
                    "expected": ["show()", "xlabel", "title", "show"],
                    "domain": "data_science"
                }
            ],
            "machine_learning": [
                {
                    "prompt": "from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nmodel = LinearRegression()\nmodel.",
                    "expected": ["fit(", "fit", "predict"],
                    "domain": "ml"
                }
            ],
            "algorithms": [
                {
                    "prompt": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right =",
                    "expected": ["mid - 1", "mid-1"],
                    "domain": "algorithms"
                }
            ],
            "database": [
                {
                    "prompt": "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > '2023-01-01'\n",
                    "expected": ["GROUP BY", "group by", "GROUP"],
                    "domain": "database"
                }
            ]
        }

        try:
            adapter = AdaptiveBenchmarkAdapter(model_name, base_url=self.base_url)

            domain_results = {}

            for domain, tests in domain_tests.items():
                print(f"   Testing domain: {domain}")
                domain_results[domain] = []

                for test in tests:
                    try:
                        response = adapter.generate_with_details(test["prompt"])
                        completion = response.get("text", "").lower()
                        correct = any(exp.lower() in completion for exp in test["expected"])

                        domain_results[domain].append({
                            "completion": response.get("text", ""),
                            "correct": correct,
                            "quality": response.get("quality_score", 0.0),
                            "strategy": response.get("selected_strategy", "unknown")
                        })

                    except Exception as e:
                        domain_results[domain].append({"error": str(e)})

            # Calculate domain performance
            domain_summary = {}
            for domain, results in domain_results.items():
                successful = [r for r in results if "error" not in r]
                correct = [r for r in successful if r.get("correct", False)]

                domain_summary[domain] = {
                    "tested": len(results),
                    "successful": len(successful),
                    "correct": len(correct),
                    "accuracy": len(correct) / len(successful) if successful else 0,
                    "avg_quality": sum(r.get("quality", 0) for r in successful) / len(successful) if successful else 0
                }

            return {
                "model": model_name,
                "test_type": "domain_specific",
                "domain_summary": domain_summary,
                "detailed_results": domain_results
            }

        except Exception as e:
            return {"error": f"Domain-specific tests failed: {e}", "model": model_name}

    def run_adaptive_learning_test(self, model_name: str, num_iterations: int = 30) -> Dict[str, Any]:
        """Test adaptive learning capabilities over multiple iterations"""
        print(f"🧠 Running Adaptive Learning Test for {model_name} ({num_iterations} iterations)")

        try:
            interface = AdaptiveOllamaInterface(model_name, base_url=self.base_url)

            test_prompts = [
                "def fibonacci(n):",
                "def factorial(n):",
                "def is_prime(n):",
                "def binary_search(arr, target):",
                "def merge_sort(arr):",
                "class Stack:",
                "def quicksort(arr):",
                "def gcd(a, b):",
                "def dfs(graph, start):",
                "def count_words(text):"
            ]

            learning_history = []
            strategy_evolution = []

            for iteration in range(num_iterations):
                prompt = test_prompts[iteration % len(test_prompts)]

                try:
                    response = interface.generate_adaptive(prompt)

                    learning_history.append({
                        "iteration": iteration + 1,
                        "prompt": prompt,
                        "strategy": response.selected_strategy,
                        "confidence": response.strategy_confidence,
                        "quality": response.quality_score,
                        "exploration": response.is_exploration,
                        "effectiveness": response.adaptation_effectiveness
                    })

                    strategy_evolution.append(response.selected_strategy)

                    if (iteration + 1) % 10 == 0:
                        print(f"   Iteration {iteration + 1}: Strategy={response.selected_strategy}, Quality={response.quality_score:.3f}")

                except Exception as e:
                    print(f"   ❌ Iteration {iteration + 1} failed: {e}")

            # Analyze learning progression
            exploration_stats = interface.bandit_selector.get_exploration_stats()
            strategy_performance = interface.bandit_selector.get_strategy_performance()
            analytics = interface.get_adaptation_analytics()

            # Calculate learning metrics
            early_quality = sum(h["quality"] for h in learning_history[:10]) / 10 if len(learning_history) >= 10 else 0
            late_quality = sum(h["quality"] for h in learning_history[-10:]) / 10 if len(learning_history) >= 10 else 0
            quality_improvement = late_quality - early_quality

            unique_strategies_used = len(set(strategy_evolution))

            return {
                "model": model_name,
                "test_type": "adaptive_learning",
                "iterations_completed": len(learning_history),
                "quality_improvement": quality_improvement,
                "early_avg_quality": early_quality,
                "late_avg_quality": late_quality,
                "strategies_used": unique_strategies_used,
                "final_exploration_rate": exploration_stats.get("exploration_rate", 0),
                "final_diversity": exploration_stats.get("recent_diversity", 0),
                "strategy_performance": strategy_performance,
                "adaptation_analytics": analytics,
                "learning_history": learning_history
            }

        except Exception as e:
            return {"error": f"Adaptive learning test failed: {e}", "model": model_name}

    def run_comprehensive_model_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Run complete evaluation suite for a single model"""
        print(f"\n🚀 COMPREHENSIVE EVALUATION: {model_name}")
        print("=" * 60)

        model_results = {
            "model": model_name,
            "timestamp": self.session_timestamp,
            "evaluation_start": time.time()
        }

        # Test 1: Simple Test Suite
        print("1️⃣ Simple Test Suite")
        model_results["simple_suite"] = self.run_simple_test_suite(model_name)

        # Test 2: HumanEval Subset
        print("\n2️⃣ HumanEval Subset")
        model_results["humaneval_subset"] = self.run_humaneval_subset(model_name, 15)

        # Test 3: Domain-Specific Tests
        print("\n3️⃣ Domain-Specific Tests")
        model_results["domain_specific"] = self.run_domain_specific_tests(model_name)

        # Test 4: Adaptive Learning
        print("\n4️⃣ Adaptive Learning Test")
        model_results["adaptive_learning"] = self.run_adaptive_learning_test(model_name, 25)

        model_results["evaluation_end"] = time.time()
        model_results["total_evaluation_time"] = model_results["evaluation_end"] - model_results["evaluation_start"]

        # Generate model summary
        model_results["summary"] = self.generate_model_summary(model_results)

        return model_results

    def generate_model_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for a model"""
        summary = {}

        # Simple suite summary
        if "simple_suite" in results and "accuracy" in results["simple_suite"]:
            summary["simple_accuracy"] = results["simple_suite"]["accuracy"]
            summary["simple_quality"] = results["simple_suite"].get("avg_quality_score", 0)

        # HumanEval summary
        if "humaneval_subset" in results and "accuracy" in results["humaneval_subset"]:
            summary["humaneval_accuracy"] = results["humaneval_subset"]["accuracy"]
            summary["humaneval_quality"] = results["humaneval_subset"].get("avg_quality", 0)

        # Domain performance
        if "domain_specific" in results and "domain_summary" in results["domain_specific"]:
            domain_accuracies = []
            for domain, stats in results["domain_specific"]["domain_summary"].items():
                domain_accuracies.append(stats.get("accuracy", 0))
            summary["avg_domain_accuracy"] = sum(domain_accuracies) / len(domain_accuracies) if domain_accuracies else 0

        # Adaptive learning
        if "adaptive_learning" in results and "quality_improvement" in results["adaptive_learning"]:
            summary["quality_improvement"] = results["adaptive_learning"]["quality_improvement"]
            summary["strategies_explored"] = results["adaptive_learning"].get("strategies_used", 0)
            summary["final_diversity"] = results["adaptive_learning"].get("final_diversity", 0)

        # Overall score (weighted average)
        scores = [
            summary.get("simple_accuracy", 0) * 0.3,
            summary.get("humaneval_accuracy", 0) * 0.4,
            summary.get("avg_domain_accuracy", 0) * 0.2,
            min(summary.get("quality_improvement", 0) * 10, 0.1) * 0.1  # Cap learning bonus
        ]
        summary["overall_score"] = sum(scores)

        return summary

    def run_all_models_comprehensive(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on all available models"""
        print("🎯 COMPREHENSIVE MODEL BENCHMARKING")
        print("=" * 50)

        models = self.get_available_models()

        if not models:
            return {
                "error": "No models available",
                "suggestion": "Start Ollama and pull models: ollama pull qwen2.5-coder:3b"
            }

        print(f"Found {len(models)} models to benchmark:")
        for model in models:
            name = model.get('name', 'unknown')
            size = model.get('size', 0) / (1024**3)
            print(f"  • {name} ({size:.1f}GB)")

        all_results = {
            "session_timestamp": self.session_timestamp,
            "total_models": len(models),
            "models_evaluated": {},
            "comparative_analysis": {}
        }

        # Evaluate each model
        for i, model in enumerate(models):
            model_name = model.get('name', f'model_{i}')
            print(f"\n📊 EVALUATING MODEL {i+1}/{len(models)}: {model_name}")

            try:
                model_results = self.run_comprehensive_model_evaluation(model_name)
                all_results["models_evaluated"][model_name] = model_results

                # Save individual model results
                model_file = self.results_dir / f"{model_name.replace(':', '_')}_{self.session_timestamp}.json"
                with open(model_file, 'w') as f:
                    json.dump(model_results, f, indent=2, default=str)

                print(f"✅ {model_name} evaluation complete")

            except Exception as e:
                print(f"❌ {model_name} evaluation failed: {e}")
                all_results["models_evaluated"][model_name] = {"error": str(e)}

        # Generate comparative analysis
        all_results["comparative_analysis"] = self.generate_comparative_analysis(all_results["models_evaluated"])

        # Save comprehensive results
        comprehensive_file = self.results_dir / f"comprehensive_benchmark_{self.session_timestamp}.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        return all_results

    def generate_comparative_analysis(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across all models"""
        comparison = {
            "model_rankings": {},
            "best_performers": {},
            "category_leaders": {},
            "insights": []
        }

        model_summaries = {}
        for model_name, results in model_results.items():
            if "summary" in results:
                model_summaries[model_name] = results["summary"]

        if not model_summaries:
            return comparison

        # Overall rankings
        ranked_models = sorted(model_summaries.items(),
                             key=lambda x: x[1].get("overall_score", 0),
                             reverse=True)

        comparison["model_rankings"] = {
            model: {"rank": i+1, "score": summary.get("overall_score", 0)}
            for i, (model, summary) in enumerate(ranked_models)
        }

        # Category leaders
        categories = ["simple_accuracy", "humaneval_accuracy", "avg_domain_accuracy", "quality_improvement"]
        for category in categories:
            best_model = max(model_summaries.items(),
                           key=lambda x: x[1].get(category, 0))
            comparison["category_leaders"][category] = {
                "model": best_model[0],
                "score": best_model[1].get(category, 0)
            }

        # Generate insights
        if len(model_summaries) > 1:
            top_model = ranked_models[0][0]
            top_score = ranked_models[0][1].get("overall_score", 0)

            comparison["insights"].append(f"Best overall performer: {top_model} (score: {top_score:.3f})")

            # Find model with best learning
            best_learner = max(model_summaries.items(),
                             key=lambda x: x[1].get("quality_improvement", 0))
            comparison["insights"].append(f"Best adaptive learner: {best_learner[0]} (improvement: {best_learner[1].get('quality_improvement', 0):.3f})")

        return comparison

    def display_results_summary(self, results: Dict[str, Any]):
        """Display formatted summary of all results"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BENCHMARKING RESULTS SUMMARY")
        print("="*80)

        if "error" in results:
            print(f"❌ {results['error']}")
            if "suggestion" in results:
                print(f"💡 {results['suggestion']}")
            return

        models_evaluated = results.get("models_evaluated", {})
        comparative = results.get("comparative_analysis", {})

        print(f"🔢 Total Models Evaluated: {len(models_evaluated)}")
        print(f"⏰ Session: {results.get('session_timestamp', 'unknown')}")

        # Model rankings
        if "model_rankings" in comparative:
            print("\n🏆 OVERALL RANKINGS:")
            for model, data in comparative["model_rankings"].items():
                rank = data["rank"]
                score = data["score"]
                print(f"  {rank}. {model}: {score:.3f}")

        # Category leaders
        if "category_leaders" in comparative:
            print("\n🎯 CATEGORY LEADERS:")
            category_names = {
                "simple_accuracy": "Simple Tests",
                "humaneval_accuracy": "HumanEval",
                "avg_domain_accuracy": "Domain-Specific",
                "quality_improvement": "Adaptive Learning"
            }

            for category, data in comparative["category_leaders"].items():
                name = category_names.get(category, category)
                model = data["model"]
                score = data["score"]
                print(f"  🥇 {name}: {model} ({score:.3f})")

        # Insights
        if "insights" in comparative:
            print("\n💡 KEY INSIGHTS:")
            for insight in comparative["insights"]:
                print(f"  • {insight}")

        # Individual model summaries
        print(f"\n📋 INDIVIDUAL MODEL PERFORMANCE:")
        for model_name, model_data in models_evaluated.items():
            if "error" in model_data:
                print(f"  ❌ {model_name}: {model_data['error']}")
            elif "summary" in model_data:
                summary = model_data["summary"]
                print(f"  📊 {model_name}:")
                print(f"     Simple Tests: {summary.get('simple_accuracy', 0):.1%}")
                print(f"     HumanEval: {summary.get('humaneval_accuracy', 0):.1%}")
                print(f"     Domain Avg: {summary.get('avg_domain_accuracy', 0):.1%}")
                print(f"     Learning: {summary.get('quality_improvement', 0):+.3f}")
                print(f"     Overall: {summary.get('overall_score', 0):.3f}")

        print(f"\n💾 Results saved to: {self.results_dir}/")
        print("="*80)


def main():
    """Main entry point for comprehensive benchmarking"""
    benchmark = ComprehensiveBenchmark()

    try:
        print("🚀 Starting Comprehensive Model Benchmarking...")

        # Check if Ollama is available
        models = benchmark.get_available_models()
        if not models:
            print("\n❌ No models available for benchmarking")
            print("To run comprehensive benchmarking:")
            print("1. Start Ollama: ollama serve")
            print("2. Pull models: ollama pull qwen2.5-coder:3b")
            print("3. Re-run this script")
            return

        # Run comprehensive evaluation
        results = benchmark.run_all_models_comprehensive()

        # Display results
        benchmark.display_results_summary(results)

    except KeyboardInterrupt:
        print("\n🛑 Benchmarking interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()