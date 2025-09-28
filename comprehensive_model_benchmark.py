#!/usr/bin/env python3
"""
Comprehensive Model Benchmark

Tests ALL currently loaded Ollama models with our fixed enhanced prompting
to provide actual performance data in a clear table format.
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface, PromptingStrategy

# Comprehensive test cases for code completion
BENCHMARK_TESTS = [
    {
        "name": "simple_add",
        "prompt": "def add(a, b):\n    return ",
        "expected": ["a + b", "a+b"],
        "category": "Basic Arithmetic"
    },
    {
        "name": "even_check",
        "prompt": "def is_even(n):\n    return ",
        "expected": ["n % 2 == 0", "n%2==0", "not n % 2"],
        "category": "Boolean Logic"
    },
    {
        "name": "max_function",
        "prompt": "def max_three(a, b, c):\n    return ",
        "expected": ["max(a, b, c)", "max([a, b, c])"],
        "category": "Built-in Functions"
    },
    {
        "name": "list_sum",
        "prompt": "def sum_list(numbers):\n    return ",
        "expected": ["sum(numbers)"],
        "category": "List Operations"
    },
    {
        "name": "string_length",
        "prompt": "def get_length(text):\n    return ",
        "expected": ["len(text)", "text.__len__()"],
        "category": "String Operations"
    },
    {
        "name": "factorial",
        "prompt": "def factorial(n):\n    if n <= 1:\n        return 1\n    return ",
        "expected": ["n * factorial(n - 1)", "n * factorial(n-1)"],
        "category": "Recursion"
    },
    {
        "name": "list_append",
        "prompt": "def add_item(lst, item):\n    lst.",
        "expected": ["append(item)", "append"],
        "category": "Method Calls"
    },
    {
        "name": "dict_access",
        "prompt": "def get_value(data, key):\n    return data",
        "expected": ["[key]", ".get(key)"],
        "category": "Dict Operations"
    }
]

def check_contains_expected(response: str, expected_list: List[str]) -> bool:
    """Check if response contains any expected completion"""
    if not response:
        return False

    response_clean = response.replace(" ", "").lower()
    return any(exp.replace(" ", "").lower() in response_clean for exp in expected_list)

def benchmark_model(model_name: str) -> Dict[str, Any]:
    """Benchmark a single model across all test cases"""
    print(f"\n🧪 Benchmarking {model_name}...")

    interface = FixedEnhancedOllamaInterface(model_name)

    if not interface.is_available():
        return {
            "model": model_name,
            "status": "unavailable",
            "error": "Ollama server not available"
        }

    results = []
    total_time = 0
    successful_tests = 0
    conversational_responses = 0

    for i, test in enumerate(BENCHMARK_TESTS):
        print(f"   {i+1}/{len(BENCHMARK_TESTS)} {test['name']}...", end=" ")

        try:
            start_time = time.time()
            response = interface.generate_auto_best(test['prompt'], max_tokens=100)
            execution_time = time.time() - start_time

            total_time += execution_time

            # Evaluate response
            has_content = len(response.text.strip()) > 0
            contains_expected = check_contains_expected(response.text, test['expected'])
            is_successful = response.success and has_content and contains_expected and not response.is_conversational

            if is_successful:
                successful_tests += 1

            if response.is_conversational:
                conversational_responses += 1

            results.append({
                "test_name": test['name'],
                "category": test['category'],
                "success": response.success,
                "has_content": has_content,
                "contains_expected": contains_expected,
                "is_conversational": response.is_conversational,
                "execution_time": execution_time,
                "strategy": response.prompting_strategy,
                "response": response.text[:50] + "..." if len(response.text) > 50 else response.text,
                "overall_success": is_successful
            })

            status = "✅" if is_successful else "❌"
            conv_indicator = "🗣️" if response.is_conversational else "🤖"
            print(f"{status} {conv_indicator} ({execution_time:.2f}s)")

        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}...")
            results.append({
                "test_name": test['name'],
                "category": test['category'],
                "success": False,
                "has_content": False,
                "contains_expected": False,
                "is_conversational": False,
                "execution_time": 0,
                "strategy": "error",
                "response": f"Error: {str(e)}",
                "overall_success": False
            })

    # Calculate summary metrics
    total_tests = len(BENCHMARK_TESTS)
    avg_time = total_time / total_tests if total_tests > 0 else 0
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    conversational_rate = (conversational_responses / total_tests) * 100 if total_tests > 0 else 0

    return {
        "model": model_name,
        "status": "tested",
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": success_rate,
        "conversational_responses": conversational_responses,
        "conversational_rate": conversational_rate,
        "avg_execution_time": avg_time,
        "total_time": total_time,
        "results": results
    }

def format_performance_table(benchmark_results: List[Dict[str, Any]]) -> str:
    """Format results into a comprehensive table"""

    # Filter out unavailable models
    available_results = [r for r in benchmark_results if r['status'] == 'tested']
    unavailable_models = [r for r in benchmark_results if r['status'] == 'unavailable']

    if not available_results:
        return "❌ No models available for testing"

    # Create main performance table
    table = "\n" + "="*120 + "\n"
    table += "🚀 COMPREHENSIVE MODEL PERFORMANCE BENCHMARK\n"
    table += "="*120 + "\n"

    # Header
    table += f"{'Model':<25} {'Size':<8} {'Success':<8} {'Conv%':<6} {'AvgTime':<8} {'Strategy':<15} {'Status':<10}\n"
    table += "-"*120 + "\n"

    # Model sizes (from ollama list)
    model_sizes = {
        "phi3.5:latest": "2.2GB",
        "tinyllama:1.1b": "637MB",
        "qwen2.5:0.5b": "397MB",
        "qwen2.5-coder:3b": "1.9GB",
        "codellama:13b-instruct": "7.4GB",
        "mistral:7b-instruct": "4.4GB",
        "phi3:latest": "2.2GB"
    }

    # Sort by success rate (descending)
    available_results.sort(key=lambda x: x['success_rate'], reverse=True)

    for result in available_results:
        model = result['model']
        size = model_sizes.get(model, "Unknown")
        success_rate = f"{result['success_rate']:.1f}%"
        conv_rate = f"{result['conversational_rate']:.1f}%"
        avg_time = f"{result['avg_execution_time']:.2f}s"

        # Get most common strategy
        strategies = [r['strategy'] for r in result['results'] if r['strategy'] != 'error']
        common_strategy = max(set(strategies), key=strategies.count) if strategies else "error"

        # Status indicator
        if result['success_rate'] >= 75:
            status = "🟢 Excellent"
        elif result['success_rate'] >= 50:
            status = "🟡 Good"
        elif result['success_rate'] >= 25:
            status = "🟠 Fair"
        else:
            status = "🔴 Poor"

        table += f"{model:<25} {size:<8} {success_rate:<8} {conv_rate:<6} {avg_time:<8} {common_strategy:<15} {status:<10}\n"

    # Add detailed breakdown
    table += "\n" + "="*120 + "\n"
    table += "📊 DETAILED PERFORMANCE BREAKDOWN\n"
    table += "="*120 + "\n"

    for result in available_results[:3]:  # Top 3 performers
        model = result['model']
        table += f"\n🏆 {model} - Detailed Results:\n"
        table += f"{'Test':<20} {'Category':<15} {'Success':<8} {'Conv':<5} {'Time':<7} {'Strategy':<12}\n"
        table += "-"*80 + "\n"

        for test_result in result['results']:
            test_name = test_result['test_name'][:18]
            category = test_result['category'][:13]
            success = "✅" if test_result['overall_success'] else "❌"
            conv = "🗣️" if test_result['is_conversational'] else "🤖"
            time_str = f"{test_result['execution_time']:.2f}s"
            strategy = test_result['strategy'][:10]

            table += f"{test_name:<20} {category:<15} {success:<8} {conv:<5} {time_str:<7} {strategy:<12}\n"

    # Summary statistics
    table += "\n" + "="*120 + "\n"
    table += "📈 SUMMARY STATISTICS\n"
    table += "="*120 + "\n"

    if available_results:
        best_model = available_results[0]
        worst_model = available_results[-1]
        avg_success = sum(r['success_rate'] for r in available_results) / len(available_results)
        avg_conv = sum(r['conversational_rate'] for r in available_results) / len(available_results)

        table += f"🥇 Best Performer: {best_model['model']} ({best_model['success_rate']:.1f}% success)\n"
        table += f"🥉 Needs Improvement: {worst_model['model']} ({worst_model['success_rate']:.1f}% success)\n"
        table += f"📊 Average Success Rate: {avg_success:.1f}%\n"
        table += f"🗣️ Average Conversational Rate: {avg_conv:.1f}%\n"
        table += f"✅ Models Tested: {len(available_results)}\n"

    if unavailable_models:
        table += f"❌ Unavailable Models: {len(unavailable_models)}\n"

    table += "="*120 + "\n"

    return table

def main():
    """Run comprehensive benchmark on all available models"""
    print("🚀 COMPREHENSIVE MODEL BENCHMARK")
    print("="*60)
    print("Testing ALL currently loaded Ollama models with fixed enhanced prompting")

    # Get available models
    models = [
        "phi3.5:latest",
        "tinyllama:1.1b",
        "qwen2.5:0.5b",
        "qwen2.5-coder:3b",
        "codellama:13b-instruct",
        "mistral:7b-instruct",
        "phi3:latest"
    ]

    print(f"\nFound {len(models)} models to benchmark:")
    for model in models:
        print(f"  - {model}")

    # Run benchmarks
    benchmark_results = []

    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {i}/{len(models)}: {model}")
        print(f"{'='*60}")

        try:
            result = benchmark_model(model)
            benchmark_results.append(result)
        except Exception as e:
            print(f"❌ Failed to benchmark {model}: {e}")
            benchmark_results.append({
                "model": model,
                "status": "error",
                "error": str(e)
            })

    # Generate and display results table
    table = format_performance_table(benchmark_results)
    print(table)

    # Save detailed results to JSON
    timestamp = int(time.time())
    results_file = f"benchmark_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_file}")

    # Return success if at least one model worked well
    successful_models = [r for r in benchmark_results if r.get('status') == 'tested' and r.get('success_rate', 0) > 50]
    return len(successful_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)