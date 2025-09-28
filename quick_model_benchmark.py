#!/usr/bin/env python3
"""
Quick Model Performance Benchmark

Tests key models with a focused set of tests to provide immediate performance data.
"""

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface

# Focused test cases for quick evaluation
QUICK_TESTS = [
    {
        "name": "add_function",
        "prompt": "def add(a, b):\n    return ",
        "expected": ["a + b", "a+b"],
    },
    {
        "name": "even_check",
        "prompt": "def is_even(n):\n    return ",
        "expected": ["n % 2 == 0", "n%2==0"],
    },
    {
        "name": "max_function",
        "prompt": "def max_three(a, b, c):\n    return ",
        "expected": ["max(a, b, c)", "max([a, b, c])"],
    }
]

def quick_test_model(model_name: str, timeout: int = 15) -> dict:
    """Quick test of a single model"""
    print(f"🧪 Testing {model_name}...", end=" ")

    interface = FixedEnhancedOllamaInterface(model_name)

    if not interface.is_available():
        print("❌ Unavailable")
        return {"model": model_name, "status": "unavailable"}

    results = []
    total_time = 0
    successes = 0

    for test in QUICK_TESTS:
        try:
            start_time = time.time()
            response = interface.generate_auto_best(test['prompt'], max_tokens=50, timeout=timeout)
            exec_time = time.time() - start_time

            total_time += exec_time

            # Check success
            has_content = len(response.text.strip()) > 0
            contains_expected = any(exp.lower() in response.text.lower() for exp in test['expected'])
            is_successful = response.success and has_content and contains_expected and not response.is_conversational

            if is_successful:
                successes += 1

            results.append({
                "test": test['name'],
                "success": is_successful,
                "conversational": response.is_conversational,
                "time": exec_time,
                "strategy": response.prompting_strategy,
                "response": response.text[:30] + "..." if len(response.text) > 30 else response.text
            })

        except Exception as e:
            print(f"❌ Error: {str(e)[:20]}")
            return {"model": model_name, "status": "error", "error": str(e)}

    success_rate = (successes / len(QUICK_TESTS)) * 100
    avg_time = total_time / len(QUICK_TESTS)
    conversational_count = sum(1 for r in results if r['conversational'])

    print(f"✅ {success_rate:.0f}% success, {avg_time:.1f}s avg")

    return {
        "model": model_name,
        "status": "tested",
        "success_rate": success_rate,
        "avg_time": avg_time,
        "conversational_count": conversational_count,
        "results": results
    }

def main():
    """Run quick benchmark"""
    print("⚡ QUICK MODEL PERFORMANCE BENCHMARK")
    print("="*60)

    # Test models in order of size (smallest first for speed)
    models = [
        "qwen2.5:0.5b",      # 397MB - Fastest
        "tinyllama:1.1b",    # 637MB - Very fast
        "qwen2.5-coder:3b",  # 1.9GB - Fast, code-focused
        "phi3.5:latest",     # 2.2GB - Medium
        "phi3:latest",       # 2.2GB - Medium
        "mistral:7b-instruct", # 4.4GB - Slower but reliable
        "codellama:13b-instruct" # 7.4GB - Slowest
    ]

    results = []

    for model in models:
        try:
            result = quick_test_model(model)
            results.append(result)
        except Exception as e:
            print(f"❌ {model}: {e}")
            results.append({"model": model, "status": "error", "error": str(e)})

    # Create performance table
    print(f"\n{'='*80}")
    print("📊 MODEL PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    print(f"{'Model':<25} {'Size':<8} {'Success':<8} {'AvgTime':<8} {'Conv':<5} {'Status':<12}")
    print("-"*80)

    # Model sizes
    sizes = {
        "qwen2.5:0.5b": "397MB",
        "tinyllama:1.1b": "637MB",
        "qwen2.5-coder:3b": "1.9GB",
        "phi3.5:latest": "2.2GB",
        "phi3:latest": "2.2GB",
        "mistral:7b-instruct": "4.4GB",
        "codellama:13b-instruct": "7.4GB"
    }

    for result in results:
        model = result['model']
        size = sizes.get(model, "Unknown")

        if result['status'] == 'tested':
            success = f"{result['success_rate']:.0f}%"
            avg_time = f"{result['avg_time']:.1f}s"
            conv = f"{result['conversational_count']}/3"
            status = "🟢 Working" if result['success_rate'] > 50 else "🟡 Limited"
        elif result['status'] == 'unavailable':
            success = avg_time = conv = "-"
            status = "❌ Unavailable"
        else:
            success = avg_time = conv = "-"
            status = "❌ Error"

        print(f"{model:<25} {size:<8} {success:<8} {avg_time:<8} {conv:<5} {status:<12}")

    # Best performers
    working_models = [r for r in results if r['status'] == 'tested' and r['success_rate'] > 0]

    if working_models:
        print(f"\n🏆 TOP PERFORMERS:")
        working_models.sort(key=lambda x: x['success_rate'], reverse=True)

        for i, model in enumerate(working_models[:3], 1):
            print(f"   {i}. {model['model']}: {model['success_rate']:.0f}% success, {model['avg_time']:.1f}s avg")

    print(f"\n📈 SUMMARY:")
    working_count = len(working_models)
    total_models = len([r for r in results if r['status'] != 'unavailable'])
    print(f"   Working models: {working_count}/{total_models}")

    if working_models:
        avg_success = sum(r['success_rate'] for r in working_models) / len(working_models)
        print(f"   Average success rate: {avg_success:.1f}%")

    return working_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)