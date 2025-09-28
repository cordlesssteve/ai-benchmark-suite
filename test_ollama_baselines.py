#!/usr/bin/env python3
"""
Enhanced Ollama Baseline Testing

Direct integration with local Ollama models for comprehensive baseline testing.
"""

import json
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
    except:
        return []
    return []

def test_code_completion(model_name: str, prompt: str, expected_contains: str = None, temperature: float = 0.1):
    """Test a single code completion"""
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 150,
                "stop": ["\n\n", "def ", "class "]
            }
        }, timeout=60)

        if response.status_code == 200:
            data = response.json()
            generated = data.get('response', '').strip()

            passed = True
            if expected_contains:
                passed = expected_contains.lower() in generated.lower()

            return {
                "success": True,
                "generated": generated,
                "passed": passed,
                "evaluation_time": data.get('eval_duration', 0) / 1e9  # nanoseconds to seconds
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "passed": False
        }

def run_humaneval_style_tests(model_name: str):
    """Run HumanEval-style coding problems"""

    problems = [
        {
            "id": "simple_add",
            "prompt": "def add(a, b):\n    \"\"\"\n    Add two numbers a and b.\n    >>> add(1, 2)\n    3\n    \"\"\"\n    return",
            "expected": "a + b",
            "description": "Simple addition"
        },
        {
            "id": "is_even",
            "prompt": "def is_even(n):\n    \"\"\"\n    Check whether a number n is even.\n    >>> is_even(4)\n    True\n    >>> is_even(5)\n    False\n    \"\"\"\n    return",
            "expected": "% 2 == 0",
            "description": "Check if number is even"
        },
        {
            "id": "max_three",
            "prompt": "def max_of_three(a, b, c):\n    \"\"\"\n    Return the maximum of three numbers.\n    >>> max_of_three(1, 2, 3)\n    3\n    \"\"\"\n    return",
            "expected": "max(",
            "description": "Maximum of three numbers"
        },
        {
            "id": "reverse_string",
            "prompt": "def reverse_string(s):\n    \"\"\"\n    Reverse a string.\n    >>> reverse_string('hello')\n    'olleh'\n    \"\"\"\n    return",
            "expected": "[::-1]",
            "description": "String reversal"
        },
        {
            "id": "factorial",
            "prompt": "def factorial(n):\n    \"\"\"\n    Compute factorial of n.\n    >>> factorial(5)\n    120\n    \"\"\"\n    if n == 0:\n        return 1\n    return",
            "expected": "n * factorial(n-1)",
            "description": "Recursive factorial"
        },
        {
            "id": "fibonacci",
            "prompt": "def fibonacci(n):\n    \"\"\"\n    Return the n-th Fibonacci number.\n    >>> fibonacci(6)\n    8\n    \"\"\"\n    if n <= 1:\n        return n\n    return",
            "expected": "fibonacci(n-1) + fibonacci(n-2)",
            "description": "Fibonacci sequence"
        },
        {
            "id": "list_sum",
            "prompt": "def sum_list(numbers):\n    \"\"\"\n    Return the sum of a list of numbers.\n    >>> sum_list([1, 2, 3, 4])\n    10\n    \"\"\"\n    total = 0\n    for num in numbers:\n        total +=",
            "expected": "num",
            "description": "Sum of list elements"
        },
        {
            "id": "count_vowels",
            "prompt": "def count_vowels(text):\n    \"\"\"\n    Count vowels in a string.\n    >>> count_vowels('hello')\n    2\n    \"\"\"\n    vowels = 'aeiou'\n    count = 0\n    for char in text.lower():\n        if char in",
            "expected": "vowels",
            "description": "Count vowels in text"
        }
    ]

    print(f"🚀 Testing {model_name} with {len(problems)} coding problems...")

    results = {
        "model": model_name,
        "timestamp": time.time(),
        "total_problems": len(problems),
        "results": [],
        "summary": {}
    }

    total_time = 0
    passed_count = 0

    for i, problem in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] {problem['description']}...")

        start_time = time.time()
        result = test_code_completion(
            model_name,
            problem['prompt'] + " ",
            problem['expected']
        )
        execution_time = time.time() - start_time

        result.update({
            "problem_id": problem['id'],
            "description": problem['description'],
            "prompt": problem['prompt'],
            "expected": problem['expected'],
            "execution_time": execution_time
        })

        results["results"].append(result)
        total_time += execution_time

        if result['passed']:
            passed_count += 1
            print(f"    ✅ PASS - Generated: {result.get('generated', '')[:60]}...")
        else:
            print(f"    ❌ FAIL - Generated: {result.get('generated', '')[:60]}...")

    # Calculate summary statistics
    score = passed_count / len(problems)
    avg_time = total_time / len(problems)

    results["summary"] = {
        "passed": passed_count,
        "total": len(problems),
        "score": score,
        "percentage": score * 100,
        "total_time": total_time,
        "avg_time_per_problem": avg_time
    }

    return results

def run_multiple_models(models: List[str], output_dir: str = "baseline_results"):
    """Run baseline tests on multiple models"""

    Path(output_dir).mkdir(exist_ok=True)

    all_results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing Model: {model}")
        print(f"{'='*60}")

        try:
            model_results = run_humaneval_style_tests(model)
            all_results.append(model_results)

            # Save individual results
            safe_name = model.replace(":", "_").replace("/", "_")
            output_file = Path(output_dir) / f"{safe_name}_baseline_{int(time.time())}.json"

            with open(output_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            print(f"\n🎯 Results for {model}:")
            print(f"   Score: {model_results['summary']['passed']}/{model_results['summary']['total']} ({model_results['summary']['percentage']:.1f}%)")
            print(f"   Avg time per problem: {model_results['summary']['avg_time_per_problem']:.2f}s")
            print(f"   Results saved: {output_file}")

        except Exception as e:
            print(f"❌ Failed to test {model}: {e}")
            all_results.append({
                "model": model,
                "error": str(e),
                "timestamp": time.time()
            })

    # Save comparative results
    comparison_file = Path(output_dir) / f"baseline_comparison_{int(time.time())}.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison summary
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*60}")

    successful_results = [r for r in all_results if 'summary' in r]
    if successful_results:
        sorted_results = sorted(successful_results, key=lambda x: x['summary']['score'], reverse=True)

        print(f"{'Model':<25} {'Score':<12} {'Avg Time':<12} {'Pass Rate'}")
        print("-" * 60)

        for result in sorted_results:
            model = result['model']
            summary = result['summary']
            print(f"{model:<25} {summary['passed']}/{summary['total']:<8} {summary['avg_time_per_problem']:<8.2f}s    {summary['percentage']:.1f}%")

    print(f"\nComparison results saved: {comparison_file}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations on Ollama models")
    parser.add_argument("--model", help="Single model to test")
    parser.add_argument("--models", nargs="+", help="Multiple models to test")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument("--code-focused", action="store_true", help="Test only code-focused models")
    parser.add_argument("--output-dir", default="baseline_results", help="Output directory for results")

    args = parser.parse_args()

    # Check if Ollama is running
    available_models = get_ollama_models()
    if not available_models:
        print("❌ Ollama server not responding or no models available.")
        print("Make sure Ollama is running with 'ollama serve' and you have models installed.")
        return

    print(f"Available models: {', '.join(available_models)}")

    # Determine which models to test
    models_to_test = []

    if args.model:
        if args.model in available_models:
            models_to_test = [args.model]
        else:
            print(f"❌ Model '{args.model}' not found in available models.")
            return
    elif args.models:
        models_to_test = [m for m in args.models if m in available_models]
        missing = [m for m in args.models if m not in available_models]
        if missing:
            print(f"⚠️  Models not found: {', '.join(missing)}")
    elif args.code_focused:
        # Focus on code-specialized models
        code_models = [m for m in available_models if any(keyword in m.lower()
                      for keyword in ['coder', 'code', 'llama', 'phi'])]
        models_to_test = code_models[:3]  # Test top 3 code-focused models
    elif args.all:
        models_to_test = available_models
    else:
        # Default: test a few representative models
        priority_models = ['qwen2.5-coder:3b', 'codellama:13b-instruct', 'phi3.5:latest', 'mistral:7b-instruct']
        models_to_test = [m for m in priority_models if m in available_models]
        if not models_to_test:
            models_to_test = available_models[:3]  # Fallback to first 3

    if not models_to_test:
        print("❌ No models to test.")
        return

    print(f"\n🎯 Testing models: {', '.join(models_to_test)}")

    # Run tests
    run_multiple_models(models_to_test, args.output_dir)

if __name__ == "__main__":
    main()