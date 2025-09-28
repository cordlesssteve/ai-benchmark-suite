#!/usr/bin/env python3
"""
Direct Code Completion Test

Simple, direct code completion testing for Ollama models.
"""

import json
import requests
import time
from pathlib import Path

def test_direct_completion(model_name: str):
    """Test direct code completion with minimal prompting"""

    problems = [
        {
            "id": "add",
            "code": "def add(a, b):\n    return ",
            "expect": "a + b"
        },
        {
            "id": "even",
            "code": "def is_even(n):\n    return n ",
            "expect": "% 2 == 0"
        },
        {
            "id": "max3",
            "code": "def max_three(a, b, c):\n    return ",
            "expect": "max("
        },
        {
            "id": "reverse",
            "code": "def reverse_string(s):\n    return s",
            "expect": "[::-1]"
        },
        {
            "id": "fib",
            "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + ",
            "expect": "fib(n-2)"
        }
    ]

    print(f"🚀 Testing {model_name} with direct code completion...")

    results = []
    total_time = 0

    for i, problem in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] {problem['id']}...")

        start_time = time.time()

        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": problem["code"],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 20,
                    "stop": ["\n", "def", "class", "#"],
                    "top_p": 0.9
                }
            }, timeout=30)

            execution_time = time.time() - start_time
            total_time += execution_time

            if response.status_code == 200:
                data = response.json()
                generated = data.get('response', '').strip()

                # Simple pattern matching
                passed = problem['expect'].lower() in generated.lower()

                result = {
                    "problem_id": problem['id'],
                    "prompt": problem['code'],
                    "generated": generated,
                    "expected": problem['expect'],
                    "passed": passed,
                    "execution_time": execution_time
                }

                results.append(result)

                status = "✅" if passed else "❌"
                print(f"    {status} '{generated[:30]}...' (expected: {problem['expect']})")

            else:
                print(f"    ❌ API Error: {response.status_code}")
                results.append({
                    "problem_id": problem['id'],
                    "error": f"API Error: {response.status_code}",
                    "passed": False,
                    "execution_time": execution_time
                })

        except Exception as e:
            execution_time = time.time() - start_time
            total_time += execution_time
            print(f"    ❌ Error: {e}")
            results.append({
                "problem_id": problem['id'],
                "error": str(e),
                "passed": False,
                "execution_time": execution_time
            })

    # Calculate score
    passed_count = sum(1 for r in results if r.get("passed", False))
    score = passed_count / len(results) if results else 0

    summary = {
        "model": model_name,
        "timestamp": time.time(),
        "total_problems": len(problems),
        "passed": passed_count,
        "score": score,
        "percentage": score * 100,
        "total_time": total_time,
        "avg_time": total_time / len(problems),
        "results": results
    }

    # Save results
    Path("direct_results").mkdir(exist_ok=True)
    safe_name = model_name.replace(":", "_").replace("/", "_")
    output_file = f"direct_results/{safe_name}_direct_{int(time.time())}.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎯 Results for {model_name}:")
    print(f"   Score: {passed_count}/{len(problems)} ({score:.1%})")
    print(f"   Avg time: {total_time/len(problems):.2f}s")
    print(f"   Saved: {output_file}")

    return summary

def test_all_models():
    """Test all available models"""

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not responding")
            return

        models = [m['name'] for m in response.json().get('models', [])]
        print(f"Available models: {', '.join(models)}")

        all_results = []

        for model in models:
            print(f"\n{'='*50}")
            print(f"Testing: {model}")
            print(f"{'='*50}")

            try:
                result = test_direct_completion(model)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Failed: {e}")

        # Summary comparison
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")

        if all_results:
            sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

            print(f"{'Model':<25} {'Score':<8} {'Time':<8}")
            print("-" * 45)

            for result in sorted_results:
                print(f"{result['model']:<25} {result['passed']}/{result['total_problems']:<5} {result['avg_time']:.2f}s")

            # Save comparison
            with open(f"direct_results/comparison_{int(time.time())}.json", 'w') as f:
                json.dump(all_results, f, indent=2)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_all_models()