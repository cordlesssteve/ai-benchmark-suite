#!/usr/bin/env python3
"""
Improved Ollama Testing with Better Prompting

Test different prompting strategies to get direct code completion instead of explanations.
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any

def test_prompting_strategies(model_name: str):
    """Test different prompting strategies for code completion"""

    # Base test problem
    test_code = "def add(a, b):\n    return "
    expected = "a + b"

    strategies = [
        {
            "name": "Direct",
            "prompt": test_code,
            "options": {"temperature": 0.1, "num_predict": 10, "stop": ["\n"]}
        },
        {
            "name": "Code-only instruction",
            "prompt": "# Complete this code with minimal output, no explanation:\n" + test_code,
            "options": {"temperature": 0.1, "num_predict": 20, "stop": ["\n", "#"]}
        },
        {
            "name": "System instruction",
            "prompt": "You are a code completion engine. Complete only the missing code:\n" + test_code,
            "options": {"temperature": 0.1, "num_predict": 15, "stop": ["\n"]}
        },
        {
            "name": "Few-shot example",
            "prompt": "Complete the code:\n\ndef multiply(x, y):\n    return x * y\n\ndef add(a, b):\n    return ",
            "options": {"temperature": 0.1, "num_predict": 15, "stop": ["\n"]}
        },
        {
            "name": "Fill-in-middle style",
            "prompt": "<|fim_prefix|>def add(a, b):\n    return <|fim_suffix|>\n\n# Test\nprint(add(2, 3))<|fim_middle|>",
            "options": {"temperature": 0.1, "num_predict": 10, "stop": ["<|fim", "\n"]}
        },
        {
            "name": "Terse instruction",
            "prompt": "CODE ONLY:\n" + test_code,
            "options": {"temperature": 0.0, "num_predict": 8, "stop": ["\n", " #"]}
        },
        {
            "name": "Continue format",
            "prompt": test_code + "# CONTINUE:",
            "options": {"temperature": 0.1, "num_predict": 12, "stop": ["\n", "#"]}
        }
    ]

    print(f"🧪 Testing prompting strategies on {model_name}...")

    results = []

    for i, strategy in enumerate(strategies, 1):
        print(f"  [{i}/{len(strategies)}] {strategy['name']}...")

        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": strategy["prompt"],
                "stream": False,
                "options": strategy["options"]
            }, timeout=30)

            if response.status_code == 200:
                data = response.json()
                generated = data.get('response', '').strip()

                # Check if it contains the expected code
                passed = expected in generated.lower()

                result = {
                    "strategy": strategy["name"],
                    "prompt": strategy["prompt"],
                    "generated": generated,
                    "passed": passed,
                    "contains_expected": expected.lower() in generated.lower()
                }

                results.append(result)

                status = "✅" if passed else "❌"
                print(f"    {status} '{generated[:40]}...'")

            else:
                print(f"    ❌ API Error: {response.status_code}")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    return results

def test_improved_prompts(model_name: str):
    """Test with the best prompting strategy on multiple problems"""

    # Based on testing, use the most direct approach
    problems = [
        {
            "id": "add",
            "prompt": "# Code completion only:\ndef add(a, b):\n    return ",
            "expected": "a + b"
        },
        {
            "id": "even",
            "prompt": "# Code completion only:\ndef is_even(n):\n    return n ",
            "expected": "% 2 == 0"
        },
        {
            "id": "max",
            "prompt": "# Code completion only:\ndef max_three(a, b, c):\n    return ",
            "expected": "max("
        },
        {
            "id": "reverse",
            "prompt": "# Code completion only:\ndef reverse_string(s):\n    return s",
            "expected": "[::-1]"
        },
        {
            "id": "fibonacci",
            "prompt": "# Code completion only:\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + ",
            "expected": "fib(n-2)"
        },
        {
            "id": "factorial",
            "prompt": "# Code completion only:\ndef factorial(n):\n    if n == 0: return 1\n    return n * ",
            "expected": "factorial(n-1)"
        },
        {
            "id": "sum_list",
            "prompt": "# Code completion only:\ndef sum_list(lst):\n    return ",
            "expected": "sum(lst)"
        },
        {
            "id": "count_vowels",
            "prompt": "# Code completion only:\ndef count_vowels(text):\n    vowels = 'aeiou'\n    return sum(1 for c in text.lower() if c in ",
            "expected": "vowels"
        }
    ]

    print(f"🚀 Testing improved prompts on {model_name}...")

    results = []
    passed_count = 0

    for i, problem in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] {problem['id']}...")

        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": problem["prompt"],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 20,
                    "stop": ["\n", "#", "def"],
                    "top_p": 0.9
                }
            }, timeout=30)

            if response.status_code == 200:
                data = response.json()
                generated = data.get('response', '').strip()

                # More flexible checking
                passed = (problem['expected'].lower() in generated.lower() or
                         generated.strip().endswith(problem['expected']) or
                         problem['expected'] in generated)

                if passed:
                    passed_count += 1

                result = {
                    "problem_id": problem['id'],
                    "prompt": problem['prompt'],
                    "generated": generated,
                    "expected": problem['expected'],
                    "passed": passed
                }

                results.append(result)

                status = "✅" if passed else "❌"
                print(f"    {status} '{generated[:50]}...' (expected: {problem['expected']})")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "problem_id": problem['id'],
                "error": str(e),
                "passed": False
            })

    score = passed_count / len(problems) if problems else 0

    summary = {
        "model": model_name,
        "timestamp": time.time(),
        "strategy": "improved_prompts",
        "total_problems": len(problems),
        "passed": passed_count,
        "score": score,
        "percentage": score * 100,
        "results": results
    }

    return summary

def test_humaneval_style_prompt(model_name: str, problem_text: str, function_name: str):
    """Test with HumanEval-style prompting"""

    # Different prompt formats for HumanEval-style problems
    prompt_formats = [
        f"# Complete this Python function:\n{problem_text}",
        f"# Code completion:\n{problem_text}",
        f"{problem_text}",  # Direct
        f"```python\n{problem_text}",
    ]

    best_result = None
    best_score = -1

    for prompt in prompt_formats:
        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200,
                    "stop": ["\ndef ", "\nclass ", "\n# ", "\nif __name__"]
                }
            }, timeout=60)

            if response.status_code == 200:
                data = response.json()
                generated = data.get('response', '').strip()

                # Simple scoring: longer code that contains function logic
                score = len(generated) if function_name in generated else 0

                if score > best_score:
                    best_score = score
                    best_result = {
                        "prompt_format": prompt,
                        "generated": generated,
                        "score": score
                    }

        except Exception as e:
            continue

    return best_result

def run_improved_baseline():
    """Run improved baseline tests on available models"""

    # Get available models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not responding")
            return

        models = [m['name'] for m in response.json().get('models', [])]
        print(f"Available models: {', '.join(models)}")

    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return

    Path("improved_results").mkdir(exist_ok=True)

    # First test prompting strategies on the best model from before
    if "mistral:7b-instruct" in models:
        print(f"\n{'='*60}")
        print("TESTING PROMPTING STRATEGIES")
        print(f"{'='*60}")

        strategy_results = test_prompting_strategies("mistral:7b-instruct")

        with open("improved_results/prompting_strategies.json", 'w') as f:
            json.dump(strategy_results, f, indent=2)

        print("\n📊 Strategy Results:")
        for result in strategy_results:
            status = "✅" if result.get('passed', False) else "❌"
            print(f"  {status} {result['strategy']}: '{result.get('generated', '')[:30]}...'")

    # Now test improved prompts on all models
    print(f"\n{'='*60}")
    print("IMPROVED BASELINE EVALUATION")
    print(f"{'='*60}")

    all_results = []

    for model in models:
        print(f"\nTesting: {model}")
        print("-" * 50)

        try:
            result = test_improved_prompts(model)
            all_results.append(result)

            # Save individual results
            safe_name = model.replace(":", "_").replace("/", "_")
            with open(f"improved_results/{safe_name}_improved.json", 'w') as f:
                json.dump(result, f, indent=2)

            print(f"🎯 {model}: {result['passed']}/{result['total_problems']} ({result['percentage']:.1f}%)")

        except Exception as e:
            print(f"❌ Failed: {e}")

    # Save comparison
    with open("improved_results/improved_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison
    print(f"\n{'='*60}")
    print("IMPROVED RESULTS COMPARISON")
    print(f"{'='*60}")

    if all_results:
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

        print(f"{'Model':<25} {'Score':<8} {'Improvement'}")
        print("-" * 50)

        # Previous baseline scores for comparison
        previous_scores = {
            "mistral:7b-instruct": 20.0,
            "phi3.5:latest": 0.0,
            "tinyllama:1.1b": 0.0,
            "qwen2.5:0.5b": 0.0,
            "phi3:latest": 0.0,
            "qwen2.5-coder:3b": 0.0,
            "codellama:13b-instruct": 0.0
        }

        for result in sorted_results:
            model = result['model']
            new_score = result['percentage']
            old_score = previous_scores.get(model, 0.0)
            improvement = new_score - old_score

            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            print(f"{model:<25} {result['passed']}/{result['total_problems']:<6} {improvement_str}")

if __name__ == "__main__":
    run_improved_baseline()