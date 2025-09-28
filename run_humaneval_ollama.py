#!/usr/bin/env python3
"""
HumanEval Evaluation for Ollama Models

Direct implementation of HumanEval evaluation using the actual HumanEval dataset.
"""

import json
import requests
import time
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_humaneval_problems():
    """Load HumanEval problems from the dataset"""

    # Check if we have the HumanEval dataset
    humaneval_file = Path("data/HumanEval.jsonl")

    if not humaneval_file.exists():
        # Try to download or find it in BigCode harness
        bigcode_humaneval = Path("harnesses/bigcode-evaluation-harness/data/humaneval.jsonl")
        if bigcode_humaneval.exists():
            humaneval_file = bigcode_humaneval
        else:
            print("❌ HumanEval dataset not found. Please ensure it's available.")
            return []

    problems = []
    try:
        with open(humaneval_file, 'r') as f:
            for line in f:
                problems.append(json.loads(line.strip()))
    except:
        # Create a subset of HumanEval problems manually for testing
        problems = [
            {
                "task_id": "HumanEval/0",
                "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n\ncheck(has_close_elements)"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n\n    return result\n",
                "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [\n        '(()())', '((()))', '()', '((())()())'\n    ]\n    assert candidate('() (()) ((())) (((())))') == [\n        '()', '(())', '((()))', '(((())))'\n    ]\n    assert candidate('(()(())((())))') == [\n        '(()(())((())))'\n    ]\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n\ncheck(separate_paren_groups)"
            },
            {
                "task_id": "HumanEval/2",
                "prompt": "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
                "canonical_solution": "    return number % 1.0\n",
                "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n\ncheck(truncate_number)"
            }
        ]

    return problems

def generate_code_with_ollama(model_name: str, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
    """Generate code completion using Ollama"""

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["\ndef ", "\nclass ", "\nif __name__", "\n# ", "\n\n\n"]
            }
        }, timeout=120)

        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            return f"# Error: HTTP {response.status_code}"

    except Exception as e:
        return f"# Error: {str(e)}"

def test_generated_code(problem: Dict[str, Any], generated_code: str) -> Dict[str, Any]:
    """Test generated code against the problem's test cases"""

    task_id = problem["task_id"]
    test_code = problem["test"]

    # Create the complete function
    function_name = problem["prompt"].split("def ")[1].split("(")[0]
    full_code = problem["prompt"] + generated_code

    # Create test file
    test_script = f"""
import sys
import traceback
import signal
import os

def timeout_handler(signum, frame):
    raise TimeoutError("Test execution timed out")

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

try:
{chr(10).join('    ' + line for line in full_code.split(chr(10)))}

{chr(10).join('    ' + line for line in test_code.split(chr(10)))}

    print("PASSED")
except Exception as e:
    print(f"FAILED: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
finally:
    signal.alarm(0)  # Cancel timeout
"""

    # Execute test
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name

        result = subprocess.run(
            ["python3", test_file],
            capture_output=True,
            text=True,
            timeout=15
        )

        os.unlink(test_file)

        if result.returncode == 0 and "PASSED" in result.stdout:
            return {
                "task_id": task_id,
                "passed": True,
                "result": "passed",
                "completion": generated_code
            }
        else:
            return {
                "task_id": task_id,
                "passed": False,
                "result": f"failed: {result.stdout.strip()} {result.stderr.strip()}",
                "completion": generated_code
            }

    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "result": f"execution_error: {str(e)}",
            "completion": generated_code
        }

def evaluate_model_on_humaneval(model_name: str, num_problems: int = 10, temperature: float = 0.1):
    """Evaluate a model on HumanEval problems"""

    print(f"🚀 Evaluating {model_name} on HumanEval (first {num_problems} problems)...")

    problems = load_humaneval_problems()
    if not problems:
        print("❌ No problems loaded")
        return None

    # Limit to specified number of problems
    problems = problems[:num_problems]

    results = []
    passed_count = 0
    total_time = 0

    for i, problem in enumerate(problems, 1):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        print(f"  [{i}/{len(problems)}] {task_id}...")

        start_time = time.time()

        # Generate code
        generated_code = generate_code_with_ollama(model_name, prompt, temperature)

        # Test the generated code
        test_result = test_generated_code(problem, generated_code)

        execution_time = time.time() - start_time
        total_time += execution_time

        test_result["execution_time"] = execution_time
        test_result["model"] = model_name

        results.append(test_result)

        if test_result["passed"]:
            passed_count += 1
            print(f"    ✅ PASSED ({execution_time:.1f}s)")
        else:
            print(f"    ❌ FAILED ({execution_time:.1f}s): {test_result['result'][:50]}...")

    # Calculate metrics
    pass_at_1 = passed_count / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0

    summary = {
        "model": model_name,
        "timestamp": time.time(),
        "total_problems": len(problems),
        "passed": passed_count,
        "pass_at_1": pass_at_1,
        "percentage": pass_at_1 * 100,
        "total_time": total_time,
        "avg_time_per_problem": avg_time,
        "temperature": temperature,
        "results": results
    }

    # Save results
    Path("humaneval_results").mkdir(exist_ok=True)
    safe_name = model_name.replace(":", "_").replace("/", "_")
    output_file = f"humaneval_results/{safe_name}_humaneval_{int(time.time())}.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎯 Results for {model_name}:")
    print(f"   Pass@1: {passed_count}/{len(problems)} ({pass_at_1:.1%})")
    print(f"   Avg time: {avg_time:.2f}s per problem")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Results saved: {output_file}")

    return summary

def run_humaneval_on_available_models():
    """Run HumanEval evaluation on all available Ollama models"""

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

    all_results = []

    # Test a subset of models (fastest ones first)
    priority_models = [
        "phi3.5:latest",      # Fast and capable
        "mistral:7b-instruct", # Previously best performer
        "qwen2.5-coder:3b",   # Code-focused
        "tinyllama:1.1b"      # Very fast
    ]

    models_to_test = [m for m in priority_models if m in models]
    if not models_to_test:
        models_to_test = models[:3]  # Fallback to first 3

    print(f"\n🎯 Testing models: {', '.join(models_to_test)}")

    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model}")
        print(f"{'='*60}")

        try:
            result = evaluate_model_on_humaneval(model, num_problems=5)  # Start with 5 problems
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to evaluate {model}: {e}")

    # Save comparison
    if all_results:
        comparison_file = f"humaneval_results/comparison_{int(time.time())}.json"
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print comparison
        print(f"\n{'='*60}")
        print("HUMANEVAL COMPARISON SUMMARY")
        print(f"{'='*60}")

        sorted_results = sorted(all_results, key=lambda x: x['pass_at_1'], reverse=True)

        print(f"{'Model':<25} {'Pass@1':<12} {'Avg Time':<12}")
        print("-" * 60)

        for result in sorted_results:
            model = result['model']
            pass_at_1 = result['pass_at_1']
            avg_time = result['avg_time_per_problem']
            print(f"{model:<25} {pass_at_1:.1%}      {avg_time:.2f}s")

        print(f"\nComparison saved: {comparison_file}")

if __name__ == "__main__":
    run_humaneval_on_available_models()