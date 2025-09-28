#!/usr/bin/env python3
"""
Real HumanEval Testing with Actual Dataset

Use the actual HumanEval dataset from BigCode harness for proper evaluation.
"""

import json
import requests
import time
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add BigCode harness to path
bigcode_path = Path(__file__).parent / "harnesses" / "bigcode-evaluation-harness"
sys.path.insert(0, str(bigcode_path))

def load_humaneval_from_bigcode():
    """Load HumanEval dataset from BigCode harness"""

    try:
        # Try to import and load from BigCode harness
        from datasets import load_dataset

        print("📂 Loading HumanEval dataset from BigCode harness...")

        # Load the dataset
        dataset = load_dataset("openai_humaneval", split="test")

        problems = []
        for item in dataset:
            problems.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"]
            })

        print(f"✅ Loaded {len(problems)} HumanEval problems")
        return problems

    except Exception as e:
        print(f"⚠️  Could not load from BigCode harness: {e}")
        print("📂 Using fallback problems...")

        # Fallback to embedded problems
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n\ncheck(has_close_elements)",
                "entry_point": "has_close_elements"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n\n    return result\n",
                "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [\n        '(()())', '((()))', '()', '((())()())'\n    ]\n    assert candidate('() (()) ((())) (((())))') == [\n        '()', '(())', '((()))', '(((())))'\n    ]\n    assert candidate('(()(())((())))') == [\n        '(()(())((())))'\n    ]\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n\ncheck(separate_paren_groups)",
                "entry_point": "separate_paren_groups"
            },
            {
                "task_id": "HumanEval/2",
                "prompt": "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
                "canonical_solution": "    return number % 1.0\n",
                "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n\ncheck(truncate_number)",
                "entry_point": "truncate_number"
            },
            {
                "task_id": "HumanEval/3",
                "prompt": "from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
                "canonical_solution": "    balance = 0\n\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, 2, -4, 5]) == True\n    assert candidate([1, 2, -3]) == False\n    assert candidate([1, 2, -3, 1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -3, 1, 2, -3, 1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -3, 1, 2, -3, 1, 2, -3, 1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -3, 1, 2, -3, 1, 2, -3, 1, 2, -3, 1, 2]) == False\n\ncheck(below_zero)",
                "entry_point": "below_zero"
            },
            {
                "task_id": "HumanEval/4",
                "prompt": "from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
                "canonical_solution": "    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)\n",
                "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6\n\ncheck(mean_absolute_deviation)",
                "entry_point": "mean_absolute_deviation"
            }
        ]

def generate_code_completion(model_name: str, prompt: str, temperature: float = 0.1) -> str:
    """Generate code completion with improved prompting"""

    # Try different prompt formats for better code completion
    prompt_formats = [
        f"# Complete this Python function:\n{prompt}",
        f"```python\n{prompt}",
        prompt,  # Direct
        f"# Python code completion:\n{prompt}"
    ]

    for attempt, formatted_prompt in enumerate(prompt_formats):
        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 300,
                    "stop": ["\ndef ", "\nclass ", "\nif __name__", "\n\n\n"]
                }
            }, timeout=60)

            if response.status_code == 200:
                generated = response.json().get('response', '').strip()

                # Filter out conversational responses
                if not any(phrase in generated.lower() for phrase in [
                    "here's", "here is", "certainly", "sure", "i'll help", "let me", "to complete"
                ]):
                    return generated

            # If we got a conversational response, try next format
            if attempt < len(prompt_formats) - 1:
                continue
            else:
                # Return the last attempt even if conversational
                return generated if 'generated' in locals() else f"# Error: No valid response"

        except Exception as e:
            if attempt == len(prompt_formats) - 1:  # Last attempt
                return f"# Error: {str(e)}"
            continue

def execute_test(problem: Dict[str, Any], generated_code: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute the test case with proper sandboxing"""

    task_id = problem["task_id"]
    entry_point = problem["entry_point"]
    test_code = problem["test"]

    # Build complete function
    complete_function = problem["prompt"] + generated_code

    # Create isolated test script
    test_script = f'''
import sys
import signal
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

def timeout_handler(signum, frame):
    raise TimeoutError("Test execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    # Capture outputs
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        # Execute the function
{chr(10).join("        " + line for line in complete_function.split(chr(10)))}

        # Execute the test
{chr(10).join("        " + line for line in test_code.split(chr(10)))}

    print("PASSED")

except TimeoutError:
    print("TIMEOUT")
except Exception as e:
    print(f"FAILED: {{type(e).__name__}}: {{str(e)}}")

finally:
    signal.alarm(0)
'''

    try:
        # Write and execute test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name

        result = subprocess.run(
            ["python3", test_file],
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )

        os.unlink(test_file)

        # Parse result
        if result.returncode == 0 and "PASSED" in result.stdout:
            return {
                "task_id": task_id,
                "passed": True,
                "result": "passed",
                "completion": generated_code,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        else:
            failure_reason = result.stdout.strip() or result.stderr.strip() or "Unknown failure"
            return {
                "task_id": task_id,
                "passed": False,
                "result": failure_reason,
                "completion": generated_code,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }

    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id,
            "passed": False,
            "result": "execution_timeout",
            "completion": generated_code
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "result": f"execution_error: {str(e)}",
            "completion": generated_code
        }

def evaluate_model_on_humaneval(model_name: str, num_problems: int = 10):
    """Evaluate model on HumanEval with real dataset"""

    print(f"🚀 Evaluating {model_name} on HumanEval (first {num_problems} problems)...")

    # Load problems
    problems = load_humaneval_from_bigcode()
    if not problems:
        print("❌ No problems available")
        return None

    # Limit problems for testing
    problems = problems[:num_problems]

    results = []
    passed_count = 0
    total_time = 0

    for i, problem in enumerate(problems, 1):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        print(f"  [{i}/{len(problems)}] {task_id}...")

        start_time = time.time()

        # Generate completion
        generated_code = generate_code_completion(model_name, prompt)

        # Test the completion
        test_result = execute_test(problem, generated_code)

        execution_time = time.time() - start_time
        total_time += execution_time

        test_result["execution_time"] = execution_time
        test_result["model"] = model_name

        results.append(test_result)

        if test_result["passed"]:
            passed_count += 1
            print(f"    ✅ PASSED ({execution_time:.1f}s)")
        else:
            print(f"    ❌ FAILED ({execution_time:.1f}s): {test_result['result']}")

    # Calculate metrics
    pass_at_1 = passed_count / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0

    summary = {
        "model": model_name,
        "evaluation_type": "humaneval_real",
        "timestamp": time.time(),
        "dataset_source": "openai_humaneval",
        "total_problems": len(problems),
        "passed": passed_count,
        "failed": len(problems) - passed_count,
        "pass_at_1": pass_at_1,
        "percentage": pass_at_1 * 100,
        "total_time": total_time,
        "avg_time_per_problem": avg_time,
        "results": results
    }

    # Save results
    Path("real_humaneval_results").mkdir(exist_ok=True)
    safe_name = model_name.replace(":", "_").replace("/", "_")
    output_file = f"real_humaneval_results/{safe_name}_real_humaneval_{int(time.time())}.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎯 Real HumanEval Results for {model_name}:")
    print(f"   Pass@1: {passed_count}/{len(problems)} ({pass_at_1:.1%})")
    print(f"   Average time: {avg_time:.2f}s per problem")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Results saved: {output_file}")

    return summary

def run_real_humaneval_benchmark():
    """Run real HumanEval benchmark on available models"""

    # Get available models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not responding")
            return

        models = [m['name'] for m in response.json().get('models', [])]
        print(f"Available models: {', '.join(models)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Test priority models
    priority_models = [
        "phi3.5:latest",        # Fast
        "mistral:7b-instruct",  # Previous best
        "qwen2.5-coder:3b",     # Code-focused
    ]

    models_to_test = [m for m in priority_models if m in models]
    if not models_to_test:
        models_to_test = models[:3]

    print(f"\n🎯 Testing models on REAL HumanEval: {', '.join(models_to_test)}")

    all_results = []

    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"REAL HUMANEVAL EVALUATION: {model}")
        print(f"{'='*60}")

        try:
            result = evaluate_model_on_humaneval(model, num_problems=5)  # Start with 5
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")

    # Save comparison
    if all_results:
        comparison_file = f"real_humaneval_results/comparison_{int(time.time())}.json"
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("REAL HUMANEVAL COMPARISON")
        print(f"{'='*60}")

        sorted_results = sorted(all_results, key=lambda x: x['pass_at_1'], reverse=True)

        print(f"{'Model':<25} {'Pass@1':<10} {'Time':<8} {'Dataset'}")
        print("-" * 60)

        for result in sorted_results:
            model = result['model']
            pass_at_1 = result['pass_at_1']
            avg_time = result['avg_time_per_problem']
            dataset = result['dataset_source']
            print(f"{model:<25} {pass_at_1:.1%}    {avg_time:.2f}s   {dataset}")

        print(f"\nComparison saved: {comparison_file}")
        print("\n✅ REAL HumanEval evaluation completed!")

if __name__ == "__main__":
    run_real_humaneval_benchmark()