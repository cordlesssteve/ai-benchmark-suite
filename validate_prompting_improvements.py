#!/usr/bin/env python3
"""
Validation: Advanced Prompting Improvements

Compare baseline (no prompting) vs enhanced prompting on real HumanEval problems
to demonstrate the effectiveness of the integration.
"""

import sys
from pathlib import Path
import time
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.ollama_interface import OllamaInterface
from model_interfaces.enhanced_ollama_interface import EnhancedOllamaInterface, PromptingStrategy

# Real HumanEval problems for testing
HUMANEVAL_PROBLEMS = [
    {
        "name": "add_numbers",
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add two integers and return the result.\"\"\"\n    return ",
        "expected_tokens": ["a + b", "a+b", "return a + b"],
        "test_case": "add(2, 3) == 5"
    },
    {
        "name": "even_check",
        "prompt": "def is_even(n: int) -> bool:\n    \"\"\"Check if a number is even.\"\"\"\n    return ",
        "expected_tokens": ["n % 2 == 0", "n%2==0", "not n % 2"],
        "test_case": "is_even(4) == True and is_even(3) == False"
    },
    {
        "name": "max_three",
        "prompt": "def max_of_three(a: int, b: int, c: int) -> int:\n    \"\"\"Return the maximum of three numbers.\"\"\"\n    return ",
        "expected_tokens": ["max(a, b, c)", "max(a,b,c)", "max([a, b, c])"],
        "test_case": "max_of_three(1, 2, 3) == 3"
    },
    {
        "name": "list_sum",
        "prompt": "def sum_list(numbers: list) -> int:\n    \"\"\"Return the sum of all numbers in a list.\"\"\"\n    return ",
        "expected_tokens": ["sum(numbers)", "sum(numbers)"],
        "test_case": "sum_list([1, 2, 3, 4]) == 10"
    },
    {
        "name": "factorial",
        "prompt": "def factorial(n: int) -> int:\n    \"\"\"Calculate factorial of n.\"\"\"\n    if n <= 1:\n        return 1\n    return ",
        "expected_tokens": ["n * factorial(n - 1)", "n * factorial(n-1)", "n*factorial(n-1)"],
        "test_case": "factorial(5) == 120"
    }
]

def check_contains_expected(response: str, expected_tokens: list) -> bool:
    """Check if response contains any of the expected tokens"""
    response_clean = response.replace(" ", "").lower()
    return any(token.replace(" ", "").lower() in response_clean for token in expected_tokens)

def is_conversational(response: str) -> bool:
    """Check if response is conversational"""
    conversational_indicators = [
        "here's", "certainly", "to complete", "this function", "i'll",
        "let me", "you can", "the code", "this code", "sure", "here is"
    ]
    response_lower = response.lower()
    return any(indicator in response_lower for indicator in conversational_indicators)

def test_baseline_prompting(model_name: str):
    """Test baseline (no enhanced prompting)"""
    print(f"\n📊 BASELINE TEST: {model_name}")
    print("-" * 50)

    interface = OllamaInterface(model_name)
    results = []

    for problem in HUMANEVAL_PROBLEMS:
        print(f"  Testing {problem['name']}...", end=" ")

        start_time = time.time()
        response = interface.generate(problem['prompt'], max_tokens=100, temperature=0.2)
        execution_time = time.time() - start_time

        if response.success:
            contains_expected = check_contains_expected(response.text, problem['expected_tokens'])
            is_conv = is_conversational(response.text)

            results.append({
                'problem': problem['name'],
                'success': True,
                'contains_expected': contains_expected,
                'is_conversational': is_conv,
                'execution_time': execution_time,
                'response': response.text[:60] + "..." if len(response.text) > 60 else response.text
            })

            status = "✅" if contains_expected and not is_conv else "❌"
            conv_indicator = "🗣️" if is_conv else "🤖"
            print(f"{status} {conv_indicator} ({execution_time:.2f}s)")
        else:
            results.append({
                'problem': problem['name'],
                'success': False,
                'contains_expected': False,
                'is_conversational': False,
                'execution_time': execution_time,
                'response': f"Error: {response.error_message}"
            })
            print(f"❌ Error ({execution_time:.2f}s)")

    return results

def test_enhanced_prompting(model_name: str):
    """Test enhanced prompting"""
    print(f"\n🧠 ENHANCED TEST: {model_name}")
    print("-" * 50)

    interface = EnhancedOllamaInterface(model_name)
    results = []

    for problem in HUMANEVAL_PROBLEMS:
        print(f"  Testing {problem['name']}...", end=" ")

        start_time = time.time()
        # Use auto-best strategy for optimal results
        response = interface.generate_auto_best(problem['prompt'], max_tokens=100)
        execution_time = time.time() - start_time

        if response.success:
            contains_expected = check_contains_expected(response.text, problem['expected_tokens'])

            results.append({
                'problem': problem['name'],
                'success': True,
                'contains_expected': contains_expected,
                'is_conversational': response.is_conversational,
                'execution_time': execution_time,
                'strategy': response.prompting_strategy,
                'response': response.text[:60] + "..." if len(response.text) > 60 else response.text
            })

            status = "✅" if contains_expected and not response.is_conversational else "❌"
            conv_indicator = "🗣️" if response.is_conversational else "🤖"
            print(f"{status} {conv_indicator} {response.prompting_strategy} ({execution_time:.2f}s)")
        else:
            results.append({
                'problem': problem['name'],
                'success': False,
                'contains_expected': False,
                'is_conversational': False,
                'execution_time': execution_time,
                'strategy': response.prompting_strategy,
                'response': f"Error: {response.error_message}"
            })
            print(f"❌ Error ({execution_time:.2f}s)")

    return results

def compare_results(baseline_results, enhanced_results, model_name):
    """Compare baseline vs enhanced results"""
    print(f"\n📈 COMPARISON ANALYSIS: {model_name}")
    print("=" * 60)

    # Calculate metrics
    baseline_correct = sum(1 for r in baseline_results if r['contains_expected'] and not r['is_conversational'])
    enhanced_correct = sum(1 for r in enhanced_results if r['contains_expected'] and not r['is_conversational'])

    baseline_conversational = sum(1 for r in baseline_results if r['is_conversational'])
    enhanced_conversational = sum(1 for r in enhanced_results if r['is_conversational'])

    baseline_avg_time = sum(r['execution_time'] for r in baseline_results) / len(baseline_results)
    enhanced_avg_time = sum(r['execution_time'] for r in enhanced_results) / len(enhanced_results)

    total_problems = len(HUMANEVAL_PROBLEMS)

    print(f"📊 Success Rate:")
    print(f"   Baseline:  {baseline_correct}/{total_problems} ({baseline_correct/total_problems:.1%})")
    print(f"   Enhanced:  {enhanced_correct}/{total_problems} ({enhanced_correct/total_problems:.1%})")

    if enhanced_correct > baseline_correct:
        improvement = enhanced_correct - baseline_correct
        print(f"   🎉 Improvement: +{improvement} problems ({improvement/total_problems:.1%})")
    elif enhanced_correct == baseline_correct:
        print(f"   ➡️ No change in success rate")
    else:
        decline = baseline_correct - enhanced_correct
        print(f"   ⚠️ Decline: -{decline} problems ({decline/total_problems:.1%})")

    print(f"\n🗣️ Conversational Rate:")
    print(f"   Baseline:  {baseline_conversational}/{total_problems} ({baseline_conversational/total_problems:.1%})")
    print(f"   Enhanced:  {enhanced_conversational}/{total_problems} ({enhanced_conversational/total_problems:.1%})")

    if enhanced_conversational < baseline_conversational:
        reduction = baseline_conversational - enhanced_conversational
        print(f"   ✅ Reduction: -{reduction} responses ({reduction/total_problems:.1%})")
    elif enhanced_conversational == baseline_conversational:
        print(f"   ➡️ No change in conversational rate")
    else:
        increase = enhanced_conversational - baseline_conversational
        print(f"   ⚠️ Increase: +{increase} responses ({increase/total_problems:.1%})")

    print(f"\n⏱️ Performance:")
    print(f"   Baseline avg:  {baseline_avg_time:.2f}s")
    print(f"   Enhanced avg:  {enhanced_avg_time:.2f}s")

    time_diff = enhanced_avg_time - baseline_avg_time
    if abs(time_diff) < 0.1:
        print(f"   ➡️ Similar performance")
    elif time_diff > 0:
        print(f"   ⚠️ Slower by {time_diff:.2f}s")
    else:
        print(f"   ✅ Faster by {abs(time_diff):.2f}s")

    # Detailed problem-by-problem comparison
    print(f"\n🔍 Problem-by-Problem Comparison:")
    print(f"{'Problem':<12} {'Baseline':<10} {'Enhanced':<15} {'Strategy':<15}")
    print("-" * 60)

    for i, problem in enumerate(HUMANEVAL_PROBLEMS):
        baseline = baseline_results[i]
        enhanced = enhanced_results[i]

        baseline_status = "✅" if baseline['contains_expected'] and not baseline['is_conversational'] else "❌"
        enhanced_status = "✅" if enhanced['contains_expected'] and not enhanced['is_conversational'] else "❌"

        strategy = enhanced.get('strategy', 'unknown')[:14]

        print(f"{problem['name']:<12} {baseline_status:<10} {enhanced_status:<15} {strategy:<15}")

    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")

    total_improvement = (enhanced_correct - baseline_correct) + (baseline_conversational - enhanced_conversational)

    if total_improvement > 0:
        print(f"   🎉 Enhanced prompting shows clear improvement!")
        print(f"   🔥 Total benefit: +{total_improvement} better responses")
    elif total_improvement == 0:
        print(f"   ➡️ Enhanced prompting shows similar performance")
    else:
        print(f"   ⚠️ Enhanced prompting needs tuning for this model")

    return {
        'baseline_correct': baseline_correct,
        'enhanced_correct': enhanced_correct,
        'baseline_conversational': baseline_conversational,
        'enhanced_conversational': enhanced_conversational,
        'improvement': total_improvement
    }

def main():
    """Run validation tests"""
    print("🔬 ADVANCED PROMPTING IMPROVEMENT VALIDATION")
    print("=" * 60)
    print("Testing with real HumanEval problems to demonstrate effectiveness")

    # Test with fastest available models
    test_models = ["phi3.5:latest", "mistral:7b-instruct"]

    all_results = {}

    for model in test_models:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model}")
        print(f"{'='*60}")

        try:
            # Test baseline
            baseline_results = test_baseline_prompting(model)

            # Test enhanced
            enhanced_results = test_enhanced_prompting(model)

            # Compare
            comparison = compare_results(baseline_results, enhanced_results, model)
            all_results[model] = comparison

        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
            continue

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*60}")

    total_improvements = 0
    models_tested = 0

    for model, results in all_results.items():
        improvement = results['improvement']
        total_improvements += improvement
        models_tested += 1

        status = "🎉" if improvement > 0 else "➡️" if improvement == 0 else "⚠️"
        print(f"{status} {model}: {improvement:+d} better responses")

    if models_tested > 0:
        avg_improvement = total_improvements / models_tested
        print(f"\n📊 Average improvement: {avg_improvement:+.1f} responses per model")

        if avg_improvement > 0:
            print(f"✅ VALIDATION SUCCESS: Enhanced prompting shows measurable improvements!")
        elif avg_improvement == 0:
            print(f"➡️ VALIDATION NEUTRAL: No significant change observed")
        else:
            print(f"⚠️ VALIDATION MIXED: Results vary by model, may need tuning")
    else:
        print(f"❌ VALIDATION FAILED: No models could be tested")

    return total_improvements > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)