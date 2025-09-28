#!/usr/bin/env python3
"""
Diagnostic Test - Investigate Binary Results

Analyzes why we're getting 0%/100% results instead of gradual performance metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface

def detailed_analysis():
    """Detailed analysis of model responses"""
    print("🔍 DIAGNOSTIC ANALYSIS - Binary Results Investigation")
    print("="*70)

    # Test with the working models
    models_to_diagnose = ["qwen2.5:0.5b", "phi3.5:latest"]

    # Multiple test cases with varying difficulty
    test_cases = [
        {
            "prompt": "def add(a, b):\n    return ",
            "expected": ["a + b", "a+b"],
            "difficulty": "trivial"
        },
        {
            "prompt": "def multiply(x, y):\n    return ",
            "expected": ["x * y", "x*y"],
            "difficulty": "trivial"
        },
        {
            "prompt": "def factorial(n):\n    if n <= 1:\n        return 1\n    return ",
            "expected": ["n * factorial(n - 1)", "n * factorial(n-1)"],
            "difficulty": "medium"
        },
        {
            "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
            "expected": ["fibonacci(n-1) + fibonacci(n-2)", "fibonacci(n - 1) + fibonacci(n - 2)"],
            "difficulty": "medium"
        },
        {
            "prompt": "class Calculator:\n    def __init__(self):\n        self.",
            "expected": ["value = 0", "result = 0", "total = 0"],
            "difficulty": "hard"
        }
    ]

    for model in models_to_diagnose:
        print(f"\n🧪 DIAGNOSING: {model}")
        print("-" * 50)

        interface = FixedEnhancedOllamaInterface(model)
        if not interface.is_available():
            print("❌ Ollama not available")
            continue

        model_results = []

        for i, test in enumerate(test_cases, 1):
            print(f"\n  Test {i}/{len(test_cases)} ({test['difficulty']})")
            print(f"  Prompt: {repr(test['prompt'])}")
            print(f"  Expected: {test['expected']}")

            try:
                response = interface.generate_auto_best(test['prompt'], max_tokens=100, timeout=8)

                # Detailed analysis
                has_content = len(response.text.strip()) > 0

                # Check each expected answer individually
                individual_matches = []
                for exp in test['expected']:
                    # Original logic - exact substring match
                    exact_match = exp.lower() in response.text.lower()

                    # More flexible logic - cleaned comparison
                    response_clean = response.text.replace(" ", "").replace("\n", "").lower()
                    exp_clean = exp.replace(" ", "").lower()
                    flexible_match = exp_clean in response_clean

                    individual_matches.append({
                        "expected": exp,
                        "exact_match": exact_match,
                        "flexible_match": flexible_match
                    })

                # Overall evaluation
                any_exact_match = any(m["exact_match"] for m in individual_matches)
                any_flexible_match = any(m["flexible_match"] for m in individual_matches)

                # Current binary logic
                current_success = response.success and has_content and any_exact_match and not response.is_conversational

                print(f"    Raw Response: {repr(response.text[:100])}")
                print(f"    Success Flag: {response.success}")
                print(f"    Has Content: {has_content}")
                print(f"    Conversational: {response.is_conversational}")
                print(f"    Strategy: {response.prompting_strategy}")

                print(f"    Match Analysis:")
                for match in individual_matches:
                    print(f"      '{match['expected']}': exact={match['exact_match']}, flexible={match['flexible_match']}")

                print(f"    Any Exact Match: {any_exact_match}")
                print(f"    Any Flexible Match: {any_flexible_match}")
                print(f"    Current Binary Result: {'✅' if current_success else '❌'}")

                # Store for summary
                model_results.append({
                    "difficulty": test['difficulty'],
                    "current_success": current_success,
                    "has_content": has_content,
                    "any_exact_match": any_exact_match,
                    "any_flexible_match": any_flexible_match,
                    "conversational": response.is_conversational,
                    "response_length": len(response.text)
                })

            except Exception as e:
                print(f"    ❌ Error: {e}")
                model_results.append({
                    "difficulty": test['difficulty'],
                    "current_success": False,
                    "error": str(e)
                })

        # Summary for this model
        print(f"\n  📊 SUMMARY FOR {model}:")
        success_by_difficulty = {}
        for result in model_results:
            if "error" not in result:
                diff = result["difficulty"]
                if diff not in success_by_difficulty:
                    success_by_difficulty[diff] = []
                success_by_difficulty[diff].append(result["current_success"])

        for difficulty, successes in success_by_difficulty.items():
            success_rate = (sum(successes) / len(successes)) * 100 if successes else 0
            print(f"    {difficulty.capitalize()}: {success_rate:.1f}% success ({sum(successes)}/{len(successes)})")

        # Identify patterns
        has_content_count = sum(1 for r in model_results if "error" not in r and r["has_content"])
        conversational_count = sum(1 for r in model_results if "error" not in r and r["conversational"])

        print(f"    Response Pattern:")
        print(f"      Non-empty responses: {has_content_count}/{len([r for r in model_results if 'error' not in r])}")
        print(f"      Conversational responses: {conversational_count}/{len([r for r in model_results if 'error' not in r])}")

def main():
    detailed_analysis()

    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    print("="*70)
    print("Potential causes of binary 0%/100% results:")
    print("1. ❌ SINGLE TEST CASE - Only testing one trivial prompt")
    print("2. ❌ BINARY EVALUATION - All-or-nothing success criteria")
    print("3. ❌ STRING MATCHING - Exact substring match requirements")
    print("4. ❌ CONVERSATIONAL FLAG - Blanket rejection of conversational responses")
    print("5. ❌ TIMEOUT SENSITIVITY - Network/VRAM issues causing false negatives")
    print("\nSolution: Use multiple test cases with graduated difficulty and partial scoring")

if __name__ == "__main__":
    main()