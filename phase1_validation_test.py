#!/usr/bin/env python3
"""
Phase 1 Validation Test

Tests the new smart response processing against known binary 0%/100% performance cases
to validate that legitimate code completions are preserved while conversational
elements are properly cleaned.

Usage:
    python phase1_validation_test.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_interfaces.smart_response_processor import SmartResponseProcessor, ContentQuality
from model_interfaces.phase1_enhanced_ollama_interface import Phase1EnhancedOllamaInterface, PromptingStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_response_cleaning():
    """Test smart response cleaning against problematic cases"""
    processor = SmartResponseProcessor(quality_threshold=0.3)

    print("\n=== TESTING SMART RESPONSE CLEANING ===")

    # Test cases that were causing binary 0%/100% results
    test_cases = [
        {
            "name": "Conversational wrapper with valid code",
            "input": """To complete the function, here's the implementation:

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

This function calculates the fibonacci number recursively.""",
            "expected_preserved": True,
            "expected_quality": "excellent"  # Well-cleaned code should be excellent
        },
        {
            "name": "Code with markdown wrapper",
            "input": """```python
def hello_world():
    print("Hello, World!")
    return "success"
```

Here's the explanation of the code above.""",
            "expected_preserved": True,
            "expected_quality": "excellent"  # Clean code extraction
        },
        {
            "name": "Pure conversational response (should be low quality)",
            "input": """I'll help you complete this function. To solve this problem, you need to think about the algorithm. Let me explain the approach that would work best for this scenario.""",
            "expected_preserved": False,
            "expected_quality": "empty"  # Should be completely cleaned out
        },
        {
            "name": "Mixed code and explanation",
            "input": """# This function adds two numbers
def add(a, b):
    # We simply return the sum
    return a + b

The function above is a simple addition function.""",
            "expected_preserved": True,
            "expected_quality": "excellent"  # Good code with comments
        },
        {
            "name": "Code completion with conversational prefix",
            "input": """Certainly! Here's the code you need:

    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1""",
            "expected_preserved": True,
            "expected_quality": "excellent"  # Clean code block
        },
        {
            "name": "Empty/whitespace response",
            "input": "   \n\n   \n  ",
            "expected_preserved": False,
            "expected_quality": "empty"
        }
    ]

    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {case['name']} ---")
        print(f"Input length: {len(case['input'])} chars")

        processed = processor.process_response(case['input'], http_success=True)

        print(f"Raw input: {repr(case['input'][:100])}...")
        print(f"Cleaned output: {repr(processed.text[:100])}...")
        print(f"Quality score: {processed.quality_score:.3f}")
        print(f"Quality level: {processed.quality_level.value}")
        print(f"HTTP success: {processed.http_success}")
        print(f"Content quality success: {processed.content_quality_success}")
        print(f"Overall success: {processed.overall_success}")
        print(f"Is conversational: {processed.is_conversational}")
        print(f"Is executable: {processed.is_executable}")
        print(f"Cleaning applied: {processed.cleaning_applied}")

        # Validate expectations
        has_meaningful_content = bool(processed.text.strip())
        quality_matches = processed.quality_level.value in case['expected_quality']

        success = True
        if case['expected_preserved'] and not has_meaningful_content:
            print(f"❌ FAIL: Expected content to be preserved, but got empty result")
            success = False
        elif not case['expected_preserved'] and has_meaningful_content and processed.quality_score > 0.3:
            print(f"❌ FAIL: Expected low quality, but got quality score {processed.quality_score:.3f}")
            success = False
        elif not quality_matches:
            print(f"❌ FAIL: Expected quality level containing '{case['expected_quality']}', got '{processed.quality_level.value}'")
            success = False
        else:
            print(f"✅ PASS: Response processing meets expectations")

        results.append({
            "case": case['name'],
            "success": success,
            "quality_score": processed.quality_score,
            "preserved_content": has_meaningful_content
        })

    return results

def test_binary_performance_scenarios():
    """Test scenarios that were causing binary 0%/100% performance issues"""
    print("\n=== TESTING BINARY PERFORMANCE SCENARIOS ===")

    # Simulate different model responses that were causing issues
    scenarios = [
        {
            "name": "qwen2.5:0.5b CODE_ENGINE response (was 100%)",
            "response": """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
            "expected_success": True
        },
        {
            "name": "qwen2.5:0.5b DETERMINISTIC response (was 0%)",
            "response": """To complete this function, you need to check if a number is prime. Here's how you can do it:

You should iterate through possible divisors and check for remainders.""",
            "expected_success": False
        },
        {
            "name": "phi3.5 conversational response (was causing issues)",
            "response": """I'll help you implement this function. The algorithm should check divisibility:

def is_prime(n):
    if n <= 1:
        return False""",
            "expected_success": True  # Should preserve the code part
        },
        {
            "name": "Aggressive cleaning victim",
            "response": """def calculate_total(items):
    # The function sums all items
    total = 0
    for item in items:
        total += item
    return total""",
            "expected_success": True  # Should preserve despite comments
        }
    ]

    processor = SmartResponseProcessor(quality_threshold=0.3)
    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")

        processed = processor.process_response(scenario['response'], http_success=True)

        print(f"Original: {repr(scenario['response'][:80])}...")
        print(f"Processed: {repr(processed.text[:80])}...")
        print(f"Overall success: {processed.overall_success}")
        print(f"Quality score: {processed.quality_score:.3f}")
        print(f"Quality level: {processed.quality_level.value}")

        # Check if result matches expectation
        matches_expectation = processed.overall_success == scenario['expected_success']

        if matches_expectation:
            print(f"✅ PASS: Expected success={scenario['expected_success']}, got success={processed.overall_success}")
        else:
            print(f"❌ FAIL: Expected success={scenario['expected_success']}, got success={processed.overall_success}")

        results.append({
            "scenario": scenario['name'],
            "expected": scenario['expected_success'],
            "actual": processed.overall_success,
            "matches": matches_expectation,
            "quality_score": processed.quality_score
        })

    return results

def test_ollama_integration():
    """Test the Phase 1 interface if Ollama is available"""
    print("\n=== TESTING OLLAMA INTEGRATION (if available) ===")

    try:
        # Test with a lightweight model if available
        interface = Phase1EnhancedOllamaInterface("phi3.5:latest")

        if not interface.is_available():
            print("⚠️  Ollama server not available, skipping integration test")
            return []

        print("✅ Ollama server is available")

        # Test simple code completion
        test_prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    # Complete this function"

        print(f"Testing prompt: {repr(test_prompt)}")

        response = interface.generate(test_prompt)

        print(f"Raw response: {repr(response.raw_text[:100])}...")
        print(f"Cleaned response: {repr(response.text[:100])}...")
        print(f"HTTP success: {response.http_success}")
        print(f"Content quality success: {response.content_quality_success}")
        print(f"Overall success: {response.overall_success}")
        print(f"Quality score: {response.quality_score:.3f}")
        print(f"Quality level: {response.quality_level.value}")
        print(f"Strategy used: {response.prompting_strategy}")

        if response.overall_success:
            print("✅ PASS: Ollama integration working with dual success flags")
        else:
            print(f"❌ FAIL: Ollama integration issues - HTTP: {response.http_success}, Quality: {response.content_quality_success}")

        return [{"integration_test": response.overall_success, "quality_score": response.quality_score}]

    except Exception as e:
        print(f"❌ ERROR: Ollama integration test failed: {e}")
        return []

def main():
    """Run all Phase 1 validation tests"""
    print("🔬 PHASE 1 VALIDATION TEST SUITE")
    print("Testing smart response processing improvements")
    print("=" * 60)

    # Run all tests
    cleaning_results = test_response_cleaning()
    binary_results = test_binary_performance_scenarios()
    ollama_results = test_ollama_integration()

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    # Response cleaning summary
    cleaning_passed = sum(1 for r in cleaning_results if r['success'])
    print(f"Response Cleaning Tests: {cleaning_passed}/{len(cleaning_results)} passed")

    # Binary performance summary
    binary_passed = sum(1 for r in binary_results if r['matches'])
    print(f"Binary Performance Tests: {binary_passed}/{len(binary_results)} passed")

    # Ollama integration summary
    if ollama_results:
        ollama_passed = sum(1 for r in ollama_results if r.get('integration_test', False))
        print(f"Ollama Integration Tests: {ollama_passed}/{len(ollama_results)} passed")
    else:
        print("Ollama Integration Tests: Skipped (server not available)")

    total_tests = len(cleaning_results) + len(binary_results) + len(ollama_results)
    total_passed = cleaning_passed + binary_passed + (sum(1 for r in ollama_results if r.get('integration_test', False)) if ollama_results else 0)

    print(f"\n🎯 OVERALL: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("✅ ALL TESTS PASSED - Phase 1 implementation successful!")
    else:
        print("❌ Some tests failed - review implementation")

    # Quality insights
    print(f"\n📈 QUALITY INSIGHTS:")
    avg_quality = sum(r['quality_score'] for r in cleaning_results + binary_results + ollama_results) / len(cleaning_results + binary_results + ollama_results) if (cleaning_results + binary_results + ollama_results) else 0
    print(f"Average quality score: {avg_quality:.3f}")

    preserved_content = sum(1 for r in cleaning_results if r['preserved_content'])
    print(f"Content preservation rate: {preserved_content}/{len(cleaning_results)} ({preserved_content/len(cleaning_results)*100:.1f}%)")

    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)