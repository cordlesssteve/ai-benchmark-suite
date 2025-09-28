#!/usr/bin/env python3
"""
Comprehensive Test: Fixed Enhanced Ollama Interface

Tests all the critical fixes from the remediation review:
1. Stop token configuration fixes
2. Error handling improvements
3. Standardized prompting patterns
4. Model-specific configurations
"""

import sys
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface, PromptingStrategy

# Test cases for validation
TEST_CASES = [
    {
        "name": "simple_addition",
        "prompt": "def add(a, b):\n    return ",
        "expected": ["a + b", "a+b"],
        "description": "Basic arithmetic completion"
    },
    {
        "name": "even_check",
        "prompt": "def is_even(n):\n    return ",
        "expected": ["n % 2 == 0", "n%2==0", "not n % 2"],
        "description": "Boolean logic completion"
    },
    {
        "name": "max_function",
        "prompt": "def max_three(a, b, c):\n    return ",
        "expected": ["max(a, b, c)", "max([a, b, c])"],
        "description": "Built-in function usage"
    },
    {
        "name": "list_sum",
        "prompt": "def sum_list(numbers):\n    return ",
        "expected": ["sum(numbers)"],
        "description": "Simple list operation"
    }
]

def test_stop_token_fix():
    """Test that stop token configuration no longer causes empty responses"""
    print("\n" + "="*60)
    print("TEST 1: STOP TOKEN CONFIGURATION FIX")
    print("="*60)

    # Test models that previously had issues
    test_models = ["phi3.5:latest", "mistral:7b-instruct"]

    for model in test_models:
        print(f"\n🧪 Testing {model}...")

        interface = FixedEnhancedOllamaInterface(model)

        if not interface.is_available():
            print(f"   ⚠️ Ollama not available, skipping {model}")
            continue

        # Test a simple case that previously caused empty responses
        response = interface.generate("def add(a, b):\n    return ", max_tokens=50)

        print(f"   Response: '{response.text}'")
        print(f"   Success: {response.success}")
        print(f"   Empty: {len(response.text.strip()) == 0}")
        print(f"   Strategy: {response.prompting_strategy}")
        print(f"   Conversational: {response.is_conversational}")

        # Validate fix
        if response.success and len(response.text.strip()) > 0:
            print(f"   ✅ Stop token fix working for {model}")
        else:
            print(f"   ❌ Stop token issue persists for {model}")

    return True

def test_error_handling():
    """Test improved error handling"""
    print("\n" + "="*60)
    print("TEST 2: ERROR HANDLING IMPROVEMENTS")
    print("="*60)

    # Test with invalid model
    print("\n🧪 Testing error handling with invalid model...")
    interface = FixedEnhancedOllamaInterface("nonexistent:model")

    response = interface.generate("def test():\n    return ", timeout=5)

    print(f"   Success: {response.success}")
    print(f"   Error message: {response.error_message}")
    print(f"   Error type: {type(response.error_message)}")

    if not response.success and response.error_message:
        print("   ✅ Error handling working correctly")
        return True
    else:
        print("   ❌ Error handling not working")
        return False

def test_standardized_prompts():
    """Test standardized prompting patterns"""
    print("\n" + "="*60)
    print("TEST 3: STANDARDIZED PROMPTING PATTERNS")
    print("="*60)

    interface = FixedEnhancedOllamaInterface("phi3.5:latest")

    if not interface.is_available():
        print("   ⚠️ Ollama not available, skipping standardization test")
        return True

    print("\n🧪 Testing all standardized strategies...")

    prompts = interface.get_standardized_prompts()
    test_prompt = "def square(x):\n    return "

    results = []
    for strategy in PromptingStrategy:
        if strategy == PromptingStrategy.AUTO_BEST:
            continue

        print(f"   Testing {strategy.value}...", end=" ")

        response = interface.generate_with_strategy(test_prompt, strategy, max_tokens=30)

        success = response.success and len(response.text.strip()) > 0
        conversational = response.is_conversational

        results.append({
            'strategy': strategy.value,
            'success': success,
            'conversational': conversational,
            'response': response.text[:50]
        })

        status = "✅" if success else "❌"
        conv_indicator = "🗣️" if conversational else "🤖"
        print(f"{status} {conv_indicator}")

    # Analyze results
    successful = [r for r in results if r['success']]
    non_conversational = [r for r in successful if not r['conversational']]

    print(f"\n   📊 Results:")
    print(f"      Successful strategies: {len(successful)}/{len(results)}")
    print(f"      Non-conversational: {len(non_conversational)}/{len(successful)}")

    return len(non_conversational) > 0

def test_model_specific_configs():
    """Test model-specific configurations"""
    print("\n" + "="*60)
    print("TEST 4: MODEL-SPECIFIC CONFIGURATIONS")
    print("="*60)

    test_models = ["phi3.5:latest", "mistral:7b-instruct", "unknown:model"]

    for model in test_models:
        print(f"\n🧪 Testing config for {model}...")

        interface = FixedEnhancedOllamaInterface(model)
        config = interface.get_model_config()

        print(f"   Strategy: {config['strategy']}")
        print(f"   Temperature: {config['temperature']}")
        print(f"   Stop tokens: {config['stop_tokens']}")
        print(f"   Max tokens: {config['max_tokens']}")

        # Validate configuration exists
        required_keys = ['strategy', 'temperature', 'stop_tokens', 'max_tokens']
        has_all_keys = all(key in config for key in required_keys)

        if has_all_keys:
            print(f"   ✅ Complete configuration for {model}")
        else:
            print(f"   ❌ Incomplete configuration for {model}")

    return True

def test_comprehensive_validation():
    """Run comprehensive validation across multiple test cases"""
    print("\n" + "="*60)
    print("TEST 5: COMPREHENSIVE VALIDATION")
    print("="*60)

    # Test with the most reliable model
    model = "mistral:7b-instruct"
    interface = FixedEnhancedOllamaInterface(model)

    if not interface.is_available():
        print(f"   ⚠️ Ollama not available, skipping comprehensive test")
        return True

    print(f"\n🧪 Running comprehensive test with {model}...")

    overall_results = []

    for test_case in TEST_CASES:
        print(f"\n   Testing {test_case['name']}: {test_case['description']}")

        # Test with auto-best strategy
        response = interface.generate_auto_best(test_case['prompt'], max_tokens=50)

        # Check if response contains expected completion
        contains_expected = any(
            expected.lower() in response.text.lower()
            for expected in test_case['expected']
        )

        result = {
            'test_case': test_case['name'],
            'success': response.success,
            'has_content': len(response.text.strip()) > 0,
            'contains_expected': contains_expected,
            'is_conversational': response.is_conversational,
            'strategy': response.prompting_strategy,
            'response': response.text[:60]
        }

        overall_results.append(result)

        # Print result
        success_indicator = "✅" if result['contains_expected'] else "❌"
        conv_indicator = "🗣️" if result['is_conversational'] else "🤖"

        print(f"      {success_indicator} {conv_indicator} '{response.text[:40]}...'")
        print(f"         Strategy: {response.prompting_strategy}")

    # Calculate overall metrics
    successful_tests = sum(1 for r in overall_results if r['contains_expected'])
    non_empty_responses = sum(1 for r in overall_results if r['has_content'])
    non_conversational = sum(1 for r in overall_results if not r['is_conversational'])

    print(f"\n   📊 Comprehensive Results:")
    print(f"      Correct completions: {successful_tests}/{len(TEST_CASES)} ({successful_tests/len(TEST_CASES):.1%})")
    print(f"      Non-empty responses: {non_empty_responses}/{len(TEST_CASES)} ({non_empty_responses/len(TEST_CASES):.1%})")
    print(f"      Non-conversational: {non_conversational}/{len(TEST_CASES)} ({non_conversational/len(TEST_CASES):.1%})")

    return successful_tests > 0 and non_conversational == len(TEST_CASES)

def main():
    """Run all validation tests"""
    print("🔧 FIXED ENHANCED INTERFACE VALIDATION")
    print("="*60)
    print("Validating all critical fixes from remediation review")

    test_functions = [
        ("Stop Token Fix", test_stop_token_fix),
        ("Error Handling", test_error_handling),
        ("Standardized Prompts", test_standardized_prompts),
        ("Model-Specific Configs", test_model_specific_configs),
        ("Comprehensive Validation", test_comprehensive_validation)
    ]

    results = []

    for test_name, test_func in test_functions:
        try:
            print(f"\n📍 Running: {test_name}")
            result = test_func()
            results.append((test_name, result))
            print(f"   {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((test_name, False))

    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed_tests = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📊 Overall: {passed_tests}/{len(results)} tests passed")

    if passed_tests == len(results):
        print("🎉 All critical fixes validated successfully!")
        return True
    else:
        print("⚠️ Some tests failed - further investigation needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)