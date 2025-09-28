#!/usr/bin/env python3
"""
Test Enhanced Prompting Integration

Quick validation script to test the enhanced prompting capabilities
before running full evaluations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.enhanced_ollama_interface import EnhancedOllamaInterface, PromptingStrategy
import requests
import time

def test_ollama_connection():
    """Test if Ollama is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"✅ Ollama connected. Available models: {models}")
            return models
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return []

def test_enhanced_interface():
    """Test the enhanced Ollama interface"""
    print("\n" + "="*60)
    print("TESTING ENHANCED OLLAMA INTERFACE")
    print("="*60)

    available_models = test_ollama_connection()
    if not available_models:
        print("❌ Cannot test without Ollama connection")
        return

    # Test with the first available model
    test_model = available_models[0]
    print(f"\n🧪 Testing with model: {test_model}")

    interface = EnhancedOllamaInterface(test_model)

    # Test simple code completion
    test_prompt = "def add_numbers(a, b):\n    return "
    expected = "a + b"

    print(f"\n📝 Test prompt: {test_prompt.strip()}")
    print(f"🎯 Expected completion: {expected}")

    print(f"\n🧠 Testing auto-best strategy...")
    response = interface.generate_auto_best(test_prompt, max_tokens=50)

    print(f"   Strategy used: {response.prompting_strategy}")
    print(f"   Success: {response.success}")
    print(f"   Conversational: {response.is_conversational}")
    print(f"   Execution time: {response.execution_time:.2f}s")
    print(f"   Response: '{response.text[:100]}{'...' if len(response.text) > 100 else ''}'")

    if expected.lower() in response.text.lower():
        print("   ✅ Contains expected completion")
    else:
        print("   ❌ Does not contain expected completion")

    return response.success and not response.is_conversational

def test_strategy_comparison():
    """Test different prompting strategies"""
    print("\n" + "="*60)
    print("TESTING STRATEGY COMPARISON")
    print("="*60)

    available_models = test_ollama_connection()
    if not available_models:
        return

    test_model = available_models[0]
    interface = EnhancedOllamaInterface(test_model)

    test_prompt = "def is_even(n):\n    return n "
    expected = "% 2 == 0"

    print(f"\n📝 Test prompt: {test_prompt.strip()}")
    print(f"🎯 Expected completion: {expected}")

    strategies_to_test = [
        PromptingStrategy.CODE_ENGINE,
        PromptingStrategy.ROLE_BASED,
        PromptingStrategy.DETERMINISTIC,
        PromptingStrategy.NEGATIVE_PROMPT
    ]

    results = []
    for strategy in strategies_to_test:
        print(f"\n🧠 Testing {strategy.value} strategy...")

        response = interface.generate_with_strategy(test_prompt, strategy, max_tokens=50)

        success = response.success and not response.is_conversational
        contains_expected = expected.lower() in response.text.lower()

        results.append({
            'strategy': strategy.value,
            'success': success,
            'contains_expected': contains_expected,
            'time': response.execution_time
        })

        status = "✅" if success else "❌"
        expected_check = "✅" if contains_expected else "❌"
        conv_indicator = "🗣️" if response.is_conversational else "🤖"

        print(f"   {status} {conv_indicator} Success: {success}")
        print(f"   {expected_check} Contains expected: {contains_expected}")
        print(f"   ⏱️ Time: {response.execution_time:.2f}s")
        print(f"   📝 Response: '{response.text[:60]}{'...' if len(response.text) > 60 else ''}'")

    # Summary
    print(f"\n📊 STRATEGY COMPARISON SUMMARY:")
    successful_strategies = [r for r in results if r['success'] and r['contains_expected']]
    print(f"   Successful strategies: {len(successful_strategies)}/{len(results)}")

    if successful_strategies:
        best = min(successful_strategies, key=lambda x: x['time'])
        print(f"   Fastest successful: {best['strategy']} ({best['time']:.2f}s)")

    return len(successful_strategies) > 0

def test_enhanced_unified_runner():
    """Test the enhanced unified runner"""
    print("\n" + "="*60)
    print("TESTING ENHANCED UNIFIED RUNNER")
    print("="*60)

    available_models = test_ollama_connection()
    if not available_models:
        return

    # Import and test the enhanced runner
    try:
        from enhanced_unified_runner import EnhancedUnifiedRunner
        runner = EnhancedUnifiedRunner()

        test_model = available_models[0]
        print(f"\n🧪 Testing enhanced runner with model: {test_model}")

        # Test direct interface
        print(f"\n🎯 Testing direct interface...")
        result = runner.run_benchmark("test_task", test_model,
                                    prompting_strategy="auto_best",
                                    limit=1, safe_mode=False)

        print(f"   Harness: {result.harness}")
        print(f"   Score: {result.score}")
        print(f"   Strategy: {result.prompting_strategy}")
        print(f"   Conversational: {result.is_conversational}")
        print(f"   Time: {result.execution_time:.2f}s")

        success = result.score > 0 and not result.is_conversational
        print(f"   {'✅' if success else '❌'} Overall success: {success}")

        return success

    except Exception as e:
        print(f"❌ Enhanced runner test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ENHANCED PROMPTING INTEGRATION TESTS")
    print("="*60)

    test_results = []

    # Test 1: Enhanced interface
    print("\n📍 TEST 1: Enhanced Interface")
    result1 = test_enhanced_interface()
    test_results.append(("Enhanced Interface", result1))

    # Test 2: Strategy comparison
    print("\n📍 TEST 2: Strategy Comparison")
    result2 = test_strategy_comparison()
    test_results.append(("Strategy Comparison", result2))

    # Test 3: Enhanced runner
    print("\n📍 TEST 3: Enhanced Runner")
    result3 = test_enhanced_unified_runner()
    test_results.append(("Enhanced Runner", result3))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1

    print(f"\n📊 Overall: {passed_tests}/{len(test_results)} tests passed")

    if passed_tests == len(test_results):
        print("🎉 All tests passed! Enhanced prompting integration is working.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)