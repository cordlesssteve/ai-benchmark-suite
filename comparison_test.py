#!/usr/bin/env python3
"""
Before/After Comparison Test

Demonstrates the improvements made by the fixes:
- Original enhanced interface (with issues)
- Fixed enhanced interface (with remediation fixes)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.enhanced_ollama_interface import EnhancedOllamaInterface
from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface

def comparison_test():
    """Compare original vs fixed implementation"""
    print("🔍 BEFORE/AFTER COMPARISON TEST")
    print("="*60)

    test_prompt = "def add(a, b):\n    return "
    model = "phi3.5:latest"

    print(f"Test prompt: {test_prompt.strip()}")
    print(f"Model: {model}")
    print()

    # Test original implementation
    print("📊 BEFORE (Original Implementation):")
    print("-" * 40)

    try:
        original = EnhancedOllamaInterface(model)

        if original.is_available():
            original_response = original.generate_auto_best(test_prompt, max_tokens=50)

            print(f"   Success: {original_response.success}")
            print(f"   Response: '{original_response.text}'")
            print(f"   Length: {len(original_response.text)}")
            print(f"   Strategy: {original_response.prompting_strategy}")
            print(f"   Conversational: {original_response.is_conversational}")
            print(f"   Time: {original_response.execution_time:.2f}s")

            original_success = original_response.success and len(original_response.text.strip()) > 0
        else:
            print("   ❌ Ollama not available")
            original_success = False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        original_success = False

    print()

    # Test fixed implementation
    print("🔧 AFTER (Fixed Implementation):")
    print("-" * 40)

    try:
        fixed = FixedEnhancedOllamaInterface(model)

        if fixed.is_available():
            fixed_response = fixed.generate_auto_best(test_prompt, max_tokens=50)

            print(f"   Success: {fixed_response.success}")
            print(f"   Response: '{fixed_response.text}'")
            print(f"   Length: {len(fixed_response.text)}")
            print(f"   Strategy: {fixed_response.prompting_strategy}")
            print(f"   Conversational: {fixed_response.is_conversational}")
            print(f"   Time: {fixed_response.execution_time:.2f}s")

            fixed_success = fixed_response.success and len(fixed_response.text.strip()) > 0
        else:
            print("   ❌ Ollama not available")
            fixed_success = False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        fixed_success = False

    print()

    # Comparison summary
    print("📈 IMPROVEMENT SUMMARY:")
    print("-" * 40)

    if original_success and fixed_success:
        print("   ✅ Both implementations working")
        if len(fixed_response.text) > len(original_response.text):
            print("   📈 Fixed version produces more content")
        if not fixed_response.is_conversational and original_response.is_conversational:
            print("   🎯 Fixed version reduces conversational responses")
    elif not original_success and fixed_success:
        print("   🎉 MAJOR IMPROVEMENT: Fixed version works, original failed")
    elif original_success and not fixed_success:
        print("   ⚠️ REGRESSION: Fixed version fails, original worked")
    else:
        print("   ❌ Both implementations have issues")

    return fixed_success

if __name__ == "__main__":
    success = comparison_test()
    print(f"\n{'✅ FIXES SUCCESSFUL' if success else '❌ FIXES NEED MORE WORK'}")