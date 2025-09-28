#!/usr/bin/env python3
"""
Manual Model Performance Test

Quick manual test of specific models to get actual performance data.
"""

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface

def test_single_model(model_name: str):
    """Test a single model with one simple prompt"""
    print(f"\n🧪 Testing {model_name}")
    print("-" * 50)

    try:
        interface = FixedEnhancedOllamaInterface(model_name)

        if not interface.is_available():
            print("❌ Ollama server not available")
            return None

        # Simple test
        test_prompt = "def add(a, b):\n    return "
        expected = "a + b"

        print(f"Prompt: {test_prompt.strip()}")
        print(f"Expected: {expected}")

        start_time = time.time()
        response = interface.generate_auto_best(test_prompt, max_tokens=50, timeout=10)
        execution_time = time.time() - start_time

        print(f"\nResult:")
        print(f"  Success: {response.success}")
        print(f"  Strategy: {response.prompting_strategy}")
        print(f"  Time: {execution_time:.2f}s")
        print(f"  Conversational: {response.is_conversational}")
        print(f"  Response: '{response.text}'")

        # Evaluate
        has_content = len(response.text.strip()) > 0
        contains_expected = expected.lower() in response.text.lower()
        overall_success = response.success and has_content and contains_expected and not response.is_conversational

        print(f"  Has content: {has_content}")
        print(f"  Contains '{expected}': {contains_expected}")
        print(f"  Overall success: {'✅' if overall_success else '❌'}")

        return {
            "model": model_name,
            "success": overall_success,
            "execution_time": execution_time,
            "strategy": response.prompting_strategy,
            "conversational": response.is_conversational,
            "response": response.text[:50]
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "model": model_name,
            "success": False,
            "execution_time": 0,
            "error": str(e)
        }

def main():
    """Test the fastest/most reliable models manually"""
    print("🚀 MANUAL MODEL PERFORMANCE TEST")
    print("="*60)

    # Test fastest models first
    models_to_test = [
        "phi3.5:latest",    # Known to work well
        "mistral:7b-instruct",  # Known to work well
        "qwen2.5-coder:3b", # Code-focused, should be good
        "tinyllama:1.1b",   # Smallest, fastest
        "qwen2.5:0.5b",     # Very small
    ]

    results = []

    for model in models_to_test:
        result = test_single_model(model)
        if result:
            results.append(result)

    # Create summary table
    print(f"\n{'='*80}")
    print("📊 MANUAL TEST RESULTS SUMMARY")
    print(f"{'='*80}")

    if results:
        print(f"{'Model':<25} {'Success':<8} {'Time':<8} {'Strategy':<15} {'Conv':<5}")
        print("-"*80)

        for result in results:
            success = "✅" if result['success'] else "❌"
            time_str = f"{result['execution_time']:.1f}s"
            strategy = result.get('strategy', 'error')[:14]
            conv = "🗣️" if result.get('conversational', False) else "🤖"

            print(f"{result['model']:<25} {success:<8} {time_str:<8} {strategy:<15} {conv:<5}")

        # Best performer
        working_models = [r for r in results if r['success']]
        if working_models:
            best = min(working_models, key=lambda x: x['execution_time'])
            print(f"\n🏆 Best performer: {best['model']} ({best['execution_time']:.1f}s)")

            # Create performance table
            print(f"\n📈 ACTUAL MODEL PERFORMANCE TABLE")
            print(f"{'='*80}")
            print(f"{'Model':<25} {'Size':<8} {'Success':<8} {'AvgTime':<8} {'Status':<12}")
            print("-"*80)

            # Model sizes
            sizes = {
                "phi3.5:latest": "2.2GB",
                "mistral:7b-instruct": "4.4GB",
                "qwen2.5-coder:3b": "1.9GB",
                "tinyllama:1.1b": "637MB",
                "qwen2.5:0.5b": "397MB"
            }

            for result in sorted(results, key=lambda x: x['execution_time'] if x['success'] else 999):
                model = result['model']
                size = sizes.get(model, "Unknown")
                success = "Yes" if result['success'] else "No"
                time_str = f"{result['execution_time']:.1f}s" if result['success'] else "Failed"

                if result['success']:
                    status = "🟢 Working"
                else:
                    status = "❌ Failed"

                print(f"{model:<25} {size:<8} {success:<8} {time_str:<8} {status:<12}")

        else:
            print("\n❌ No models working successfully")

    else:
        print("❌ No results collected")

    return len([r for r in results if r['success']]) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)