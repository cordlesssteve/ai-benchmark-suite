#!/usr/bin/env python3
"""
Phase 2 Validation Script
Comprehensive validation of the adaptive prompting system with real Ollama models.
"""

import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_interfaces.adaptive_ollama_interface import AdaptiveOllamaInterface
from model_interfaces.adaptive_benchmark_adapter import AdaptiveBenchmarkAdapter
from prompting.bandit_strategy_selector import PromptingStrategy


class Phase2Validator:
    """Comprehensive Phase 2 validation with real Ollama models"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.validation_results = {}

    def check_ollama_availability(self) -> Dict[str, Any]:
        """Check if Ollama is running and get available models"""
        print("🔍 OLLAMA AVAILABILITY CHECK")
        print("=" * 40)

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])

                print(f"✅ Ollama is running at {self.base_url}")
                if models:
                    print(f"✅ Found {len(models)} models:")
                    for model in models:
                        name = model.get('name', 'unknown')
                        size = model.get('size', 0)
                        size_gb = size / (1024**3) if size > 0 else 0
                        print(f"   - {name} ({size_gb:.1f}GB)")

                    return {
                        "available": True,
                        "models": [m['name'] for m in models],
                        "total_models": len(models)
                    }
                else:
                    print("⚠️  No models found")
                    print("   Run: ollama pull qwen2.5-coder:3b")
                    return {"available": True, "models": [], "total_models": 0}
            else:
                print(f"❌ Ollama responded with status {response.status_code}")
                return {"available": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama")
            print("   Solution 1: Start Ollama with 'ollama serve'")
            print("   Solution 2: Pull a model with 'ollama pull qwen2.5-coder:3b'")
            return {"available": False, "error": "Connection refused"}

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {"available": False, "error": str(e)}

    def validate_without_ollama(self) -> Dict[str, Any]:
        """Validate core system functionality without requiring Ollama"""
        print("\n🧪 CORE SYSTEM VALIDATION (No Ollama Required)")
        print("=" * 50)

        results = {}

        # Test 1: Interface creation and configuration
        print("\n1. Interface Creation:")
        try:
            interface = AdaptiveOllamaInterface(
                model_name="test_model",
                base_url=self.base_url
            )
            print("   ✅ AdaptiveOllamaInterface created")

            adapter = AdaptiveBenchmarkAdapter("test_model")
            print("   ✅ AdaptiveBenchmarkAdapter created")

            results["interface_creation"] = {"success": True}
        except Exception as e:
            print(f"   ❌ Interface creation failed: {e}")
            results["interface_creation"] = {"success": False, "error": str(e)}

        # Test 2: Context analysis pipeline
        print("\n2. Context Analysis Pipeline:")
        try:
            test_prompts = [
                "def fibonacci(n):",
                "class DataProcessor:",
                "import pandas as pd\ndf = pd.DataFrame()",
                "const component = () => { return <div>",
                "SELECT * FROM users WHERE"
            ]

            context_results = []
            for prompt in test_prompts:
                features = interface.context_analyzer.extract_features(prompt, "test_model")
                domain_score = features.code_domain
                completion_type = features.completion_type

                context_results.append({
                    "prompt": prompt[:30] + "...",
                    "domain_score": domain_score,
                    "completion_type": completion_type,
                    "feature_vector_length": len(features.to_vector())
                })

            print(f"   ✅ Analyzed {len(test_prompts)} prompts")
            print(f"   ✅ Feature vectors: {len(features.to_vector())} dimensions")

            results["context_analysis"] = {
                "success": True,
                "prompts_analyzed": len(test_prompts),
                "feature_dimensions": len(features.to_vector()),
                "results": context_results
            }
        except Exception as e:
            print(f"   ❌ Context analysis failed: {e}")
            results["context_analysis"] = {"success": False, "error": str(e)}

        # Test 3: Strategy selection and bandit learning
        print("\n3. Strategy Selection & Learning:")
        try:
            bandit = interface.bandit_selector

            # Test strategy selection for different contexts
            strategy_results = []
            for prompt in test_prompts:
                features = interface.context_analyzer.extract_features(prompt, "test_model")
                strategy, confidence, predicted = bandit.select_strategy(features)

                strategy_results.append({
                    "prompt_type": prompt[:20],
                    "strategy": strategy.value,
                    "confidence": confidence,
                    "predicted_reward": predicted
                })

                # Simulate learning with varied feedback
                quality = 0.6 + (hash(prompt) % 30) / 100  # Deterministic but varied
                bandit.update_reward(strategy, features, quality, 1.0, quality > 0.6)

            # Check exploration stats
            exploration_stats = bandit.get_exploration_stats()
            strategy_performance = bandit.get_strategy_performance()

            print(f"   ✅ Selected strategies for {len(test_prompts)} prompts")
            print(f"   ✅ Exploration rate: {exploration_stats['exploration_rate']:.2f}")
            print(f"   ✅ Strategy diversity: {exploration_stats['recent_diversity']:.2f}")

            results["strategy_learning"] = {
                "success": True,
                "strategies_tested": len(strategy_results),
                "exploration_rate": exploration_stats['exploration_rate'],
                "strategy_diversity": exploration_stats['recent_diversity'],
                "strategy_results": strategy_results
            }
        except Exception as e:
            print(f"   ❌ Strategy learning failed: {e}")
            results["strategy_learning"] = {"success": False, "error": str(e)}

        # Test 4: Error handling and fallbacks
        print("\n4. Error Handling:")
        try:
            from model_interfaces.error_handling import AdaptiveErrorHandler

            handler = AdaptiveErrorHandler()

            # Test different error types
            test_errors = [
                (ConnectionError("Test connection error"), "ollama_connection"),
                (ValueError("Test bandit error"), "bandit_system"),
                (ImportError("Test import error"), "system_config")
            ]

            fallback_results = []
            for error, component in test_errors:
                context = {"component": component, "prompt": "test prompt"}
                response = handler.handle_error(error, context)

                fallback_results.append({
                    "error_type": type(error).__name__,
                    "component": component,
                    "fallback_used": response.get("fallback_used", "unknown"),
                    "error_recovered": response.get("error_recovered", False)
                })

            error_stats = handler.get_error_statistics()

            print(f"   ✅ Tested {len(test_errors)} error scenarios")
            print(f"   ✅ Recovery rate: {error_stats.get('recovery_rate', 0):.1%}")

            results["error_handling"] = {
                "success": True,
                "errors_tested": len(test_errors),
                "recovery_rate": error_stats.get('recovery_rate', 0),
                "fallback_results": fallback_results
            }
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            results["error_handling"] = {"success": False, "error": str(e)}

        return results

    def validate_with_ollama(self, model_name: str) -> Dict[str, Any]:
        """Validate with actual Ollama model"""
        print(f"\n🚀 REAL OLLAMA VALIDATION: {model_name}")
        print("=" * 50)

        results = {}

        # Test 1: Basic generation
        print("\n1. Basic Generation Test:")
        try:
            interface = AdaptiveOllamaInterface(
                model_name=model_name,
                base_url=self.base_url
            )

            test_prompt = "def add_numbers(a, b):\n    \"\"\"Add two numbers and return the result\"\"\"\n    return"

            start_time = time.time()
            response = interface.generate_adaptive(test_prompt)
            generation_time = time.time() - start_time

            print(f"   ✅ Generation completed in {generation_time:.2f}s")
            print(f"   ✅ Strategy used: {response.selected_strategy}")
            print(f"   ✅ Quality score: {response.quality_score:.3f}")
            print(f"   ✅ Success: {response.overall_success}")
            print(f"   📝 Completion: {response.text[:50]}...")

            results["basic_generation"] = {
                "success": True,
                "generation_time": generation_time,
                "strategy_used": response.selected_strategy,
                "quality_score": response.quality_score,
                "overall_success": response.overall_success,
                "completion_length": len(response.text)
            }

        except Exception as e:
            print(f"   ❌ Basic generation failed: {e}")
            results["basic_generation"] = {"success": False, "error": str(e)}

        # Test 2: Multi-prompt learning
        print("\n2. Multi-Prompt Learning Test:")
        try:
            test_prompts = [
                "def fibonacci(n):\n    \"\"\"Calculate fibonacci number\"\"\"\n    if n <= 1:\n        return",
                "def is_even(number):\n    \"\"\"Check if number is even\"\"\"\n    return",
                "def reverse_string(s):\n    \"\"\"Reverse a string\"\"\"\n    return",
                "def count_vowels(text):\n    \"\"\"Count vowels in text\"\"\"\n    vowels = 'aeiou'\n    count = 0\n    for char in text.lower():\n        if char in",
                "class Calculator:\n    \"\"\"Simple calculator class\"\"\"\n    def __init__(self):\n        self."
            ]

            learning_results = []
            strategies_used = []

            for i, prompt in enumerate(test_prompts):
                print(f"   Processing prompt {i+1}/{len(test_prompts)}...")

                response = interface.generate_adaptive(prompt)
                strategies_used.append(response.selected_strategy)

                learning_results.append({
                    "prompt_id": i,
                    "strategy": response.selected_strategy,
                    "quality": response.quality_score,
                    "confidence": response.strategy_confidence,
                    "success": response.overall_success
                })

                time.sleep(0.5)  # Brief pause between requests

            # Analyze learning
            unique_strategies = len(set(strategies_used))
            avg_quality = sum(r['quality'] for r in learning_results) / len(learning_results)

            exploration_stats = interface.bandit_selector.get_exploration_stats()

            print(f"   ✅ Processed {len(test_prompts)} prompts")
            print(f"   ✅ Used {unique_strategies}/{len(list(PromptingStrategy))} strategies")
            print(f"   ✅ Average quality: {avg_quality:.3f}")
            print(f"   ✅ Final exploration rate: {exploration_stats['exploration_rate']:.2f}")

            results["multi_prompt_learning"] = {
                "success": True,
                "prompts_processed": len(test_prompts),
                "unique_strategies": unique_strategies,
                "total_strategies": len(list(PromptingStrategy)),
                "average_quality": avg_quality,
                "exploration_rate": exploration_stats['exploration_rate'],
                "learning_results": learning_results
            }

        except Exception as e:
            print(f"   ❌ Multi-prompt learning failed: {e}")
            results["multi_prompt_learning"] = {"success": False, "error": str(e)}

        # Test 3: Adaptation analytics
        print("\n3. Adaptation Analytics:")
        try:
            analytics = interface.get_adaptation_analytics()
            strategy_performance = interface.bandit_selector.get_strategy_performance()

            print(f"   ✅ Total requests: {analytics.get('total_requests', 0)}")
            print(f"   ✅ Successful adaptations: {analytics.get('successful_adaptations', 0)}")

            # Show strategy performance
            print("   📊 Strategy Performance:")
            for strategy, perf in strategy_performance.items():
                if perf['total_trials'] > 0:
                    print(f"      {strategy}: {perf['total_trials']} trials, "
                          f"quality={perf['mean_reward']:.3f}")

            results["adaptation_analytics"] = {
                "success": True,
                "analytics": analytics,
                "strategy_performance": strategy_performance
            }

        except Exception as e:
            print(f"   ❌ Analytics test failed: {e}")
            results["adaptation_analytics"] = {"success": False, "error": str(e)}

        # Test 4: CLI integration
        print("\n4. CLI Integration Test:")
        try:
            import subprocess

            # Test CLI with actual model
            cli_result = subprocess.run([
                'python3', 'adaptive_ollama_cli.py', 'generate',
                'def hello_world():', '--model', model_name
            ], capture_output=True, text=True, timeout=30)

            if cli_result.returncode == 0:
                print("   ✅ CLI generation successful")
                print(f"   📝 CLI output length: {len(cli_result.stdout)} chars")
            else:
                print(f"   ⚠️  CLI returned code {cli_result.returncode}")
                print(f"   📝 Error: {cli_result.stderr[:100]}...")

            results["cli_integration"] = {
                "success": cli_result.returncode == 0,
                "return_code": cli_result.returncode,
                "output_length": len(cli_result.stdout),
                "error_output": cli_result.stderr[:200] if cli_result.stderr else None
            }

        except Exception as e:
            print(f"   ❌ CLI integration test failed: {e}")
            results["cli_integration"] = {"success": False, "error": str(e)}

        return results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🔬 PHASE 2 COMPREHENSIVE VALIDATION")
        print("=" * 60)

        all_results = {
            "timestamp": time.time(),
            "validation_type": "comprehensive"
        }

        # Step 1: Check Ollama availability
        ollama_status = self.check_ollama_availability()
        all_results["ollama_status"] = ollama_status

        # Step 2: Always run core validation (works without Ollama)
        core_results = self.validate_without_ollama()
        all_results["core_validation"] = core_results

        # Step 3: Run Ollama validation if available
        if ollama_status["available"] and ollama_status["total_models"] > 0:
            # Use first available model
            model_name = ollama_status["models"][0]
            print(f"\n🎯 Using model: {model_name}")

            ollama_results = self.validate_with_ollama(model_name)
            all_results["ollama_validation"] = ollama_results
            all_results["model_tested"] = model_name
        else:
            print("\n⚠️  Skipping Ollama validation - no models available")
            all_results["ollama_validation"] = {"skipped": True, "reason": "No models available"}

        return all_results

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("📊 PHASE 2 VALIDATION REPORT")
        report.append("=" * 40)

        # Overall status
        core_success = all(
            test.get("success", False)
            for test in results.get("core_validation", {}).values()
            if isinstance(test, dict)
        )

        ollama_success = False
        if "ollama_validation" in results and not results["ollama_validation"].get("skipped"):
            ollama_success = all(
                test.get("success", False)
                for test in results["ollama_validation"].values()
                if isinstance(test, dict)
            )

        report.append(f"\n🎯 OVERALL STATUS:")
        report.append(f"   Core System: {'✅ PASSED' if core_success else '❌ FAILED'}")
        if results["ollama_status"]["available"]:
            report.append(f"   Ollama Integration: {'✅ PASSED' if ollama_success else '❌ FAILED'}")
        else:
            report.append(f"   Ollama Integration: ⚠️  SKIPPED (Ollama not available)")

        # Core validation details
        if "core_validation" in results:
            report.append(f"\n🧪 CORE SYSTEM VALIDATION:")
            for test_name, test_result in results["core_validation"].items():
                if isinstance(test_result, dict):
                    status = "✅ PASS" if test_result.get("success") else "❌ FAIL"
                    report.append(f"   {test_name}: {status}")

        # Ollama validation details
        if "ollama_validation" in results and not results["ollama_validation"].get("skipped"):
            report.append(f"\n🚀 OLLAMA VALIDATION:")
            for test_name, test_result in results["ollama_validation"].items():
                if isinstance(test_result, dict):
                    status = "✅ PASS" if test_result.get("success") else "❌ FAIL"
                    report.append(f"   {test_name}: {status}")

        # Performance metrics
        if "ollama_validation" in results and "multi_prompt_learning" in results["ollama_validation"]:
            learning = results["ollama_validation"]["multi_prompt_learning"]
            if learning.get("success"):
                report.append(f"\n📈 PERFORMANCE METRICS:")
                report.append(f"   Strategy Diversity: {learning['unique_strategies']}/{learning['total_strategies']} strategies used")
                report.append(f"   Average Quality: {learning['average_quality']:.3f}")
                report.append(f"   Exploration Rate: {learning['exploration_rate']:.2f}")

        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if not results["ollama_status"]["available"]:
            report.append(f"   • Start Ollama: ollama serve")
            report.append(f"   • Pull a model: ollama pull qwen2.5-coder:3b")
        elif results["ollama_status"]["total_models"] == 0:
            report.append(f"   • Pull a model: ollama pull qwen2.5-coder:3b")
        elif core_success and ollama_success:
            report.append(f"   • ✅ System is ready for production deployment!")
            report.append(f"   • Monitor learning progress with: python adaptive_ollama_cli.py status")
        else:
            report.append(f"   • Review failed tests and check logs")

        return "\n".join(report)


def main():
    """Main validation entry point"""
    validator = Phase2Validator()

    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()

        # Generate and display report
        report = validator.generate_validation_report(results)
        print("\n" + report)

        # Save detailed results
        with open("phase2_validation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: phase2_validation_results.json")

    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()