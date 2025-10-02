#!/usr/bin/env python3
"""
Phase 2 Validation Test Suite

Tests the adaptive prompting system against known strategy-dependent performance
variations to validate that the contextual multi-armed bandit learns optimal
strategies for different models and contexts.

Usage:
    python phase2_validation_test.py
"""

import sys
import os
import time
import random
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_interfaces.adaptive_ollama_interface import AdaptiveOllamaInterface, create_adaptive_interface
from prompting.context_analyzer import PromptContextAnalyzer
from prompting.bandit_strategy_selector import PromptingStrategy
from prompting.continuous_learning_framework import ContinuousLearningFramework
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_contextual_feature_extraction():
    """Test contextual feature extraction with various prompt types"""
    print("\n=== TESTING CONTEXTUAL FEATURE EXTRACTION ===")

    analyzer = PromptContextAnalyzer()

    test_prompts = [
        {
            "name": "Simple function completion",
            "prompt": "def fibonacci(n):",
            "expected_features": {
                "has_function_def": 1.0,
                "completion_type": 0.5,  # Function body completion
                "code_domain": 0.0       # General algorithms
            }
        },
        {
            "name": "Complex class with data science context",
            "prompt": """import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe

    def analyze_data(self):
        # Complete this method""",
            "expected_features": {
                "has_class_def": 1.0,
                "code_domain": 0.5,      # Data science domain
                "prompt_complexity": 0.7, # Complex prompt
                "nesting_level": 0.4     # Some nesting
            }
        },
        {
            "name": "Web development context",
            "prompt": """from flask import Flask, request
app = Flask(__name__)

@app.route('/api/data')
def get_data():
    # Complete the API endpoint""",
            "expected_features": {
                "code_domain": 0.5,      # Web development
                "has_function_def": 1.0,
                "keyword_density": 0.3   # Python keywords present
            }
        },
        {
            "name": "Simple expression",
            "prompt": "x = ",
            "expected_features": {
                "prompt_complexity": 0.1, # Very simple
                "completion_type": 0.2,   # Statement completion
                "context_length": 0.0     # Very short
            }
        }
    ]

    results = []
    for i, case in enumerate(test_prompts, 1):
        print(f"\n--- Test Case {i}: {case['name']} ---")

        features = analyzer.extract_features(case['prompt'], "test_model")
        feature_dict = features.to_dict()

        print(f"Extracted features:")
        for key, value in feature_dict.items():
            print(f"  {key}: {value:.3f}")

        # Validate expected features
        success = True
        for expected_key, expected_range in case['expected_features'].items():
            actual_value = feature_dict.get(expected_key, 0.0)

            # Allow 20% tolerance for feature extraction
            tolerance = 0.2
            if isinstance(expected_range, (int, float)):
                if abs(actual_value - expected_range) > tolerance:
                    print(f"❌ FAIL: {expected_key} expected ~{expected_range}, got {actual_value:.3f}")
                    success = False
                else:
                    print(f"✅ PASS: {expected_key} = {actual_value:.3f} (expected ~{expected_range})")

        results.append({
            "case": case['name'],
            "success": success,
            "features": feature_dict
        })

    return results

def test_bandit_learning_simulation():
    """Test bandit learning with simulated strategy-dependent performance"""
    print("\n=== TESTING BANDIT LEARNING SIMULATION ===")

    from prompting.bandit_strategy_selector import create_bandit_selector
    from prompting.context_analyzer import ContextFeatures

    # Create simulated performance patterns for different strategies
    strategy_performance_patterns = {
        PromptingStrategy.CODE_ENGINE: {
            "base_performance": 0.8,
            "context_preferences": {"prompt_complexity": "high", "code_domain": "general"}
        },
        PromptingStrategy.ROLE_BASED: {
            "base_performance": 0.7,
            "context_preferences": {"has_function_def": "true", "completion_type": "high"}
        },
        PromptingStrategy.DETERMINISTIC: {
            "base_performance": 0.6,
            "context_preferences": {"prompt_complexity": "low", "context_length": "short"}
        },
        PromptingStrategy.NEGATIVE_PROMPT: {
            "base_performance": 0.5,
            "context_preferences": {"code_domain": "specialized"}
        },
        PromptingStrategy.SILENT_GENERATOR: {
            "base_performance": 0.6,
            "context_preferences": {"context_length": "short", "code_domain": "general"}
        },
        PromptingStrategy.FORMAT_CONSTRAINT: {
            "base_performance": 0.5,
            "context_preferences": {"completion_type": "low", "prompt_complexity": "low"}
        }
    }

    def simulate_quality_score(strategy: PromptingStrategy, context: ContextFeatures) -> float:
        """Simulate quality score based on strategy and context"""
        pattern = strategy_performance_patterns[strategy]
        base_score = pattern["base_performance"]

        # Add context-dependent bonus/penalty
        context_dict = context.to_dict()
        bonus = 0.0

        prefs = pattern["context_preferences"]
        if "prompt_complexity" in prefs:
            if prefs["prompt_complexity"] == "high" and context_dict["prompt_complexity"] > 0.7:
                bonus += 0.1
            elif prefs["prompt_complexity"] == "low" and context_dict["prompt_complexity"] < 0.3:
                bonus += 0.1

        if "has_function_def" in prefs and context_dict["has_function_def"] > 0.5:
            bonus += 0.1

        if "code_domain" in prefs:
            if prefs["code_domain"] == "general" and context_dict["code_domain"] < 0.3:
                bonus += 0.1
            elif prefs["code_domain"] == "specialized" and context_dict["code_domain"] > 0.7:
                bonus += 0.1

        # Add noise
        noise = random.uniform(-0.1, 0.1)
        final_score = max(0.0, min(1.0, base_score + bonus + noise))

        return final_score

    # Test bandit learning
    bandit = create_bandit_selector("test_model", alpha=1.0)
    analyzer = PromptContextAnalyzer()

    # Create diverse test contexts
    test_contexts = [
        "def fibonacci(n):",  # Simple function
        """import pandas as pd
class DataProcessor:
    def analyze(self):""",  # Complex class
        "x = ",  # Simple expression
        """from flask import Flask
@app.route('/api')
def endpoint():""",  # Web context
        """for i in range(10):
    if condition:
        # Complete this""",  # Complex control flow
    ]

    print(f"Training bandit with {len(test_contexts)} context types over 100 iterations...")

    learning_results = []

    for iteration in range(100):
        # Choose random context
        prompt = random.choice(test_contexts)
        context_features = analyzer.extract_features(prompt, "test_model")

        # Bandit selects strategy
        selected_strategy, confidence, predicted = bandit.select_strategy(context_features)

        # Simulate quality score
        actual_quality = simulate_quality_score(selected_strategy, context_features)

        # Update bandit
        bandit.update_reward(selected_strategy, context_features, actual_quality)

        learning_results.append({
            'iteration': iteration,
            'strategy': selected_strategy.value,
            'predicted': predicted,
            'actual': actual_quality,
            'confidence': confidence
        })

        if iteration % 20 == 19:  # Progress updates
            recent_performance = [r['actual'] for r in learning_results[-20:]]
            avg_performance = sum(recent_performance) / len(recent_performance)
            print(f"Iteration {iteration + 1}: Avg quality = {avg_performance:.3f}")

    # Analyze learning results
    print("\n📊 LEARNING ANALYSIS:")

    # Strategy performance analysis
    final_performance = bandit.get_strategy_performance()
    print("Final strategy performance:")
    for strategy, stats in final_performance.items():
        print(f"  {strategy}: {stats['mean_reward']:.3f} avg ({stats['total_trials']} trials)")

    # Learning curve analysis
    early_performance = [r['actual'] for r in learning_results[:20]]
    late_performance = [r['actual'] for r in learning_results[-20:]]

    early_avg = sum(early_performance) / len(early_performance)
    late_avg = sum(late_performance) / len(late_performance)
    improvement = late_avg - early_avg

    print(f"\nLearning curve:")
    print(f"  Early performance (1-20): {early_avg:.3f}")
    print(f"  Late performance (81-100): {late_avg:.3f}")
    print(f"  Improvement: {improvement:.3f}")

    # Convergence analysis
    convergence_threshold = 0.05
    converged = abs(improvement) < convergence_threshold and late_avg > 0.6

    if converged:
        print("✅ PASS: Bandit converged to good performance")
    else:
        print("❌ FAIL: Bandit did not converge or performance too low")

    return {
        "converged": converged,
        "improvement": improvement,
        "final_performance": late_avg,
        "strategy_performance": final_performance
    }

def test_adaptive_interface_integration():
    """Test full adaptive interface integration (simulation mode)"""
    print("\n=== TESTING ADAPTIVE INTERFACE INTEGRATION ===")

    # Create adaptive interface in simulation mode (no actual Ollama calls)
    class SimulatedAdaptiveInterface(AdaptiveOllamaInterface):
        """Adaptive interface that simulates responses instead of calling Ollama"""

        def _generate_with_strategy(self, prompt: str, strategy: PromptingStrategy, **kwargs):
            """Simulate response generation with strategy-dependent quality"""
            time.sleep(0.1)  # Simulate generation time

            # Strategy-dependent response simulation
            if strategy == PromptingStrategy.CODE_ENGINE and "def " in prompt:
                response = "return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
                quality_multiplier = 0.9
            elif strategy == PromptingStrategy.ROLE_BASED and "class " in prompt:
                response = "self.data = data\n        self.processed = False"
                quality_multiplier = 0.8
            elif strategy == PromptingStrategy.DETERMINISTIC:
                response = "# Simple implementation\npass"
                quality_multiplier = 0.6
            else:
                response = "# Generic completion"
                quality_multiplier = 0.5

            # Add noise to quality
            noise = random.uniform(-0.1, 0.1)
            base_quality = min(0.9, max(0.1, quality_multiplier + noise))

            return {
                'raw_text': response,
                'http_success': True
            }

    # Test adaptive interface
    interface = SimulatedAdaptiveInterface(
        model_name="test_model",
        quality_threshold=0.3,
        exploration_alpha=1.0
    )

    test_prompts = [
        "def fibonacci(n):",
        "class DataProcessor:",
        "import pandas as pd\ndf = ",
        "for i in range(10):",
        "def calculate_sum(numbers):"
    ]

    print("Testing adaptive interface with 25 interactions...")

    results = []
    for i in range(25):
        prompt = random.choice(test_prompts)

        response = interface.generate_adaptive(prompt)

        results.append({
            'iteration': i + 1,
            'prompt_type': prompt.split()[0] if prompt.split() else '',
            'selected_strategy': response.selected_strategy,
            'quality_score': response.quality_score,
            'overall_success': response.overall_success,
            'exploration': response.bandit_exploration
        })

        if i % 5 == 4:  # Progress updates
            recent_quality = [r['quality_score'] for r in results[-5:]]
            avg_quality = sum(recent_quality) / len(recent_quality)
            print(f"Iterations {i-4}-{i+1}: Avg quality = {avg_quality:.3f}")

    # Analyze integration results
    print("\n📊 INTEGRATION ANALYSIS:")

    # Quality improvement
    early_quality = [r['quality_score'] for r in results[:10]]
    late_quality = [r['quality_score'] for r in results[-10:]]

    early_avg = sum(early_quality) / len(early_quality)
    late_avg = sum(late_quality) / len(late_quality)

    print(f"Quality improvement: {early_avg:.3f} → {late_avg:.3f} ({late_avg - early_avg:+.3f})")

    # Strategy diversity
    strategies_used = set(r['selected_strategy'] for r in results)
    strategy_diversity = len(strategies_used) / len(PromptingStrategy)

    print(f"Strategy diversity: {strategy_diversity:.1%} ({len(strategies_used)} strategies)")

    # Success rate
    success_rate = sum(1 for r in results if r['overall_success']) / len(results)
    print(f"Overall success rate: {success_rate:.1%}")

    # Exploration rate
    exploration_rate = sum(1 for r in results if r['exploration']) / len(results)
    print(f"Exploration rate: {exploration_rate:.1%}")

    # Get analytics
    analytics = interface.get_adaptation_analytics()
    print(f"\nAdaptation efficiency: {analytics['summary']['adaptation_efficiency']:.3f}")

    # Validate integration
    integration_success = (
        late_avg > early_avg and  # Quality improved
        strategy_diversity > 0.5 and  # Good strategy diversity
        success_rate > 0.6 and  # Good success rate
        0.1 <= exploration_rate <= 0.4  # Reasonable exploration
    )

    if integration_success:
        print("✅ PASS: Adaptive interface integration successful")
    else:
        print("❌ FAIL: Adaptive interface integration issues")

    return {
        "integration_success": integration_success,
        "quality_improvement": late_avg - early_avg,
        "strategy_diversity": strategy_diversity,
        "success_rate": success_rate,
        "exploration_rate": exploration_rate
    }

def test_strategy_dependent_performance():
    """Test that system identifies and adapts to strategy-dependent performance patterns"""
    print("\n=== TESTING STRATEGY-DEPENDENT PERFORMANCE ADAPTATION ===")

    # This test validates the core issue that Phase 2 was designed to solve:
    # Models showing 0%-100% performance based on strategy choice

    from prompting.bandit_strategy_selector import create_bandit_selector
    from prompting.context_analyzer import PromptContextAnalyzer

    analyzer = PromptContextAnalyzer()
    bandit = create_bandit_selector("qwen2.5:0.5b", alpha=1.5)  # High exploration

    # Simulate the exact problem we found: binary performance based on strategy
    def simulate_binary_performance(strategy: PromptingStrategy, context_features) -> float:
        """Simulate the binary 0%/100% performance issue"""
        context_dict = context_features.to_dict()

        # For qwen2.5:0.5b model (the problematic case):
        if strategy == PromptingStrategy.CODE_ENGINE:
            # CODE_ENGINE always works well for this model
            return random.uniform(0.8, 1.0)
        elif strategy == PromptingStrategy.DETERMINISTIC:
            # DETERMINISTIC always fails for this model
            return random.uniform(0.0, 0.2)
        elif strategy == PromptingStrategy.ROLE_BASED:
            # ROLE_BASED works well for function definitions
            if context_dict["has_function_def"] > 0.5:
                return random.uniform(0.7, 0.9)
            else:
                return random.uniform(0.1, 0.3)
        else:
            # Other strategies have medium performance
            return random.uniform(0.4, 0.6)

    # Test with function definition prompts (the problematic case)
    function_prompts = [
        "def fibonacci(n):",
        "def calculate_sum(numbers):",
        "def is_prime(num):",
        "def binary_search(arr, target):",
        "def merge_sort(arr):"
    ]

    print("Training bandit to discover strategy-dependent performance...")

    adaptation_results = []

    for iteration in range(60):  # 60 iterations to learn
        prompt = random.choice(function_prompts)
        context_features = analyzer.extract_features(prompt, "qwen2.5:0.5b")

        # Bandit selects strategy
        selected_strategy, confidence, predicted = bandit.select_strategy(context_features)

        # Simulate binary performance
        actual_quality = simulate_binary_performance(selected_strategy, context_features)
        success = actual_quality > 0.5

        # Update bandit
        bandit.update_reward(selected_strategy, context_features, actual_quality, success=success)

        adaptation_results.append({
            'iteration': iteration,
            'strategy': selected_strategy.value,
            'quality': actual_quality,
            'success': success,
            'confidence': confidence
        })

    # Analyze adaptation to binary performance
    print("\n📊 ADAPTATION TO BINARY PERFORMANCE:")

    final_performance = bandit.get_strategy_performance()
    print("Final strategy rankings:")

    strategy_rankings = sorted(
        final_performance.items(),
        key=lambda x: x[1]['mean_reward'],
        reverse=True
    )

    for i, (strategy, stats) in enumerate(strategy_rankings, 1):
        print(f"  {i}. {strategy}: {stats['mean_reward']:.3f} avg "
              f"({stats['success_rate']:.1%} success, {stats['total_trials']} trials)")

    # Validate that bandit learned the patterns
    best_strategy = strategy_rankings[0][0]
    worst_strategy = strategy_rankings[-1][0]

    best_performance = strategy_rankings[0][1]['mean_reward']
    worst_performance = strategy_rankings[-1][1]['mean_reward']

    # Check if bandit correctly identified CODE_ENGINE as best and DETERMINISTIC as worst
    learned_correctly = (
        best_strategy == "code_engine" and
        worst_strategy == "deterministic" and
        best_performance > 0.7 and
        worst_performance < 0.3
    )

    # Check convergence to good strategies
    last_10_strategies = [r['strategy'] for r in adaptation_results[-10:]]
    good_strategy_rate = sum(1 for s in last_10_strategies if s == "code_engine") / len(last_10_strategies)

    converged_to_good = good_strategy_rate > 0.6

    print(f"\nLearning validation:")
    print(f"  Correctly identified best strategy: {'✅' if best_strategy == 'code_engine' else '❌'}")
    print(f"  Correctly identified worst strategy: {'✅' if worst_strategy == 'deterministic' else '❌'}")
    print(f"  Performance separation: {best_performance:.3f} vs {worst_performance:.3f}")
    print(f"  Converged to good strategies: {'✅' if converged_to_good else '❌'} ({good_strategy_rate:.1%})")

    adaptation_success = learned_correctly and converged_to_good

    if adaptation_success:
        print("✅ PASS: Successfully adapted to strategy-dependent performance")
    else:
        print("❌ FAIL: Failed to adapt to strategy-dependent performance")

    return {
        "adaptation_success": adaptation_success,
        "learned_correctly": learned_correctly,
        "converged_to_good": converged_to_good,
        "best_strategy": best_strategy,
        "performance_separation": best_performance - worst_performance
    }

def main():
    """Run comprehensive Phase 2 validation"""
    print("🧠 PHASE 2 VALIDATION TEST SUITE")
    print("Testing adaptive prompting with contextual multi-armed bandits")
    print("=" * 70)

    # Run all test suites
    feature_results = test_contextual_feature_extraction()
    bandit_results = test_bandit_learning_simulation()
    integration_results = test_adaptive_interface_integration()
    strategy_results = test_strategy_dependent_performance()

    # Overall summary
    print("\n" + "=" * 70)
    print("📊 PHASE 2 VALIDATION SUMMARY")
    print("=" * 70)

    # Feature extraction summary
    feature_passed = sum(1 for r in feature_results if r['success'])
    print(f"Contextual Feature Extraction: {feature_passed}/{len(feature_results)} tests passed")

    # Bandit learning summary
    bandit_passed = bandit_results['converged']
    print(f"Bandit Learning Simulation: {'✅ PASSED' if bandit_passed else '❌ FAILED'}")
    print(f"  Quality improvement: {bandit_results['improvement']:+.3f}")

    # Integration summary
    integration_passed = integration_results['integration_success']
    print(f"Adaptive Interface Integration: {'✅ PASSED' if integration_passed else '❌ FAILED'}")
    print(f"  Quality improvement: {integration_results['quality_improvement']:+.3f}")

    # Strategy adaptation summary
    strategy_passed = strategy_results['adaptation_success']
    print(f"Strategy-Dependent Performance: {'✅ PASSED' if strategy_passed else '❌ FAILED'}")
    print(f"  Performance separation: {strategy_results['performance_separation']:.3f}")

    # Overall assessment
    total_tests = len(feature_results) + 3  # 3 major test suites
    total_passed = feature_passed + sum([bandit_passed, integration_passed, strategy_passed])

    print(f"\n🎯 OVERALL PHASE 2 VALIDATION: {total_passed}/{total_tests} components passed")

    # Phase 2 success criteria (from roadmap)
    phase2_criteria = {
        "performance_improvement": bandit_results['improvement'] > 0.05,  # >5% improvement
        "adaptation_speed": bandit_passed,  # Converged within test iterations
        "strategy_optimization": strategy_passed,  # Found optimal strategy mix
        "feature_extraction": feature_passed >= len(feature_results) * 0.8  # 80% feature tests pass
    }

    criteria_met = sum(phase2_criteria.values())
    print(f"\nPhase 2 Success Criteria: {criteria_met}/{len(phase2_criteria)} met")

    for criterion, met in phase2_criteria.items():
        status = "✅" if met else "❌"
        print(f"  {status} {criterion.replace('_', ' ').title()}")

    if criteria_met == len(phase2_criteria):
        print("\n🎉 PHASE 2 IMPLEMENTATION SUCCESSFUL!")
        print("Adaptive prompting system meets all validation criteria")
    else:
        print("\n⚠️  PHASE 2 IMPLEMENTATION NEEDS REFINEMENT")
        print("Some validation criteria not met - review implementation")

    print(f"\n🚀 NEXT STEPS:")
    if criteria_met == len(phase2_criteria):
        print("✅ Ready for production integration")
        print("✅ Ready for real model testing (when Ollama available)")
        print("✅ Phase 2 academic targets achieved")
    else:
        print("🔧 Address failing validation criteria")
        print("🧪 Additional testing and tuning needed")

    return criteria_met == len(phase2_criteria)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)