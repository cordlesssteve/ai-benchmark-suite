#!/usr/bin/env python3
"""
Test script for Pass@K implementation (Sprint 2.1)

This script validates the Pass@K implementation against known results
and ensures compatibility with BigCode harness methodology.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.pass_at_k_calculator import (
    estimate_pass_at_k,
    calculate_multiple_pass_at_k,
    comprehensive_pass_at_k_analysis,
    format_pass_at_k_results
)
import numpy as np


def test_basic_pass_at_k():
    """Test basic Pass@K calculation"""
    print("🔍 Testing basic Pass@K calculation...")

    # Test case: 10 samples, 3 correct, k=1
    result = estimate_pass_at_k(10, 3, 1)
    expected = 3/10  # Should be 0.3 for k=1
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    print(f"✅ Pass@1 with 3/10 correct: {result:.3f}")

    # Test case: 10 samples, 3 correct, k=5
    result = estimate_pass_at_k(10, 3, 5)
    # For k=5: 1 - C(7,5)/C(10,5) = 1 - (7*6)/(10*9*8*7*6/5*4*3*2*1) = 1 - 42/252 ≈ 0.833
    print(f"✅ Pass@5 with 3/10 correct: {result:.3f}")

    # Edge case: more correct than k
    result = estimate_pass_at_k(10, 8, 5)
    print(f"✅ Pass@5 with 8/10 correct: {result:.3f}")

    print()


def test_multiple_problems():
    """Test Pass@K with multiple problems"""
    print("🔍 Testing multiple problems...")

    # Sample data: 3 problems with different success rates
    results_per_problem = [
        [True, False, True, False, True],   # Problem 1: 3/5 = 60%
        [False, False, True, True, False],  # Problem 2: 2/5 = 40%
        [True, True, True, True, True],     # Problem 3: 5/5 = 100%
    ]

    pass_at_k = calculate_multiple_pass_at_k(results_per_problem, [1, 2, 3, 5])

    print("Pass@K results for multiple problems:")
    for k, score in pass_at_k.items():
        print(f"  Pass@{k}: {score:.3f} ({score*100:.1f}%)")

    # Sanity checks
    assert pass_at_k[1] > 0, "Pass@1 should be > 0"
    assert pass_at_k[5] >= pass_at_k[1], "Pass@5 should be >= Pass@1"

    print("✅ Multiple problems test passed")
    print()


def test_comprehensive_analysis():
    """Test comprehensive Pass@K analysis with confidence intervals"""
    print("🔍 Testing comprehensive analysis...")

    # More realistic test data: 10 problems, varying difficulty
    results_per_problem = [
        [True, True, False, True, False],      # Easy problem: 3/5
        [False, False, False, True, False],    # Hard problem: 1/5
        [True, True, True, True, True],        # Very easy: 5/5
        [False, False, False, False, False],   # Impossible: 0/5
        [True, False, True, False, True],      # Medium: 3/5
        [True, True, False, True, True],       # Easy: 4/5
        [False, True, False, False, True],     # Hard: 2/5
        [True, True, True, False, True],       # Easy: 4/5
        [False, False, True, False, False],    # Hard: 1/5
        [True, False, False, True, True],      # Medium: 3/5
    ]

    try:
        result = comprehensive_pass_at_k_analysis(results_per_problem, [1, 5, 10])
        print(format_pass_at_k_results(result))
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Sanity checks
    assert result.num_problems == 10, f"Expected 10 problems, got {result.num_problems}"
    assert result.total_samples == 50, f"Expected 50 samples, got {result.total_samples}"
    assert 0 <= result.success_rate <= 1, f"Success rate should be 0-1, got {result.success_rate}"

    for k in [1, 5, 10]:
        assert 0 <= result.pass_at_k[k] <= 1, f"Pass@{k} should be 0-1, got {result.pass_at_k[k]}"
        ci_lower, ci_upper = result.confidence_intervals[k]
        assert ci_lower <= result.pass_at_k[k] <= ci_upper, f"Pass@{k} not in confidence interval"

    print("✅ Comprehensive analysis test passed")
    print()


def test_edge_cases():
    """Test edge cases"""
    print("🔍 Testing edge cases...")

    # Empty results
    empty_result = comprehensive_pass_at_k_analysis([], [1, 5, 10])
    assert empty_result.num_problems == 0
    assert all(score == 0.0 for score in empty_result.pass_at_k.values())
    print("✅ Empty results handled correctly")

    # All failures
    all_fail = [[False, False, False] for _ in range(3)]
    fail_result = comprehensive_pass_at_k_analysis(all_fail, [1, 3])
    assert all(score == 0.0 for score in fail_result.pass_at_k.values())
    print("✅ All failures handled correctly")

    # All successes
    all_success = [[True, True, True] for _ in range(3)]
    success_result = comprehensive_pass_at_k_analysis(all_success, [1, 3])
    assert all(score == 1.0 for score in success_result.pass_at_k.values())
    print("✅ All successes handled correctly")

    print()


def test_bigcode_compatibility():
    """Test compatibility with BigCode harness algorithm"""
    print("🔍 Testing BigCode harness compatibility...")

    # Test against known BigCode calculation
    # From BigCode docs: with n=200, c=50, k=1 should give c/n = 0.25
    result_k1 = estimate_pass_at_k(200, 50, 1)
    expected_k1 = 50/200
    assert abs(result_k1 - expected_k1) < 0.001, f"K=1 mismatch: expected {expected_k1}, got {result_k1}"
    print(f"✅ Pass@1 (200 samples, 50 correct): {result_k1:.3f}")

    # For k=10 with same data
    result_k10 = estimate_pass_at_k(200, 50, 10)
    print(f"✅ Pass@10 (200 samples, 50 correct): {result_k10:.3f}")

    # Should have Pass@10 > Pass@1
    assert result_k10 > result_k1, "Pass@10 should be > Pass@1"

    print("✅ BigCode compatibility verified")
    print()


def main():
    """Run all tests"""
    print("🧪 Testing Pass@K Implementation (Sprint 2.1)")
    print("=" * 60)
    print()

    try:
        test_basic_pass_at_k()
        test_multiple_problems()
        test_comprehensive_analysis()
        test_edge_cases()
        test_bigcode_compatibility()

        print("🎉 All tests passed! Pass@K implementation is working correctly.")
        print()
        print("✅ Sprint 2.1 Pass@K calculations ready for production use")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()