#!/usr/bin/env python3
"""
Pass@K Statistical Calculator
Sprint 2.1: Dedicated Pass@K implementation following BigCode harness methodology

This module provides Pass@K calculations compatible with the BigCode evaluation harness,
implementing the official algorithm from Chen et al. (2021) "Evaluating Large Language Models Trained on Code"
"""

import numpy as np
import itertools
from typing import List, Dict, Union, Tuple
import math
from dataclasses import dataclass


@dataclass
class PassAtKResult:
    """Result structure for Pass@K calculations"""
    k_values: List[int]
    pass_at_k: Dict[int, float]
    num_problems: int
    total_samples: int
    success_rate: float
    confidence_intervals: Dict[int, Tuple[float, float]]


def estimate_pass_at_k(num_samples: Union[int, np.ndarray],
                       num_correct: Union[int, np.ndarray],
                       k: int) -> Union[float, np.ndarray]:
    """
    Estimates pass@k of each problem and returns them in an array.

    This implements the official BigCode harness algorithm from Chen et al.
    Formula: pass@k = 1 - comb(n-c, k) / comb(n, k)

    Args:
        num_samples: Number of samples per problem (can be array for multiple problems)
        num_correct: Number of correct samples per problem (can be array for multiple problems)
        k: The k value for pass@k calculation

    Returns:
        Pass@k estimate(s)
    """

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        # Handle case where num_correct might be a scalar or array
        if hasattr(num_correct, '__len__'):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            num_samples_it = itertools.repeat(num_samples, 1)
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    if isinstance(num_correct, int):
        return estimator(num_samples, num_correct, k)
    else:
        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def calculate_multiple_pass_at_k(results_per_problem: List[List[bool]],
                                k_values: List[int] = [1, 10, 100]) -> Dict[int, float]:
    """
    Calculate Pass@K for multiple k values given results per problem.

    Args:
        results_per_problem: List of lists, where each inner list contains
                           boolean results for all samples of one problem
        k_values: List of k values to calculate pass@k for

    Returns:
        Dictionary mapping k -> pass@k score
    """
    if not results_per_problem:
        return {k: 0.0 for k in k_values}

    # Convert to numpy arrays for BigCode algorithm
    num_samples = np.array([len(problem_results) for problem_results in results_per_problem])
    num_correct = np.array([sum(problem_results) for problem_results in results_per_problem])

    pass_at_k = {}
    for k in k_values:
        if (num_samples >= k).all():
            # Only calculate if all problems have at least k samples
            estimates = estimate_pass_at_k(num_samples, num_correct, k)
            pass_at_k[k] = float(estimates.mean())
        else:
            # If some problems have fewer than k samples, use available samples
            individual_estimates = []
            for n, c in zip(num_samples, num_correct):
                # Handle individual problem estimation
                n_val = int(n)
                c_val = int(c)
                k_val = min(k, n_val)
                if n_val - c_val < k_val:
                    estimate = 1.0
                else:
                    estimate = 1.0 - np.prod(1.0 - k_val / np.arange(n_val - c_val + 1, n_val + 1))
                individual_estimates.append(estimate)
            pass_at_k[k] = float(np.mean(individual_estimates))

    return pass_at_k


def bootstrap_confidence_interval(results_per_problem: List[List[bool]],
                                k: int,
                                n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for Pass@K.

    Args:
        results_per_problem: List of lists with boolean results per problem
        k: The k value for pass@k
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval
    """
    if not results_per_problem:
        return (0.0, 0.0)

    np.random.seed(42)  # For reproducibility
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample: sample problems with replacement
        bootstrap_indices = np.random.choice(len(results_per_problem),
                                           size=len(results_per_problem),
                                           replace=True)
        bootstrap_problems = [results_per_problem[i] for i in bootstrap_indices]

        # Calculate pass@k for this bootstrap sample
        pass_k_dict = calculate_multiple_pass_at_k(bootstrap_problems, [k])
        bootstrap_scores.append(pass_k_dict[k])

    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    return (float(ci_lower), float(ci_upper))


def comprehensive_pass_at_k_analysis(results_per_problem: List[List[bool]],
                                    k_values: List[int] = [1, 5, 10, 100]) -> PassAtKResult:
    """
    Perform comprehensive Pass@K analysis with confidence intervals.

    Args:
        results_per_problem: List of lists with boolean results per problem
        k_values: List of k values to analyze

    Returns:
        PassAtKResult with complete analysis
    """
    if not results_per_problem:
        return PassAtKResult(
            k_values=k_values,
            pass_at_k={k: 0.0 for k in k_values},
            num_problems=0,
            total_samples=0,
            success_rate=0.0,
            confidence_intervals={k: (0.0, 0.0) for k in k_values}
        )

    # Basic statistics
    num_problems = len(results_per_problem)
    total_samples = sum(len(problem_results) for problem_results in results_per_problem)
    total_correct = sum(sum(problem_results) for problem_results in results_per_problem)
    success_rate = total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate Pass@K for all k values
    pass_at_k = calculate_multiple_pass_at_k(results_per_problem, k_values)

    # Calculate confidence intervals
    confidence_intervals = {}
    for k in k_values:
        confidence_intervals[k] = bootstrap_confidence_interval(results_per_problem, k)

    return PassAtKResult(
        k_values=k_values,
        pass_at_k=pass_at_k,
        num_problems=num_problems,
        total_samples=total_samples,
        success_rate=success_rate,
        confidence_intervals=confidence_intervals
    )


def format_pass_at_k_results(result: PassAtKResult) -> str:
    """Format Pass@K results for display"""
    lines = []
    lines.append(f"📊 Pass@K Analysis Results")
    lines.append(f"=" * 50)
    lines.append(f"Problems evaluated: {result.num_problems}")
    lines.append(f"Total samples: {result.total_samples}")
    lines.append(f"Overall success rate: {result.success_rate:.3f} ({result.success_rate*100:.1f}%)")
    lines.append("")
    lines.append("Pass@K Metrics:")

    for k in result.k_values:
        if k in result.pass_at_k:
            score = result.pass_at_k[k]
            ci_lower, ci_upper = result.confidence_intervals.get(k, (0.0, 0.0))
            lines.append(f"  Pass@{k:3d}: {score:.3f} ({score*100:5.1f}%) "
                        f"[95% CI: {ci_lower:.3f}-{ci_upper:.3f}]")

    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_results = [
        [True, False, True],   # Problem 1: 2/3 correct
        [False, False, True],  # Problem 2: 1/3 correct
        [True, True, True],    # Problem 3: 3/3 correct
        [False, False, False], # Problem 4: 0/3 correct
    ]

    result = comprehensive_pass_at_k_analysis(sample_results, [1, 3, 5, 10])
    print(format_pass_at_k_results(result))