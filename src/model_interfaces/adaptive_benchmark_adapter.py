#!/usr/bin/env python3
"""
Adaptive Benchmark Adapter
Drop-in replacement for existing Ollama interfaces that adds adaptive prompting.

This adapter provides the same interface as existing benchmark runners
but uses the Phase 2 adaptive system underneath.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .adaptive_ollama_interface import AdaptiveOllamaInterface, AdaptiveOllamaResponse

logger = logging.getLogger(__name__)


class AdaptiveBenchmarkAdapter:
    """
    Adaptive benchmark adapter that provides compatibility with existing runners
    while using the Phase 2 adaptive prompting system underneath.
    """

    def __init__(self,
                 model_name: str,
                 base_url: str = "http://localhost:11434",
                 **kwargs):
        """
        Initialize adaptive benchmark adapter

        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        self.base_url = base_url

        # Create underlying adaptive interface
        self.adaptive_interface = AdaptiveOllamaInterface(
            model_name=model_name,
            base_url=base_url,
            **kwargs
        )

        # Track compatibility metrics
        self.total_requests = 0
        self.successful_requests = 0

    # Drop-in compatibility methods for existing benchmark runners

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate completion (compatibility method for existing runners)

        Returns just the text completion for compatibility with existing code.
        """
        try:
            response = self.adaptive_interface.generate_adaptive(prompt, **kwargs)
            self.total_requests += 1

            if response.overall_success:
                self.successful_requests += 1
                return response.text
            else:
                # Return raw text even if quality is low for compatibility
                return response.raw_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.total_requests += 1
            return ""

    def generate_with_details(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate completion with detailed response information

        This method provides access to all adaptive features while maintaining
        compatibility with advanced benchmark runners.
        """
        try:
            response = self.adaptive_interface.generate_adaptive(prompt, **kwargs)
            self.total_requests += 1

            if response.overall_success:
                self.successful_requests += 1

            # Return comprehensive details
            return {
                # Basic compatibility fields
                "text": response.text,
                "success": response.overall_success,
                "execution_time": response.execution_time,

                # Adaptive system details
                "raw_text": response.raw_text,
                "http_success": response.http_success,
                "content_quality_success": response.content_quality_success,
                "quality_score": response.quality_score,
                "quality_level": response.quality_level.value,

                # Strategy information
                "selected_strategy": response.selected_strategy.value,
                "strategy_confidence": response.strategy_confidence,
                "predicted_reward": response.predicted_reward,
                "is_exploration": response.is_exploration,

                # Context features
                "context_features": response.context_features.to_dict(),

                # Adaptation info
                "adaptation_effectiveness": response.adaptation_effectiveness
            }

        except Exception as e:
            logger.error(f"Generation with details failed: {e}")
            self.total_requests += 1
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.adaptive_interface.is_available()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.adaptive_interface.get_model_info()

    # Batch processing methods for benchmark efficiency

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate completions for batch of prompts

        Optimized for benchmark processing with progress tracking.
        """
        results = []
        total = len(prompts)

        logger.info(f"Processing batch of {total} prompts with adaptive strategies")

        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

            result = self.generate(prompt, **kwargs)
            results.append(result)

            # Periodic tuning to optimize exploration during batch
            if i > 0 and i % 50 == 0:
                self.adaptive_interface.tune_exploration()

        logger.info(f"Batch completed: {self.successful_requests}/{self.total_requests} successful")
        return results

    def generate_batch_with_details(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate batch with detailed responses"""
        results = []
        total = len(prompts)

        logger.info(f"Processing detailed batch of {total} prompts")

        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

            result = self.generate_with_details(prompt, **kwargs)
            results.append(result)

            # Periodic tuning
            if i > 0 and i % 50 == 0:
                self.adaptive_interface.tune_exploration()

        return results

    # Analytics and monitoring methods

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        bandit_stats = self.adaptive_interface.bandit_selector.get_exploration_stats()
        performance = self.adaptive_interface.bandit_selector.get_strategy_performance()
        analytics = self.adaptive_interface.get_adaptation_analytics()

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "bandit_stats": bandit_stats,
            "strategy_performance": performance,
            "adaptation_analytics": analytics
        }

    def get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies used"""
        actions = self.adaptive_interface.bandit_selector.action_history
        distribution = {}

        for action in actions:
            strategy = action.strategy.value
            distribution[strategy] = distribution.get(strategy, 0) + 1

        return distribution

    def save_analytics(self, filepath: str):
        """Save adaptation analytics to file"""
        import json

        analytics = self.get_adaptation_stats()

        with open(filepath, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)

        logger.info(f"Analytics saved to {filepath}")

    # Configuration and tuning methods

    def tune_exploration(self):
        """Manually trigger exploration parameter tuning"""
        self.adaptive_interface.tune_exploration()

    def reset_adaptation(self):
        """Reset bandit learning (for testing or retraining)"""
        self.adaptive_interface.bandit_selector.reset_bandit()
        logger.info("Bandit learning reset")

    def get_current_alpha(self) -> float:
        """Get current exploration parameter"""
        return self.adaptive_interface.bandit_selector.alpha

    def set_exploration_alpha(self, alpha: float):
        """Set exploration parameter manually"""
        self.adaptive_interface.bandit_selector.alpha = max(1.0, min(alpha, 3.5))
        logger.info(f"Exploration alpha set to {alpha}")

    # Compatibility methods for specific benchmark frameworks

    def humaneval_generate(self, prompt: str, **kwargs) -> str:
        """HumanEval-compatible generation method"""
        return self.generate(prompt, **kwargs)

    def bigcode_generate(self, prompt: str, **kwargs) -> str:
        """BigCode-compatible generation method"""
        return self.generate(prompt, **kwargs)

    def lm_eval_generate(self, prompt: str, **kwargs) -> str:
        """LM-Eval-compatible generation method"""
        return self.generate(prompt, **kwargs)

    # Context manager support for batch processing

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with analytics logging"""
        if self.total_requests > 0:
            logger.info(f"Session summary: {self.successful_requests}/{self.total_requests} "
                       f"successful ({self.successful_requests/self.total_requests*100:.1f}%)")

            # Log strategy distribution
            distribution = self.get_strategy_distribution()
            logger.info(f"Strategy distribution: {distribution}")


# Factory functions for easy integration

def create_adaptive_humaneval_adapter(model_name: str, **kwargs) -> AdaptiveBenchmarkAdapter:
    """Create adapter optimized for HumanEval benchmarks"""
    return AdaptiveBenchmarkAdapter(
        model_name=model_name,
        quality_threshold=0.6,
        exploration_alpha=2.0,
        **kwargs
    )

def create_adaptive_bigcode_adapter(model_name: str, **kwargs) -> AdaptiveBenchmarkAdapter:
    """Create adapter optimized for BigCode benchmarks"""
    return AdaptiveBenchmarkAdapter(
        model_name=model_name,
        quality_threshold=0.65,
        exploration_alpha=2.2,
        **kwargs
    )

def create_adaptive_lm_eval_adapter(model_name: str, **kwargs) -> AdaptiveBenchmarkAdapter:
    """Create adapter optimized for LM-Eval benchmarks"""
    return AdaptiveBenchmarkAdapter(
        model_name=model_name,
        quality_threshold=0.55,
        exploration_alpha=1.8,
        **kwargs
    )


# Compatibility function for drop-in replacement
def create_ollama_interface(model_name: str, **kwargs):
    """
    Drop-in replacement function for existing Ollama interface creation.

    This can be used to replace existing ollama interface creation
    throughout the codebase with minimal changes.
    """
    return AdaptiveBenchmarkAdapter(model_name=model_name, **kwargs)