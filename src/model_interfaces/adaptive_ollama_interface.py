#!/usr/bin/env python3
"""
Adaptive Ollama Interface with Contextual Multi-Armed Bandit

Integrates Phase 1 smart response processing with Phase 2 adaptive strategy selection.
Uses contextual bandits to automatically learn optimal prompting strategies
based on prompt characteristics and quality feedback.

Based on academic research:
- ProCC Framework (2024): Contextual multi-armed bandits for code completion
- Bandit-Based Prompt Design (March 2025): Autonomous optimization
- Multi-Armed Bandits Meet LLMs (May 2025): Comprehensive adaptive approaches

Version: 1.0 (Phase 2 Implementation)
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Phase 1 imports
from .smart_response_processor import SmartResponseProcessor, ProcessedResponse, ContentQuality

# Phase 2 imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prompting.context_analyzer import PromptContextAnalyzer, ContextFeatures
from prompting.bandit_strategy_selector import LinUCBBandit, PromptingStrategy, create_bandit_selector

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveOllamaResponse:
    """Enhanced response with adaptive strategy information and quality metrics"""
    # Core response data
    text: str                           # Cleaned response text
    raw_text: str                      # Original response text
    execution_time: float              # Time taken for request

    # Phase 1: Dual success flags and quality
    http_success: bool                 # HTTP request succeeded
    content_quality_success: bool      # Content meets quality threshold
    overall_success: bool              # Combined success indicator
    quality_level: ContentQuality      # Detailed quality assessment
    quality_score: float               # Numerical quality score (0.0-1.0)
    is_conversational: bool           # Contains conversational elements
    is_executable: bool               # Appears to be executable code
    has_syntax_errors: bool           # Contains syntax errors
    cleaning_applied: bool            # Whether cleaning was applied

    # Phase 2: Adaptive strategy information
    selected_strategy: str             # Strategy chosen by bandit
    strategy_confidence: float         # Confidence in strategy selection
    predicted_reward: float           # Predicted quality before generation
    context_features: Dict[str, float] # Extracted context features
    bandit_exploration: bool          # Whether this was exploration vs exploitation
    is_exploration: bool              # Alias for bandit_exploration for compatibility
    adaptation_effectiveness: float   # How effective this adaptation was

    # Additional metadata
    error_message: Optional[str] = None
    adaptation_metadata: Optional[Dict[str, Any]] = None

class AdaptiveOllamaInterface:
    """Adaptive Ollama interface with contextual multi-armed bandit strategy selection"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434",
                 quality_threshold: float = 0.3, exploration_alpha: float = 1.0):
        """
        Initialize adaptive interface

        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            quality_threshold: Minimum quality score for content success
            exploration_alpha: Bandit exploration parameter (higher = more exploration)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        # Phase 1: Smart response processing
        self.response_processor = SmartResponseProcessor(quality_threshold=quality_threshold)

        # Phase 2: Adaptive components
        self.context_analyzer = PromptContextAnalyzer()
        self.bandit_selector = create_bandit_selector(model_name, alpha=exploration_alpha)

        # Strategy configurations (conservative for Phase 2 stability)
        self.strategy_prompts = {
            PromptingStrategy.CODE_ENGINE: {
                "prompt": "You are a code completion engine. Complete the code with only the missing parts. No explanations.",
                "temperature": 0.0,
                "stop_tokens": ["\n\n"]
            },
            PromptingStrategy.SILENT_GENERATOR: {
                "prompt": "Complete the code. Output only the completion.",
                "temperature": 0.1,
                "stop_tokens": ["\n\n"]
            },
            PromptingStrategy.DETERMINISTIC: {
                "prompt": "Complete this code with the most likely continuation:",
                "temperature": 0.0,
                "stop_tokens": ["\n\n"]
            },
            PromptingStrategy.NEGATIVE_PROMPT: {
                "prompt": "Complete the code. Do NOT include explanations, comments, or markdown. Code only.",
                "temperature": 0.1,
                "stop_tokens": ["\n\n"]
            },
            PromptingStrategy.FORMAT_CONSTRAINT: {
                "prompt": "Format: code_only. Complete: ",
                "temperature": 0.1,
                "stop_tokens": ["\n\n"]
            },
            PromptingStrategy.ROLE_BASED: {
                "prompt": "As a code autocomplete tool, complete this code:",
                "temperature": 0.0,
                "stop_tokens": ["\n\n"]
            }
        }

        # Performance tracking
        self.adaptation_history: List[Dict[str, Any]] = []
        self.total_requests = 0
        self.successful_adaptations = 0

    def generate_adaptive(self, prompt: str, **kwargs) -> AdaptiveOllamaResponse:
        """
        Generate response using adaptive strategy selection

        This is the main method that combines Phase 1 and Phase 2:
        1. Extract contextual features from prompt
        2. Use bandit to select optimal strategy
        3. Generate response with selected strategy
        4. Process response with smart cleaning and quality evaluation
        5. Update bandit with observed quality feedback
        """
        start_time = time.time()
        self.total_requests += 1

        try:
            # Phase 2: Extract contextual features
            context_features = self.context_analyzer.extract_features(prompt, self.model_name)

            # Phase 2: Select strategy using bandit
            selected_strategy, confidence, predicted_reward = self.bandit_selector.select_strategy(context_features)

            logger.info(f"Adaptive selection: {selected_strategy.value} "
                       f"(confidence: {confidence:.3f}, predicted: {predicted_reward:.3f})")

            # Generate response with selected strategy
            response = self._generate_with_strategy(prompt, selected_strategy, **kwargs)

            # Calculate total execution time
            total_execution_time = time.time() - start_time

            # Phase 1: Process response with smart cleaning and quality evaluation
            processed = self.response_processor.process_response(response.get('raw_text', ''),
                                                               http_success=response.get('http_success', False))

            # Determine if this was exploration (confidence below threshold)
            exploration_threshold = 0.6  # Configurable
            is_exploration = confidence < exploration_threshold

            # Calculate adaptation effectiveness
            adaptation_effectiveness = min(processed.quality_score / max(predicted_reward, 0.1), 2.0)

            # Create adaptive response
            adaptive_response = AdaptiveOllamaResponse(
                text=processed.text,
                raw_text=processed.raw_text,
                execution_time=total_execution_time,
                http_success=processed.http_success,
                content_quality_success=processed.content_quality_success,
                overall_success=processed.overall_success,
                quality_level=processed.quality_level,
                quality_score=processed.quality_score,
                is_conversational=processed.is_conversational,
                is_executable=processed.is_executable,
                has_syntax_errors=processed.has_syntax_errors,
                cleaning_applied=processed.cleaning_applied,
                selected_strategy=selected_strategy.value,
                strategy_confidence=confidence,
                predicted_reward=predicted_reward,
                context_features=context_features.to_dict(),
                bandit_exploration=is_exploration,
                is_exploration=is_exploration,  # Compatibility alias
                adaptation_effectiveness=adaptation_effectiveness,
                adaptation_metadata=processed.metadata
            )

            # Phase 2: Update bandit with observed quality feedback
            self.bandit_selector.update_reward(
                strategy=selected_strategy,
                context_features=context_features,
                quality_score=processed.quality_score,
                execution_time=total_execution_time,
                success=processed.overall_success
            )

            # Track adaptation success
            if processed.overall_success:
                self.successful_adaptations += 1

            # Record adaptation history
            self.adaptation_history.append({
                'timestamp': time.time(),
                'strategy': selected_strategy.value,
                'confidence': confidence,
                'predicted_reward': predicted_reward,
                'actual_quality': processed.quality_score,
                'success': processed.overall_success,
                'exploration': is_exploration,
                'context_summary': self._summarize_context(context_features)
            })

            return adaptive_response

        except Exception as e:
            logger.error(f"Adaptive generation failed: {e}")
            return self._create_error_response(str(e), time.time() - start_time)

    def _generate_with_strategy(self, prompt: str, strategy: PromptingStrategy, **kwargs) -> Dict[str, Any]:
        """Generate response using specific strategy"""
        try:
            strategy_config = self.strategy_prompts[strategy]

            # Create enhanced prompt
            enhanced_prompt = f"{strategy_config['prompt']}\n\n{prompt}"

            # Configure options
            options = {
                "temperature": kwargs.get("temperature", strategy_config["temperature"]),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", 100),
            }

            # Add stop tokens
            if strategy_config["stop_tokens"]:
                options["stop"] = strategy_config["stop_tokens"]

            payload = {
                "model": self.model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": options
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", 30)
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            return {
                'raw_text': raw_response,
                'http_success': True
            }

        except requests.RequestException as e:
            logger.error(f"HTTP request failed for strategy {strategy.value}: {e}")
            return {
                'raw_text': "",
                'http_success': False,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Generation failed for strategy {strategy.value}: {e}")
            return {
                'raw_text': "",
                'http_success': False,
                'error': str(e)
            }

    def get_adaptation_analytics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation analytics and insights"""
        return {
            'summary': {
                'total_requests': self.total_requests,
                'successful_adaptations': self.successful_adaptations,
                'success_rate': self.successful_adaptations / max(self.total_requests, 1),
                'adaptation_efficiency': self._calculate_adaptation_efficiency()
            },
            'bandit_performance': self.bandit_selector.get_strategy_performance(),
            'exploration_stats': self.bandit_selector.get_exploration_stats(),
            'context_insights': self.bandit_selector.get_context_insights(),
            'recent_adaptations': self.adaptation_history[-20:],  # Last 20
            'learning_trends': self._analyze_learning_trends(),
            'strategy_evolution': self._analyze_strategy_evolution()
        }

    def _calculate_adaptation_efficiency(self) -> float:
        """Calculate how efficiently the system is adapting (learning curve)"""
        if len(self.adaptation_history) < 10:
            return 0.0

        # Compare recent performance to early performance
        early_performance = [entry['actual_quality'] for entry in self.adaptation_history[:10]]
        recent_performance = [entry['actual_quality'] for entry in self.adaptation_history[-10:]]

        early_avg = sum(early_performance) / len(early_performance)
        recent_avg = sum(recent_performance) / len(recent_performance)

        # Efficiency is improvement rate
        improvement = recent_avg - early_avg
        return max(0.0, min(improvement * 2.0, 1.0))  # Normalize to 0-1

    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """Analyze learning trends over time"""
        if len(self.adaptation_history) < 5:
            return {'insufficient_data': True}

        # Quality trend
        qualities = [entry['actual_quality'] for entry in self.adaptation_history]
        recent_window = min(20, len(qualities) // 2)

        if len(qualities) >= recent_window * 2:
            early_qualities = qualities[:recent_window]
            recent_qualities = qualities[-recent_window:]

            trend = {
                'early_avg_quality': sum(early_qualities) / len(early_qualities),
                'recent_avg_quality': sum(recent_qualities) / len(recent_qualities),
                'quality_improvement': (sum(recent_qualities) / len(recent_qualities)) -
                                     (sum(early_qualities) / len(early_qualities)),
                'learning_trajectory': 'improving' if sum(recent_qualities) > sum(early_qualities) else 'stable'
            }
        else:
            trend = {'insufficient_data_for_trend': True}

        # Exploration vs exploitation trend
        explorations = [entry['exploration'] for entry in self.adaptation_history[-50:]]
        exploration_rate = sum(explorations) / len(explorations) if explorations else 0.0

        return {
            'quality_trend': trend,
            'current_exploration_rate': exploration_rate,
            'total_learning_sessions': len(self.adaptation_history)
        }

    def _analyze_strategy_evolution(self) -> Dict[str, Any]:
        """Analyze how strategy preferences have evolved"""
        if len(self.adaptation_history) < 10:
            return {'insufficient_data': True}

        # Strategy usage over time
        strategy_timeline = {}
        window_size = max(10, len(self.adaptation_history) // 5)

        for i in range(0, len(self.adaptation_history), window_size):
            window = self.adaptation_history[i:i + window_size]
            window_strategies = [entry['strategy'] for entry in window]

            period_name = f"period_{i//window_size + 1}"
            strategy_timeline[period_name] = {
                strategy: window_strategies.count(strategy) / len(window_strategies)
                for strategy in set(window_strategies)
            }

        return {
            'strategy_timeline': strategy_timeline,
            'current_favorite': self._get_current_favorite_strategy(),
            'strategy_diversity': len(set(entry['strategy'] for entry in self.adaptation_history[-20:]))
        }

    def _get_current_favorite_strategy(self) -> str:
        """Get currently most successful strategy"""
        performance = self.bandit_selector.get_strategy_performance()
        if not performance:
            return "unknown"

        best_strategy = max(performance.keys(),
                           key=lambda s: performance[s]['mean_reward'])
        return best_strategy

    def _summarize_context(self, context_features: ContextFeatures) -> Dict[str, str]:
        """Create human-readable context summary"""
        return {
            'complexity': 'high' if context_features.prompt_complexity > 0.7 else
                         'medium' if context_features.prompt_complexity > 0.3 else 'low',
            'domain': 'specialized' if context_features.code_domain > 0.5 else 'general',
            'structure': 'complex' if context_features.nesting_level > 0.5 else 'simple'
        }

    def _create_error_response(self, error_message: str, execution_time: float) -> AdaptiveOllamaResponse:
        """Create error response with consistent structure"""
        empty_features = ContextFeatures(
            prompt_complexity=0.0, context_length=0.0, completion_type=0.0,
            code_domain=0.0, indentation_level=0.0, nesting_level=0.0,
            has_function_def=0.0, has_class_def=0.0, keyword_density=0.0,
            variable_complexity=0.0, model_preference=0.0
        )

        return AdaptiveOllamaResponse(
            text="", raw_text="", execution_time=execution_time,
            http_success=False, content_quality_success=False, overall_success=False,
            quality_level=ContentQuality.EMPTY, quality_score=0.0,
            is_conversational=False, is_executable=False, has_syntax_errors=False,
            cleaning_applied=False, selected_strategy="error", strategy_confidence=0.0,
            predicted_reward=0.0, context_features=empty_features.to_dict(),
            bandit_exploration=False, is_exploration=False, adaptation_effectiveness=0.0,
            error_message=error_message
        )

    def tune_exploration(self, target_rate: float = 0.2):
        """Tune exploration parameter to achieve target exploration rate"""
        self.bandit_selector.tune_exploration_parameter(target_rate)

    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            for model in models:
                if model.get("name", "").startswith(self.model_name):
                    return model
            return {}
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

def create_adaptive_interface(model_name: str,
                            quality_threshold: float = 0.3,
                            exploration_alpha: float = 1.0) -> AdaptiveOllamaInterface:
    """Factory function to create adaptive interface"""
    return AdaptiveOllamaInterface(
        model_name=model_name,
        quality_threshold=quality_threshold,
        exploration_alpha=exploration_alpha
    )

def create_research_interface(model_name: str) -> AdaptiveOllamaInterface:
    """Create interface optimized for research (high exploration)"""
    return AdaptiveOllamaInterface(
        model_name=model_name,
        quality_threshold=0.1,  # Lenient for research
        exploration_alpha=2.0   # High exploration
    )

def create_production_interface(model_name: str) -> AdaptiveOllamaInterface:
    """Create interface optimized for production (balanced)"""
    return AdaptiveOllamaInterface(
        model_name=model_name,
        quality_threshold=0.5,  # Stricter for production
        exploration_alpha=0.5   # Lower exploration
    )