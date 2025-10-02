#!/usr/bin/env python3
"""
Contextual Multi-Armed Bandit for Adaptive Strategy Selection

Implements LinUCB (Linear Upper Confidence Bound) algorithm for dynamic
prompting strategy selection based on contextual features and quality feedback.

Based on:
- ProCC Framework (2024): 7.92%-10.1% improvement using contextual bandits
- Bandit-Based Prompt Design (March 2025): ArXiv 2503.01163
- Multi-Armed Bandits Meet LLMs (May 2025): ArXiv 2505.13355

Version: 1.0 (Phase 2 Implementation)
"""

import numpy as np
import json
import logging
import time
import sqlite3
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .context_analyzer import ContextFeatures, PromptContextAnalyzer

logger = logging.getLogger(__name__)

class PromptingStrategy(Enum):
    """Available prompting strategies for bandit selection"""
    CODE_ENGINE = "code_engine"
    SILENT_GENERATOR = "silent_generator"
    DETERMINISTIC = "deterministic"
    NEGATIVE_PROMPT = "negative_prompt"
    FORMAT_CONSTRAINT = "format_constraint"
    ROLE_BASED = "role_based"

@dataclass
class BanditAction:
    """Represents a bandit action (strategy selection) with context"""
    strategy: PromptingStrategy
    context_features: ContextFeatures
    confidence: float
    predicted_reward: float
    timestamp: float

@dataclass
class BanditFeedback:
    """Feedback for bandit learning"""
    strategy: PromptingStrategy
    context_features: ContextFeatures
    quality_score: float
    execution_time: float
    success: bool
    timestamp: float

class LinUCBBandit:
    """Linear Upper Confidence Bound bandit for contextual strategy selection"""

    def __init__(self, strategies: List[PromptingStrategy],
                 alpha: float = 2.0,
                 feature_dim: int = 11,
                 model_name: str = "unknown"):
        """
        Initialize LinUCB bandit

        Args:
            strategies: Available prompting strategies
            alpha: Exploration parameter (higher = more exploration)
            feature_dim: Dimension of context feature vector
            model_name: Model name for persistence
        """
        self.strategies = strategies
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.model_name = model_name

        # LinUCB parameters for each strategy (arm)
        self.A = {s: np.eye(feature_dim) for s in strategies}  # Feature covariance
        self.b = {s: np.zeros(feature_dim) for s in strategies}  # Feature-reward products
        self.theta = {s: np.zeros(feature_dim) for s in strategies}  # Learned parameters

        # Performance tracking
        self.action_history: List[BanditAction] = []
        self.feedback_history: List[BanditFeedback] = []
        self.strategy_stats = {s: {'count': 0, 'total_reward': 0.0, 'success_count': 0}
                              for s in strategies}

        # Persistence
        self.db_path = f"bandit_data_{model_name.replace(':', '_')}.db"
        self._init_database()
        self._load_state()

    def select_strategy(self, context_features: ContextFeatures) -> Tuple[PromptingStrategy, float, float]:
        """
        Select optimal strategy using LinUCB algorithm

        Returns:
            Tuple of (selected_strategy, confidence_bound, predicted_reward)
        """
        x = np.array(context_features.to_vector())

        best_strategy = None
        best_confidence = -np.inf
        best_predicted = 0.0

        strategy_scores = {}

        for strategy in self.strategies:
            try:
                # Update theta estimate
                A_inv = np.linalg.inv(self.A[strategy])
                self.theta[strategy] = A_inv @ self.b[strategy]

                # Predicted reward
                predicted_reward = x.T @ self.theta[strategy]

                # Confidence interval width
                confidence_width = self.alpha * np.sqrt(x.T @ A_inv @ x)

                # Upper confidence bound
                upper_confidence = predicted_reward + confidence_width

                strategy_scores[strategy] = {
                    'predicted': predicted_reward,
                    'confidence_width': confidence_width,
                    'upper_bound': upper_confidence
                }

                if upper_confidence > best_confidence:
                    best_confidence = upper_confidence
                    best_predicted = predicted_reward
                    best_strategy = strategy

            except np.linalg.LinAlgError as e:
                logger.warning(f"LinAlg error for strategy {strategy}: {e}, using fallback")
                # Fallback to basic selection
                if best_strategy is None:
                    best_strategy = strategy
                    best_confidence = 0.5
                    best_predicted = 0.5

        # Apply diversity enforcement if needed
        best_strategy = self._apply_diversity_enforcement(best_strategy, strategy_scores)

        # Record action
        action = BanditAction(
            strategy=best_strategy,
            context_features=context_features,
            confidence=best_confidence,
            predicted_reward=best_predicted,
            timestamp=time.time()
        )
        self.action_history.append(action)

        logger.debug(f"Selected strategy: {best_strategy.value}, "
                    f"confidence: {best_confidence:.3f}, "
                    f"predicted: {best_predicted:.3f}")

        return best_strategy, best_confidence, best_predicted

    def update_reward(self, strategy: PromptingStrategy,
                     context_features: ContextFeatures,
                     quality_score: float,
                     execution_time: float = 0.0,
                     success: bool = True):
        """
        Update bandit with observed reward (quality score)

        Args:
            strategy: Strategy that was used
            context_features: Context features for the request
            quality_score: Observed quality score (0.0-1.0)
            execution_time: Time taken for generation
            success: Whether the generation was successful
        """
        x = np.array(context_features.to_vector())

        # Update LinUCB parameters
        self.A[strategy] += np.outer(x, x)
        self.b[strategy] += quality_score * x

        # Update statistics
        self.strategy_stats[strategy]['count'] += 1
        self.strategy_stats[strategy]['total_reward'] += quality_score
        if success:
            self.strategy_stats[strategy]['success_count'] += 1

        # Record feedback
        feedback = BanditFeedback(
            strategy=strategy,
            context_features=context_features,
            quality_score=quality_score,
            execution_time=execution_time,
            success=success,
            timestamp=time.time()
        )
        self.feedback_history.append(feedback)

        logger.debug(f"Updated reward for {strategy.value}: quality={quality_score:.3f}, "
                    f"success={success}")

        # Save state immediately for critical learning data
        self._save_state()

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive performance analytics for each strategy"""
        performance = {}

        for strategy in self.strategies:
            stats = self.strategy_stats[strategy]

            # Calculate metrics
            count = stats['count']
            if count > 0:
                mean_reward = stats['total_reward'] / count
                success_rate = stats['success_count'] / count

                # Calculate recent performance (last 20 trials)
                recent_feedback = [f for f in self.feedback_history[-20:]
                                 if f.strategy == strategy]

                if recent_feedback:
                    recent_rewards = [f.quality_score for f in recent_feedback]
                    recent_mean = np.mean(recent_rewards)
                    recent_std = np.std(recent_rewards)
                else:
                    recent_mean = mean_reward
                    recent_std = 0.0

                performance[strategy.value] = {
                    'total_trials': count,
                    'mean_reward': mean_reward,
                    'success_rate': success_rate,
                    'recent_mean_reward': recent_mean,
                    'recent_std_reward': recent_std,
                    'theta_norm': np.linalg.norm(self.theta[strategy]),
                    'confidence_trace': np.trace(np.linalg.inv(self.A[strategy]))
                }
            else:
                performance[strategy.value] = {
                    'total_trials': 0,
                    'mean_reward': 0.0,
                    'success_rate': 0.0,
                    'recent_mean_reward': 0.0,
                    'recent_std_reward': 0.0,
                    'theta_norm': 0.0,
                    'confidence_trace': 1.0
                }

        return performance

    def get_exploration_stats(self) -> Dict[str, float]:
        """Get exploration vs exploitation statistics"""
        if not self.action_history:
            return {'exploration_rate': 0.0, 'total_actions': 0}

        # Count how often we selected the current best strategy
        strategy_counts = {}
        for action in self.action_history[-50:]:  # Last 50 actions
            strategy = action.strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        total_recent = len(self.action_history[-50:])
        if total_recent == 0:
            return {'exploration_rate': 0.0, 'total_actions': len(self.action_history)}

        # Current best strategy (highest average reward)
        performance = self.get_strategy_performance()
        best_strategy = max(self.strategies,
                           key=lambda s: performance[s.value]['mean_reward'])

        best_count = strategy_counts.get(best_strategy, 0)
        exploitation_rate = best_count / total_recent
        exploration_rate = 1.0 - exploitation_rate

        return {
            'exploration_rate': exploration_rate,
            'exploitation_rate': exploitation_rate,
            'total_actions': len(self.action_history),
            'recent_diversity': len(strategy_counts) / len(self.strategies)
        }

    def get_context_insights(self) -> Dict[str, Any]:
        """Analyze which contexts lead to better performance for each strategy"""
        insights = {}

        for strategy in self.strategies:
            strategy_feedback = [f for f in self.feedback_history if f.strategy == strategy]

            if len(strategy_feedback) < 5:  # Need minimum data
                insights[strategy.value] = {'insufficient_data': True}
                continue

            # Separate high and low quality responses
            high_quality = [f for f in strategy_feedback if f.quality_score > 0.7]
            low_quality = [f for f in strategy_feedback if f.quality_score < 0.3]

            if high_quality and low_quality:
                # Average context features for high vs low quality
                high_contexts = [f.context_features.to_dict() for f in high_quality]
                low_contexts = [f.context_features.to_dict() for f in low_quality]

                # Calculate feature differences
                feature_diffs = {}
                for feature in high_contexts[0].keys():
                    high_avg = np.mean([ctx[feature] for ctx in high_contexts])
                    low_avg = np.mean([ctx[feature] for ctx in low_contexts])
                    feature_diffs[feature] = high_avg - low_avg

                insights[strategy.value] = {
                    'high_quality_count': len(high_quality),
                    'low_quality_count': len(low_quality),
                    'feature_preferences': feature_diffs,
                    'optimal_context': {k: np.mean([ctx[k] for ctx in high_contexts])
                                      for k in high_contexts[0].keys()}
                }
            else:
                insights[strategy.value] = {'insufficient_contrast': True}

        return insights

    def tune_exploration_parameter(self, target_exploration_rate: float = 0.3,
                                 target_diversity: float = 0.6):
        """
        Automatically tune alpha parameter based on recent exploration rate and strategy diversity

        Args:
            target_exploration_rate: Desired exploration rate (0.0-1.0)
            target_diversity: Desired strategy diversity (0.0-1.0)
        """
        current_stats = self.get_exploration_stats()
        current_exploration = current_stats['exploration_rate']
        current_diversity = current_stats['recent_diversity']

        # Check both exploration rate and diversity
        needs_more_exploration = (current_exploration < target_exploration_rate - 0.05 or
                                current_diversity < target_diversity - 0.1)

        needs_less_exploration = (current_exploration > target_exploration_rate + 0.1 and
                                current_diversity > target_diversity + 0.2)

        if needs_more_exploration:
            # Increase exploration more aggressively if diversity is low
            multiplier = 1.2 if current_diversity < 0.4 else 1.1
            self.alpha = min(self.alpha * multiplier, 3.5)
            logger.info(f"Increased exploration: alpha={self.alpha:.3f} "
                       f"(diversity={current_diversity:.2f}, exploration={current_exploration:.2f})")
        elif needs_less_exploration:
            # Decrease exploration only if we have good diversity
            self.alpha = max(self.alpha * 0.9, 1.0)  # Don't go below 1.0
            logger.info(f"Decreased exploration: alpha={self.alpha:.3f}")

    def _apply_diversity_enforcement(self, selected_strategy: PromptingStrategy,
                                   strategy_scores: Dict) -> PromptingStrategy:
        """
        Apply diversity enforcement to prevent over-exploitation of single strategies

        Args:
            selected_strategy: Strategy chosen by LinUCB
            strategy_scores: All strategy scores for potential override

        Returns:
            Final strategy (may be overridden for diversity)
        """
        # Only apply enforcement after sufficient history
        if len(self.action_history) < 10:
            return selected_strategy

        # Check recent strategy usage (last 20 actions)
        recent_actions = self.action_history[-20:]
        strategy_counts = {}
        for action in recent_actions:
            strategy_counts[action.strategy] = strategy_counts.get(action.strategy, 0) + 1

        # Calculate diversity metrics
        total_recent = len(recent_actions)
        used_strategies = len(strategy_counts)
        total_strategies = len(self.strategies)
        recent_diversity = used_strategies / total_strategies

        # Check if selected strategy is over-used
        selected_count = strategy_counts.get(selected_strategy, 0)
        selected_ratio = selected_count / total_recent

        # Diversity enforcement triggers
        low_diversity = recent_diversity < 0.4  # Less than 40% of strategies used
        over_exploitation = selected_ratio > 0.6  # One strategy >60% of recent use

        if low_diversity or over_exploitation:
            # Find under-used strategies with reasonable scores
            underused_strategies = []
            for strategy in self.strategies:
                usage_ratio = strategy_counts.get(strategy, 0) / total_recent
                if usage_ratio < 0.2 and strategy in strategy_scores:  # Used <20% of time
                    underused_strategies.append((strategy, strategy_scores[strategy]['upper_bound']))

            if underused_strategies:
                # Select best underused strategy
                underused_strategies.sort(key=lambda x: x[1], reverse=True)
                chosen_strategy = underused_strategies[0][0]

                logger.info(f"Diversity enforcement: {selected_strategy.value} → {chosen_strategy.value} "
                           f"(diversity={recent_diversity:.2f}, exploitation={selected_ratio:.2f})")
                return chosen_strategy

        return selected_strategy

    def reset_bandit(self):
        """Reset bandit to initial state (for testing or retraining)"""
        self.A = {s: np.eye(self.feature_dim) for s in self.strategies}
        self.b = {s: np.zeros(self.feature_dim) for s in self.strategies}
        self.theta = {s: np.zeros(self.feature_dim) for s in self.strategies}

        self.action_history = []
        self.feedback_history = []
        self.strategy_stats = {s: {'count': 0, 'total_reward': 0.0, 'success_count': 0}
                              for s in self.strategies}

        logger.info("Bandit reset to initial state")

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS bandit_state (
                        strategy TEXT PRIMARY KEY,
                        A_matrix TEXT,
                        b_vector TEXT,
                        theta_vector TEXT
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS feedback_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy TEXT,
                        context_features TEXT,
                        quality_score REAL,
                        execution_time REAL,
                        success INTEGER,
                        timestamp REAL
                    )
                ''')
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")

    def _save_state(self):
        """Save bandit state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for strategy in self.strategies:
                    conn.execute('''
                        INSERT OR REPLACE INTO bandit_state
                        (strategy, A_matrix, b_vector, theta_vector)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        strategy.value,
                        json.dumps(self.A[strategy].tolist()),
                        json.dumps(self.b[strategy].tolist()),
                        json.dumps(self.theta[strategy].tolist())
                    ))

                # Save recent feedback (last 100 entries)
                for feedback in self.feedback_history[-100:]:
                    conn.execute('''
                        INSERT OR IGNORE INTO feedback_history
                        (strategy, context_features, quality_score, execution_time, success, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        feedback.strategy.value,
                        json.dumps(feedback.context_features.to_dict()),
                        feedback.quality_score,
                        feedback.execution_time,
                        int(feedback.success),
                        feedback.timestamp
                    ))

        except Exception as e:
            logger.warning(f"State save failed: {e}")

    def _load_state(self):
        """Load bandit state from database"""
        try:
            if not os.path.exists(self.db_path):
                return

            with sqlite3.connect(self.db_path) as conn:
                # Load bandit parameters
                cursor = conn.execute('SELECT strategy, A_matrix, b_vector, theta_vector FROM bandit_state')
                for row in cursor:
                    strategy_name, A_json, b_json, theta_json = row
                    try:
                        strategy = PromptingStrategy(strategy_name)
                        self.A[strategy] = np.array(json.loads(A_json))
                        self.b[strategy] = np.array(json.loads(b_json))
                        self.theta[strategy] = np.array(json.loads(theta_json))
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning(f"Failed to load strategy {strategy_name}: {e}")

                # Load recent feedback
                cursor = conn.execute('''
                    SELECT strategy, context_features, quality_score, execution_time, success, timestamp
                    FROM feedback_history
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''')

                for row in cursor:
                    try:
                        strategy_name, context_json, quality_score, execution_time, success, timestamp = row
                        strategy = PromptingStrategy(strategy_name)
                        context_dict = json.loads(context_json)
                        context_features = ContextFeatures(**context_dict)

                        feedback = BanditFeedback(
                            strategy=strategy,
                            context_features=context_features,
                            quality_score=quality_score,
                            execution_time=execution_time,
                            success=bool(success),
                            timestamp=timestamp
                        )
                        self.feedback_history.append(feedback)

                        # Update statistics
                        self.strategy_stats[strategy]['count'] += 1
                        self.strategy_stats[strategy]['total_reward'] += quality_score
                        if success:
                            self.strategy_stats[strategy]['success_count'] += 1

                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning(f"Failed to load feedback: {e}")

                logger.info(f"Loaded bandit state with {len(self.feedback_history)} feedback entries")

        except Exception as e:
            logger.warning(f"State load failed: {e}")

def create_bandit_selector(model_name: str,
                          alpha: float = 1.0) -> LinUCBBandit:
    """Factory function to create bandit selector"""
    strategies = list(PromptingStrategy)
    return LinUCBBandit(strategies, alpha=alpha, model_name=model_name)

# Testing utilities
def test_bandit_functionality():
    """Test bandit with simulated data"""
    from .context_analyzer import PromptContextAnalyzer

    analyzer = PromptContextAnalyzer()
    bandit = create_bandit_selector("test_model", alpha=1.0)

    test_prompts = [
        "def fibonacci(n):",
        "class DataProcessor:",
        "import pandas as pd",
        "for i in range(10):"
    ]

    print("Testing bandit functionality...")

    for i, prompt in enumerate(test_prompts * 5):  # Repeat for learning
        # Extract features
        features = analyzer.extract_features(prompt, "test_model")

        # Select strategy
        strategy, confidence, predicted = bandit.select_strategy(features)

        # Simulate quality score (better for certain strategies on certain prompts)
        if "def " in prompt and strategy == PromptingStrategy.CODE_ENGINE:
            quality_score = 0.9
        elif "class " in prompt and strategy == PromptingStrategy.ROLE_BASED:
            quality_score = 0.8
        else:
            quality_score = np.random.uniform(0.3, 0.7)

        # Update bandit
        bandit.update_reward(strategy, features, quality_score, success=True)

        print(f"Iteration {i+1}: {strategy.value} -> quality {quality_score:.3f}")

    # Show results
    print("\nFinal Performance:")
    performance = bandit.get_strategy_performance()
    for strategy, stats in performance.items():
        print(f"{strategy}: {stats['mean_reward']:.3f} avg, {stats['total_trials']} trials")

if __name__ == "__main__":
    test_bandit_functionality()