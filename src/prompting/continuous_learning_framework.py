#!/usr/bin/env python3
"""
Continuous Learning Framework for Adaptive Prompting

Provides monitoring, optimization, and long-term learning capabilities
for the contextual multi-armed bandit system. Implements automatic
parameter tuning, performance tracking, and strategy evolution.

Version: 1.0 (Phase 2 Implementation)
"""

import json
import time
import logging
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum

from .bandit_strategy_selector import LinUCBBandit, PromptingStrategy
from .context_analyzer import ContextFeatures

logger = logging.getLogger(__name__)

class LearningPhase(Enum):
    """Learning phases for different optimization strategies"""
    EXPLORATION = "exploration"        # High exploration for initial learning
    EXPLOITATION = "exploitation"     # Focus on best known strategies
    BALANCED = "balanced"             # Balanced exploration/exploitation
    REFINEMENT = "refinement"         # Fine-tuning with low exploration

@dataclass
class LearningSession:
    """Represents a learning session with metrics"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_interactions: int
    successful_interactions: int
    average_quality: float
    learning_phase: LearningPhase
    strategies_explored: List[str]
    best_strategy: str
    improvement_rate: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    session_count: int
    total_interactions: int
    overall_success_rate: float
    average_quality_score: float
    strategy_diversity: float
    adaptation_efficiency: float
    exploration_efficiency: float
    learning_velocity: float

class ContinuousLearningFramework:
    """Framework for continuous learning and optimization"""

    def __init__(self, bandit: LinUCBBandit, model_name: str):
        """
        Initialize continuous learning framework

        Args:
            bandit: The LinUCB bandit instance to monitor and optimize
            model_name: Model name for persistence and tracking
        """
        self.bandit = bandit
        self.model_name = model_name

        # Learning session management
        self.current_session: Optional[LearningSession] = None
        self.session_history: List[LearningSession] = []
        self.session_counter = 0

        # Performance tracking
        self.interaction_buffer = deque(maxlen=1000)  # Last 1000 interactions
        self.quality_history = deque(maxlen=500)      # Quality score history
        self.strategy_performance_history = defaultdict(list)

        # Learning phase management
        self.current_phase = LearningPhase.EXPLORATION
        self.phase_transition_thresholds = {
            'exploration_to_balanced': {'min_interactions': 50, 'success_rate': 0.3},
            'balanced_to_exploitation': {'min_interactions': 100, 'success_rate': 0.6},
            'exploitation_to_refinement': {'min_interactions': 200, 'success_rate': 0.7}
        }

        # Optimization parameters
        self.auto_tuning_enabled = True
        self.performance_window = 50  # Window for performance evaluation
        self.adaptation_targets = {
            'min_success_rate': 0.6,
            'target_exploration_rate': 0.2,
            'quality_improvement_threshold': 0.05
        }

        # Persistence
        self.db_path = f"continuous_learning_{model_name.replace(':', '_')}.db"
        self._init_database()
        self._load_learning_history()

    def start_learning_session(self, phase: Optional[LearningPhase] = None) -> str:
        """Start a new learning session"""
        # End current session if active
        if self.current_session and self.current_session.end_time is None:
            self.end_learning_session()

        self.session_counter += 1
        session_id = f"session_{self.model_name}_{self.session_counter}_{int(time.time())}"

        # Determine learning phase
        if phase is None:
            phase = self._determine_optimal_phase()

        self.current_session = LearningSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=None,
            total_interactions=0,
            successful_interactions=0,
            average_quality=0.0,
            learning_phase=phase,
            strategies_explored=[],
            best_strategy="",
            improvement_rate=0.0
        )

        # Adjust bandit parameters for learning phase
        self._configure_bandit_for_phase(phase)

        logger.info(f"Started learning session {session_id} in {phase.value} phase")
        return session_id

    def record_interaction(self, strategy: PromptingStrategy,
                          context_features: ContextFeatures,
                          quality_score: float,
                          success: bool,
                          execution_time: float = 0.0):
        """Record an interaction for learning analysis"""
        interaction_data = {
            'timestamp': time.time(),
            'strategy': strategy.value,
            'quality_score': quality_score,
            'success': success,
            'execution_time': execution_time,
            'context_features': context_features.to_dict()
        }

        self.interaction_buffer.append(interaction_data)
        self.quality_history.append(quality_score)
        self.strategy_performance_history[strategy.value].append(quality_score)

        # Update current session
        if self.current_session:
            self.current_session.total_interactions += 1
            if success:
                self.current_session.successful_interactions += 1

            # Update session metrics
            self._update_session_metrics()

        # Check for phase transitions
        if self.auto_tuning_enabled:
            self._check_phase_transition()
            self._auto_tune_parameters()

    def end_learning_session(self) -> LearningSession:
        """End current learning session and analyze results"""
        if not self.current_session:
            raise ValueError("No active learning session")

        self.current_session.end_time = time.time()

        # Final session analysis
        self._finalize_session_metrics()

        # Store session
        self.session_history.append(self.current_session)
        self._save_learning_session(self.current_session)

        logger.info(f"Ended learning session {self.current_session.session_id}")
        logger.info(f"Session results: {self.current_session.total_interactions} interactions, "
                   f"{self.current_session.average_quality:.3f} avg quality")

        completed_session = self.current_session
        self.current_session = None

        return completed_session

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        if not self.interaction_buffer:
            return PerformanceMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        recent_interactions = list(self.interaction_buffer)[-self.performance_window:]

        # Calculate metrics
        total_interactions = len(self.interaction_buffer)
        session_count = len(self.session_history)

        success_rate = sum(1 for i in recent_interactions if i['success']) / len(recent_interactions)
        avg_quality = sum(i['quality_score'] for i in recent_interactions) / len(recent_interactions)

        # Strategy diversity (unique strategies in recent window)
        recent_strategies = set(i['strategy'] for i in recent_interactions)
        strategy_diversity = len(recent_strategies) / len(self.bandit.strategies)

        # Adaptation efficiency (improvement over time)
        adaptation_efficiency = self._calculate_adaptation_efficiency()

        # Exploration efficiency (quality during exploration vs exploitation)
        exploration_efficiency = self._calculate_exploration_efficiency()

        # Learning velocity (rate of improvement)
        learning_velocity = self._calculate_learning_velocity()

        return PerformanceMetrics(
            session_count=session_count,
            total_interactions=total_interactions,
            overall_success_rate=success_rate,
            average_quality_score=avg_quality,
            strategy_diversity=strategy_diversity,
            adaptation_efficiency=adaptation_efficiency,
            exploration_efficiency=exploration_efficiency,
            learning_velocity=learning_velocity
        )

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights and recommendations"""
        metrics = self.get_performance_metrics()
        bandit_performance = self.bandit.get_strategy_performance()

        # Strategy evolution analysis
        strategy_evolution = self._analyze_strategy_evolution()

        # Context-strategy mapping insights
        context_insights = self._analyze_context_strategy_patterns()

        # Learning phase recommendations
        phase_recommendations = self._recommend_learning_phase()

        # Performance trends
        performance_trends = self._analyze_performance_trends()

        return {
            'current_metrics': asdict(metrics),
            'current_phase': self.current_phase.value,
            'bandit_performance': bandit_performance,
            'strategy_evolution': strategy_evolution,
            'context_insights': context_insights,
            'phase_recommendations': phase_recommendations,
            'performance_trends': performance_trends,
            'optimization_opportunities': self._identify_optimization_opportunities()
        }

    def optimize_parameters(self) -> Dict[str, Any]:
        """Automatically optimize bandit and learning parameters"""
        optimizations = {}

        # Optimize exploration parameter
        current_exploration = self.bandit.get_exploration_stats()['exploration_rate']
        target_exploration = self.adaptation_targets['target_exploration_rate']

        if abs(current_exploration - target_exploration) > 0.1:
            old_alpha = self.bandit.alpha
            self.bandit.tune_exploration_parameter(target_exploration)
            optimizations['exploration_tuning'] = {
                'old_alpha': old_alpha,
                'new_alpha': self.bandit.alpha,
                'old_exploration_rate': current_exploration,
                'target_exploration_rate': target_exploration
            }

        # Optimize learning phase
        recommended_phase = self._recommend_learning_phase()
        if recommended_phase != self.current_phase:
            old_phase = self.current_phase
            self._transition_to_phase(recommended_phase)
            optimizations['phase_transition'] = {
                'old_phase': old_phase.value,
                'new_phase': self.current_phase.value,
                'reason': 'performance_optimization'
            }

        # Optimize quality threshold
        performance_metrics = self.get_performance_metrics()
        if performance_metrics.overall_success_rate < self.adaptation_targets['min_success_rate']:
            # Consider adjusting quality threshold or strategy selection
            optimizations['quality_threshold_recommendation'] = {
                'current_success_rate': performance_metrics.overall_success_rate,
                'target_success_rate': self.adaptation_targets['min_success_rate'],
                'recommendation': 'consider_lowering_quality_threshold'
            }

        logger.info(f"Applied {len(optimizations)} optimizations")
        return optimizations

    def reset_learning(self, preserve_best_strategies: bool = True):
        """Reset learning state while optionally preserving best strategies"""
        if preserve_best_strategies:
            # Save current best strategy information
            best_strategies = self._get_best_strategies()
            logger.info(f"Preserving {len(best_strategies)} best strategies")

        # Reset bandit
        self.bandit.reset_bandit()

        # Clear learning history
        self.interaction_buffer.clear()
        self.quality_history.clear()
        self.strategy_performance_history.clear()

        # Reset to exploration phase
        self.current_phase = LearningPhase.EXPLORATION
        self._configure_bandit_for_phase(self.current_phase)

        # Optionally restore best strategies with reduced confidence
        if preserve_best_strategies:
            self._restore_best_strategies(best_strategies)

        logger.info("Learning state reset")

    def _determine_optimal_phase(self) -> LearningPhase:
        """Determine optimal learning phase based on current performance"""
        if not self.interaction_buffer:
            return LearningPhase.EXPLORATION

        recent_interactions = list(self.interaction_buffer)[-self.performance_window:]
        success_rate = sum(1 for i in recent_interactions if i['success']) / len(recent_interactions)
        interaction_count = len(self.interaction_buffer)

        # Phase transition logic
        if interaction_count < 50:
            return LearningPhase.EXPLORATION
        elif interaction_count < 100 or success_rate < 0.5:
            return LearningPhase.BALANCED
        elif success_rate < 0.7:
            return LearningPhase.EXPLOITATION
        else:
            return LearningPhase.REFINEMENT

    def _configure_bandit_for_phase(self, phase: LearningPhase):
        """Configure bandit parameters for learning phase"""
        phase_configs = {
            LearningPhase.EXPLORATION: {'alpha': 2.0},
            LearningPhase.BALANCED: {'alpha': 1.0},
            LearningPhase.EXPLOITATION: {'alpha': 0.5},
            LearningPhase.REFINEMENT: {'alpha': 0.2}
        }

        config = phase_configs.get(phase, {'alpha': 1.0})
        self.bandit.alpha = config['alpha']
        self.current_phase = phase

        logger.debug(f"Configured bandit for {phase.value} phase: alpha={self.bandit.alpha}")

    def _update_session_metrics(self):
        """Update current session metrics"""
        if not self.current_session:
            return

        # Update average quality
        session_interactions = [i for i in self.interaction_buffer
                               if i['timestamp'] >= self.current_session.start_time]

        if session_interactions:
            self.current_session.average_quality = sum(
                i['quality_score'] for i in session_interactions
            ) / len(session_interactions)

            # Update strategies explored
            strategies = set(i['strategy'] for i in session_interactions)
            self.current_session.strategies_explored = list(strategies)

            # Update best strategy
            strategy_performance = defaultdict(list)
            for interaction in session_interactions:
                strategy_performance[interaction['strategy']].append(interaction['quality_score'])

            if strategy_performance:
                best_strategy = max(strategy_performance.keys(),
                                  key=lambda s: sum(strategy_performance[s]) / len(strategy_performance[s]))
                self.current_session.best_strategy = best_strategy

    def _finalize_session_metrics(self):
        """Finalize session metrics at session end"""
        if not self.current_session:
            return

        # Calculate improvement rate
        session_interactions = [i for i in self.interaction_buffer
                               if i['timestamp'] >= self.current_session.start_time]

        if len(session_interactions) >= 10:
            early_quality = sum(i['quality_score'] for i in session_interactions[:5]) / 5
            late_quality = sum(i['quality_score'] for i in session_interactions[-5:]) / 5
            self.current_session.improvement_rate = late_quality - early_quality
        else:
            self.current_session.improvement_rate = 0.0

    def _check_phase_transition(self):
        """Check if learning phase should transition"""
        if not self.interaction_buffer:
            return

        current_performance = self.get_performance_metrics()
        transition_needed = False
        new_phase = self.current_phase

        # Check transition conditions
        if (self.current_phase == LearningPhase.EXPLORATION and
            current_performance.total_interactions >= 50 and
            current_performance.overall_success_rate >= 0.3):
            new_phase = LearningPhase.BALANCED
            transition_needed = True

        elif (self.current_phase == LearningPhase.BALANCED and
              current_performance.total_interactions >= 100 and
              current_performance.overall_success_rate >= 0.6):
            new_phase = LearningPhase.EXPLOITATION
            transition_needed = True

        elif (self.current_phase == LearningPhase.EXPLOITATION and
              current_performance.total_interactions >= 200 and
              current_performance.overall_success_rate >= 0.7):
            new_phase = LearningPhase.REFINEMENT
            transition_needed = True

        if transition_needed:
            self._transition_to_phase(new_phase)

    def _transition_to_phase(self, new_phase: LearningPhase):
        """Transition to new learning phase"""
        old_phase = self.current_phase
        self._configure_bandit_for_phase(new_phase)
        logger.info(f"Transitioned from {old_phase.value} to {new_phase.value} phase")

    def _auto_tune_parameters(self):
        """Automatically tune parameters based on performance"""
        if len(self.interaction_buffer) % 25 != 0:  # Tune every 25 interactions
            return

        # Auto-tune exploration
        self.bandit.tune_exploration_parameter(self.adaptation_targets['target_exploration_rate'])

    def _calculate_adaptation_efficiency(self) -> float:
        """Calculate adaptation efficiency (learning curve)"""
        if len(self.quality_history) < 20:
            return 0.0

        # Compare recent vs early performance
        early_window = list(self.quality_history)[:10]
        recent_window = list(self.quality_history)[-10:]

        early_avg = sum(early_window) / len(early_window)
        recent_avg = sum(recent_window) / len(recent_window)

        improvement = recent_avg - early_avg
        return max(0.0, min(improvement * 2.0, 1.0))  # Normalize to 0-1

    def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency"""
        if not self.interaction_buffer:
            return 0.0

        # Get exploration vs exploitation quality
        exploration_threshold = 0.6  # Confidence threshold for exploration
        bandit_performance = self.bandit.get_strategy_performance()

        exploration_qualities = []
        exploitation_qualities = []

        for interaction in list(self.interaction_buffer)[-100:]:  # Recent 100
            strategy = interaction['strategy']
            quality = interaction['quality_score']

            # Estimate if this was exploration based on strategy performance
            if strategy in bandit_performance:
                trials = bandit_performance[strategy]['total_trials']
                if trials < 10:  # Likely exploration
                    exploration_qualities.append(quality)
                else:  # Likely exploitation
                    exploitation_qualities.append(quality)

        if not exploration_qualities or not exploitation_qualities:
            return 0.5  # Default

        exploration_avg = sum(exploration_qualities) / len(exploration_qualities)
        exploitation_avg = sum(exploitation_qualities) / len(exploitation_qualities)

        # Efficiency is how close exploration quality is to exploitation quality
        efficiency = 1.0 - abs(exploration_avg - exploitation_avg)
        return max(0.0, min(efficiency, 1.0))

    def _calculate_learning_velocity(self) -> float:
        """Calculate learning velocity (rate of improvement)"""
        if len(self.quality_history) < 20:
            return 0.0

        # Calculate slope of quality improvement
        qualities = list(self.quality_history)[-50:]  # Last 50 interactions
        x = np.arange(len(qualities))
        y = np.array(qualities)

        # Simple linear regression for slope
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return max(0.0, min(slope * 100, 1.0))  # Normalize and scale
        else:
            return 0.0

    def _analyze_strategy_evolution(self) -> Dict[str, Any]:
        """Analyze how strategy preferences have evolved"""
        if not self.strategy_performance_history:
            return {'insufficient_data': True}

        evolution = {}
        for strategy, performance_list in self.strategy_performance_history.items():
            if len(performance_list) >= 10:
                early_perf = sum(performance_list[:5]) / 5
                recent_perf = sum(performance_list[-5:]) / 5
                evolution[strategy] = {
                    'early_performance': early_perf,
                    'recent_performance': recent_perf,
                    'improvement': recent_perf - early_perf,
                    'total_uses': len(performance_list)
                }

        return evolution

    def _analyze_context_strategy_patterns(self) -> Dict[str, Any]:
        """Analyze context-strategy performance patterns"""
        # This would analyze which strategies work best for which contexts
        # Simplified implementation for now
        return {'pattern_analysis': 'implemented_in_future_version'}

    def _recommend_learning_phase(self) -> LearningPhase:
        """Recommend optimal learning phase"""
        return self._determine_optimal_phase()

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.quality_history) < 10:
            return {'insufficient_data': True}

        qualities = list(self.quality_history)
        recent_trend = 'stable'

        if len(qualities) >= 20:
            recent_10 = qualities[-10:]
            previous_10 = qualities[-20:-10]

            recent_avg = sum(recent_10) / len(recent_10)
            previous_avg = sum(previous_10) / len(previous_10)

            if recent_avg > previous_avg + 0.05:
                recent_trend = 'improving'
            elif recent_avg < previous_avg - 0.05:
                recent_trend = 'declining'

        return {
            'recent_trend': recent_trend,
            'average_quality': sum(qualities) / len(qualities),
            'quality_variance': np.var(qualities)
        }

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for optimization"""
        opportunities = []

        metrics = self.get_performance_metrics()

        # Low success rate opportunity
        if metrics.overall_success_rate < 0.6:
            opportunities.append({
                'type': 'success_rate_improvement',
                'description': 'Success rate below target',
                'current_value': metrics.overall_success_rate,
                'target_value': 0.6,
                'suggested_action': 'increase_exploration_or_adjust_quality_threshold'
            })

        # Low strategy diversity opportunity
        if metrics.strategy_diversity < 0.5:
            opportunities.append({
                'type': 'strategy_diversity',
                'description': 'Low strategy diversity',
                'current_value': metrics.strategy_diversity,
                'target_value': 0.7,
                'suggested_action': 'increase_exploration_parameter'
            })

        return opportunities

    def _get_best_strategies(self) -> Dict[str, Any]:
        """Get information about best performing strategies"""
        performance = self.bandit.get_strategy_performance()
        best_strategies = {}

        for strategy, stats in performance.items():
            if stats['total_trials'] >= 5 and stats['mean_reward'] > 0.6:
                best_strategies[strategy] = {
                    'mean_reward': stats['mean_reward'],
                    'total_trials': stats['total_trials'],
                    'theta': self.bandit.theta[PromptingStrategy(strategy)].tolist()
                }

        return best_strategies

    def _restore_best_strategies(self, best_strategies: Dict[str, Any]):
        """Restore best strategies with reduced confidence"""
        for strategy_name, info in best_strategies.items():
            try:
                strategy = PromptingStrategy(strategy_name)
                # Restore with reduced confidence (multiply by 0.5)
                self.bandit.theta[strategy] = np.array(info['theta']) * 0.5
                logger.debug(f"Restored strategy {strategy_name} with reduced confidence")
            except ValueError:
                logger.warning(f"Could not restore strategy {strategy_name}")

    def _init_database(self):
        """Initialize database for learning persistence"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time REAL,
                        end_time REAL,
                        total_interactions INTEGER,
                        successful_interactions INTEGER,
                        average_quality REAL,
                        learning_phase TEXT,
                        strategies_explored TEXT,
                        best_strategy TEXT,
                        improvement_rate REAL
                    )
                ''')
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")

    def _save_learning_session(self, session: LearningSession):
        """Save learning session to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO learning_sessions
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    session.start_time,
                    session.end_time,
                    session.total_interactions,
                    session.successful_interactions,
                    session.average_quality,
                    session.learning_phase.value,
                    json.dumps(session.strategies_explored),
                    session.best_strategy,
                    session.improvement_rate
                ))
        except Exception as e:
            logger.warning(f"Failed to save learning session: {e}")

    def _load_learning_history(self):
        """Load learning history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM learning_sessions ORDER BY start_time DESC LIMIT 10')
                for row in cursor:
                    session = LearningSession(
                        session_id=row[0],
                        start_time=row[1],
                        end_time=row[2],
                        total_interactions=row[3],
                        successful_interactions=row[4],
                        average_quality=row[5],
                        learning_phase=LearningPhase(row[6]),
                        strategies_explored=json.loads(row[7]),
                        best_strategy=row[8],
                        improvement_rate=row[9]
                    )
                    self.session_history.append(session)

                logger.info(f"Loaded {len(self.session_history)} learning sessions")
        except Exception as e:
            logger.warning(f"Failed to load learning history: {e}")

def create_learning_framework(bandit: LinUCBBandit, model_name: str) -> ContinuousLearningFramework:
    """Factory function to create continuous learning framework"""
    return ContinuousLearningFramework(bandit, model_name)