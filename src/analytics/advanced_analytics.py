#!/usr/bin/env python3
"""
Advanced Analytics and Visualization (Sprint 4.0)

Enterprise-grade analytics system for AI benchmark results with:
- Interactive visualizations and dashboards
- Statistical analysis and significance testing
- Cross-model and cross-language comparative analysis
- Performance trend analysis and forecasting
- Publication-ready report generation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class AnalyticsConfig:
    """Configuration for analytics system"""
    results_dir: Path
    output_dir: Path
    confidence_level: float = 0.95
    min_samples: int = 5
    enable_statistical_testing: bool = True
    enable_trend_analysis: bool = True
    chart_theme: str = "plotly_white"
    export_formats: List[str] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "png", "pdf"]


@dataclass
class StatisticalTest:
    """Results of statistical significance testing"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    pass_at_k: Dict[str, float]
    execution_time_stats: Dict[str, float]


class AdvancedAnalytics:
    """
    Advanced analytics system for AI benchmark evaluation results.

    Provides comprehensive statistical analysis, visualization, and reporting
    capabilities for enterprise AI model evaluation workflows.
    """

    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize plotting settings
        self._setup_plotting()

        # Data cache
        self._results_cache = {}
        self._analysis_cache = {}

    def _setup_plotting(self):
        """Setup plotting configuration"""
        # Set plotly theme
        import plotly.io as pio
        pio.templates.default = self.config.chart_theme

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def load_evaluation_results(self, pattern: str = "*.json") -> pd.DataFrame:
        """Load and consolidate evaluation results from files"""
        results_files = list(self.config.results_dir.glob(pattern))

        if not results_files:
            self.logger.warning(f"No results files found matching pattern: {pattern}")
            return pd.DataFrame()

        all_results = []

        for file_path in results_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract individual results
                if 'results' in data:
                    for result in data['results']:
                        result['file_source'] = file_path.name
                        result['suite_name'] = data.get('suite_name', 'unknown')
                        result['timestamp'] = data.get('timestamp', file_path.stat().st_mtime)
                        all_results.append(result)

            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")

        if not all_results:
            self.logger.warning("No valid results found in files")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)

        # Data cleaning and preprocessing
        df = self._preprocess_results(df)

        self.logger.info(f"Loaded {len(df)} evaluation results from {len(results_files)} files")
        return df

    def _preprocess_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean evaluation results"""
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Extract metrics
        if 'metrics' in df.columns:
            # Expand metrics into separate columns
            metrics_df = pd.json_normalize(df['metrics'])
            df = pd.concat([df, metrics_df], axis=1)

        # Clean up data types
        numeric_columns = ['score', 'execution_time']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill missing values
        df['score'] = df['score'].fillna(0.0)
        df['execution_time'] = df['execution_time'].fillna(0.0)

        # Add derived columns
        df['success'] = df['score'] > 0
        df['language'] = df['task'].apply(self._extract_language_from_task)

        return df

    def _extract_language_from_task(self, task: str) -> str:
        """Extract programming language from task name"""
        language_map = {
            'humaneval': 'python',
            'multiple-js': 'javascript',
            'multiple-java': 'java',
            'multiple-cpp': 'cpp',
            'multiple-go': 'go',
            'multiple-rs': 'rust',
            'multiple-ts': 'typescript'
        }
        return language_map.get(task, 'unknown')

    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        self.logger.info("Generating comprehensive analytics report")

        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_evaluations': len(df),
                'date_range': self._get_date_range(df),
                'models_analyzed': df['model'].nunique() if 'model' in df.columns else 0,
                'tasks_analyzed': df['task'].nunique() if 'task' in df.columns else 0
            },
            'summary_statistics': self._calculate_summary_statistics(df),
            'model_comparison': self._analyze_model_performance(df),
            'language_analysis': self._analyze_language_performance(df),
            'temporal_analysis': self._analyze_temporal_trends(df),
            'statistical_tests': self._perform_statistical_tests(df),
            'performance_insights': self._generate_performance_insights(df)
        }

        # Save report
        report_path = self.config.output_dir / f"comprehensive_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Report saved to {report_path}")
        return report

    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get date range of the data"""
        if 'timestamp' in df.columns and not df['timestamp'].empty:
            return {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        return {'start': 'unknown', 'end': 'unknown'}

    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall summary statistics"""
        if df.empty:
            return {}

        stats = {}

        # Score statistics
        if 'score' in df.columns:
            scores = df['score'].dropna()
            stats['score'] = {
                'mean': float(scores.mean()),
                'median': float(scores.median()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'q25': float(scores.quantile(0.25)),
                'q75': float(scores.quantile(0.75))
            }

        # Success rate
        if 'success' in df.columns:
            stats['success_rate'] = float(df['success'].mean())

        # Execution time statistics
        if 'execution_time' in df.columns:
            times = df['execution_time'].dropna()
            stats['execution_time'] = {
                'mean': float(times.mean()),
                'median': float(times.median()),
                'std': float(times.std())
            }

        # Pass@K statistics
        pass_at_k_cols = [col for col in df.columns if col.startswith('pass@')]
        if pass_at_k_cols:
            stats['pass_at_k'] = {}
            for col in pass_at_k_cols:
                values = df[col].dropna()
                if not values.empty:
                    stats['pass_at_k'][col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }

        return stats

    def _analyze_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across different models"""
        if 'model' not in df.columns or df.empty:
            return {}

        analysis = {}

        # Group by model
        model_groups = df.groupby('model')

        # Performance metrics by model
        model_stats = {}
        for model, group in model_groups:
            if 'score' in group.columns:
                scores = group['score'].dropna()
                model_stats[model] = {
                    'sample_size': len(group),
                    'mean_score': float(scores.mean()) if not scores.empty else 0.0,
                    'std_score': float(scores.std()) if not scores.empty else 0.0,
                    'success_rate': float(group['success'].mean()) if 'success' in group.columns else 0.0
                }

                # Pass@K metrics
                pass_at_k_cols = [col for col in group.columns if col.startswith('pass@')]
                if pass_at_k_cols:
                    model_stats[model]['pass_at_k'] = {}
                    for col in pass_at_k_cols:
                        values = group[col].dropna()
                        if not values.empty:
                            model_stats[model]['pass_at_k'][col] = float(values.mean())

        analysis['model_statistics'] = model_stats

        # Model ranking
        if model_stats:
            ranked_models = sorted(
                model_stats.items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            analysis['model_ranking'] = [
                {'model': model, 'mean_score': stats['mean_score']}
                for model, stats in ranked_models
            ]

        return analysis

    def _analyze_language_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across programming languages"""
        if 'language' not in df.columns or df.empty:
            return {}

        analysis = {}

        # Group by language
        language_groups = df.groupby('language')

        # Performance metrics by language
        language_stats = {}
        for language, group in language_groups:
            if 'score' in group.columns:
                scores = group['score'].dropna()
                language_stats[language] = {
                    'sample_size': len(group),
                    'mean_score': float(scores.mean()) if not scores.empty else 0.0,
                    'std_score': float(scores.std()) if not scores.empty else 0.0,
                    'success_rate': float(group['success'].mean()) if 'success' in group.columns else 0.0
                }

                # Model diversity in language
                language_stats[language]['models_tested'] = group['model'].nunique() if 'model' in group.columns else 0

        analysis['language_statistics'] = language_stats

        # Language difficulty ranking (lower mean score = harder)
        if language_stats:
            difficulty_ranking = sorted(
                language_stats.items(),
                key=lambda x: x[1]['mean_score']
            )
            analysis['difficulty_ranking'] = [
                {'language': lang, 'mean_score': stats['mean_score'], 'difficulty': 'hard' if stats['mean_score'] < 0.5 else 'easy'}
                for lang, stats in difficulty_ranking
            ]

        return analysis

    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal trends in performance"""
        if 'timestamp' not in df.columns or df.empty:
            return {}

        analysis = {}

        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')

        # Weekly aggregations
        df_sorted['week'] = df_sorted['timestamp'].dt.to_period('W')
        weekly_stats = df_sorted.groupby('week').agg({
            'score': ['mean', 'std', 'count'],
            'execution_time': ['mean', 'std']
        }).round(3)

        analysis['weekly_trends'] = weekly_stats.to_dict()

        # Trend detection using linear regression
        if len(df_sorted) >= self.config.min_samples:
            # Convert timestamp to numeric for regression
            df_sorted['timestamp_numeric'] = df_sorted['timestamp'].astype(np.int64) // 10**9

            if 'score' in df_sorted.columns:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_sorted['timestamp_numeric'], df_sorted['score']
                )

                analysis['score_trend'] = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significant': p_value < (1 - self.config.confidence_level),
                    'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
                }

        return analysis

    def _perform_statistical_tests(self, df: pd.DataFrame) -> List[StatisticalTest]:
        """Perform statistical significance tests"""
        if not self.config.enable_statistical_testing or df.empty:
            return []

        tests = []

        # Model comparison tests
        if 'model' in df.columns and 'score' in df.columns:
            models = df['model'].unique()
            if len(models) >= 2:
                # Kruskal-Wallis test for multiple model comparison
                model_scores = [df[df['model'] == model]['score'].dropna() for model in models]
                if all(len(scores) >= self.config.min_samples for scores in model_scores):
                    try:
                        statistic, p_value = kruskal(*model_scores)
                        tests.append(StatisticalTest(
                            test_name="Kruskal-Wallis (Model Comparison)",
                            statistic=float(statistic),
                            p_value=float(p_value),
                            significant=p_value < (1 - self.config.confidence_level),
                            interpretation="Significant differences between model performances" if p_value < 0.05 else "No significant differences between models"
                        ))
                    except Exception as e:
                        self.logger.warning(f"Kruskal-Wallis test failed: {e}")

        # Language comparison tests
        if 'language' in df.columns and 'score' in df.columns:
            languages = df['language'].unique()
            if len(languages) >= 2:
                # Kruskal-Wallis test for multiple language comparison
                language_scores = [df[df['language'] == lang]['score'].dropna() for lang in languages]
                if all(len(scores) >= self.config.min_samples for scores in language_scores):
                    try:
                        statistic, p_value = kruskal(*language_scores)
                        tests.append(StatisticalTest(
                            test_name="Kruskal-Wallis (Language Comparison)",
                            statistic=float(statistic),
                            p_value=float(p_value),
                            significant=p_value < (1 - self.config.confidence_level),
                            interpretation="Significant differences between language difficulties" if p_value < 0.05 else "No significant differences between languages"
                        ))
                    except Exception as e:
                        self.logger.warning(f"Language comparison test failed: {e}")

        return tests

    def _generate_performance_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable performance insights"""
        insights = []

        if df.empty:
            return ["No data available for insights generation"]

        # Overall performance insights
        if 'score' in df.columns:
            mean_score = df['score'].mean()
            if mean_score > 0.8:
                insights.append("🟢 Excellent overall performance - models are performing very well")
            elif mean_score > 0.6:
                insights.append("🟡 Good overall performance with room for improvement")
            else:
                insights.append("🔴 Low overall performance - significant optimization needed")

        # Model performance insights
        if 'model' in df.columns and 'score' in df.columns:
            model_performance = df.groupby('model')['score'].mean().sort_values(ascending=False)
            if len(model_performance) > 1:
                best_model = model_performance.index[0]
                worst_model = model_performance.index[-1]
                performance_gap = model_performance.iloc[0] - model_performance.iloc[-1]

                insights.append(f"🏆 Best performing model: {best_model} ({model_performance.iloc[0]:.3f})")
                insights.append(f"📉 Lowest performing model: {worst_model} ({model_performance.iloc[-1]:.3f})")

                if performance_gap > 0.3:
                    insights.append("⚠️ Large performance gap between models - consider model selection optimization")

        # Language difficulty insights
        if 'language' in df.columns and 'score' in df.columns:
            language_difficulty = df.groupby('language')['score'].mean().sort_values()
            if len(language_difficulty) > 1:
                hardest_lang = language_difficulty.index[0]
                easiest_lang = language_difficulty.index[-1]

                insights.append(f"🔴 Most challenging language: {hardest_lang} ({language_difficulty.iloc[0]:.3f})")
                insights.append(f"🟢 Easiest language: {easiest_lang} ({language_difficulty.iloc[-1]:.3f})")

        # Pass@K insights
        pass_at_k_cols = [col for col in df.columns if col.startswith('pass@')]
        if pass_at_k_cols and len(pass_at_k_cols) > 1:
            # Compare pass@1 vs pass@10 improvement
            if 'pass@1' in df.columns and 'pass@10' in df.columns:
                pass1_mean = df['pass@1'].mean()
                pass10_mean = df['pass@10'].mean()
                improvement = (pass10_mean - pass1_mean) / pass1_mean if pass1_mean > 0 else 0

                if improvement > 0.5:
                    insights.append("🎯 High Pass@K improvement - models benefit significantly from multiple attempts")
                elif improvement > 0.2:
                    insights.append("📈 Moderate Pass@K improvement - multiple attempts provide some benefit")
                else:
                    insights.append("📊 Low Pass@K improvement - models are consistent but may lack diversity")

        # Execution time insights
        if 'execution_time' in df.columns:
            mean_time = df['execution_time'].mean()
            if mean_time > 60:
                insights.append("⏱️ High execution times - consider performance optimization")
            elif mean_time < 5:
                insights.append("⚡ Excellent execution performance - very fast evaluations")

        return insights

    def create_interactive_dashboard(self, df: pd.DataFrame) -> str:
        """Create interactive HTML dashboard"""
        if df.empty:
            return ""

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Model Performance Comparison",
                "Language Difficulty Analysis",
                "Score Distribution",
                "Pass@K Metrics",
                "Execution Time vs Score",
                "Performance Over Time"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )

        # Model performance comparison
        if 'model' in df.columns and 'score' in df.columns:
            model_stats = df.groupby('model')['score'].agg(['mean', 'std']).reset_index()
            fig.add_trace(
                go.Bar(
                    x=model_stats['model'],
                    y=model_stats['mean'],
                    error_y=dict(type='data', array=model_stats['std']),
                    name="Model Performance",
                    showlegend=False
                ),
                row=1, col=1
            )

        # Language difficulty analysis
        if 'language' in df.columns and 'score' in df.columns:
            lang_stats = df.groupby('language')['score'].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=lang_stats['language'],
                    y=lang_stats['score'],
                    name="Language Difficulty",
                    showlegend=False
                ),
                row=1, col=2
            )

        # Score distribution
        if 'score' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['score'],
                    nbinsx=20,
                    name="Score Distribution",
                    showlegend=False
                ),
                row=2, col=1
            )

        # Pass@K metrics
        pass_at_k_cols = [col for col in df.columns if col.startswith('pass@')]
        if pass_at_k_cols:
            for col in pass_at_k_cols[:3]:  # Show first 3 Pass@K metrics
                values = df[col].dropna()
                if not values.empty:
                    fig.add_trace(
                        go.Bar(
                            x=[col],
                            y=[values.mean()],
                            name=col,
                            showlegend=True
                        ),
                        row=2, col=2
                    )

        # Execution time vs score scatter plot
        if 'execution_time' in df.columns and 'score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['execution_time'],
                    y=df['score'],
                    mode='markers',
                    text=df['model'] if 'model' in df.columns else None,
                    name="Time vs Score",
                    showlegend=False
                ),
                row=3, col=1
            )

        # Performance over time
        if 'timestamp' in df.columns and 'score' in df.columns:
            df_sorted = df.sort_values('timestamp')
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['score'],
                    mode='markers+lines',
                    name="Performance Trend",
                    showlegend=False
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text="AI Benchmark Suite - Advanced Analytics Dashboard",
            showlegend=True
        )

        # Save interactive dashboard
        dashboard_path = self.config.output_dir / f"interactive_dashboard_{int(time.time())}.html"
        fig.write_html(str(dashboard_path))

        self.logger.info(f"Interactive dashboard saved to {dashboard_path}")
        return str(dashboard_path)

    def export_publication_ready_plots(self, df: pd.DataFrame) -> List[str]:
        """Export publication-ready static plots"""
        if df.empty:
            return []

        exported_files = []

        # Model comparison plot
        if 'model' in df.columns and 'score' in df.columns:
            plt.figure(figsize=(12, 8))
            model_stats = df.groupby('model')['score'].agg(['mean', 'std']).reset_index()

            plt.bar(model_stats['model'], model_stats['mean'], yerr=model_stats['std'], capsize=5)
            plt.xlabel('Model')
            plt.ylabel('Mean Score')
            plt.title('Model Performance Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()

            filename = self.config.output_dir / f"model_comparison_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            exported_files.append(str(filename))
            plt.close()

        # Language difficulty heatmap
        if 'model' in df.columns and 'language' in df.columns and 'score' in df.columns:
            pivot_table = df.pivot_table(values='score', index='model', columns='language', aggfunc='mean')

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f')
            plt.title('Model Performance Across Languages')
            plt.tight_layout()

            filename = self.config.output_dir / f"language_heatmap_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            exported_files.append(str(filename))
            plt.close()

        self.logger.info(f"Exported {len(exported_files)} publication-ready plots")
        return exported_files


# Factory function
def create_analytics_system(results_dir: Path, output_dir: Path, **kwargs) -> AdvancedAnalytics:
    """Factory function to create analytics system with optimal settings"""
    config = AnalyticsConfig(
        results_dir=results_dir,
        output_dir=output_dir,
        **kwargs
    )
    return AdvancedAnalytics(config)


# Testing and demonstration
if __name__ == "__main__":
    print("📊 Advanced Analytics Demo")
    print("=" * 50)

    # Create demo data
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        output_dir = Path(temp_dir) / "analytics"
        results_dir.mkdir()

        # Create sample data
        sample_data = {
            "suite_name": "demo_suite",
            "timestamp": time.time(),
            "results": [
                {"model": "qwen-coder", "task": "humaneval", "score": 0.85, "execution_time": 15.2, "metrics": {"pass@1": 0.85, "pass@10": 0.92}},
                {"model": "codellama", "task": "humaneval", "score": 0.78, "execution_time": 18.5, "metrics": {"pass@1": 0.78, "pass@10": 0.88}},
                {"model": "qwen-coder", "task": "multiple-js", "score": 0.72, "execution_time": 12.8, "metrics": {"pass@1": 0.72, "pass@10": 0.85}},
                {"model": "codellama", "task": "multiple-js", "score": 0.68, "execution_time": 16.2, "metrics": {"pass@1": 0.68, "pass@10": 0.82}},
            ]
        }

        # Save sample data
        with open(results_dir / "demo_results.json", 'w') as f:
            json.dump(sample_data, f)

        # Create analytics system
        analytics = create_analytics_system(results_dir, output_dir)

        # Load and analyze data
        df = analytics.load_evaluation_results()
        print(f"Loaded {len(df)} evaluation results")

        # Generate comprehensive report
        report = analytics.generate_comprehensive_report(df)
        print("Generated comprehensive analytics report")

        # Create interactive dashboard
        dashboard_path = analytics.create_interactive_dashboard(df)
        print(f"Created interactive dashboard: {dashboard_path}")

        # Export publication plots
        plots = analytics.export_publication_ready_plots(df)
        print(f"Exported {len(plots)} publication-ready plots")

        print("\n✅ Advanced analytics demo completed!")
        print("📊 Generated comprehensive analytics with statistical testing and visualization")