#!/usr/bin/env python3
"""
Enterprise Monitoring Dashboard (Sprint 4.0)

Real-time monitoring dashboard for AI Benchmark Suite with:
- Live system metrics and performance monitoring
- Interactive evaluation tracking and results visualization
- Resource usage monitoring and alerting
- Cache performance and optimization insights
- WebSocket-based real-time updates
"""

import asyncio
import json
import time
import psutil
import redis
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Streamlit for dashboard
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Import our modules
try:
    from model_interfaces.performance_benchmarker import PerformanceBenchmarker
    from model_interfaces.result_cache_manager import ResultCacheManager
    from model_interfaces.memory_optimizer import MemoryOptimizer
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")


class MonitoringDashboard:
    """
    Enterprise monitoring dashboard for real-time system insights.
    """

    def __init__(self):
        self.redis_client = None
        self.cache_manager = None
        self.memory_optimizer = None
        self.update_interval = 5  # seconds

        # Initialize connections
        self._init_connections()

        # Dashboard state
        self.metrics_history = []
        self.max_history_points = 100

    def _init_connections(self):
        """Initialize connections to Redis and other services"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            st.error(f"Failed to connect to Redis: {e}")

        try:
            cache_dir = PROJECT_ROOT / "cache"
            self.cache_manager = ResultCacheManager(cache_dir)
        except Exception as e:
            st.error(f"Failed to initialize cache manager: {e}")

        try:
            self.memory_optimizer = MemoryOptimizer()
        except Exception as e:
            st.error(f"Failed to initialize memory optimizer: {e}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Network stats
        net_io = psutil.net_io_counters()

        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'process_memory_mb': process_memory,
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
        }

        # Add cache metrics if available
        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_cache_stats()
                metrics.update({
                    'cache_hit_rate': cache_stats.get('hit_rate', 0),
                    'cache_size_mb': cache_stats.get('cache_size_mb', 0),
                    'cache_entries': cache_stats.get('total_cached_results', 0)
                })
            except Exception:
                pass

        # Add Redis metrics if available
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                metrics.update({
                    'redis_used_memory_mb': redis_info.get('used_memory', 0) / (1024**2),
                    'redis_connected_clients': redis_info.get('connected_clients', 0),
                    'redis_total_commands': redis_info.get('total_commands_processed', 0)
                })
            except Exception:
                pass

        return metrics

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get evaluation-specific metrics"""
        metrics = {
            'active_evaluations': 0,
            'completed_evaluations': 0,
            'failed_evaluations': 0,
            'total_evaluation_time': 0,
            'average_evaluation_time': 0,
            'recent_evaluations': []
        }

        if self.redis_client:
            try:
                # Get active evaluations
                active_keys = self.redis_client.keys("active:*")
                metrics['active_evaluations'] = len(active_keys)

                # Get completed evaluations
                completed_keys = self.redis_client.keys("completed:*")
                metrics['completed_evaluations'] = len(completed_keys)

                # Get recent evaluations
                recent_evaluations = []
                for key in completed_keys[-10:]:  # Last 10 completed
                    try:
                        eval_data = json.loads(self.redis_client.get(key))
                        recent_evaluations.append(eval_data)
                    except:
                        pass

                metrics['recent_evaluations'] = recent_evaluations

                # Calculate average evaluation time
                eval_times = [e.get('execution_time', 0) for e in recent_evaluations if e.get('execution_time')]
                if eval_times:
                    metrics['average_evaluation_time'] = sum(eval_times) / len(eval_times)
                    metrics['total_evaluation_time'] = sum(eval_times)

            except Exception as e:
                st.error(f"Error fetching evaluation metrics: {e}")

        return metrics

    def render_dashboard(self):
        """Render the main dashboard"""
        st.set_page_config(
            page_title="AI Benchmark Suite - Enterprise Dashboard",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚀 AI Benchmark Suite - Enterprise Dashboard")
        st.subheader("Sprint 4.0 Real-Time Monitoring")

        # Sidebar configuration
        self._render_sidebar()

        # Auto-refresh
        if st.sidebar.checkbox("Auto-refresh (5s)", value=True):
            time.sleep(5)
            st.experimental_rerun()

        # Get current metrics
        system_metrics = self.get_system_metrics()
        evaluation_metrics = self.get_evaluation_metrics()

        # Update metrics history
        self.metrics_history.append(system_metrics)
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history.pop(0)

        # Render main dashboard sections
        self._render_overview_cards(system_metrics, evaluation_metrics)
        self._render_system_metrics_charts()
        self._render_evaluation_tracking(evaluation_metrics)
        self._render_performance_insights()
        self._render_cache_analytics()

    def _render_sidebar(self):
        """Render sidebar with controls and information"""
        st.sidebar.header("Dashboard Controls")

        # System information
        st.sidebar.subheader("System Info")
        st.sidebar.text(f"Uptime: {time.time() - psutil.boot_time():.0f}s")
        st.sidebar.text(f"Python: {sys.version[:5]}")
        st.sidebar.text(f"Platform: {sys.platform}")

        # Connection status
        st.sidebar.subheader("Service Status")
        redis_status = "🟢 Connected" if self.redis_client else "🔴 Disconnected"
        st.sidebar.text(f"Redis: {redis_status}")

        cache_status = "🟢 Available" if self.cache_manager else "🔴 Unavailable"
        st.sidebar.text(f"Cache: {cache_status}")

        # Dashboard settings
        st.sidebar.subheader("Settings")
        self.update_interval = st.sidebar.slider("Update Interval (s)", 1, 30, 5)
        self.max_history_points = st.sidebar.slider("History Points", 50, 500, 100)

    def _render_overview_cards(self, system_metrics: Dict, evaluation_metrics: Dict):
        """Render overview cards with key metrics"""
        st.subheader("📊 System Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "CPU Usage",
                f"{system_metrics['cpu_percent']:.1f}%",
                delta=None
            )

        with col2:
            st.metric(
                "Memory Usage",
                f"{system_metrics['memory_percent']:.1f}%",
                delta=None
            )

        with col3:
            st.metric(
                "Active Evaluations",
                evaluation_metrics['active_evaluations'],
                delta=None
            )

        with col4:
            cache_hit_rate = system_metrics.get('cache_hit_rate', 0)
            st.metric(
                "Cache Hit Rate",
                f"{cache_hit_rate:.1%}",
                delta=None
            )

    def _render_system_metrics_charts(self):
        """Render system metrics charts"""
        if not self.metrics_history:
            return

        st.subheader("📈 System Metrics")

        # Create DataFrame from metrics history
        df = pd.DataFrame(self.metrics_history)

        # System usage over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CPU Usage", "Memory Usage", "Disk Usage", "Process Memory"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # CPU Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name="CPU %", line=dict(color='#FF6B6B')),
            row=1, col=1
        )

        # Memory Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_percent'], name="Memory %", line=dict(color='#4ECDC4')),
            row=1, col=2
        )

        # Disk Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk_percent'], name="Disk %", line=dict(color='#45B7D1')),
            row=2, col=1
        )

        # Process Memory
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['process_memory_mb'], name="Process MB", line=dict(color='#96CEB4')),
            row=2, col=2
        )

        fig.update_layout(height=500, showlegend=False, title_text="System Resource Usage")
        st.plotly_chart(fig, use_container_width=True)

    def _render_evaluation_tracking(self, evaluation_metrics: Dict):
        """Render evaluation tracking and results"""
        st.subheader("🎯 Evaluation Tracking")

        col1, col2 = st.columns(2)

        with col1:
            # Evaluation status pie chart
            labels = ['Active', 'Completed', 'Failed']
            values = [
                evaluation_metrics['active_evaluations'],
                evaluation_metrics['completed_evaluations'],
                evaluation_metrics['failed_evaluations']
            ]

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(title="Evaluation Status Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Recent evaluations table
            st.write("**Recent Evaluations**")
            recent_evals = evaluation_metrics.get('recent_evaluations', [])

            if recent_evals:
                eval_df = pd.DataFrame([
                    {
                        'Task': e.get('task', 'Unknown'),
                        'Model': e.get('model', 'Unknown'),
                        'Status': e.get('status', 'Unknown'),
                        'Score': e.get('result', {}).get('score', 0) if e.get('result') else 0,
                        'Time (s)': e.get('result', {}).get('execution_time', 0) if e.get('result') else 0
                    }
                    for e in recent_evals[-5:]  # Last 5 evaluations
                ])
                st.dataframe(eval_df, use_container_width=True)
            else:
                st.info("No recent evaluations found")

    def _render_performance_insights(self):
        """Render performance insights and optimization recommendations"""
        st.subheader("⚡ Performance Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Sprint 3.0 Optimizations Active**")
            optimizations = [
                "🔄 Parallel Container Execution",
                "💾 Intelligent Result Caching",
                "🧠 Memory Optimization",
                "📊 Performance Benchmarking",
                "🎯 Batch Processing"
            ]
            for opt in optimizations:
                st.success(opt)

        with col2:
            st.write("**Performance Recommendations**")

            # Dynamic recommendations based on current metrics
            recommendations = []
            if len(self.metrics_history) > 10:
                recent_cpu = [m['cpu_percent'] for m in self.metrics_history[-10:]]
                recent_memory = [m['memory_percent'] for m in self.metrics_history[-10:]]

                avg_cpu = sum(recent_cpu) / len(recent_cpu)
                avg_memory = sum(recent_memory) / len(recent_memory)

                if avg_cpu > 80:
                    recommendations.append("🔴 High CPU usage - consider scaling horizontally")
                elif avg_cpu < 20:
                    recommendations.append("🟢 Low CPU usage - system is well optimized")

                if avg_memory > 85:
                    recommendations.append("🔴 High memory usage - enable memory optimization")
                elif avg_memory < 30:
                    recommendations.append("🟢 Efficient memory usage")

                cache_hit_rate = self.metrics_history[-1].get('cache_hit_rate', 0)
                if cache_hit_rate > 0.7:
                    recommendations.append("🟢 Excellent cache performance")
                elif cache_hit_rate < 0.3:
                    recommendations.append("🟡 Low cache hit rate - consider cache tuning")

            if not recommendations:
                recommendations = ["✅ System performing optimally"]

            for rec in recommendations:
                if rec.startswith("🔴"):
                    st.error(rec[2:])
                elif rec.startswith("🟡"):
                    st.warning(rec[2:])
                else:
                    st.success(rec[2:])

    def _render_cache_analytics(self):
        """Render cache performance analytics"""
        st.subheader("💾 Cache Analytics")

        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_cache_stats()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Cache Size",
                        f"{cache_stats.get('cache_size_mb', 0):.1f} MB"
                    )

                with col2:
                    st.metric(
                        "Cache Entries",
                        cache_stats.get('total_cached_results', 0)
                    )

                with col3:
                    st.metric(
                        "Hit Rate",
                        f"{cache_stats.get('hit_rate', 0):.1%}"
                    )

                # Language breakdown
                lang_breakdown = cache_stats.get('language_breakdown', {})
                if lang_breakdown:
                    st.write("**Cache Usage by Language**")
                    lang_df = pd.DataFrame([
                        {
                            'Language': lang.upper(),
                            'Cached Results': stats['cached_results'],
                            'Total Accesses': stats['total_accesses'],
                            'Avg Time (s)': stats['avg_execution_time']
                        }
                        for lang, stats in lang_breakdown.items()
                    ])
                    st.dataframe(lang_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading cache analytics: {e}")
        else:
            st.info("Cache manager not available")


def main():
    """Main dashboard application"""
    dashboard = MonitoringDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()