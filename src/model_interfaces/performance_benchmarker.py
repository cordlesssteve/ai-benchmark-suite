#!/usr/bin/env python3
"""
Performance Benchmarker (Sprint 3.0)

Comprehensive performance measurement and benchmarking tools for validating
Sprint 3.0 optimizations and measuring actual performance improvements.
"""

import time
import psutil
import statistics
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import logging
from datetime import datetime
import threading
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .optimized_unified_runner import OptimizedUnifiedRunner, OptimizationConfig
    from .parallel_execution_manager import ExecutionStrategy
    from .result_cache_manager import CacheStrategy
    from .memory_optimizer import MemoryStrategy
except ImportError:
    # For standalone testing
    from optimized_unified_runner import OptimizedUnifiedRunner, OptimizationConfig
    from parallel_execution_manager import ExecutionStrategy
    from result_cache_manager import CacheStrategy
    from memory_optimizer import MemoryStrategy

from unified_runner import UnifiedRunner


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    total_execution_time: float
    average_task_time: float
    median_task_time: float
    min_task_time: float
    max_task_time: float
    successful_tasks: int
    failed_tasks: int
    success_rate: float
    throughput_tasks_per_second: float

    # Memory metrics
    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency_score: float

    # Cache metrics
    cache_hit_rate: float
    cache_miss_rate: float
    time_saved_by_cache: float

    # Parallel execution metrics
    parallel_efficiency: float
    container_utilization: float

    # System resource metrics
    cpu_usage_percent: float
    disk_io_mb: float
    network_io_mb: Optional[float] = None


@dataclass
class BenchmarkComparison:
    """Comparison between baseline and optimized performance"""
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_factor: float
    time_reduction_percent: float
    memory_reduction_percent: float
    throughput_improvement_percent: float
    optimization_effectiveness: Dict[str, float]


class BenchmarkType(Enum):
    """Types of performance benchmarks"""
    MICRO = "micro"          # Single task, single model
    SUITE = "suite"          # Full benchmark suite
    STRESS = "stress"        # Large-scale stress test
    MEMORY = "memory"        # Memory-focused benchmark
    PARALLEL = "parallel"    # Parallel execution benchmark


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking system for AI evaluation suite.

    Measures and validates Sprint 3.0 performance improvements across multiple dimensions:
    - Execution time improvements
    - Memory usage optimization
    - Cache effectiveness
    - Parallel execution efficiency
    - Resource utilization
    """

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.current_metrics: Dict[str, Any] = {}
        self.baseline_results: Dict[str, PerformanceMetrics] = {}
        self.optimized_results: Dict[str, PerformanceMetrics] = {}

        # Resource monitoring
        self.process = psutil.Process()
        self.monitoring_data: List[Dict[str, float]] = []
        self.monitoring_active = False
        self._monitor_thread = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def performance_measurement(self, benchmark_name: str, benchmark_type: BenchmarkType = BenchmarkType.MICRO):
        """Context manager for measuring performance of operations"""

        self.logger.info(f"🔬 Starting performance measurement: {benchmark_name} ({benchmark_type.value})")

        # Start monitoring
        self._start_monitoring()

        # Initial measurements
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu_time = self.process.cpu_times()

        # Track task-level metrics
        task_times = []
        task_results = []

        try:
            # Provide measurement context
            measurement_context = PerformanceMeasurementContext(
                benchmark_name=benchmark_name,
                task_times=task_times,
                task_results=task_results,
                start_time=start_time
            )

            yield measurement_context

        finally:
            # Final measurements
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_cpu_time = self.process.cpu_times()

            # Stop monitoring
            self._stop_monitoring()

            # Calculate metrics
            metrics = self._calculate_performance_metrics(
                start_time, end_time, start_memory, end_memory,
                start_cpu_time, end_cpu_time, task_times, task_results
            )

            # Store results
            self.current_metrics[benchmark_name] = metrics

            self.logger.info(f"✅ Performance measurement completed: {benchmark_name}")
            self.logger.info(f"   Total time: {metrics.total_execution_time:.2f}s")
            self.logger.info(f"   Success rate: {metrics.success_rate:.1%}")
            self.logger.info(f"   Throughput: {metrics.throughput_tasks_per_second:.2f} tasks/s")

    def benchmark_baseline_vs_optimized(self, suite_name: str, models: List[str],
                                      test_params: Dict[str, Any] = None) -> BenchmarkComparison:
        """
        Compare baseline (original) vs optimized performance.

        This is the key method for validating Sprint 3.0 improvements.
        """

        test_params = test_params or {'limit': 3, 'n_samples': 5}

        self.logger.info(f"🏁 Starting baseline vs optimized benchmark")
        self.logger.info(f"   Suite: {suite_name}")
        self.logger.info(f"   Models: {models}")
        self.logger.info(f"   Test params: {test_params}")

        # Run baseline benchmark
        baseline_metrics = self._run_baseline_benchmark(suite_name, models, test_params)

        # Run optimized benchmark
        optimized_metrics = self._run_optimized_benchmark(suite_name, models, test_params)

        # Calculate comparison
        comparison = self._calculate_benchmark_comparison(baseline_metrics, optimized_metrics)

        # Save results
        self._save_benchmark_comparison(suite_name, comparison)

        return comparison

    def _run_baseline_benchmark(self, suite_name: str, models: List[str],
                               test_params: Dict[str, Any]) -> PerformanceMetrics:
        """Run benchmark with original (baseline) implementation"""

        with self.performance_measurement(f"baseline_{suite_name}", BenchmarkType.SUITE) as ctx:

            # Create original runner
            runner = UnifiedRunner()

            # Track individual task performance
            for model in models:
                for task in self._get_suite_tasks(runner, suite_name):
                    task_start = time.time()

                    try:
                        result = runner.run_benchmark(task, model, **test_params)
                        task_time = time.time() - task_start

                        ctx.task_times.append(task_time)
                        ctx.task_results.append({
                            'success': result.score > 0,
                            'task': task,
                            'model': model,
                            'score': result.score,
                            'execution_time': result.execution_time
                        })

                    except Exception as e:
                        task_time = time.time() - task_start
                        ctx.task_times.append(task_time)
                        ctx.task_results.append({
                            'success': False,
                            'task': task,
                            'model': model,
                            'error': str(e),
                            'score': 0.0,
                            'execution_time': 0.0
                        })

        return self.current_metrics[f"baseline_{suite_name}"]

    def _run_optimized_benchmark(self, suite_name: str, models: List[str],
                                test_params: Dict[str, Any]) -> PerformanceMetrics:
        """Run benchmark with Sprint 3.0 optimizations"""

        with self.performance_measurement(f"optimized_{suite_name}", BenchmarkType.SUITE) as ctx:

            # Create optimized runner with all optimizations enabled
            opt_config = OptimizationConfig(
                enable_parallel_execution=True,
                enable_result_caching=True,
                enable_memory_optimization=True,
                max_parallel_workers=4,
                execution_strategy=ExecutionStrategy.CONCURRENT_LANGUAGES,
                cache_strategy=CacheStrategy.CONSERVATIVE,
                memory_strategy=MemoryStrategy.BALANCED
            )

            runner = OptimizedUnifiedRunner(optimization_config=opt_config)

            # Run optimized suite
            try:
                results = runner.run_suite_optimized(suite_name, models, **test_params)

                # Extract task performance data
                for result in results:
                    ctx.task_results.append({
                        'success': result.score > 0,
                        'task': result.task,
                        'model': result.model,
                        'score': result.score,
                        'execution_time': result.execution_time,
                        'cached': result.metadata.get('cached', False)
                    })

                    # Individual task times are included in execution_time
                    ctx.task_times.append(result.execution_time)

            except Exception as e:
                self.logger.error(f"Optimized benchmark failed: {e}")
                ctx.task_results.append({
                    'success': False,
                    'error': str(e),
                    'score': 0.0,
                    'execution_time': 0.0
                })

            finally:
                runner.cleanup_optimizations()

        return self.current_metrics[f"optimized_{suite_name}"]

    def _get_suite_tasks(self, runner: UnifiedRunner, suite_name: str) -> List[str]:
        """Get tasks for a benchmark suite"""
        suite_def = runner.suite_config.get('suites', {}).get(suite_name, {})
        return suite_def.get('tasks', ['humaneval'])  # Default to humaneval if no suite found

    def _start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring_active = True
        self.monitoring_data.clear()

        def monitor_resources():
            while self.monitoring_active:
                try:
                    memory_info = self.process.memory_info()
                    cpu_percent = self.process.cpu_percent()

                    self.monitoring_data.append({
                        'timestamp': time.time(),
                        'memory_mb': memory_info.rss / 1024 / 1024,
                        'cpu_percent': cpu_percent
                    })

                    time.sleep(0.5)  # Monitor every 500ms

                except Exception as e:
                    self.logger.warning(f"Monitoring error: {e}")

        self._monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self._monitor_thread.start()

    def _stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _calculate_performance_metrics(self, start_time: float, end_time: float,
                                     start_memory: float, end_memory: float,
                                     start_cpu: Any, end_cpu: Any,
                                     task_times: List[float], task_results: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        total_time = end_time - start_time
        successful_tasks = sum(1 for r in task_results if r.get('success', False))
        failed_tasks = len(task_results) - successful_tasks

        # Task timing statistics
        if task_times:
            avg_task_time = statistics.mean(task_times)
            median_task_time = statistics.median(task_times)
            min_task_time = min(task_times)
            max_task_time = max(task_times)
        else:
            avg_task_time = median_task_time = min_task_time = max_task_time = 0.0

        # Memory statistics from monitoring data
        if self.monitoring_data:
            memory_values = [d['memory_mb'] for d in self.monitoring_data]
            peak_memory = max(memory_values)
            avg_memory = statistics.mean(memory_values)
            cpu_values = [d['cpu_percent'] for d in self.monitoring_data if d['cpu_percent'] > 0]
            avg_cpu = statistics.mean(cpu_values) if cpu_values else 0.0
        else:
            peak_memory = max(start_memory, end_memory)
            avg_memory = (start_memory + end_memory) / 2
            avg_cpu = 0.0

        # Cache metrics (if available from task results)
        cached_results = sum(1 for r in task_results if r.get('cached', False))
        cache_hit_rate = cached_results / len(task_results) if task_results else 0.0
        cache_miss_rate = 1.0 - cache_hit_rate

        # Time saved by cache (estimated)
        time_saved_by_cache = sum(
            r.get('execution_time', 0) for r in task_results
            if r.get('cached', False)
        )

        # Calculate throughput
        throughput = len(task_results) / total_time if total_time > 0 else 0.0

        # Success rate
        success_rate = successful_tasks / len(task_results) if task_results else 0.0

        # Memory efficiency (lower is better)
        memory_efficiency = avg_memory / max(successful_tasks, 1)

        # Parallel efficiency (estimated based on CPU utilization)
        parallel_efficiency = min(avg_cpu / 100.0, 1.0) if avg_cpu > 0 else 0.0

        return PerformanceMetrics(
            total_execution_time=total_time,
            average_task_time=avg_task_time,
            median_task_time=median_task_time,
            min_task_time=min_task_time,
            max_task_time=max_task_time,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            success_rate=success_rate,
            throughput_tasks_per_second=throughput,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            memory_efficiency_score=memory_efficiency,
            cache_hit_rate=cache_hit_rate,
            cache_miss_rate=cache_miss_rate,
            time_saved_by_cache=time_saved_by_cache,
            parallel_efficiency=parallel_efficiency,
            container_utilization=0.0,  # Would need container-specific monitoring
            cpu_usage_percent=avg_cpu,
            disk_io_mb=0.0  # Would need disk monitoring
        )

    def _calculate_benchmark_comparison(self, baseline: PerformanceMetrics,
                                      optimized: PerformanceMetrics) -> BenchmarkComparison:
        """Calculate comparison metrics between baseline and optimized runs"""

        # Overall improvement factor
        improvement_factor = baseline.total_execution_time / optimized.total_execution_time if optimized.total_execution_time > 0 else 1.0

        # Time reduction percentage
        time_reduction = ((baseline.total_execution_time - optimized.total_execution_time) / baseline.total_execution_time) * 100

        # Memory reduction percentage
        memory_reduction = ((baseline.peak_memory_mb - optimized.peak_memory_mb) / baseline.peak_memory_mb) * 100 if baseline.peak_memory_mb > 0 else 0.0

        # Throughput improvement
        throughput_improvement = ((optimized.throughput_tasks_per_second - baseline.throughput_tasks_per_second) / baseline.throughput_tasks_per_second) * 100 if baseline.throughput_tasks_per_second > 0 else 0.0

        # Optimization effectiveness breakdown
        optimization_effectiveness = {
            'caching_benefit': optimized.cache_hit_rate * 100,
            'parallel_efficiency': optimized.parallel_efficiency * 100,
            'memory_optimization': memory_reduction,
            'overall_speedup': improvement_factor
        }

        return BenchmarkComparison(
            baseline_metrics=baseline,
            optimized_metrics=optimized,
            improvement_factor=improvement_factor,
            time_reduction_percent=time_reduction,
            memory_reduction_percent=memory_reduction,
            throughput_improvement_percent=throughput_improvement,
            optimization_effectiveness=optimization_effectiveness
        )

    def _save_benchmark_comparison(self, suite_name: str, comparison: BenchmarkComparison):
        """Save benchmark comparison results to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_file = self.results_dir / f"benchmark_comparison_{suite_name}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(comparison), f, indent=2, default=str)

        # Save CSV summary
        csv_file = self.results_dir / f"benchmark_summary_{suite_name}_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Baseline', 'Optimized', 'Improvement'])

            writer.writerow(['Execution Time (s)',
                           f"{comparison.baseline_metrics.total_execution_time:.2f}",
                           f"{comparison.optimized_metrics.total_execution_time:.2f}",
                           f"{comparison.improvement_factor:.2f}x"])

            writer.writerow(['Throughput (tasks/s)',
                           f"{comparison.baseline_metrics.throughput_tasks_per_second:.2f}",
                           f"{comparison.optimized_metrics.throughput_tasks_per_second:.2f}",
                           f"{comparison.throughput_improvement_percent:+.1f}%"])

            writer.writerow(['Peak Memory (MB)',
                           f"{comparison.baseline_metrics.peak_memory_mb:.1f}",
                           f"{comparison.optimized_metrics.peak_memory_mb:.1f}",
                           f"{comparison.memory_reduction_percent:+.1f}%"])

            writer.writerow(['Cache Hit Rate',
                           f"{comparison.baseline_metrics.cache_hit_rate:.1%}",
                           f"{comparison.optimized_metrics.cache_hit_rate:.1%}",
                           f"{comparison.optimized_metrics.cache_hit_rate - comparison.baseline_metrics.cache_hit_rate:+.1%}"])

        # Save markdown report
        md_file = self.results_dir / f"benchmark_report_{suite_name}_{timestamp}.md"
        self._generate_markdown_report(comparison, suite_name, md_file)

        self.logger.info(f"📊 Benchmark results saved:")
        self.logger.info(f"   JSON: {json_file}")
        self.logger.info(f"   CSV: {csv_file}")
        self.logger.info(f"   Markdown: {md_file}")

    def _generate_markdown_report(self, comparison: BenchmarkComparison,
                                suite_name: str, output_file: Path):
        """Generate comprehensive markdown report"""

        with open(output_file, 'w') as f:
            f.write(f"# Sprint 3.0 Performance Benchmark Report\n\n")
            f.write(f"**Suite:** {suite_name}  \n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Overall Improvement:** {comparison.improvement_factor:.2f}x speedup  \n\n")

            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            if comparison.improvement_factor >= 5.0:
                f.write(f"✅ **SUCCESS**: Achieved {comparison.improvement_factor:.1f}x speedup (target: 5x+)\n\n")
            elif comparison.improvement_factor >= 3.0:
                f.write(f"🟡 **GOOD**: Achieved {comparison.improvement_factor:.1f}x speedup (approaching 5x target)\n\n")
            else:
                f.write(f"🔴 **NEEDS WORK**: {comparison.improvement_factor:.1f}x speedup (target: 5x+)\n\n")

            # Performance Comparison Table
            f.write("## 📊 Performance Comparison\n\n")
            f.write("| Metric | Baseline | Optimized | Improvement |\n")
            f.write("|--------|----------|-----------|-------------|\n")

            b = comparison.baseline_metrics
            o = comparison.optimized_metrics

            f.write(f"| **Execution Time** | {b.total_execution_time:.2f}s | {o.total_execution_time:.2f}s | **{comparison.improvement_factor:.2f}x** |\n")
            f.write(f"| **Throughput** | {b.throughput_tasks_per_second:.2f} tasks/s | {o.throughput_tasks_per_second:.2f} tasks/s | {comparison.throughput_improvement_percent:+.1f}% |\n")
            f.write(f"| **Success Rate** | {b.success_rate:.1%} | {o.success_rate:.1%} | {(o.success_rate - b.success_rate)*100:+.1f}% |\n")
            f.write(f"| **Peak Memory** | {b.peak_memory_mb:.1f} MB | {o.peak_memory_mb:.1f} MB | {comparison.memory_reduction_percent:+.1f}% |\n")
            f.write(f"| **Cache Hit Rate** | {b.cache_hit_rate:.1%} | {o.cache_hit_rate:.1%} | {(o.cache_hit_rate - b.cache_hit_rate)*100:+.1f}% |\n\n")

            # Optimization Effectiveness
            f.write("## ⚡ Optimization Effectiveness\n\n")
            for opt_name, effectiveness in comparison.optimization_effectiveness.items():
                f.write(f"- **{opt_name.replace('_', ' ').title()}**: {effectiveness:.1f}%\n")
            f.write("\n")

            # Detailed Analysis
            f.write("## 🔍 Detailed Analysis\n\n")
            f.write("### Task-Level Performance\n")
            f.write(f"- **Average task time (baseline)**: {b.average_task_time:.2f}s\n")
            f.write(f"- **Average task time (optimized)**: {o.average_task_time:.2f}s\n")
            f.write(f"- **Time improvement per task**: {((b.average_task_time - o.average_task_time) / b.average_task_time * 100):+.1f}%\n\n")

            f.write("### Resource Utilization\n")
            f.write(f"- **Parallel efficiency**: {o.parallel_efficiency:.1%}\n")
            f.write(f"- **Memory efficiency**: {o.memory_efficiency_score:.1f} MB/task\n")
            f.write(f"- **CPU utilization**: {o.cpu_usage_percent:.1f}%\n\n")

            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            if comparison.improvement_factor < 5.0:
                f.write("To achieve 5x+ improvement target:\n")
                if comparison.optimized_metrics.cache_hit_rate < 0.5:
                    f.write("- Increase cache hit rate through longer evaluation runs\n")
                if comparison.optimized_metrics.parallel_efficiency < 0.7:
                    f.write("- Optimize parallel execution strategy\n")
                if comparison.memory_reduction_percent < 20:
                    f.write("- Apply more aggressive memory optimizations\n")
            else:
                f.write("🎉 **Target achieved!** Consider:\n")
                f.write("- Scaling to larger evaluation workloads\n")
                f.write("- Adding more languages/models\n")
                f.write("- Implementing additional optimizations\n")


class PerformanceMeasurementContext:
    """Context object provided during performance measurement"""

    def __init__(self, benchmark_name: str, task_times: List[float],
                 task_results: List[Dict], start_time: float):
        self.benchmark_name = benchmark_name
        self.task_times = task_times
        self.task_results = task_results
        self.start_time = start_time


# Testing and demonstration
def demo_performance_benchmarker():
    """Demonstration of performance benchmarker"""
    print("🚀 Performance Benchmarker Demo")
    print("=" * 50)

    benchmarker = PerformanceBenchmarker(Path("/tmp/benchmark_demo"))

    print("📊 Performance benchmarker initialized")
    print(f"   Results directory: {benchmarker.results_dir}")

    # Simulate a simple performance measurement
    with benchmarker.performance_measurement("demo_test", BenchmarkType.MICRO) as ctx:
        print(f"   Running simulated evaluation...")

        # Simulate some work
        for i in range(3):
            task_start = time.time()
            time.sleep(0.1)  # Simulate work
            task_time = time.time() - task_start

            ctx.task_times.append(task_time)
            ctx.task_results.append({
                'success': True,
                'task': f'task_{i}',
                'model': 'demo_model',
                'score': 0.85,
                'execution_time': task_time
            })

    # Show results
    metrics = benchmarker.current_metrics['demo_test']
    print(f"\n📈 Demo Results:")
    print(f"   Total time: {metrics.total_execution_time:.2f}s")
    print(f"   Successful tasks: {metrics.successful_tasks}")
    print(f"   Throughput: {metrics.throughput_tasks_per_second:.2f} tasks/s")
    print(f"   Peak memory: {metrics.peak_memory_mb:.1f} MB")

    print(f"\n✅ Ready to benchmark Sprint 3.0 improvements!")


if __name__ == "__main__":
    demo_performance_benchmarker()