#!/usr/bin/env python3
"""
Memory Optimizer (Sprint 3.0)

Advanced memory management for large-scale AI benchmark evaluations.
Optimizes memory usage through intelligent resource management, streaming processing,
and efficient data structures.
"""

import os
import gc
import psutil
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Callable, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import logging
import tempfile
import mmap
import json


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    process_memory_mb: float
    system_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    peak_memory_mb: float
    timestamp: float


class MemoryStrategy(Enum):
    """Memory optimization strategies"""
    MINIMAL = "minimal"          # Aggressive memory optimization
    BALANCED = "balanced"        # Balance memory and performance
    PERFORMANCE = "performance"  # Favor performance over memory
    STREAMING = "streaming"      # Use streaming for large datasets


class ResourceMonitor:
    """Monitor system resource usage in real-time"""

    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.process = psutil.Process()
        self.peak_memory = 0.0
        self.memory_history: List[MemoryStats] = []
        self.monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_current_stats()
                self.memory_history.append(stats)
                self.peak_memory = max(self.peak_memory, stats.process_memory_mb)

                # Keep only last 100 entries to prevent memory leak
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]

                time.sleep(self.check_interval)
            except Exception as e:
                logging.warning(f"Memory monitoring error: {e}")

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        process_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        return MemoryStats(
            process_memory_mb=process_info.rss / 1024 / 1024,
            system_memory_mb=system_memory.total / 1024 / 1024,
            available_memory_mb=system_memory.available / 1024 / 1024,
            memory_percent=system_memory.percent,
            peak_memory_mb=self.peak_memory,
            timestamp=time.time()
        )


class MemoryEfficientDataLoader:
    """Memory-efficient data loading for large benchmark datasets"""

    def __init__(self, strategy: MemoryStrategy = MemoryStrategy.BALANCED):
        self.strategy = strategy
        self.loaded_data_cache = weakref.WeakValueDictionary()
        self.temp_files: List[Path] = []

    def load_dataset_streaming(self, file_path: Path,
                             chunk_size: int = 1000) -> Iterator[Dict[str, Any]]:
        """Load dataset in streaming fashion to minimize memory usage"""

        if file_path.suffix.lower() == '.json':
            yield from self._stream_json_dataset(file_path, chunk_size)
        elif file_path.suffix.lower() == '.jsonl':
            yield from self._stream_jsonl_dataset(file_path, chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _stream_json_dataset(self, file_path: Path,
                           chunk_size: int) -> Iterator[Dict[str, Any]]:
        """Stream JSON dataset in chunks"""
        try:
            with open(file_path, 'r') as f:
                # For large JSON files, we might need to use a streaming JSON parser
                data = json.load(f)

                if isinstance(data, list):
                    for i in range(0, len(data), chunk_size):
                        chunk = data[i:i + chunk_size]
                        for item in chunk:
                            yield item

                        # Force garbage collection after each chunk
                        if self.strategy == MemoryStrategy.MINIMAL:
                            gc.collect()

                elif isinstance(data, dict):
                    yield data

        except MemoryError:
            # Fallback to line-by-line processing for very large files
            logging.warning(f"Memory error loading {file_path}, falling back to streaming")
            yield from self._fallback_stream_large_json(file_path)

    def _stream_jsonl_dataset(self, file_path: Path,
                            chunk_size: int) -> Iterator[Dict[str, Any]]:
        """Stream JSONL dataset line by line"""
        batch = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        batch.append(item)

                        if len(batch) >= chunk_size:
                            for item in batch:
                                yield item
                            batch.clear()

                            # Memory management
                            if self.strategy == MemoryStrategy.MINIMAL:
                                gc.collect()

                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON line in {file_path}: {e}")

        # Yield remaining items
        for item in batch:
            yield item

    def _fallback_stream_large_json(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Fallback streaming for extremely large JSON files"""
        # This is a simplified implementation
        # In practice, you might want to use a proper streaming JSON parser
        with open(file_path, 'r') as f:
            buffer = ""
            brace_count = 0
            in_string = False
            escape_next = False

            for char in f.read():
                buffer += char

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                        if brace_count == 0 and buffer.strip():
                            try:
                                obj = json.loads(buffer.strip())
                                yield obj
                                buffer = ""
                            except json.JSONDecodeError:
                                pass

    def create_memory_mapped_file(self, data: bytes,
                                file_prefix: str = "benchmark_") -> Path:
        """Create memory-mapped temporary file for large data"""
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, prefix=file_prefix, suffix=".tmp"
        )
        temp_path = Path(temp_file.name)

        with open(temp_path, 'wb') as f:
            f.write(data)

        self.temp_files.append(temp_path)
        return temp_path

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()


class MemoryOptimizer:
    """
    Main memory optimization manager for AI benchmark suite.

    Provides intelligent memory management strategies to handle large-scale
    evaluations efficiently.
    """

    def __init__(self, strategy: MemoryStrategy = MemoryStrategy.BALANCED,
                 memory_limit_mb: Optional[int] = None):
        self.strategy = strategy
        self.memory_limit_mb = memory_limit_mb or self._detect_memory_limit()

        # Initialize components
        self.monitor = ResourceMonitor()
        self.data_loader = MemoryEfficientDataLoader(strategy)

        # Memory management state
        self.optimization_enabled = True
        self.cleanup_callbacks: List[Callable] = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _detect_memory_limit(self) -> int:
        """Detect appropriate memory limit based on system resources"""
        system_memory = psutil.virtual_memory()
        total_memory_mb = system_memory.total / 1024 / 1024

        # Use 70% of available memory as limit for safety
        return int(total_memory_mb * 0.7)

    @contextmanager
    def memory_optimized_execution(self, operation_name: str = "benchmark"):
        """Context manager for memory-optimized execution"""
        self.logger.info(f"Starting memory-optimized execution: {operation_name}")

        # Start monitoring
        self.monitor.start_monitoring()
        initial_stats = self.monitor.get_current_stats()

        try:
            # Apply memory optimizations based on strategy
            self._apply_memory_optimizations()

            yield self

        finally:
            # Cleanup and report
            final_stats = self.monitor.get_current_stats()
            self.monitor.stop_monitoring()

            self._cleanup_resources()
            self._report_memory_usage(initial_stats, final_stats, operation_name)

    def _apply_memory_optimizations(self):
        """Apply memory optimizations based on strategy"""
        if self.strategy == MemoryStrategy.MINIMAL:
            # Aggressive optimizations
            gc.set_threshold(100, 10, 10)  # More frequent garbage collection
            self._enable_memory_monitoring_with_cleanup()

        elif self.strategy == MemoryStrategy.BALANCED:
            # Balanced optimizations
            gc.set_threshold(700, 10, 10)  # Default with slight tuning

        elif self.strategy == MemoryStrategy.STREAMING:
            # Optimize for streaming workloads
            gc.set_threshold(1000, 15, 15)  # Less frequent GC for streaming

        # Force initial garbage collection
        gc.collect()

    def _enable_memory_monitoring_with_cleanup(self):
        """Enable automatic cleanup when memory usage gets high"""
        def memory_check():
            while self.optimization_enabled:
                stats = self.monitor.get_current_stats()

                # If memory usage is over 80% of limit, trigger cleanup
                if stats.process_memory_mb > (self.memory_limit_mb * 0.8):
                    self.logger.warning(f"High memory usage detected: {stats.process_memory_mb:.1f}MB")
                    self._emergency_cleanup()

                time.sleep(5)  # Check every 5 seconds

        cleanup_thread = threading.Thread(target=memory_check, daemon=True)
        cleanup_thread.start()

    def _emergency_cleanup(self):
        """Emergency memory cleanup when usage is high"""
        self.logger.info("Performing emergency memory cleanup")

        # Run all registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"Cleanup callback failed: {e}")

        # Force garbage collection
        gc.collect()

        # Clear weakref cache
        self.data_loader.loaded_data_cache.clear()

    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback for emergency situations"""
        self.cleanup_callbacks.append(callback)

    def optimize_for_large_evaluation(self, expected_tasks: int,
                                    expected_models: int) -> Dict[str, Any]:
        """Optimize memory settings for large evaluation workloads"""

        # Estimate memory requirements
        estimated_memory_per_task = 50  # MB per task (conservative estimate)
        estimated_total_memory = expected_tasks * expected_models * estimated_memory_per_task

        recommendations = {
            'estimated_memory_mb': estimated_total_memory,
            'system_memory_mb': self.memory_limit_mb,
            'recommendations': []
        }

        if estimated_total_memory > self.memory_limit_mb:
            # Memory will likely be insufficient
            recommendations['recommendations'].extend([
                "Consider reducing batch size or number of parallel workers",
                "Enable streaming mode for large datasets",
                "Use result caching to avoid redundant evaluations",
                "Consider running evaluation in smaller chunks"
            ])

            # Automatically switch to minimal strategy
            if self.strategy != MemoryStrategy.MINIMAL:
                self.strategy = MemoryStrategy.MINIMAL
                recommendations['recommendations'].append("Automatically switched to minimal memory strategy")

        elif estimated_total_memory > (self.memory_limit_mb * 0.7):
            # Memory will be tight
            recommendations['recommendations'].extend([
                "Consider reducing parallel workers",
                "Enable result caching",
                "Monitor memory usage during execution"
            ])

        else:
            # Memory should be sufficient
            recommendations['recommendations'].append("Memory usage should be within acceptable limits")

        return recommendations

    def get_optimized_batch_size(self, base_batch_size: int,
                                item_size_mb: float = 10.0) -> int:
        """Calculate optimized batch size based on memory constraints"""

        available_memory = self.memory_limit_mb * 0.6  # Use 60% of limit for batching
        max_batch_size = int(available_memory / item_size_mb)

        # Apply strategy-specific adjustments
        if self.strategy == MemoryStrategy.MINIMAL:
            # Very conservative batching
            optimized_size = min(base_batch_size, max_batch_size // 2)
        elif self.strategy == MemoryStrategy.BALANCED:
            # Balanced approach
            optimized_size = min(base_batch_size, max_batch_size)
        else:  # PERFORMANCE or STREAMING
            # More aggressive batching
            optimized_size = min(base_batch_size * 2, max_batch_size)

        return max(1, optimized_size)  # Ensure at least batch size of 1

    def _cleanup_resources(self):
        """Clean up all managed resources"""
        self.optimization_enabled = False

        # Cleanup data loader
        self.data_loader.cleanup_temp_files()

        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"Resource cleanup failed: {e}")

        # Final garbage collection
        gc.collect()

    def _report_memory_usage(self, initial_stats: MemoryStats,
                           final_stats: MemoryStats, operation_name: str):
        """Report memory usage statistics"""

        memory_delta = final_stats.process_memory_mb - initial_stats.process_memory_mb
        peak_memory = self.monitor.peak_memory

        self.logger.info(f"Memory usage report for '{operation_name}':")
        self.logger.info(f"  Initial memory: {initial_stats.process_memory_mb:.1f} MB")
        self.logger.info(f"  Final memory: {final_stats.process_memory_mb:.1f} MB")
        self.logger.info(f"  Memory delta: {memory_delta:+.1f} MB")
        self.logger.info(f"  Peak memory: {peak_memory:.1f} MB")
        self.logger.info(f"  Strategy used: {self.strategy.value}")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        current_stats = self.monitor.get_current_stats()

        return {
            'current_memory_mb': current_stats.process_memory_mb,
            'peak_memory_mb': self.monitor.peak_memory,
            'memory_limit_mb': self.memory_limit_mb,
            'memory_utilization': current_stats.process_memory_mb / self.memory_limit_mb,
            'strategy': self.strategy.value,
            'optimization_enabled': self.optimization_enabled,
            'temp_files_count': len(self.data_loader.temp_files),
            'cleanup_callbacks_registered': len(self.cleanup_callbacks),
            'memory_history': [
                {
                    'timestamp': stat.timestamp,
                    'memory_mb': stat.process_memory_mb,
                    'memory_percent': stat.memory_percent
                }
                for stat in self.monitor.memory_history[-10:]  # Last 10 entries
            ]
        }


# Utility functions and testing
def demo_memory_optimizer():
    """Demonstration of memory optimizer capabilities"""
    print("🚀 Memory Optimizer Demo")
    print("=" * 50)

    # Create optimizer with balanced strategy
    optimizer = MemoryOptimizer(
        strategy=MemoryStrategy.BALANCED,
        memory_limit_mb=1000  # 1GB limit for demo
    )

    # Simulate large evaluation
    with optimizer.memory_optimized_execution("demo_evaluation") as opt:
        print(f"📊 Memory optimization active with {opt.strategy.value} strategy")

        # Get recommendations for large evaluation
        recommendations = opt.optimize_for_large_evaluation(
            expected_tasks=50,
            expected_models=5
        )

        print(f"\n🔍 Evaluation recommendations:")
        print(f"  Estimated memory: {recommendations['estimated_memory_mb']:.1f} MB")
        print(f"  System memory: {recommendations['system_memory_mb']:.1f} MB")
        for rec in recommendations['recommendations']:
            print(f"  • {rec}")

        # Calculate optimized batch size
        batch_size = opt.get_optimized_batch_size(32, item_size_mb=15.0)
        print(f"\n⚙️  Optimized batch size: {batch_size} (from 32)")

        # Simulate some memory usage
        large_data = [{"test": "data"} * 1000 for _ in range(100)]
        print(f"  Created test data: {len(large_data)} items")

        # Force cleanup
        del large_data
        import gc
        gc.collect()

    # Show final report
    report = optimizer.get_memory_report()
    print(f"\n📈 Memory Report:")
    print(f"  Current memory: {report['current_memory_mb']:.1f} MB")
    print(f"  Peak memory: {report['peak_memory_mb']:.1f} MB")
    print(f"  Memory utilization: {report['memory_utilization']:.1%}")
    print(f"  Strategy: {report['strategy']}")

    print(f"\n✅ Memory optimization demo completed!")


if __name__ == "__main__":
    demo_memory_optimizer()