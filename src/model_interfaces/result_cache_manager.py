#!/usr/bin/env python3
"""
Result Cache Manager (Sprint 3.0)

Intelligent caching system for AI benchmark evaluations to avoid redundant computations
and dramatically improve performance for repeated evaluations.
"""

import json
import hashlib
import time
import sqlite3
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from contextlib import contextmanager

try:
    from .language_detector import ProgrammingLanguage
except ImportError:
    # For standalone testing
    from language_detector import ProgrammingLanguage


@dataclass
class CacheKey:
    """Represents a unique cache key for an evaluation"""
    task_name: str
    model_name: str
    language: str
    parameters_hash: str
    code_hash: Optional[str] = None  # For code-specific caching


@dataclass
class CachedResult:
    """Cached evaluation result"""
    cache_key: CacheKey
    result_data: Dict[str, Any]
    timestamp: float
    execution_time: float
    cache_version: str = "3.0"
    metadata: Optional[Dict[str, Any]] = None


class CacheStrategy(Enum):
    """Cache strategies for different scenarios"""
    AGGRESSIVE = "aggressive"      # Cache all results, very permissive matching
    CONSERVATIVE = "conservative"  # Cache only stable results, strict matching
    SELECTIVE = "selective"        # Cache based on specific criteria
    DISABLED = "disabled"          # No caching


class ResultCacheManager:
    """
    Advanced result caching system for AI benchmarks.

    Features:
    - Parameter-aware caching with intelligent key generation
    - SQLite-based persistent storage with compression
    - TTL (Time To Live) support for cache expiration
    - Cache size management and automatic cleanup
    - Performance analytics and hit rate tracking
    """

    def __init__(self, cache_dir: Path, strategy: CacheStrategy = CacheStrategy.CONSERVATIVE,
                 max_cache_size_mb: int = 1000, default_ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.strategy = strategy
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl_seconds = default_ttl_hours * 3600

        # Database setup
        self.db_path = self.cache_dir / "evaluation_cache.db"
        self._init_database()

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_saves = 0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database for cache storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key_hash TEXT UNIQUE NOT NULL,
                    task_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    parameters_hash TEXT NOT NULL,
                    code_hash TEXT,
                    result_data BLOB NOT NULL,
                    timestamp REAL NOT NULL,
                    execution_time REAL NOT NULL,
                    cache_version TEXT NOT NULL,
                    metadata BLOB,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key_hash
                ON cache_entries(cache_key_hash)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_model_lang
                ON cache_entries(task_name, model_name, language)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON cache_entries(timestamp)
            """)

    def generate_cache_key(self, task_name: str, model_name: str,
                          language: Union[str, ProgrammingLanguage],
                          parameters: Dict[str, Any],
                          code: Optional[str] = None) -> CacheKey:
        """Generate a unique cache key for the given evaluation parameters"""

        # Normalize language
        if isinstance(language, ProgrammingLanguage):
            language_str = language.value
        else:
            language_str = str(language).lower()

        # Create parameters hash (exclude non-deterministic parameters)
        cache_relevant_params = self._filter_cache_relevant_parameters(parameters)
        params_json = json.dumps(cache_relevant_params, sort_keys=True)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()[:16]

        # Create code hash if provided
        code_hash = None
        if code:
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        return CacheKey(
            task_name=task_name,
            model_name=model_name,
            language=language_str,
            parameters_hash=params_hash,
            code_hash=code_hash
        )

    def _filter_cache_relevant_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters to only include cache-relevant ones"""
        # Exclude parameters that don't affect deterministic results
        excluded_params = {
            'save_results', 'results_dir', 'debug', 'verbose',
            'timestamp', 'session_id', 'output_file'
        }

        # Include only parameters that affect evaluation results
        relevant_params = {
            k: v for k, v in parameters.items()
            if k not in excluded_params and not k.startswith('_')
        }

        # Special handling for certain parameters based on strategy
        if self.strategy == CacheStrategy.CONSERVATIVE:
            # Be very strict about what parameters matter
            critical_params = {
                'n_samples', 'temperature', 'limit', 'max_length_generation',
                'target_language', 'safe_mode', 'container_isolation'
            }
            relevant_params = {
                k: v for k, v in relevant_params.items()
                if k in critical_params
            }

        return relevant_params

    def get_cache_key_hash(self, cache_key: CacheKey) -> str:
        """Generate a hash for the cache key"""
        key_string = f"{cache_key.task_name}|{cache_key.model_name}|{cache_key.language}|{cache_key.parameters_hash}"
        if cache_key.code_hash:
            key_string += f"|{cache_key.code_hash}"

        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cached_result(self, cache_key: CacheKey,
                         max_age_hours: Optional[int] = None) -> Optional[CachedResult]:
        """Retrieve cached result if available and valid"""

        if self.strategy == CacheStrategy.DISABLED:
            return None

        cache_key_hash = self.get_cache_key_hash(cache_key)
        max_age_seconds = (max_age_hours * 3600) if max_age_hours else self.default_ttl_seconds
        min_timestamp = time.time() - max_age_seconds

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT result_data, timestamp, execution_time, cache_version, metadata
                    FROM cache_entries
                    WHERE cache_key_hash = ? AND timestamp > ?
                """, (cache_key_hash, min_timestamp))

                row = cursor.fetchone()
                if row:
                    # Update access tracking
                    conn.execute("""
                        UPDATE cache_entries
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE cache_key_hash = ?
                    """, (time.time(), cache_key_hash))

                    # Decompress and deserialize result data
                    result_data = pickle.loads(gzip.decompress(row[0]))
                    metadata = pickle.loads(gzip.decompress(row[4])) if row[4] else None

                    self.cache_hits += 1

                    cached_result = CachedResult(
                        cache_key=cache_key,
                        result_data=result_data,
                        timestamp=row[1],
                        execution_time=row[2],
                        cache_version=row[3],
                        metadata=metadata
                    )

                    self.logger.info(f"Cache HIT for {cache_key.task_name}:{cache_key.model_name}:{cache_key.language}")
                    return cached_result
                else:
                    self.cache_misses += 1
                    self.logger.debug(f"Cache MISS for {cache_key.task_name}:{cache_key.model_name}:{cache_key.language}")
                    return None

        except Exception as e:
            self.logger.warning(f"Error retrieving cached result: {e}")
            self.cache_misses += 1
            return None

    def save_result(self, cache_key: CacheKey, result_data: Dict[str, Any],
                   execution_time: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save evaluation result to cache"""

        if self.strategy == CacheStrategy.DISABLED:
            return False

        # Apply caching strategy filters
        if not self._should_cache_result(cache_key, result_data):
            return False

        cache_key_hash = self.get_cache_key_hash(cache_key)
        timestamp = time.time()

        try:
            # Compress and serialize data
            compressed_result = gzip.compress(pickle.dumps(result_data))
            compressed_metadata = gzip.compress(pickle.dumps(metadata)) if metadata else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (cache_key_hash, task_name, model_name, language, parameters_hash,
                     code_hash, result_data, timestamp, execution_time, cache_version,
                     metadata, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (
                    cache_key_hash, cache_key.task_name, cache_key.model_name,
                    cache_key.language, cache_key.parameters_hash, cache_key.code_hash,
                    compressed_result, timestamp, execution_time, "3.0",
                    compressed_metadata, timestamp
                ))

            self.cache_saves += 1
            self.logger.info(f"Cached result for {cache_key.task_name}:{cache_key.model_name}:{cache_key.language}")

            # Cleanup if cache is getting too large
            asyncio.create_task(self._cleanup_cache_if_needed())

            return True

        except Exception as e:
            self.logger.error(f"Error saving result to cache: {e}")
            return False

    def _should_cache_result(self, cache_key: CacheKey, result_data: Dict[str, Any]) -> bool:
        """Determine if a result should be cached based on strategy"""

        if self.strategy == CacheStrategy.AGGRESSIVE:
            return True

        elif self.strategy == CacheStrategy.CONSERVATIVE:
            # Only cache if result appears stable and successful
            if isinstance(result_data, dict):
                # Check for success indicators
                if result_data.get('success', True) and result_data.get('score', 0) > 0:
                    # Check for Pass@K metrics (indicating proper evaluation)
                    metrics = result_data.get('metrics', {})
                    if any(key.startswith('pass@') for key in metrics.keys()):
                        return True
            return False

        elif self.strategy == CacheStrategy.SELECTIVE:
            # Cache based on specific criteria (can be customized)
            if isinstance(result_data, dict):
                # Cache if execution was successful and took significant time
                if (result_data.get('success', False) and
                    result_data.get('execution_time', 0) > 5.0):
                    return True
            return False

        return False

    async def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limits"""
        try:
            cache_size_mb = self.get_cache_size_mb()
            if cache_size_mb > self.max_cache_size_mb:
                await self._cleanup_old_entries()
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")

    async def _cleanup_old_entries(self, target_reduction_percent: float = 0.2):
        """Remove old cache entries to reduce size"""
        with sqlite3.connect(self.db_path) as conn:
            # Delete oldest 20% of entries
            total_count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            delete_count = int(total_count * target_reduction_percent)

            if delete_count > 0:
                conn.execute("""
                    DELETE FROM cache_entries
                    WHERE id IN (
                        SELECT id FROM cache_entries
                        ORDER BY last_accessed ASC, timestamp ASC
                        LIMIT ?
                    )
                """, (delete_count,))

                self.logger.info(f"Cleaned up {delete_count} old cache entries")

    def get_cache_size_mb(self) -> float:
        """Get current cache size in MB"""
        try:
            size_bytes = self.db_path.stat().st_size
            return size_bytes / (1024 * 1024)
        except:
            return 0.0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            total_entries = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]

            # Age analysis
            current_time = time.time()
            recent_entries = conn.execute("""
                SELECT COUNT(*) FROM cache_entries
                WHERE timestamp > ?
            """, (current_time - 3600,)).fetchone()[0]  # Last hour

            # Hit rate calculation
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0.0

            # Language breakdown
            language_stats = {}
            cursor = conn.execute("""
                SELECT language, COUNT(*) as count,
                       AVG(execution_time) as avg_time,
                       SUM(access_count) as total_accesses
                FROM cache_entries
                GROUP BY language
            """)
            for row in cursor:
                language_stats[row[0]] = {
                    'cached_results': row[1],
                    'avg_execution_time': row[2],
                    'total_accesses': row[3]
                }

            # Model breakdown
            model_stats = {}
            cursor = conn.execute("""
                SELECT model_name, COUNT(*) as count,
                       SUM(access_count) as total_accesses
                FROM cache_entries
                GROUP BY model_name
            """)
            for row in cursor:
                model_stats[row[0]] = {
                    'cached_results': row[1],
                    'total_accesses': row[2]
                }

        return {
            'total_cached_results': total_entries,
            'recent_entries_1h': recent_entries,
            'cache_size_mb': self.get_cache_size_mb(),
            'hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_saves': self.cache_saves,
            'strategy': self.strategy.value,
            'language_breakdown': language_stats,
            'model_breakdown': model_stats
        }

    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            if older_than_hours:
                min_timestamp = time.time() - (older_than_hours * 3600)
                deleted = conn.execute("""
                    DELETE FROM cache_entries WHERE timestamp < ?
                """, (min_timestamp,)).rowcount
                self.logger.info(f"Cleared {deleted} cache entries older than {older_than_hours} hours")
            else:
                conn.execute("DELETE FROM cache_entries")
                self.logger.info("Cleared all cache entries")

    def export_cache_report(self, output_file: Path):
        """Export detailed cache usage report"""
        stats = self.get_cache_stats()

        with open(output_file, 'w') as f:
            f.write("# AI Benchmark Suite - Cache Performance Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Cache Overview\n")
            f.write(f"- Total cached results: {stats['total_cached_results']}\n")
            f.write(f"- Cache size: {stats['cache_size_mb']:.2f} MB\n")
            f.write(f"- Hit rate: {stats['hit_rate']:.1%}\n")
            f.write(f"- Strategy: {stats['strategy']}\n\n")

            f.write("## Language Breakdown\n")
            for lang, lang_stats in stats['language_breakdown'].items():
                f.write(f"- **{lang.upper()}**: {lang_stats['cached_results']} results, ")
                f.write(f"{lang_stats['total_accesses']} accesses, ")
                f.write(f"avg time: {lang_stats['avg_execution_time']:.2f}s\n")

            f.write("\n## Model Breakdown\n")
            for model, model_stats in stats['model_breakdown'].items():
                f.write(f"- **{model}**: {model_stats['cached_results']} results, ")
                f.write(f"{model_stats['total_accesses']} accesses\n")

    @contextmanager
    def evaluation_cache_context(self, cache_key: CacheKey, execution_func,
                                max_age_hours: Optional[int] = None):
        """Context manager for automatic caching of evaluation results"""
        # Try to get cached result first
        cached_result = self.get_cached_result(cache_key, max_age_hours)
        if cached_result:
            yield cached_result.result_data
            return

        # Execute evaluation and cache result
        start_time = time.time()
        try:
            result = execution_func()
            execution_time = time.time() - start_time

            # Save to cache
            self.save_result(cache_key, result, execution_time)

            yield result
        except Exception as e:
            # Don't cache failed results
            raise e


# Testing and demonstration
def demo_cache_manager():
    """Demonstration of cache manager capabilities"""
    print("🚀 Result Cache Manager Demo")
    print("=" * 50)

    cache_dir = Path("/tmp/benchmark_cache_demo")
    cache_manager = ResultCacheManager(
        cache_dir,
        strategy=CacheStrategy.CONSERVATIVE
    )

    # Test cache operations
    test_cases = [
        ("humaneval", "qwen-coder", "python", {"n_samples": 5, "temperature": 0.2}),
        ("multiple-js", "codellama", "javascript", {"n_samples": 10, "temperature": 0.25}),
        ("humaneval", "qwen-coder", "python", {"n_samples": 5, "temperature": 0.2}),  # Duplicate
    ]

    for i, (task, model, lang, params) in enumerate(test_cases):
        print(f"\n🔸 Test {i+1}: {task} on {model} ({lang})")

        # Generate cache key
        cache_key = cache_manager.generate_cache_key(task, model, lang, params)

        # Try to get cached result
        cached = cache_manager.get_cached_result(cache_key)
        if cached:
            print(f"   ✅ Found cached result from {time.strftime('%H:%M:%S', time.localtime(cached.timestamp))}")
        else:
            print(f"   ⏳ No cached result, simulating evaluation...")

            # Simulate evaluation result
            result_data = {
                'success': True,
                'score': 0.85,
                'metrics': {'pass@1': 0.85, 'pass@5': 0.92},
                'execution_time': 15.0
            }

            # Save to cache
            cache_manager.save_result(cache_key, result_data, 15.0)
            print(f"   💾 Result cached")

    # Show cache statistics
    print(f"\n📊 Cache Statistics:")
    stats = cache_manager.get_cache_stats()
    for key, value in stats.items():
        if key not in ['language_breakdown', 'model_breakdown']:
            print(f"  {key}: {value}")

    print(f"\n✅ Cache demo completed! Size: {stats['cache_size_mb']:.2f} MB")


if __name__ == "__main__":
    import asyncio
    demo_cache_manager()