#!/usr/bin/env python3
"""
Parallel Execution Manager (Sprint 3.0)

Manages parallel execution of evaluations across multiple containers and languages
with optimized resource utilization and container lifecycle management.
"""

import asyncio
import concurrent.futures
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import asynccontextmanager

try:
    from .language_detector import ProgrammingLanguage
    from .multi_language_executor import ExecutionResult, ExecutionMode
except ImportError:
    # For standalone testing
    from language_detector import ProgrammingLanguage
    from multi_language_executor import ExecutionResult, ExecutionMode


@dataclass
class EvaluationTask:
    """Individual evaluation task"""
    task_id: str
    task_name: str
    model_name: str
    language: ProgrammingLanguage
    parameters: Dict[str, Any]
    priority: int = 1  # Higher numbers = higher priority


@dataclass
class EvaluationResult:
    """Result from parallel evaluation"""
    task_id: str
    task_name: str
    model_name: str
    language: ProgrammingLanguage
    success: bool
    result_data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class ExecutionStrategy(Enum):
    """Execution strategies for parallel processing"""
    CONCURRENT_LANGUAGES = "concurrent_languages"    # Run different languages in parallel
    CONCURRENT_MODELS = "concurrent_models"          # Run different models in parallel
    CONCURRENT_TASKS = "concurrent_tasks"            # Run different tasks in parallel
    FULL_PARALLEL = "full_parallel"                  # Maximum parallelization


class ContainerPool:
    """Pool of reusable Docker containers for optimized execution"""

    def __init__(self, max_containers: int = 4, container_timeout: int = 300):
        self.max_containers = max_containers
        self.container_timeout = container_timeout
        self.active_containers: Dict[str, Dict[str, Any]] = {}
        self.container_lock = threading.RLock()
        self._shutdown = False

    async def get_container(self, language: ProgrammingLanguage,
                          docker_image: str) -> Dict[str, Any]:
        """Get or create a container for the specified language"""
        container_key = f"{language.value}_{docker_image}"

        with self.container_lock:
            if container_key in self.active_containers:
                container_info = self.active_containers[container_key]
                container_info['last_used'] = time.time()
                return container_info

            # Create new container if under limit
            if len(self.active_containers) < self.max_containers:
                container_info = await self._create_container(language, docker_image)
                self.active_containers[container_key] = container_info
                return container_info

            # If at limit, wait for a container to become available
            await self._wait_for_available_container()
            return await self.get_container(language, docker_image)

    async def _create_container(self, language: ProgrammingLanguage,
                              docker_image: str) -> Dict[str, Any]:
        """Create a new Docker container"""
        container_name = f"benchmark_{language.value}_{int(time.time())}"

        # Create long-running container
        create_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "none",
            "--memory", "512m",
            "--cpus", "0.5",
            "--user", "1000:1000",
            "--workdir", "/workspace",
            docker_image,
            "sleep", str(self.container_timeout)
        ]

        import subprocess
        result = await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {stderr.decode()}")

        return {
            'container_id': stdout.decode().strip(),
            'container_name': container_name,
            'language': language,
            'docker_image': docker_image,
            'created_time': time.time(),
            'last_used': time.time()
        }

    async def _wait_for_available_container(self):
        """Wait for a container to become available"""
        # Simple implementation: wait 1 second and try again
        await asyncio.sleep(1)

    async def execute_in_container(self, container_info: Dict[str, Any],
                                 command: List[str]) -> ExecutionResult:
        """Execute command in an existing container"""
        container_name = container_info['container_name']
        language = container_info['language']

        # Execute command in container
        exec_cmd = ["docker", "exec", container_name] + command

        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            execution_time = time.time() - start_time

            return ExecutionResult(
                language=language,
                success=(proc.returncode == 0),
                exit_code=proc.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                execution_time=execution_time
            )

        except Exception as e:
            return ExecutionResult(
                language=language,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    async def cleanup(self):
        """Clean up all containers in the pool"""
        self._shutdown = True

        with self.container_lock:
            for container_key, container_info in self.active_containers.items():
                try:
                    container_name = container_info['container_name']

                    # Stop and remove container
                    stop_proc = await asyncio.create_subprocess_exec(
                        "docker", "stop", container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await stop_proc.communicate()

                    remove_proc = await asyncio.create_subprocess_exec(
                        "docker", "rm", container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await remove_proc.communicate()

                except Exception as e:
                    logging.warning(f"Error cleaning up container {container_key}: {e}")

            self.active_containers.clear()


class ParallelExecutionManager:
    """
    Manages parallel execution of AI benchmark evaluations.

    Optimizes performance through:
    - Concurrent container execution
    - Container pooling and reuse
    - Intelligent task scheduling
    - Resource management
    """

    def __init__(self, max_workers: int = 4, max_containers: int = 6,
                 strategy: ExecutionStrategy = ExecutionStrategy.CONCURRENT_LANGUAGES):
        self.max_workers = max_workers
        self.max_containers = max_containers
        self.strategy = strategy
        self.container_pool = ContainerPool(max_containers)
        self.active_tasks: Dict[str, EvaluationTask] = {}
        self.completed_tasks: Dict[str, EvaluationResult] = {}
        self._shutdown = False

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def execute_parallel_evaluations(self, tasks: List[EvaluationTask],
                                         evaluation_func: Callable,
                                         progress_callback: Optional[Callable] = None) -> List[EvaluationResult]:
        """
        Execute multiple evaluation tasks in parallel.

        Args:
            tasks: List of evaluation tasks to execute
            evaluation_func: Function to call for each evaluation
            progress_callback: Optional callback for progress updates

        Returns:
            List of evaluation results
        """

        self.logger.info(f"Starting parallel execution of {len(tasks)} tasks")
        start_time = time.time()

        try:
            # Sort tasks by priority and group by strategy
            sorted_tasks = self._sort_and_group_tasks(tasks)

            # Execute tasks based on strategy
            if self.strategy == ExecutionStrategy.CONCURRENT_LANGUAGES:
                results = await self._execute_by_language(sorted_tasks, evaluation_func, progress_callback)
            elif self.strategy == ExecutionStrategy.CONCURRENT_MODELS:
                results = await self._execute_by_model(sorted_tasks, evaluation_func, progress_callback)
            elif self.strategy == ExecutionStrategy.CONCURRENT_TASKS:
                results = await self._execute_by_task(sorted_tasks, evaluation_func, progress_callback)
            else:  # FULL_PARALLEL
                results = await self._execute_full_parallel(sorted_tasks, evaluation_func, progress_callback)

            total_time = time.time() - start_time
            self.logger.info(f"Completed {len(results)} evaluations in {total_time:.2f}s")

            return results

        finally:
            await self.container_pool.cleanup()

    def _sort_and_group_tasks(self, tasks: List[EvaluationTask]) -> List[List[EvaluationTask]]:
        """Sort and group tasks based on execution strategy"""
        # Sort by priority (higher first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        if self.strategy == ExecutionStrategy.CONCURRENT_LANGUAGES:
            # Group by language
            groups = self._group_by_attribute(sorted_tasks, lambda t: t.language)
        elif self.strategy == ExecutionStrategy.CONCURRENT_MODELS:
            # Group by model
            groups = self._group_by_attribute(sorted_tasks, lambda t: t.model_name)
        elif self.strategy == ExecutionStrategy.CONCURRENT_TASKS:
            # Group by task name
            groups = self._group_by_attribute(sorted_tasks, lambda t: t.task_name)
        else:  # FULL_PARALLEL
            # Individual tasks
            groups = [[task] for task in sorted_tasks]

        return groups

    def _group_by_attribute(self, tasks: List[EvaluationTask],
                           key_func: Callable) -> List[List[EvaluationTask]]:
        """Group tasks by a specific attribute"""
        groups = {}
        for task in tasks:
            key = key_func(task)
            if key not in groups:
                groups[key] = []
            groups[key].append(task)

        return list(groups.values())

    async def _execute_by_language(self, task_groups: List[List[EvaluationTask]],
                                 evaluation_func: Callable,
                                 progress_callback: Optional[Callable]) -> List[EvaluationResult]:
        """Execute tasks grouped by language in parallel"""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def execute_language_group(group: List[EvaluationTask]) -> List[EvaluationResult]:
            async with semaphore:
                results = []
                for task in group:
                    result = await self._execute_single_task(task, evaluation_func)
                    results.append(result)
                    if progress_callback:
                        await progress_callback(task, result)
                return results

        # Execute all language groups in parallel
        group_results = await asyncio.gather(*[
            execute_language_group(group) for group in task_groups
        ])

        # Flatten results
        return [result for group_result in group_results for result in group_result]

    async def _execute_by_model(self, task_groups: List[List[EvaluationTask]],
                               evaluation_func: Callable,
                               progress_callback: Optional[Callable]) -> List[EvaluationResult]:
        """Execute tasks grouped by model in parallel"""
        return await self._execute_by_language(task_groups, evaluation_func, progress_callback)

    async def _execute_by_task(self, task_groups: List[List[EvaluationTask]],
                              evaluation_func: Callable,
                              progress_callback: Optional[Callable]) -> List[EvaluationResult]:
        """Execute tasks grouped by task name in parallel"""
        return await self._execute_by_language(task_groups, evaluation_func, progress_callback)

    async def _execute_full_parallel(self, task_groups: List[List[EvaluationTask]],
                                   evaluation_func: Callable,
                                   progress_callback: Optional[Callable]) -> List[EvaluationResult]:
        """Execute all tasks in full parallel mode"""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def execute_task_with_semaphore(task: EvaluationTask) -> EvaluationResult:
            async with semaphore:
                result = await self._execute_single_task(task, evaluation_func)
                if progress_callback:
                    await progress_callback(task, result)
                return result

        # Execute all tasks in parallel (limited by semaphore)
        all_tasks = [task for group in task_groups for task in group]
        results = await asyncio.gather(*[
            execute_task_with_semaphore(task) for task in all_tasks
        ])

        return results

    async def _execute_single_task(self, task: EvaluationTask,
                                 evaluation_func: Callable) -> EvaluationResult:
        """Execute a single evaluation task"""
        start_time = time.time()

        try:
            self.active_tasks[task.task_id] = task

            # Call the evaluation function with task parameters
            result_data = await asyncio.get_event_loop().run_in_executor(
                None, evaluation_func, task.task_name, task.model_name, **task.parameters
            )

            execution_time = time.time() - start_time

            result = EvaluationResult(
                task_id=task.task_id,
                task_name=task.task_name,
                model_name=task.model_name,
                language=task.language,
                success=True,
                result_data=result_data.__dict__ if hasattr(result_data, '__dict__') else result_data,
                execution_time=execution_time
            )

            self.completed_tasks[task.task_id] = result
            return result

        except Exception as e:
            execution_time = time.time() - start_time

            result = EvaluationResult(
                task_id=task.task_id,
                task_name=task.task_name,
                model_name=task.model_name,
                language=task.language,
                success=False,
                result_data={},
                execution_time=execution_time,
                error_message=str(e)
            )

            self.completed_tasks[task.task_id] = result
            return result

        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        completed = list(self.completed_tasks.values())
        if not completed:
            return {}

        successful = [r for r in completed if r.success]
        failed = [r for r in completed if not r.success]

        total_time = sum(r.execution_time for r in completed)
        avg_time = total_time / len(completed)

        # Group by language for analysis
        by_language = {}
        for result in completed:
            lang = result.language.value
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(result)

        language_stats = {}
        for lang, results in by_language.items():
            lang_successful = [r for r in results if r.success]
            language_stats[lang] = {
                'total': len(results),
                'successful': len(lang_successful),
                'success_rate': len(lang_successful) / len(results),
                'avg_time': sum(r.execution_time for r in results) / len(results)
            }

        return {
            'total_tasks': len(completed),
            'successful_tasks': len(successful),
            'failed_tasks': len(failed),
            'success_rate': len(successful) / len(completed),
            'total_execution_time': total_time,
            'average_execution_time': avg_time,
            'language_breakdown': language_stats,
            'strategy_used': self.strategy.value
        }


# Testing and utility functions
async def example_evaluation_function(task_name: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """Example evaluation function for testing"""
    # Simulate evaluation work
    await asyncio.sleep(0.5)  # Simulate processing time

    return {
        'task': task_name,
        'model': model_name,
        'score': 0.85,
        'metrics': {'pass@1': 0.85, 'pass@5': 0.92},
        'parameters': kwargs
    }


async def demo_parallel_execution():
    """Demonstration of parallel execution capabilities"""
    print("🚀 Parallel Execution Manager Demo")
    print("=" * 50)

    # Create test tasks
    tasks = [
        EvaluationTask("task_1", "humaneval", "qwen-coder", ProgrammingLanguage.PYTHON, {"n_samples": 5}),
        EvaluationTask("task_2", "multiple-js", "qwen-coder", ProgrammingLanguage.JAVASCRIPT, {"n_samples": 5}),
        EvaluationTask("task_3", "multiple-cpp", "codellama", ProgrammingLanguage.CPP, {"n_samples": 3}),
        EvaluationTask("task_4", "multiple-java", "phi3.5", ProgrammingLanguage.JAVA, {"n_samples": 5}),
        EvaluationTask("task_5", "humaneval", "codellama", ProgrammingLanguage.PYTHON, {"n_samples": 10}),
    ]

    # Create manager with concurrent language strategy
    manager = ParallelExecutionManager(
        max_workers=3,
        strategy=ExecutionStrategy.CONCURRENT_LANGUAGES
    )

    def progress_callback(task, result):
        status = "✅" if result.success else "❌"
        print(f"  {status} {task.task_name} on {task.model_name} ({task.language.value}) - {result.execution_time:.2f}s")

    # Execute tasks
    start_time = time.time()
    results = await manager.execute_parallel_evaluations(
        tasks, example_evaluation_function, progress_callback
    )
    total_time = time.time() - start_time

    # Show results
    print(f"\n📊 Results Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tasks completed: {len(results)}")
    print(f"Success rate: {len([r for r in results if r.success])}/{len(results)}")

    # Show performance stats
    stats = manager.get_performance_stats()
    print(f"\n📈 Performance Statistics:")
    for key, value in stats.items():
        if key != 'language_breakdown':
            print(f"  {key}: {value}")

    print(f"\n🔍 Language Breakdown:")
    for lang, lang_stats in stats['language_breakdown'].items():
        print(f"  {lang.upper()}: {lang_stats['successful']}/{lang_stats['total']} "
              f"({lang_stats['success_rate']:.1%}) - avg: {lang_stats['avg_time']:.2f}s")


if __name__ == "__main__":
    asyncio.run(demo_parallel_execution())