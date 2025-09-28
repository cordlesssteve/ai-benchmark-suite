#!/usr/bin/env python3
"""
Multi-Language Test Runner (Sprint 2.2)

Integrates with BigCode harness multi-language evaluation system to provide
language-aware test execution with proper Pass@K metrics.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tempfile

try:
    from .language_detector import ProgrammingLanguage, LanguageDetector, get_bigcode_task_name
    from .multi_language_executor import MultiLanguageExecutor, ExecutionMode, ExecutionResult
except ImportError:
    # For standalone testing
    from language_detector import ProgrammingLanguage, LanguageDetector, get_bigcode_task_name
    from multi_language_executor import MultiLanguageExecutor, ExecutionMode, ExecutionResult


@dataclass
class TestCase:
    """Represents a test case for any language"""
    input_data: str
    expected_output: str
    timeout: int = 5


@dataclass
class MultiLanguageTestResult:
    """Result of multi-language test execution"""
    problem_id: str
    language: ProgrammingLanguage
    generated_code: str
    test_results: List[bool]
    execution_results: List[ExecutionResult]
    pass_rate: float
    total_tests: int
    passed_tests: int
    error_message: Optional[str] = None


class BigCodeMultiLanguageAdapter:
    """Adapter for BigCode multi-language evaluation"""

    def __init__(self, bigcode_dir: Path, execution_mode: ExecutionMode = ExecutionMode.DOCKER):
        self.bigcode_dir = bigcode_dir
        self.execution_mode = execution_mode
        self.detector = LanguageDetector()
        self.executor = MultiLanguageExecutor(execution_mode)

        # BigCode language mappings
        self.language_task_map = {
            ProgrammingLanguage.PYTHON: "humaneval",
            ProgrammingLanguage.JAVASCRIPT: "multiple-js",
            ProgrammingLanguage.JAVA: "multiple-java",
            ProgrammingLanguage.CPP: "multiple-cpp",
            ProgrammingLanguage.GO: "multiple-go",
            ProgrammingLanguage.RUST: "multiple-rs",
            ProgrammingLanguage.TYPESCRIPT: "multiple-ts",
            ProgrammingLanguage.PHP: "multiple-php",
            ProgrammingLanguage.RUBY: "multiple-rb",
        }

    def run_multi_language_evaluation(self, model_name: str,
                                    target_language: Optional[ProgrammingLanguage] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Run multi-language evaluation using BigCode harness.

        Args:
            model_name: Name of the model to evaluate
            target_language: Specific language to test (None for auto-detection)
            **kwargs: Additional parameters (n_samples, temperature, etc.)

        Returns:
            Dictionary with evaluation results
        """

        if target_language and target_language in self.language_task_map:
            # Run specific language evaluation
            task_name = self.language_task_map[target_language]
            return self._run_bigcode_task(model_name, task_name, target_language, **kwargs)

        elif target_language is None:
            # Run evaluation for all supported languages
            results = {}
            for language, task_name in self.language_task_map.items():
                if language in self.executor.get_supported_languages():
                    try:
                        result = self._run_bigcode_task(model_name, task_name, language, **kwargs)
                        results[language.value] = result
                    except Exception as e:
                        results[language.value] = {
                            "error": str(e),
                            "language": language.value,
                            "task": task_name
                        }
            return results

        else:
            raise ValueError(f"Language {target_language.value} not supported")

    def _run_bigcode_task(self, model_name: str, task_name: str,
                         language: ProgrammingLanguage, **kwargs) -> Dict[str, Any]:
        """Run BigCode evaluation for specific language task"""

        # Prepare BigCode command
        cmd = [
            "python", "main.py",
            "--model", model_name,
            "--tasks", task_name,
            "--allow_code_execution",
            "--batch_size", "1",
            "--save_generations"
        ]

        # Add Sprint 2.1 parameters
        if n_samples := kwargs.get('n_samples', 5):
            cmd.extend(["--n_samples", str(n_samples)])

        if temperature := kwargs.get('temperature', 0.2):
            cmd.extend(["--temperature", str(temperature)])

        if limit := kwargs.get('limit', 10):
            cmd.extend(["--limit", str(limit)])

        # Add language-specific parameters
        if max_length := kwargs.get('max_length_generation', 512):
            cmd.extend(["--max_length_generation", str(max_length)])

        # Create output paths
        results_file = f"results_{language.value}_{int(time.time())}.json"
        cmd.extend(["--metric_output_path", results_file])

        try:
            # Execute BigCode harness
            result = subprocess.run(
                cmd,
                cwd=self.bigcode_dir,
                capture_output=True,
                text=True,
                timeout=kwargs.get('timeout', 600)  # 10 minute timeout
            )

            # Parse results
            results_path = self.bigcode_dir / results_file
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results_data = json.load(f)

                # Add language metadata
                results_data['language'] = language.value
                results_data['task'] = task_name
                results_data['command'] = ' '.join(cmd)

                return results_data
            else:
                return {
                    "error": "No results file generated",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "language": language.value,
                    "task": task_name
                }

        except subprocess.TimeoutExpired:
            return {
                "error": "BigCode evaluation timed out",
                "language": language.value,
                "task": task_name,
                "timeout": kwargs.get('timeout', 600)
            }
        except Exception as e:
            return {
                "error": str(e),
                "language": language.value,
                "task": task_name
            }

    def detect_and_run_tests(self, code: str, test_cases: List[TestCase],
                           problem_id: str = "unknown") -> MultiLanguageTestResult:
        """
        Detect language and run tests with appropriate executor.

        Args:
            code: Generated code to test
            test_cases: List of test cases to run
            problem_id: Identifier for the problem

        Returns:
            MultiLanguageTestResult with detailed results
        """
        # Detect language
        detection_result = self.detector.detect_language(code)
        language = detection_result.language

        if language == ProgrammingLanguage.UNKNOWN:
            return MultiLanguageTestResult(
                problem_id=problem_id,
                language=language,
                generated_code=code,
                test_results=[],
                execution_results=[],
                pass_rate=0.0,
                total_tests=len(test_cases),
                passed_tests=0,
                error_message="Could not detect programming language"
            )

        # Run tests
        test_results = []
        execution_results = []

        for i, test_case in enumerate(test_cases):
            try:
                # Create test code that includes the test case
                test_code = self._create_test_code(language, test_case, i)

                # Execute code with test
                exec_result = self.executor.execute_code(
                    code, language, test_code, test_case.timeout
                )

                execution_results.append(exec_result)

                # Determine if test passed
                test_passed = self._evaluate_test_result(exec_result, test_case)
                test_results.append(test_passed)

            except Exception as e:
                # Test execution failed
                error_result = ExecutionResult(
                    language=language,
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    execution_time=0.0,
                    error_message=str(e)
                )
                execution_results.append(error_result)
                test_results.append(False)

        # Calculate statistics
        passed_tests = sum(test_results)
        total_tests = len(test_cases)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        return MultiLanguageTestResult(
            problem_id=problem_id,
            language=language,
            generated_code=code,
            test_results=test_results,
            execution_results=execution_results,
            pass_rate=pass_rate,
            total_tests=total_tests,
            passed_tests=passed_tests
        )

    def _create_test_code(self, language: ProgrammingLanguage,
                         test_case: TestCase, test_index: int) -> str:
        """Create language-specific test code"""

        if language == ProgrammingLanguage.PYTHON:
            return f"""
# Test case {test_index}
try:
    result = {test_case.input_data}
    print(result)
except Exception as e:
    print(f"ERROR: {{e}}")
"""

        elif language == ProgrammingLanguage.JAVASCRIPT:
            return f"""
// Test case {test_index}
try {{
    let result = {test_case.input_data};
    console.log(result);
}} catch (e) {{
    console.log("ERROR: " + e.message);
}}
"""

        elif language == ProgrammingLanguage.JAVA:
            return f"""
// Test case {test_index} would be embedded in main method
        System.out.println({test_case.input_data});
"""

        elif language == ProgrammingLanguage.CPP:
            return f"""
// Test case {test_index}
    cout << {test_case.input_data} << endl;
"""

        else:
            return f"// Test case {test_index}: {test_case.input_data}"

    def _evaluate_test_result(self, exec_result: ExecutionResult,
                            test_case: TestCase) -> bool:
        """Evaluate if test case passed"""
        if not exec_result.success:
            return False

        # Simple output comparison (can be made more sophisticated)
        actual_output = exec_result.stdout.strip()
        expected_output = test_case.expected_output.strip()

        return actual_output == expected_output


class MultiLanguageTestSuite:
    """Test suite for multi-language evaluation"""

    def __init__(self, bigcode_dir: Path):
        self.adapter = BigCodeMultiLanguageAdapter(bigcode_dir)

    def run_comprehensive_evaluation(self, model_name: str,
                                   languages: Optional[List[ProgrammingLanguage]] = None,
                                   **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive multi-language evaluation.

        Args:
            model_name: Model to evaluate
            languages: Specific languages to test (None for all supported)
            **kwargs: Evaluation parameters

        Returns:
            Comprehensive evaluation results
        """
        if languages is None:
            languages = self.adapter.executor.get_supported_languages()

        results = {
            "model": model_name,
            "languages_tested": [lang.value for lang in languages],
            "parameters": kwargs,
            "results": {}
        }

        for language in languages:
            print(f"🔸 Evaluating {language.value.upper()}...")

            try:
                lang_result = self.adapter.run_multi_language_evaluation(
                    model_name, language, **kwargs
                )
                results["results"][language.value] = lang_result

                # Extract Pass@K metrics if available
                if isinstance(lang_result, dict) and "pass@1" in lang_result:
                    print(f"   Pass@1: {lang_result.get('pass@1', 0):.3f}")

            except Exception as e:
                results["results"][language.value] = {
                    "error": str(e),
                    "language": language.value
                }
                print(f"   Error: {e}")

        return results


# Testing and examples
if __name__ == "__main__":
    # Test the multi-language test runner
    print("🧪 Multi-Language Test Runner Demo")
    print("=" * 50)

    # Create test cases
    test_cases = [
        TestCase("add(2, 3)", "5"),
        TestCase("add(10, 20)", "30"),
        TestCase("add(-5, 5)", "0"),
    ]

    # Test code samples
    test_codes = {
        ProgrammingLanguage.PYTHON: """
def add(a, b):
    return a + b
""",

        ProgrammingLanguage.JAVASCRIPT: """
function add(a, b) {
    return a + b;
}
""",

        ProgrammingLanguage.CPP: """
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}
""",
    }

    # Create adapter (using direct mode for testing)
    temp_dir = Path("/tmp")  # Placeholder for BigCode directory
    adapter = BigCodeMultiLanguageAdapter(temp_dir, ExecutionMode.DIRECT)

    print("\n🔍 Testing language detection and test execution:")

    for language, code in test_codes.items():
        print(f"\n🔸 Testing {language.value.upper()}:")
        print("-" * 30)

        result = adapter.detect_and_run_tests(code, test_cases[:1], f"test_{language.value}")

        print(f"Detected Language: {result.language.value}")
        print(f"Pass Rate: {result.pass_rate:.1%}")
        print(f"Tests Passed: {result.passed_tests}/{result.total_tests}")

        if result.error_message:
            print(f"Error: {result.error_message}")

    print(f"\n✅ Multi-language test runner ready for integration!")
    print(f"Supported languages: {[lang.value for lang in adapter.executor.get_supported_languages()]}")