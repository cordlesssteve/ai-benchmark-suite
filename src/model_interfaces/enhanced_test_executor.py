#!/usr/bin/env python3
"""
Enhanced Test Case Execution Engine - Sprint 1.2

Builds on BigCode's execution framework with comprehensive enhancements:
- Function extraction from generated code
- Comprehensive test case execution framework
- Detailed error reporting for test failures
- Multiple test case batching support
- Enhanced safety and error handling
"""

import ast
import re
import sys
import time
import tempfile
import multiprocessing
import contextlib
import signal
import io
import os
import traceback
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures


class TestResult(Enum):
    """Test execution results"""
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    EXTRACTION_ERROR = "extraction_error"


@dataclass
class FunctionInfo:
    """Information about extracted function"""
    name: str
    signature: str
    body: str
    line_start: int
    line_end: int
    ast_node: Optional[ast.FunctionDef] = None


@dataclass
class TestCase:
    """Individual test case information"""
    test_id: str
    test_code: str
    expected_output: Optional[Any] = None
    timeout: float = 3.0
    description: str = ""


@dataclass
class TestExecutionResult:
    """Detailed result from test execution"""
    test_id: str
    result: TestResult
    passed: bool
    execution_time: float
    output: str = ""
    error_message: str = ""
    error_type: str = ""
    traceback: str = ""
    extracted_functions: List[FunctionInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchExecutionResult:
    """Results from batch test execution"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    timeout_tests: int
    error_tests: int
    execution_time: float
    pass_rate: float
    individual_results: List[TestExecutionResult]
    summary: Dict[str, Any] = field(default_factory=dict)


class TimeoutException(Exception):
    """Exception for test execution timeout"""
    pass


class EnhancedTestExecutor:
    """
    Enhanced test execution engine with comprehensive error reporting
    and function extraction capabilities.
    """

    def __init__(self, default_timeout: float = 3.0, max_workers: int = 4):
        self.default_timeout = default_timeout
        self.max_workers = max_workers

    def extract_functions(self, code: str) -> List[FunctionInfo]:
        """
        Extract function definitions from generated code with detailed analysis.
        """
        functions = []

        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Find all function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function_node(node, code)
                    if func_info:
                        functions.append(func_info)

        except SyntaxError as e:
            # Handle syntax errors gracefully
            print(f"⚠️ Syntax error during function extraction: {e}")

            # Try to extract functions using regex as fallback
            functions.extend(self._extract_functions_regex(code))

        except Exception as e:
            print(f"⚠️ Error during function extraction: {e}")

        return functions

    def _analyze_function_node(self, node: ast.FunctionDef, code: str) -> Optional[FunctionInfo]:
        """Analyze a function AST node and extract detailed information"""
        try:
            # Get function signature
            signature = self._get_function_signature(node)

            # Get function body
            lines = code.split('\n')
            start_line = node.lineno - 1  # AST line numbers are 1-based
            end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else len(lines) - 1

            # Extract function body
            body_lines = lines[start_line:end_line + 1]
            body = '\n'.join(body_lines)

            return FunctionInfo(
                name=node.name,
                signature=signature,
                body=body,
                line_start=start_line,
                line_end=end_line,
                ast_node=node
            )

        except Exception as e:
            print(f"⚠️ Error analyzing function node {getattr(node, 'name', 'unknown')}: {e}")
            return None

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node"""
        try:
            # Build argument list
            args = []

            # Regular arguments
            for arg in node.args.args:
                args.append(arg.arg)

            # Handle defaults
            defaults = node.args.defaults
            if defaults:
                num_defaults = len(defaults)
                for i, default in enumerate(defaults):
                    arg_index = len(node.args.args) - num_defaults + i
                    if arg_index >= 0:
                        try:
                            default_value = ast.literal_eval(default)
                            args[arg_index] = f"{args[arg_index]}={default_value}"
                        except:
                            args[arg_index] = f"{args[arg_index]}=..."

            # Handle *args and **kwargs
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            signature = f"def {node.name}({', '.join(args)}):"
            return signature

        except Exception as e:
            return f"def {node.name}(...):"

    def _extract_functions_regex(self, code: str) -> List[FunctionInfo]:
        """Fallback function extraction using regex"""
        functions = []

        # Pattern to match function definitions
        pattern = r'^(\s*)def\s+(\w+)\s*\([^)]*\)\s*:'

        lines = code.split('\n')
        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                indent, func_name = match.groups()

                # Find function body (simple indentation-based)
                body_lines = [line]
                j = i + 1
                while j < len(lines):
                    if lines[j].strip() == "":
                        body_lines.append(lines[j])
                    elif lines[j].startswith(indent + "    ") or lines[j].startswith(indent + "\t"):
                        body_lines.append(lines[j])
                    else:
                        break
                    j += 1

                functions.append(FunctionInfo(
                    name=func_name,
                    signature=line.strip(),
                    body='\n'.join(body_lines),
                    line_start=i,
                    line_end=j - 1
                ))

        return functions

    def execute_test_case(self, test_case: TestCase, generated_code: str) -> TestExecutionResult:
        """
        Execute a single test case with comprehensive error reporting.
        """
        start_time = time.time()

        # Extract functions from generated code
        extracted_functions = self.extract_functions(generated_code)

        # Create complete test program
        test_program = self._create_test_program(generated_code, test_case)

        # Execute test with multiprocessing for safety
        result = self._execute_with_multiprocessing(test_program, test_case.timeout)

        execution_time = time.time() - start_time

        # Parse execution result
        test_result, error_info = self._parse_execution_result(result)

        return TestExecutionResult(
            test_id=test_case.test_id,
            result=test_result,
            passed=(test_result == TestResult.PASSED),
            execution_time=execution_time,
            output=result.get("output", ""),
            error_message=error_info.get("message", ""),
            error_type=error_info.get("type", ""),
            traceback=error_info.get("traceback", ""),
            extracted_functions=extracted_functions,
            metadata={
                "timeout": test_case.timeout,
                "description": test_case.description,
                "function_count": len(extracted_functions),
                "code_length": len(generated_code)
            }
        )

    def _create_test_program(self, generated_code: str, test_case: TestCase) -> str:
        """Create complete test program by combining generated code and test"""

        # Clean the generated code
        clean_code = self._clean_generated_code(generated_code)

        # Combine with test case
        test_program = f"""
# Generated code
{clean_code}

# Test case
{test_case.test_code}
"""
        return test_program

    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code for safe execution"""

        # Remove dangerous imports and calls
        dangerous_patterns = [
            r'import\s+os\s*',
            r'import\s+sys\s*',
            r'import\s+subprocess\s*',
            r'from\s+os\s+import\s*',
            r'from\s+sys\s+import\s*',
            r'from\s+subprocess\s+import\s*',
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\('
        ]

        lines = code.split('\n')
        clean_lines = []

        for line in lines:
            # Check for dangerous patterns
            is_dangerous = False
            for pattern in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_dangerous = True
                    break

            if not is_dangerous:
                clean_lines.append(line)
            else:
                # Replace with comment
                clean_lines.append(f"# REMOVED DANGEROUS LINE: {line}")

        return '\n'.join(clean_lines)

    def _execute_with_multiprocessing(self, test_program: str, timeout: float) -> Dict[str, Any]:
        """Execute test program using multiprocessing for isolation"""

        manager = multiprocessing.Manager()
        result = manager.dict()

        # Create process to run the test
        process = multiprocessing.Process(
            target=self._safe_execute,
            args=(test_program, result, timeout)
        )

        process.start()
        process.join(timeout=timeout + 1)

        if process.is_alive():
            process.kill()
            process.join()
            result["status"] = "timeout"
            result["error"] = "Test execution timed out"

        # Convert manager dict to regular dict
        return dict(result)

    def _safe_execute(self, test_program: str, result, timeout: float):
        """Safely execute test program in isolated process"""

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Set up timeout handler
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(timeout))

            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute the test program
            exec_globals = {}
            exec(test_program, exec_globals)

            # If we get here, test passed
            result["status"] = "passed"
            result["output"] = stdout_capture.getvalue()

        except TimeoutException:
            result["status"] = "timeout"
            result["error"] = "Test execution timed out"

        except SyntaxError as e:
            result["status"] = "syntax_error"
            result["error"] = str(e)
            result["error_type"] = "SyntaxError"
            result["traceback"] = traceback.format_exc()

        except ImportError as e:
            result["status"] = "import_error"
            result["error"] = str(e)
            result["error_type"] = "ImportError"
            result["traceback"] = traceback.format_exc()

        except Exception as e:
            result["status"] = "runtime_error"
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            result["traceback"] = traceback.format_exc()
            result["stderr"] = stderr_capture.getvalue()

        finally:
            # Restore output
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Cancel timeout
            signal.alarm(0)

    def _timeout_handler(self, signum, frame):
        """Handle timeout signal"""
        raise TimeoutException("Test execution timed out")

    def _parse_execution_result(self, result: Dict[str, Any]) -> Tuple[TestResult, Dict[str, Any]]:
        """Parse execution result and extract error information"""

        status = result.get("status", "unknown")
        error_info = {}

        if status == "passed":
            test_result = TestResult.PASSED
        elif status == "timeout":
            test_result = TestResult.TIMEOUT
            error_info = {
                "message": result.get("error", "Timeout"),
                "type": "TimeoutError",
                "traceback": ""
            }
        elif status == "syntax_error":
            test_result = TestResult.SYNTAX_ERROR
            error_info = {
                "message": result.get("error", "Syntax error"),
                "type": result.get("error_type", "SyntaxError"),
                "traceback": result.get("traceback", "")
            }
        elif status == "import_error":
            test_result = TestResult.IMPORT_ERROR
            error_info = {
                "message": result.get("error", "Import error"),
                "type": result.get("error_type", "ImportError"),
                "traceback": result.get("traceback", "")
            }
        else:  # runtime_error or other
            test_result = TestResult.RUNTIME_ERROR
            error_info = {
                "message": result.get("error", "Runtime error"),
                "type": result.get("error_type", "RuntimeError"),
                "traceback": result.get("traceback", "")
            }

        return test_result, error_info

    def execute_test_batch(self, test_cases: List[TestCase], generated_code: str,
                          parallel: bool = True) -> BatchExecutionResult:
        """
        Execute multiple test cases with optional parallelization.
        """
        start_time = time.time()

        print(f"🧪 Executing batch of {len(test_cases)} test cases...")

        if parallel and len(test_cases) > 1:
            results = self._execute_parallel(test_cases, generated_code)
        else:
            results = self._execute_sequential(test_cases, generated_code)

        execution_time = time.time() - start_time

        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed and r.result != TestResult.TIMEOUT)
        timeout_tests = sum(1 for r in results if r.result == TestResult.TIMEOUT)
        error_tests = sum(1 for r in results if r.result in [TestResult.SYNTAX_ERROR, TestResult.IMPORT_ERROR, TestResult.RUNTIME_ERROR])

        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        # Create detailed summary
        summary = {
            "pass_rate": pass_rate,
            "avg_execution_time": sum(r.execution_time for r in results) / total_tests if total_tests > 0 else 0,
            "error_distribution": {
                "syntax_errors": sum(1 for r in results if r.result == TestResult.SYNTAX_ERROR),
                "import_errors": sum(1 for r in results if r.result == TestResult.IMPORT_ERROR),
                "runtime_errors": sum(1 for r in results if r.result == TestResult.RUNTIME_ERROR),
                "timeouts": timeout_tests
            },
            "function_analysis": self._analyze_extracted_functions(results)
        }

        print(f"✅ Batch execution complete: {passed_tests}/{total_tests} passed ({pass_rate:.1%})")

        return BatchExecutionResult(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            timeout_tests=timeout_tests,
            error_tests=error_tests,
            execution_time=execution_time,
            pass_rate=pass_rate,
            individual_results=results,
            summary=summary
        )

    def _execute_parallel(self, test_cases: List[TestCase], generated_code: str) -> List[TestExecutionResult]:
        """Execute test cases in parallel using ThreadPoolExecutor"""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test cases
            future_to_test = {
                executor.submit(self.execute_test_case, test_case, generated_code): test_case
                for test_case in test_cases
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_test):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    test_case = future_to_test[future]
                    # Create error result
                    results.append(TestExecutionResult(
                        test_id=test_case.test_id,
                        result=TestResult.RUNTIME_ERROR,
                        passed=False,
                        execution_time=0.0,
                        error_message=f"Parallel execution failed: {str(e)}",
                        error_type="ExecutionError",
                        extracted_functions=[]
                    ))

        # Sort results by test_id to maintain order
        results.sort(key=lambda r: r.test_id)
        return results

    def _execute_sequential(self, test_cases: List[TestCase], generated_code: str) -> List[TestExecutionResult]:
        """Execute test cases sequentially"""
        results = []

        for i, test_case in enumerate(test_cases):
            print(f"  Test {i+1}/{len(test_cases)}: {test_case.test_id}")
            result = self.execute_test_case(test_case, generated_code)
            results.append(result)

        return results

    def _analyze_extracted_functions(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Analyze extracted functions across all test results"""

        all_functions = []
        for result in results:
            all_functions.extend(result.extracted_functions)

        if not all_functions:
            return {"total_functions": 0}

        # Function name analysis
        function_names = [f.name for f in all_functions]
        unique_names = set(function_names)

        # Function signature analysis
        signatures = [f.signature for f in all_functions]
        unique_signatures = set(signatures)

        return {
            "total_functions": len(all_functions),
            "unique_names": len(unique_names),
            "unique_signatures": len(unique_signatures),
            "function_names": list(unique_names),
            "avg_function_length": sum(len(f.body) for f in all_functions) / len(all_functions),
            "most_common_name": max(set(function_names), key=function_names.count) if function_names else None
        }

    def create_humaneval_test_cases(self, problem_data: Dict[str, Any]) -> List[TestCase]:
        """
        Create test cases from HumanEval problem data format.
        """
        test_cases = []

        # Extract test from problem data
        test_code = problem_data.get('test', '')
        task_id = problem_data.get('task_id', 'unknown')

        if test_code:
            test_case = TestCase(
                test_id=f"{task_id}_main",
                test_code=test_code,
                timeout=self.default_timeout,
                description=f"Main test for {task_id}"
            )
            test_cases.append(test_case)

        return test_cases

    def generate_detailed_report(self, batch_result: BatchExecutionResult) -> str:
        """Generate detailed text report of batch execution results"""

        report = []
        report.append("=" * 80)
        report.append("ENHANCED TEST EXECUTION REPORT - Sprint 1.2")
        report.append("=" * 80)
        report.append("")

        # Summary
        report.append("SUMMARY:")
        report.append(f"  Total Tests: {batch_result.total_tests}")
        report.append(f"  Passed: {batch_result.passed_tests} ({batch_result.pass_rate:.1%})")
        report.append(f"  Failed: {batch_result.failed_tests}")
        report.append(f"  Timeouts: {batch_result.timeout_tests}")
        report.append(f"  Errors: {batch_result.error_tests}")
        report.append(f"  Execution Time: {batch_result.execution_time:.2f}s")
        report.append("")

        # Error Analysis
        if batch_result.summary.get("error_distribution"):
            report.append("ERROR DISTRIBUTION:")
            errors = batch_result.summary["error_distribution"]
            for error_type, count in errors.items():
                if count > 0:
                    report.append(f"  {error_type}: {count}")
            report.append("")

        # Function Analysis
        if batch_result.summary.get("function_analysis"):
            func_analysis = batch_result.summary["function_analysis"]
            report.append("FUNCTION ANALYSIS:")
            report.append(f"  Total Functions Extracted: {func_analysis.get('total_functions', 0)}")
            report.append(f"  Unique Function Names: {func_analysis.get('unique_names', 0)}")
            if func_analysis.get('most_common_name'):
                report.append(f"  Most Common Function: {func_analysis['most_common_name']}")
            report.append("")

        # Individual Results
        report.append("INDIVIDUAL TEST RESULTS:")
        for result in batch_result.individual_results:
            status = "✅ PASS" if result.passed else f"❌ {result.result.value.upper()}"
            report.append(f"  {result.test_id}: {status} ({result.execution_time:.3f}s)")

            if not result.passed and result.error_message:
                report.append(f"    Error: {result.error_message}")

            if result.extracted_functions:
                func_names = [f.name for f in result.extracted_functions]
                report.append(f"    Functions: {', '.join(func_names)}")

        report.append("")
        report.append("=" * 80)

        return '\n'.join(report)