#!/usr/bin/env python3
"""
Multi-Language Code Execution Environment (Sprint 2.2)

Executes code in different programming languages using containerized environments
for secure, isolated execution with language-specific toolchains.
"""

import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

try:
    from .language_detector import ProgrammingLanguage, LanguageDetector, get_docker_image_for_language
except ImportError:
    # For standalone testing
    from language_detector import ProgrammingLanguage, LanguageDetector, get_docker_image_for_language


@dataclass
class ExecutionResult:
    """Result of code execution"""
    language: ProgrammingLanguage
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    compile_output: Optional[str] = None
    error_message: Optional[str] = None


class ExecutionMode(Enum):
    """Execution modes"""
    DIRECT = "direct"  # Direct execution on host
    DOCKER = "docker"  # Docker container execution
    SAFE = "safe"      # Maximum security container execution


class LanguageExecutor:
    """Base class for language-specific executors"""

    def __init__(self, language: ProgrammingLanguage, mode: ExecutionMode = ExecutionMode.DOCKER):
        self.language = language
        self.mode = mode
        self.docker_image = get_docker_image_for_language(language)

    def execute(self, code: str, test_code: Optional[str] = None,
                timeout: int = 30) -> ExecutionResult:
        """Execute code with optional test code"""
        raise NotImplementedError

    def compile(self, code: str, output_path: Path) -> ExecutionResult:
        """Compile code if needed"""
        raise NotImplementedError

    def _create_temp_file(self, code: str, extension: str) -> Path:
        """Create temporary file with code"""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix=extension, delete=False, encoding='utf-8'
        )
        temp_file.write(code)
        temp_file.close()
        return Path(temp_file.name)

    def _run_command(self, command: List[str], cwd: Optional[Path] = None,
                    timeout: int = 30, env: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """Run command with timeout and capture output"""
        start_time = time.time()

        try:
            if self.mode == ExecutionMode.DOCKER:
                command = self._wrap_docker_command(command, cwd)

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                language=self.language,
                success=(result.returncode == 0),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                language=self.language,
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                execution_time=timeout,
                error_message="Timeout"
            )
        except Exception as e:
            return ExecutionResult(
                language=self.language,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def _wrap_docker_command(self, command: List[str], cwd: Optional[Path] = None) -> List[str]:
        """Wrap command for Docker execution"""
        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "none",  # No network access
            "--memory", "512m",   # Memory limit
            "--cpus", "0.5",      # CPU limit
            "--user", "1000:1000", # Non-root user
            "--workdir", "/workspace",
        ]

        # Mount working directory if specified
        if cwd:
            docker_cmd.extend(["-v", f"{cwd}:/workspace:ro"])

        docker_cmd.append(self.docker_image)
        docker_cmd.extend(command)

        return docker_cmd


class PythonExecutor(LanguageExecutor):
    """Python code executor"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.DOCKER):
        super().__init__(ProgrammingLanguage.PYTHON, mode)

    def execute(self, code: str, test_code: Optional[str] = None,
                timeout: int = 30) -> ExecutionResult:
        """Execute Python code"""
        # Combine code and test code
        full_code = code
        if test_code:
            full_code += "\n\n" + test_code

        # Create temporary file
        temp_file = self._create_temp_file(full_code, '.py')

        try:
            # Execute Python code
            result = self._run_command(["python3", str(temp_file)], timeout=timeout)
            return result

        finally:
            # Cleanup
            temp_file.unlink(missing_ok=True)

    def compile(self, code: str, output_path: Path) -> ExecutionResult:
        """Python doesn't need compilation"""
        return ExecutionResult(
            language=self.language,
            success=True,
            exit_code=0,
            stdout="Python compilation not needed",
            stderr="",
            execution_time=0.0
        )


class JavaScriptExecutor(LanguageExecutor):
    """JavaScript code executor"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.DOCKER):
        super().__init__(ProgrammingLanguage.JAVASCRIPT, mode)

    def execute(self, code: str, test_code: Optional[str] = None,
                timeout: int = 30) -> ExecutionResult:
        """Execute JavaScript code"""
        # Combine code and test code
        full_code = code
        if test_code:
            full_code += "\n\n" + test_code

        # Create temporary file
        temp_file = self._create_temp_file(full_code, '.js')

        try:
            # Execute with Node.js
            result = self._run_command(["node", str(temp_file)], timeout=timeout)
            return result

        finally:
            # Cleanup
            temp_file.unlink(missing_ok=True)

    def compile(self, code: str, output_path: Path) -> ExecutionResult:
        """JavaScript doesn't need compilation"""
        return ExecutionResult(
            language=self.language,
            success=True,
            exit_code=0,
            stdout="JavaScript compilation not needed",
            stderr="",
            execution_time=0.0
        )


class JavaExecutor(LanguageExecutor):
    """Java code executor"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.DOCKER):
        super().__init__(ProgrammingLanguage.JAVA, mode)

    def execute(self, code: str, test_code: Optional[str] = None,
                timeout: int = 30) -> ExecutionResult:
        """Execute Java code"""
        # Combine code and test code
        full_code = code
        if test_code:
            full_code += "\n\n" + test_code

        # Create temporary directory for Java files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Java file (assuming class name is "Problem")
            java_file = temp_path / "Problem.java"
            with open(java_file, 'w') as f:
                f.write(full_code)

            # Compile
            compile_result = self._run_command(
                ["javac", str(java_file)],
                cwd=temp_path,
                timeout=timeout
            )

            if not compile_result.success:
                return ExecutionResult(
                    language=self.language,
                    success=False,
                    exit_code=compile_result.exit_code,
                    stdout=compile_result.stdout,
                    stderr=compile_result.stderr,
                    execution_time=compile_result.execution_time,
                    compile_output=compile_result.stderr,
                    error_message="Compilation failed"
                )

            # Execute
            execute_result = self._run_command(
                ["java", "Problem"],
                cwd=temp_path,
                timeout=timeout
            )

            execute_result.compile_output = compile_result.stderr
            return execute_result

    def compile(self, code: str, output_path: Path) -> ExecutionResult:
        """Compile Java code"""
        java_file = output_path / "Problem.java"
        with open(java_file, 'w') as f:
            f.write(code)

        return self._run_command(["javac", str(java_file)], cwd=output_path)


class CppExecutor(LanguageExecutor):
    """C++ code executor"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.DOCKER):
        super().__init__(ProgrammingLanguage.CPP, mode)

    def execute(self, code: str, test_code: Optional[str] = None,
                timeout: int = 30) -> ExecutionResult:
        """Execute C++ code"""
        # Combine code and test code
        full_code = code
        if test_code:
            full_code += "\n\n" + test_code

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create C++ file
            cpp_file = temp_path / "program.cpp"
            executable = temp_path / "program"

            with open(cpp_file, 'w') as f:
                f.write(full_code)

            # Compile
            compile_result = self._run_command(
                ["g++", "-std=c++17", str(cpp_file), "-o", str(executable)],
                cwd=temp_path,
                timeout=timeout
            )

            if not compile_result.success:
                return ExecutionResult(
                    language=self.language,
                    success=False,
                    exit_code=compile_result.exit_code,
                    stdout=compile_result.stdout,
                    stderr=compile_result.stderr,
                    execution_time=compile_result.execution_time,
                    compile_output=compile_result.stderr,
                    error_message="Compilation failed"
                )

            # Execute
            execute_result = self._run_command(
                [str(executable)],
                cwd=temp_path,
                timeout=timeout
            )

            execute_result.compile_output = compile_result.stderr
            return execute_result

    def compile(self, code: str, output_path: Path) -> ExecutionResult:
        """Compile C++ code"""
        cpp_file = output_path / "program.cpp"
        executable = output_path / "program"

        with open(cpp_file, 'w') as f:
            f.write(code)

        return self._run_command(
            ["g++", "-std=c++17", str(cpp_file), "-o", str(executable)],
            cwd=output_path
        )


class MultiLanguageExecutor:
    """Main multi-language execution coordinator"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.DOCKER):
        self.mode = mode
        self.detector = LanguageDetector()

        # Initialize language-specific executors
        self.executors = {
            ProgrammingLanguage.PYTHON: PythonExecutor(mode),
            ProgrammingLanguage.JAVASCRIPT: JavaScriptExecutor(mode),
            ProgrammingLanguage.JAVA: JavaExecutor(mode),
            ProgrammingLanguage.CPP: CppExecutor(mode),
        }

    def execute_code(self, code: str, language: Optional[ProgrammingLanguage] = None,
                    test_code: Optional[str] = None, timeout: int = 30) -> ExecutionResult:
        """
        Execute code in the appropriate language environment.

        Args:
            code: The source code to execute
            language: Optional language hint (will auto-detect if not provided)
            test_code: Optional test code to run with the main code
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with execution details
        """
        # Detect language if not provided
        if language is None:
            detection_result = self.detector.detect_language(code)
            language = detection_result.language

        # Check if language is supported
        if language not in self.executors:
            return ExecutionResult(
                language=language,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Language {language.value} not supported yet",
                execution_time=0.0,
                error_message=f"Unsupported language: {language.value}"
            )

        # Execute with appropriate executor
        executor = self.executors[language]
        return executor.execute(code, test_code, timeout)

    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Get list of supported languages"""
        return list(self.executors.keys())

    def add_executor(self, language: ProgrammingLanguage, executor: LanguageExecutor):
        """Add support for a new language"""
        self.executors[language] = executor


# Testing and examples
if __name__ == "__main__":
    executor = MultiLanguageExecutor(ExecutionMode.DIRECT)  # Use direct mode for testing

    # Test cases
    test_cases = [
        (ProgrammingLanguage.PYTHON, """
def add(a, b):
    return a + b

print(add(2, 3))
        """),

        (ProgrammingLanguage.JAVASCRIPT, """
function add(a, b) {
    return a + b;
}

console.log(add(2, 3));
        """),

        (ProgrammingLanguage.JAVA, """
public class Problem {
    public static int add(int a, int b) {
        return a + b;
    }

    public static void main(String[] args) {
        System.out.println(add(2, 3));
    }
}
        """),

        (ProgrammingLanguage.CPP, """
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    cout << add(2, 3) << endl;
    return 0;
}
        """),
    ]

    print("🚀 Multi-Language Execution Test Results:")
    print("=" * 60)

    for language, code in test_cases:
        print(f"\n🔸 Testing {language.value.upper()}:")
        print("-" * 30)

        result = executor.execute_code(code, language, timeout=10)

        print(f"Success: {result.success}")
        print(f"Exit Code: {result.exit_code}")
        print(f"Execution Time: {result.execution_time:.3f}s")

        if result.stdout:
            print(f"Output: {result.stdout.strip()}")

        if result.stderr:
            print(f"Errors: {result.stderr.strip()}")

        if result.compile_output:
            print(f"Compile Output: {result.compile_output.strip()}")

    print(f"\n✅ Supported Languages: {[lang.value for lang in executor.get_supported_languages()]}")