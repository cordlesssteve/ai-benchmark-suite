#!/usr/bin/env python3
"""
Docker-Based Test Execution Engine - Sprint 2.0

Production-grade security using Docker container isolation:
- Docker container execution environment
- Network isolation and secure volume mounting
- Container resource limits and cleanup
- Complete isolation from host system
"""

import os
import sys
import time
import json
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
# Using subprocess to call Docker CLI directly instead of Python docker library

from .enhanced_test_executor import (
    EnhancedTestExecutor, TestCase, TestExecutionResult, BatchExecutionResult,
    TestResult, FunctionInfo
)


@dataclass
class ContainerConfig:
    """Configuration for Docker container execution"""
    image: str = "python:3.11-slim"
    memory_limit: str = "512m"
    cpu_limit: str = "1.0"
    timeout: int = 300
    network_mode: str = "none"  # No network access
    read_only_filesystem: bool = True
    no_new_privileges: bool = True
    user: str = "nobody"  # Non-root user
    working_dir: str = "/app"

    # Security settings
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=list)
    security_opt: List[str] = field(default_factory=lambda: ["no-new-privileges:true"])

    # Resource limits
    pids_limit: int = 50
    ulimits: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"Name": "nofile", "Soft": 1024, "Hard": 1024},
        {"Name": "nproc", "Soft": 50, "Hard": 50}
    ])


class DockerTestExecutor(EnhancedTestExecutor):
    """
    Container-based test execution engine using Docker for maximum isolation.

    Features:
    - Complete filesystem isolation
    - Network isolation (no network access)
    - Resource limits (memory, CPU, processes)
    - Read-only filesystem with writable tmp
    - Non-root execution
    - Security hardening
    """

    def __init__(self, container_config: Optional[ContainerConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.container_config = container_config or ContainerConfig()

        # Verify Docker CLI is available
        try:
            self._verify_docker_setup()
            print(f"🐳 Docker CLI verified successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to verify Docker setup: {e}")

        # Track created containers for cleanup
        self.created_containers: List[str] = []

    def _verify_docker_setup(self):
        """Verify Docker CLI is accessible and image is available"""
        try:
            # Test Docker CLI connection
            result = subprocess.run(['docker', 'version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"Docker CLI not accessible: {result.stderr}")

            # Check if base image exists, pull if needed
            result = subprocess.run(
                ['docker', 'images', '-q', self.container_config.image],
                capture_output=True, text=True, timeout=10
            )

            if not result.stdout.strip():
                print(f"📥 Pulling Docker image {self.container_config.image}...")
                pull_result = subprocess.run(
                    ['docker', 'pull', self.container_config.image],
                    capture_output=True, text=True, timeout=300
                )
                if pull_result.returncode != 0:
                    raise RuntimeError(f"Failed to pull image: {pull_result.stderr}")
                print(f"✅ Docker image {self.container_config.image} pulled successfully")
            else:
                print(f"✅ Docker image {self.container_config.image} available")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker CLI timeout")
        except Exception as e:
            raise RuntimeError(f"Docker setup verification failed: {e}")

    def execute_test_case(self, test_case: TestCase, generated_code: str) -> TestExecutionResult:
        """
        Execute a single test case in an isolated Docker container.
        """
        start_time = time.time()

        # Extract functions from generated code
        extracted_functions = self.extract_functions(generated_code)

        # Create test program
        test_program = self._create_test_program(generated_code, test_case)

        try:
            # Execute in Docker container
            result = self._execute_in_container(test_program, test_case.timeout)

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
                    "code_length": len(generated_code),
                    "execution_mode": "docker_container",
                    "container_image": self.container_config.image
                }
            )

        except Exception as e:
            return TestExecutionResult(
                test_id=test_case.test_id,
                result=TestResult.RUNTIME_ERROR,
                passed=False,
                execution_time=time.time() - start_time,
                error_message=f"Container execution failed: {str(e)}",
                error_type="ContainerExecutionError",
                extracted_functions=extracted_functions,
                metadata={"execution_mode": "docker_container_failed"}
            )

    def _execute_in_container(self, test_program: str, timeout: float) -> Dict[str, Any]:
        """Execute test program in isolated Docker container"""

        container_id = None
        temp_dir = None

        try:
            # Create temporary directory for code
            temp_dir = tempfile.mkdtemp(prefix="docker_test_")

            # Write test program to file with world-readable permissions
            test_file = Path(temp_dir) / "test_program.py"
            with open(test_file, 'w') as f:
                f.write(test_program)

            # Make file readable by all users (for container access)
            os.chmod(test_file, 0o644)
            os.chmod(temp_dir, 0o755)

            # Run container with Docker CLI
            container_id = self._run_container(temp_dir, timeout)

            print(f"🐳 Running test in container {container_id[:12]}...")

            try:
                # Wait for completion and get result
                result = self._wait_for_container(container_id, timeout)
                return result
            finally:
                # Always cleanup container
                self._cleanup_container(container_id)

        except Exception as e:
            return {
                "status": "container_error",
                "error": f"Container execution failed: {str(e)}",
                "output": ""
            }

        finally:
            # Cleanup container
            if container_id:
                self._cleanup_container(container_id)

            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"⚠️ Failed to cleanup temp dir {temp_dir}: {e}")

    def _run_container(self, host_code_dir: str, timeout: float) -> str:
        """Run Docker container with security hardening using CLI"""

        container_name = f"bigcode-test-{uuid.uuid4().hex[:8]}"

        # Build Docker run command with security hardening
        docker_cmd = [
            'docker', 'run',
            '--name', container_name,
            '--detach',
            '--workdir', self.container_config.working_dir,
            '--user', self.container_config.user,
            '--network', self.container_config.network_mode,
            '--read-only',  # Read-only filesystem
            '--security-opt', 'no-new-privileges:true',
            '--cap-drop', 'ALL',  # Drop all capabilities
            '--memory', self.container_config.memory_limit,
            '--cpus', self.container_config.cpu_limit,
            '--pids-limit', str(self.container_config.pids_limit),
            '--ulimit', 'nofile=1024:1024',
            '--ulimit', 'nproc=50:50',
            '--env', 'PYTHONPATH=/app',
            '--env', 'PYTHONUNBUFFERED=1',
            '--env', 'HOME=/tmp',
            '--volume', f'{host_code_dir}:/app:ro',  # Read-only mount
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=100m',  # Writable tmp
            self.container_config.image,
            'python', '/app/test_program.py'
        ]

        try:
            # Run the container
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")

            container_id = result.stdout.strip()

            # Track for cleanup
            self.created_containers.append(container_id)

            print(f"🐳 Started container {container_id[:12]} with security hardening")
            return container_id

        except subprocess.TimeoutExpired:
            raise RuntimeError("Container start timeout")
        except Exception as e:
            raise RuntimeError(f"Failed to run container: {e}")

    def _wait_for_container(self, container_id: str, timeout: float) -> Dict[str, Any]:
        """Wait for container completion and get results"""

        try:
            # Wait for container to complete
            wait_cmd = ['docker', 'wait', container_id]
            wait_result = subprocess.run(wait_cmd, capture_output=True, text=True, timeout=timeout + 10)

            if wait_result.returncode != 0:
                return {
                    "status": "container_error",
                    "error": f"Failed to wait for container: {wait_result.stderr}",
                    "output": ""
                }

            exit_code = int(wait_result.stdout.strip())

            # Get container logs
            logs_cmd = ['docker', 'logs', container_id]
            logs_result = subprocess.run(logs_cmd, capture_output=True, text=True, timeout=10)

            output = logs_result.stdout + logs_result.stderr

            if exit_code == 0:
                return {
                    "status": "passed",
                    "output": output,
                    "exit_code": exit_code
                }
            else:
                return {
                    "status": "runtime_error",
                    "output": output,
                    "error": f"Container exited with code {exit_code}",
                    "exit_code": exit_code
                }

        except subprocess.TimeoutExpired:
            # Container timed out, try to stop it
            self._stop_container(container_id)
            return {
                "status": "timeout",
                "error": f"Container execution timed out after {timeout}s",
                "output": ""
            }
        except Exception as e:
            return {
                "status": "container_error",
                "error": f"Container wait failed: {str(e)}",
                "output": ""
            }

    def _stop_container(self, container_id: str):
        """Stop a running container"""
        try:
            subprocess.run(['docker', 'stop', container_id], capture_output=True, timeout=10)
            print(f"🛑 Stopped container {container_id[:12]}")
        except Exception as e:
            print(f"⚠️ Failed to stop container {container_id[:12]}: {e}")

    def _cleanup_container(self, container_id: str):
        """Clean up Docker container using CLI"""
        try:
            # Check if container exists and is running
            inspect_cmd = ['docker', 'inspect', container_id, '--format', '{{.State.Status}}']
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=5)

            if inspect_result.returncode == 0:
                status = inspect_result.stdout.strip()

                # Stop if running
                if status == "running":
                    subprocess.run(['docker', 'stop', container_id], capture_output=True, timeout=10)

                # Remove container (if not auto-removed)
                subprocess.run(['docker', 'rm', '-f', container_id], capture_output=True, timeout=5)

            # Remove from tracking
            if container_id in self.created_containers:
                self.created_containers.remove(container_id)

            print(f"🧹 Cleaned up container {container_id[:12]}")

        except subprocess.TimeoutExpired:
            print(f"⚠️ Cleanup timeout for container {container_id[:12]}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup container {container_id[:12]}: {e}")

    def execute_test_batch(self, test_cases: List[TestCase], generated_code: str,
                          parallel: bool = False) -> BatchExecutionResult:
        """
        Execute multiple test cases in Docker containers.

        Note: Parallel execution creates multiple containers simultaneously.
        """
        print(f"🐳 Executing batch of {len(test_cases)} test cases in Docker containers...")

        # For Docker execution, we typically run sequentially to avoid
        # overwhelming the Docker daemon, unless explicitly requested
        if parallel and len(test_cases) > 1:
            print("⚠️ Parallel container execution - monitor Docker resources")

        # Use parent implementation but with container-based execution
        return super().execute_test_batch(test_cases, generated_code, parallel=False)

    def cleanup_all_containers(self):
        """Clean up all created containers"""
        print(f"🧹 Cleaning up {len(self.created_containers)} containers...")

        for container_id in self.created_containers.copy():
            self._cleanup_container(container_id)

        self.created_containers.clear()
        print("✅ All containers cleaned up")

    def get_container_stats(self) -> Dict[str, Any]:
        """Get statistics about container usage"""
        return {
            "total_containers_created": len(self.created_containers),
            "active_containers": len(self.created_containers),
            "container_config": {
                "image": self.container_config.image,
                "memory_limit": self.container_config.memory_limit,
                "cpu_limit": self.container_config.cpu_limit,
                "network_mode": self.container_config.network_mode,
                "read_only": self.container_config.read_only_filesystem
            }
        }

    @contextmanager
    def docker_batch_context(self):
        """Context manager for batch Docker operations with cleanup"""
        try:
            print("🐳 Starting Docker batch execution context")
            yield self
        finally:
            print("🧹 Docker batch execution complete, cleaning up...")
            self.cleanup_all_containers()

    def create_secure_dockerfile(self, output_path: Path) -> Path:
        """Create a hardened Dockerfile for custom images"""

        dockerfile_content = '''
# Hardened Python container for code execution
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r coderunner && useradd -r -g coderunner coderunner

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    && rm -rf /var/lib/apt/lists/*

# Set up secure directories
RUN mkdir -p /app /tmp/workspace && \\
    chown -R coderunner:coderunner /tmp/workspace && \\
    chmod 755 /app && \\
    chmod 1777 /tmp/workspace

# Security: Remove unnecessary packages and files
RUN apt-get autoremove -y && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set security limits
RUN echo "coderunner soft nproc 50" >> /etc/security/limits.conf && \\
    echo "coderunner hard nproc 50" >> /etc/security/limits.conf && \\
    echo "coderunner soft nofile 1024" >> /etc/security/limits.conf && \\
    echo "coderunner hard nofile 1024" >> /etc/security/limits.conf

# Switch to non-root user
USER coderunner

# Set working directory
WORKDIR /app

# Default command
CMD ["python"]
'''

        dockerfile_path = output_path / "Dockerfile.secure"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        print(f"📝 Created secure Dockerfile: {dockerfile_path}")
        return dockerfile_path

    def build_secure_image(self, tag: str = "bigcode-secure:latest") -> str:
        """Build a hardened Docker image for code execution using CLI"""

        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Dockerfile
            dockerfile_path = self.create_secure_dockerfile(temp_path)

            print(f"🔨 Building secure Docker image: {tag}")

            # Build image using Docker CLI
            build_cmd = [
                'docker', 'build',
                '-t', tag,
                '-f', str(dockerfile_path),
                '--rm',
                '--pull',
                str(temp_path)
            ]

            try:
                result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=600)

                if result.returncode != 0:
                    raise RuntimeError(f"Image build failed: {result.stderr}")

                # Print build output
                print(result.stdout)

                print(f"✅ Secure Docker image built: {tag}")
                return tag

            except subprocess.TimeoutExpired:
                raise RuntimeError("Docker build timeout")
            except Exception as e:
                raise RuntimeError(f"Failed to build image: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup_all_containers()