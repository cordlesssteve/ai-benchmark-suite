#!/usr/bin/env python3
"""
Safe BigCode Harness Integration with Security Framework

Sprint 1.1: Adds safety measures on top of RealBigCodeAdapter:
- Temporary directory isolation
- Timeout mechanisms and resource limits
- Error handling for malicious code attempts
- Safe execution environment for generated code

Sprint 1.2: Enhanced with comprehensive test execution engine:
- Function extraction from generated code
- Comprehensive test case execution framework
- Detailed error reporting for test failures
- Multiple test case batching support
"""

import os
import sys
import time
import tempfile
import subprocess
import json
import shutil
import resource
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager

from .real_bigcode_adapter import RealBigCodeAdapter, RealBigCodeResult
from .ollama_interface import OllamaInterface
from .enhanced_test_executor import EnhancedTestExecutor, TestCase, BatchExecutionResult
from .docker_test_executor import DockerTestExecutor, ContainerConfig

@dataclass
class SafeExecutionConfig:
    """Configuration for safe code execution"""
    max_execution_time: int = 300  # 5 minutes per evaluation
    max_memory_mb: int = 2048      # 2GB memory limit
    max_cpu_time: int = 180        # 3 minutes CPU time
    max_file_size_mb: int = 100    # 100MB file size limit
    max_processes: int = 10        # Max concurrent processes
    temp_dir_prefix: str = "safe_bigcode_"
    cleanup_on_error: bool = True

    # Sprint 1.2: Test execution settings
    test_timeout: float = 3.0      # Per-test timeout
    max_test_workers: int = 4      # Parallel test execution
    enable_function_extraction: bool = True  # Extract functions from generated code
    enable_detailed_reporting: bool = True   # Generate detailed test reports
    enable_parallel_testing: bool = True     # Use parallel test execution

    # Sprint 2.0: Container execution settings
    use_container_isolation: bool = False    # Use Docker containers for maximum isolation
    container_image: str = "python:3.11-slim"  # Docker image for containers
    container_memory_limit: str = "512m"    # Container memory limit
    container_cpu_limit: str = "0.5"        # Container CPU limit (cores)
    container_timeout: int = 300             # Container execution timeout
    container_network_isolation: bool = True # Disable network access in containers

class SafeBigCodeAdapter(RealBigCodeAdapter):
    """
    Safe BigCode harness integration with security isolation.

    Extends RealBigCodeAdapter with Sprint 1.1 safety features:
    - Isolated temporary directories for each evaluation
    - Resource limits (memory, CPU, file size)
    - Timeout protection
    - Malicious code detection and prevention

    Sprint 1.2 enhancements:
    - Function extraction from generated code
    - Comprehensive test case execution framework
    - Detailed error reporting for test failures
    - Multiple test case batching support
    """

    def __init__(self, project_root: Path, safety_config: Optional[SafeExecutionConfig] = None):
        super().__init__(project_root)
        self.safety_config = safety_config or SafeExecutionConfig()
        self.temp_dirs_created = []  # Track for cleanup

        # Sprint 1.2 & 2.0: Initialize test executor (container or enhanced)
        if self.safety_config.use_container_isolation:
            # Sprint 2.0: Use Docker containers for maximum isolation
            container_config = ContainerConfig(
                image=self.safety_config.container_image,
                memory_limit=self.safety_config.container_memory_limit,
                cpu_limit=self.safety_config.container_cpu_limit,
                timeout=self.safety_config.container_timeout,
                network_mode="none" if self.safety_config.container_network_isolation else "bridge"
            )
            self.test_executor = DockerTestExecutor(
                container_config=container_config,
                default_timeout=self.safety_config.test_timeout,
                max_workers=self.safety_config.max_test_workers
            )
            self.execution_mode = "docker_container"
            print(f"🐳 Using Docker container isolation: {self.safety_config.container_image}")
        else:
            # Sprint 1.2: Use enhanced multiprocess executor
            self.test_executor = EnhancedTestExecutor(
                default_timeout=self.safety_config.test_timeout,
                max_workers=self.safety_config.max_test_workers
            )
            self.execution_mode = "multiprocess"
            print(f"🔒 Using multiprocess isolation")

    def run_evaluation(self, task: str, model_name: str, model_interface: OllamaInterface, **kwargs) -> RealBigCodeResult:
        """
        Run safe BigCode evaluation with security isolation.
        """
        start_time = time.time()

        # Create isolated execution environment
        with self._create_safe_environment() as safe_env:
            try:
                # Enhanced safety checks
                self._validate_task_safety(task)
                self._validate_model_safety(model_name)

                # Run evaluation in safe environment
                result = self._run_safe_evaluation(
                    task, model_name, model_interface, safe_env, **kwargs
                )

                result.execution_time = time.time() - start_time
                return result

            except Exception as e:
                return RealBigCodeResult(
                    task=task,
                    model=model_name,
                    metrics={},
                    raw_output="",
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message=f"Safe execution failed: {str(e)}"
                )

    @contextmanager
    def _create_safe_environment(self):
        """Create isolated temporary environment for safe code execution"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(
            prefix=self.safety_config.temp_dir_prefix,
            dir="/tmp"  # Use system temp directory
        )
        self.temp_dirs_created.append(temp_dir)

        try:
            # Set up isolated environment
            safe_env = {
                "temp_dir": Path(temp_dir),
                "bigcode_dir": self._setup_isolated_bigcode(temp_dir),
                "original_cwd": os.getcwd()
            }

            print(f"🔒 Created safe execution environment: {temp_dir}")
            yield safe_env

        finally:
            # Cleanup temporary directory
            if self.safety_config.cleanup_on_error:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Cleaned up safe environment: {temp_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to cleanup {temp_dir}: {e}")

    def _setup_isolated_bigcode(self, temp_dir: str) -> Path:
        """Set up isolated BigCode harness copy for safe execution"""
        isolated_bigcode = Path(temp_dir) / "bigcode"

        # Create minimal BigCode harness structure
        isolated_bigcode.mkdir(parents=True)

        # Copy essential BigCode files (selective copying for security)
        essential_files = [
            "main.py",
            "bigcode_eval",  # Directory
            "requirements.txt"
        ]

        for item in essential_files:
            src = self.bigcode_dir / item
            dst = isolated_bigcode / item

            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

        return isolated_bigcode

    def _validate_task_safety(self, task: str):
        """Validate that the task is safe to execute"""
        # List of dangerous tasks/patterns
        dangerous_patterns = [
            "__import__",
            "eval(",
            "exec(",
            "subprocess",
            "os.system",
            "file:/",
            "http://",
            "https://",
        ]

        task_lower = task.lower()
        for pattern in dangerous_patterns:
            if pattern in task_lower:
                raise ValueError(f"Task contains potentially dangerous pattern: {pattern}")

    def _validate_model_safety(self, model_name: str):
        """Validate that the model name is safe"""
        # Prevent path traversal and injection attacks
        if any(char in model_name for char in ["../", "../../", ";", "&", "|", "`"]):
            raise ValueError(f"Model name contains unsafe characters: {model_name}")

    def _run_safe_evaluation(self, task: str, model_name: str, model_interface: OllamaInterface,
                           safe_env: Dict[str, Any], **kwargs) -> RealBigCodeResult:
        """Run BigCode evaluation with enhanced safety measures"""

        temp_dir = safe_env["temp_dir"]
        isolated_bigcode = safe_env["bigcode_dir"]

        try:
            # Create adapter file in isolated environment
            adapter_file = self._create_safe_ollama_adapter(model_name, isolated_bigcode)

            # Execute BigCode harness with safety measures
            result = self._execute_safe_bigcode_harness(
                task, adapter_file, isolated_bigcode, **kwargs
            )

            # Sprint 1.2: Enhanced testing mode
            use_enhanced_testing = kwargs.get('enhanced_testing', self.safety_config.enable_detailed_reporting)

            if result["success"]:
                base_result = RealBigCodeResult(
                    task=task,
                    model=model_name,
                    metrics=result["metrics"],
                    raw_output=result["output"],
                    execution_time=0.0,  # Will be set by caller
                    success=True
                )

                # Add enhanced testing if enabled
                if use_enhanced_testing:
                    enhanced_result = self._run_enhanced_testing(task, model_name, model_interface, isolated_bigcode, **kwargs)
                    base_result = self._merge_enhanced_results(base_result, enhanced_result)

                return base_result
            else:
                return RealBigCodeResult(
                    task=task,
                    model=model_name,
                    metrics={},
                    raw_output=result["output"],
                    execution_time=0.0,
                    success=False,
                    error_message=result.get("error", "Safe BigCode execution failed")
                )

        except Exception as e:
            return RealBigCodeResult(
                task=task,
                model=model_name,
                metrics={},
                raw_output="",
                execution_time=0.0,
                success=False,
                error_message=f"Safe execution error: {str(e)}"
            )

    def _create_safe_ollama_adapter(self, model_name: str, isolated_bigcode: Path) -> Path:
        """Create Ollama adapter in isolated environment with safety checks"""

        # Validate model name again
        self._validate_model_safety(model_name)

        adapter_file = isolated_bigcode / "safe_ollama_adapter.py"

        # Use the same adapter code as parent but with safety annotations
        adapter_code = f'''
"""
SAFE Ollama adapter for BigCode evaluation harness
Generated in isolated environment with safety measures
"""

import requests
import json
import logging
import warnings
import os
import signal
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch

# Safety: Set resource limits for this process
import resource

def set_resource_limits():
    """Set conservative resource limits for safety"""
    try:
        # Memory limit: {self.safety_config.max_memory_mb}MB
        resource.setrlimit(resource.RLIMIT_AS, ({self.safety_config.max_memory_mb * 1024 * 1024}, -1))

        # CPU time limit: {self.safety_config.max_cpu_time} seconds
        resource.setrlimit(resource.RLIMIT_CPU, ({self.safety_config.max_cpu_time}, -1))

        # File size limit: {self.safety_config.max_file_size_mb}MB
        resource.setrlimit(resource.RLIMIT_FSIZE, ({self.safety_config.max_file_size_mb * 1024 * 1024}, -1))

        # Process limit
        resource.setrlimit(resource.RLIMIT_NPROC, ({self.safety_config.max_processes}, -1))

        print("🔒 Resource limits set for safe execution")
    except Exception as e:
        print(f"⚠️ Failed to set resource limits: {{e}}")

# Apply safety limits immediately
set_resource_limits()

class SafeOllamaBigCodeModel:
    """Safe BigCode adapter for Ollama models with resource limits"""

    def __init__(self, model_name: str = "{model_name}", base_url: str = "http://localhost:11434"):
        # Validate inputs
        if any(char in model_name for char in ["../", ";", "&", "|"]):
            raise ValueError(f"Unsafe model name: {{model_name}}")

        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        # Set session timeout for safety
        self.session.timeout = 30

        # Create a dummy tokenizer for BigCode compatibility
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium",
                                                         padding_side="left")
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_ids: torch.Tensor, **generation_kwargs) -> torch.Tensor:
        """Generate completions with safety measures"""
        try:
            # Safety check: Limit generation time
            max_time = generation_kwargs.get('max_generation_time', 60)  # 1 minute default
            start_time = time.time()

            # Convert input_ids tensor to text
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

                batch_size = input_ids.shape[0]
                prompts = []

                for i in range(batch_size):
                    prompt = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # Safety: Check prompt for dangerous patterns
                    if self._is_prompt_safe(prompt):
                        prompts.append(prompt)
                    else:
                        print(f"⚠️ Unsafe prompt detected, using safe placeholder")
                        prompts.append("# Safe placeholder prompt")
            else:
                prompts = input_ids if isinstance(input_ids, list) else [str(input_ids)]
                batch_size = len(prompts)

            # Generate responses with Ollama
            responses = []
            for prompt in prompts:
                if time.time() - start_time > max_time:
                    print(f"⏰ Generation timeout reached")
                    responses.append("")
                    continue

                response = self._safe_ollama_generate(prompt, generation_kwargs)
                responses.append(response)

            # Convert responses back to tensor format
            output_ids = []
            for i, response in enumerate(responses):
                full_text = prompts[i] + response
                encoded = self.tokenizer.encode(full_text, return_tensors="pt")
                output_ids.append(encoded.squeeze(0))

            # Pad sequences to same length
            max_length = max(len(seq) for seq in output_ids)
            padded_outputs = []

            for seq in output_ids:
                if len(seq) < max_length:
                    padding = torch.full((max_length - len(seq),),
                                       self.tokenizer.pad_token_id,
                                       dtype=seq.dtype)
                    padded_seq = torch.cat([seq, padding])
                else:
                    padded_seq = seq[:max_length]
                padded_outputs.append(padded_seq)

            return torch.stack(padded_outputs)

        except Exception as e:
            logging.error(f"Safe Ollama generation failed: {{e}}")
            # Return safe dummy output
            dummy_length = 50
            dummy_output = torch.full((batch_size, dummy_length),
                                    self.tokenizer.pad_token_id,
                                    dtype=torch.long)
            return dummy_output

    def _is_prompt_safe(self, prompt: str) -> bool:
        """Check if prompt contains dangerous patterns"""
        dangerous_patterns = [
            "__import__", "eval(", "exec(", "subprocess", "os.system",
            "rm -rf", "del ", "format(", "file://", "http://", "https://"
        ]

        prompt_lower = prompt.lower()
        for pattern in dangerous_patterns:
            if pattern in prompt_lower:
                return False
        return True

    def _safe_ollama_generate(self, prompt: str, generation_kwargs: Dict[str, Any]) -> str:
        """Generate text using Ollama API with safety measures"""
        try:
            # Safety: Limit prompt length
            if len(prompt) > 10000:  # 10K char limit
                prompt = prompt[:10000]
                print("⚠️ Prompt truncated for safety")

            # Extract generation parameters with safe defaults
            max_new_tokens = min(generation_kwargs.get('max_new_tokens', 512), 1024)  # Cap at 1K
            temperature = max(0.0, min(generation_kwargs.get('temperature', 0.1), 2.0))  # 0-2 range
            do_sample = generation_kwargs.get('do_sample', True)

            payload = {{
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {{
                    "temperature": temperature if do_sample else 0.0,
                    "num_predict": max_new_tokens,
                }}
            }}

            response = self.session.post(
                f"{{self.base_url}}/api/generate",
                json=payload,
                timeout=30  # 30 second timeout for safety
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            # Safety: Limit response length
            if len(generated_text) > 5000:  # 5K char limit
                generated_text = generated_text[:5000]
                print("⚠️ Response truncated for safety")

            return generated_text

        except Exception as e:
            logging.error(f"Safe Ollama generation failed: {{e}}")
            return ""

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

# Create a global instance that can be imported
model = SafeOllamaBigCodeModel("{model_name}")
tokenizer = model.tokenizer
'''

        with open(adapter_file, 'w') as f:
            f.write(adapter_code)

        return adapter_file

    def _execute_safe_bigcode_harness(self, task: str, adapter_file: Path,
                                    isolated_bigcode: Path, **kwargs) -> Dict[str, Any]:
        """Execute BigCode harness with enhanced safety measures"""

        # Prepare safe BigCode command
        cmd = [
            str(self.venv_python.absolute()), "main.py",
            "--model", "safe_ollama_adapter",
            "--tasks", task,
            "--allow_code_execution",
            "--batch_size", "1",
            "--modeltype", "causal",
            "--trust_remote_code",
            "--save_generations",
            "--metric_output_path", "safe_results.json"
        ]

        # Add limit with safety cap
        limit = min(kwargs.get('limit', 1), 5)  # Cap at 5 problems for safety
        cmd.extend(["--limit", str(limit)])

        # Add other parameters with safety limits
        max_length = min(kwargs.get('max_length_generation', 512), 1024)  # Cap at 1K tokens
        cmd.extend(["--max_length_generation", str(max_length)])

        try:
            # Set up safe execution environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(isolated_bigcode)

            # Additional safety environment variables
            env["TMPDIR"] = str(isolated_bigcode.parent)  # Isolate temp files
            env["HOME"] = str(isolated_bigcode.parent)    # Isolate home directory

            print(f"🔒 Executing safe BigCode harness: {' '.join(cmd[:4])}...")

            # Execute with safety timeout
            result = subprocess.run(
                cmd,
                cwd=isolated_bigcode,
                capture_output=True,
                text=True,
                timeout=self.safety_config.max_execution_time,  # Safe timeout
                env=env
            )

            if result.returncode == 0:
                metrics = self._parse_safe_bigcode_output(result.stdout, isolated_bigcode)
                return {
                    "success": True,
                    "metrics": metrics,
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "metrics": {},
                    "output": result.stderr,
                    "error": f"Safe BigCode harness failed with return code {result.returncode}"
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "metrics": {},
                "output": "",
                "error": f"Safe BigCode execution timed out after {self.safety_config.max_execution_time}s"
            }
        except Exception as e:
            return {
                "success": False,
                "metrics": {},
                "output": "",
                "error": f"Safe execution error: {str(e)}"
            }
        finally:
            # Cleanup adapter file
            try:
                adapter_file.unlink()
            except:
                pass

            # Cleanup results file
            try:
                results_file = isolated_bigcode / "safe_results.json"
                if results_file.exists():
                    results_file.unlink()
            except:
                pass

    def _parse_safe_bigcode_output(self, output: str, isolated_bigcode: Path) -> Dict[str, Any]:
        """Parse metrics from safe BigCode harness output"""
        metrics = {}

        try:
            # Look for JSON results file in isolated environment
            results_file = isolated_bigcode / "safe_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        metrics = data
                        return metrics

            # Fallback to output parsing (same as parent class)
            metrics = super()._parse_bigcode_output(output)

            # Add safety metadata
            metrics["safety_framework"] = "sprint_1_1"
            metrics["safe_execution"] = True

        except Exception as e:
            metrics = {
                "parse_error": str(e),
                "safety_framework": "sprint_1_1",
                "safe_execution": True
            }

        return metrics

    def cleanup_temp_dirs(self):
        """Manual cleanup of temporary directories if needed"""
        for temp_dir in self.temp_dirs_created:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Manually cleaned up: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup {temp_dir}: {e}")

        self.temp_dirs_created.clear()

    def _run_enhanced_testing(self, task: str, model_name: str, model_interface: OllamaInterface,
                            isolated_bigcode: Path, **kwargs) -> Optional[BatchExecutionResult]:
        """
        Sprint 1.2: Run enhanced test execution with function extraction and detailed reporting.
        """
        try:
            print(f"🧪 Running enhanced testing for {task}...")

            # Generate test code using Ollama
            generated_code = self._generate_test_code(task, model_interface, **kwargs)

            if not generated_code:
                print("⚠️ No code generated for enhanced testing")
                return None

            # Create test cases for the task
            test_cases = self._create_test_cases_for_task(task, isolated_bigcode)

            if not test_cases:
                print("⚠️ No test cases found for enhanced testing")
                return None

            # Execute enhanced test batch with appropriate isolation
            if self.execution_mode == "docker_container":
                print(f"🐳 Executing tests in Docker containers...")
                # For container execution, use context manager for cleanup
                with self.test_executor.docker_batch_context():
                    batch_result = self.test_executor.execute_test_batch(
                        test_cases=test_cases,
                        generated_code=generated_code,
                        parallel=self.safety_config.enable_parallel_testing
                    )
            else:
                print(f"🔒 Executing tests with multiprocess isolation...")
                batch_result = self.test_executor.execute_test_batch(
                    test_cases=test_cases,
                    generated_code=generated_code,
                    parallel=self.safety_config.enable_parallel_testing
                )

            # Generate detailed report if enabled
            if self.safety_config.enable_detailed_reporting:
                report = self.test_executor.generate_detailed_report(batch_result)
                print("📊 Enhanced Testing Report:")
                print(report)

            return batch_result

        except Exception as e:
            print(f"⚠️ Enhanced testing failed: {e}")
            return None

    def _generate_test_code(self, task: str, model_interface: OllamaInterface, **kwargs) -> str:
        """Generate code for the given task using the model interface"""
        try:
            # This is a simplified version - in a real implementation,
            # we would extract the task prompt from the BigCode dataset
            if task == "humaneval":
                # Use a sample HumanEval prompt for testing
                prompt = '''def add_two_numbers(a, b):
    """
    Add two numbers and return the result.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Sum of a and b
    """
'''
                # Generate completion using Ollama
                response = model_interface.generate(prompt, max_length=200)
                return prompt + response.get("response", "")
            else:
                print(f"⚠️ Enhanced testing not implemented for task: {task}")
                return ""

        except Exception as e:
            print(f"⚠️ Code generation failed: {e}")
            return ""

    def _create_test_cases_for_task(self, task: str, isolated_bigcode: Path) -> List[TestCase]:
        """Create test cases for the given task"""
        test_cases = []

        try:
            if task == "humaneval":
                # Sample test case for HumanEval-style problems
                test_case = TestCase(
                    test_id="humaneval_sample_test",
                    test_code='''
# Test the add_two_numbers function
try:
    result = add_two_numbers(2, 3)
    assert result == 5, f"Expected 5, got {result}"

    result = add_two_numbers(-1, 1)
    assert result == 0, f"Expected 0, got {result}"

    result = add_two_numbers(0, 0)
    assert result == 0, f"Expected 0, got {result}"

    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")
    raise
''',
                    timeout=self.safety_config.test_timeout,
                    description="Sample test for add_two_numbers function"
                )
                test_cases.append(test_case)

            return test_cases

        except Exception as e:
            print(f"⚠️ Test case creation failed: {e}")
            return []

    def _merge_enhanced_results(self, base_result: RealBigCodeResult,
                              enhanced_result: Optional[BatchExecutionResult]) -> RealBigCodeResult:
        """
        Merge enhanced testing results with base BigCode results.
        """
        if not enhanced_result:
            return base_result

        # Add enhanced testing metrics to the base result
        enhanced_metrics = {
            "enhanced_testing": {
                "total_tests": enhanced_result.total_tests,
                "passed_tests": enhanced_result.passed_tests,
                "pass_rate": enhanced_result.pass_rate,
                "execution_time": enhanced_result.execution_time,
                "function_analysis": enhanced_result.summary.get("function_analysis", {}),
                "error_distribution": enhanced_result.summary.get("error_distribution", {}),
                "execution_mode": self.execution_mode,  # Sprint 2.0: Track execution mode
                "isolation_level": "docker_container" if self.execution_mode == "docker_container" else "multiprocess"
            }
        }

        # Sprint 2.0: Add container-specific metrics if using Docker
        if self.execution_mode == "docker_container" and hasattr(self.test_executor, 'get_container_stats'):
            enhanced_metrics["container_stats"] = self.test_executor.get_container_stats()

        # Merge with existing metrics
        merged_metrics = {**base_result.metrics, **enhanced_metrics}

        # Update raw output with enhanced testing information
        enhanced_output = base_result.raw_output + f"\n\n--- Enhanced Testing Results ---\n"
        enhanced_output += f"Tests: {enhanced_result.passed_tests}/{enhanced_result.total_tests} passed ({enhanced_result.pass_rate:.1%})\n"

        if enhanced_result.individual_results:
            for result in enhanced_result.individual_results:
                status = "✅ PASS" if result.passed else f"❌ {result.result.value.upper()}"
                enhanced_output += f"  {result.test_id}: {status}\n"

                if result.extracted_functions:
                    func_names = [f.name for f in result.extracted_functions]
                    enhanced_output += f"    Functions: {', '.join(func_names)}\n"

        return RealBigCodeResult(
            task=base_result.task,
            model=base_result.model,
            metrics=merged_metrics,
            raw_output=enhanced_output,
            execution_time=base_result.execution_time,
            success=base_result.success,
            error_message=base_result.error_message
        )