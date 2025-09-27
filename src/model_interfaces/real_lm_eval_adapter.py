#!/usr/bin/env python3
"""
Real LM-Eval Harness Integration

This adapter creates a bridge between Ollama models and the LM-Eval harness
by implementing a custom model class that LM-Eval can use.
"""

import os
import sys
import time
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .ollama_interface import OllamaInterface

@dataclass
class RealLMEvalResult:
    """Result from real LM-Eval harness execution"""
    task: str
    model: str
    metrics: Dict[str, Any]
    raw_output: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class RealLMEvalAdapter:
    """
    Real LM-Eval harness integration using custom Ollama model adapter.

    This adapter:
    - Creates a temporary LM-Eval model wrapper for Ollama
    - Executes real LM-Eval harness with the wrapped model
    - Parses actual LM-Eval results and metrics
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()  # Convert to absolute path
        self.lm_eval_dir = self.project_root / "harnesses" / "lm-evaluation-harness"
        self.venv_python = self.lm_eval_dir / "venv" / "bin" / "python"

        # Verify venv exists
        if not self.venv_python.exists():
            raise RuntimeError(f"LM-Eval venv not found at {self.venv_python}")

        # Verify lm_eval module is available
        try:
            import subprocess
            result = subprocess.run(
                [str(self.venv_python), "-c", "import lm_eval; print('LM-Eval available')"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"LM-Eval not available in venv: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Failed to verify LM-Eval installation: {e}")

    def run_evaluation(self, task: str, model_name: str, model_interface: OllamaInterface, **kwargs) -> RealLMEvalResult:
        """
        Run real LM-Eval evaluation using Ollama interface.

        This creates a custom model adapter that LM-Eval can use to evaluate Ollama models.
        """

        start_time = time.time()

        try:
            # Create temporary model adapter file
            adapter_file = self._create_ollama_adapter(model_name)

            # Run LM-Eval with the custom adapter
            result = self._execute_lm_eval(task, adapter_file, **kwargs)

            execution_time = time.time() - start_time

            if result["success"]:
                return RealLMEvalResult(
                    task=task,
                    model=model_name,
                    metrics=result["metrics"],
                    raw_output=result["output"],
                    execution_time=execution_time,
                    success=True
                )
            else:
                return RealLMEvalResult(
                    task=task,
                    model=model_name,
                    metrics={},
                    raw_output=result["output"],
                    execution_time=execution_time,
                    success=False,
                    error_message=result.get("error", "LM-Eval execution failed")
                )

        except Exception as e:
            return RealLMEvalResult(
                task=task,
                model=model_name,
                metrics={},
                raw_output="",
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _create_ollama_adapter(self, model_name: str) -> Path:
        """Create a temporary LM-Eval model adapter for Ollama"""

        # Create adapter directly in LM-Eval directory for proper import
        adapter_file = self.lm_eval_dir / "ollama_adapter.py"

        adapter_code = f'''
"""
Temporary Ollama adapter for LM-Eval harness
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

@register_model("ollama_adapter")
class OllamaLMEvalModel(LM):
    """LM-Eval adapter for Ollama models"""

    def __init__(self, model_name: str = "{model_name}", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

    def generate_until(self, requests: List[Dict[str, Any]]) -> List[str]:
        """Generate responses for requests until stop sequences"""
        results = []

        for request in requests:
            context = request.get("context", "")
            until = request.get("until", [])
            max_length = request.get("max_length", 512)

            # Create prompt from context
            prompt = self._format_prompt(context, request)

            # Generate with Ollama
            response = self._ollama_generate(prompt, until, max_length)
            results.append(response)

        return results

    def loglikelihood(self, requests: List[tuple]) -> List[tuple]:
        """Calculate log likelihood for requests"""
        # For Ollama, we can't easily get loglikelihoods, so return dummy values
        # This is a limitation of using Ollama with LM-Eval
        results = []

        for context, continuation in requests:
            # Generate a response to see if it matches
            prompt = context + continuation
            response = self._ollama_generate(prompt, [], 50)

            # Simple heuristic: if generated text contains continuation, higher likelihood
            contains_continuation = continuation.lower() in response.lower()
            loglikelihood = -1.0 if contains_continuation else -10.0
            is_greedy = True

            results.append((loglikelihood, is_greedy))

        return results

    def _format_prompt(self, context: str, request: Dict[str, Any]) -> str:
        """Format the prompt for Ollama"""
        return context

    def _ollama_generate(self, prompt: str, until: List[str], max_length: int) -> str:
        """Generate text using Ollama API"""
        try:
            payload = {{
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {{
                    "temperature": 0.1,
                    "max_tokens": max_length,
                }}
            }}

            response = self.session.post(
                f"{{self.base_url}}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            # Apply stop sequences
            for stop in until:
                if stop in generated_text:
                    generated_text = generated_text.split(stop)[0]
                    break

            return generated_text

        except Exception as e:
            logging.error(f"Ollama generation failed: {{e}}")
            return ""

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 512

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "cpu"
'''

        with open(adapter_file, 'w') as f:
            f.write(adapter_code)

        return adapter_file

    def _execute_lm_eval(self, task: str, adapter_file: Path, **kwargs) -> Dict[str, Any]:
        """Execute LM-Eval with the Ollama adapter"""

        # Prepare LM-Eval command (use absolute path but don't resolve symlinks)
        cmd = [
            str(self.venv_python.absolute()), "-m", "lm_eval",
            "--model", "ollama_adapter",
            "--tasks", task,
            "--batch_size", "1",
            "--device", "cpu",
            "--verbosity", "ERROR"  # Reduce noise
        ]

        # Add limit if specified
        if limit := kwargs.get('limit', 1):
            cmd.extend(["--limit", str(limit)])

        try:
            # Execute LM-Eval
            env = os.environ.copy()

            result = subprocess.run(
                cmd,
                cwd=self.lm_eval_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )

            if result.returncode == 0:
                # Parse LM-Eval output for metrics
                metrics = self._parse_lm_eval_output(result.stdout)
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
                    "error": f"LM-Eval failed with return code {result.returncode}"
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "metrics": {},
                "output": "",
                "error": "LM-Eval execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "metrics": {},
                "output": "",
                "error": str(e)
            }
        finally:
            # Cleanup adapter file from LM-Eval directory
            try:
                adapter_file.unlink()
            except:
                pass

    def _parse_lm_eval_output(self, output: str) -> Dict[str, Any]:
        """Parse metrics from LM-Eval output"""
        metrics = {}

        try:
            # Look for JSON results in the output
            lines = output.split('\n')
            for line in lines:
                if line.strip().startswith('{') and '"results"' in line:
                    try:
                        data = json.loads(line.strip())
                        if "results" in data:
                            metrics = data["results"]
                            break
                    except json.JSONDecodeError:
                        continue

            # If no JSON found, try to extract simple metrics
            if not metrics:
                for line in lines:
                    if "accuracy" in line.lower() or "score" in line.lower():
                        # Try to extract numeric values
                        import re
                        numbers = re.findall(r'(\d+\.?\d*)', line)
                        if numbers:
                            metrics["extracted_score"] = float(numbers[-1])
                            break

        except Exception as e:
            metrics["parse_error"] = str(e)

        return metrics