#!/usr/bin/env python3
"""
Real BigCode Harness Integration

This adapter creates a bridge between Ollama models and the BigCode evaluation harness
by implementing a custom model class that BigCode can use for real evaluation.
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
class RealBigCodeResult:
    """Result from real BigCode harness execution"""
    task: str
    model: str
    metrics: Dict[str, Any]
    raw_output: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class RealBigCodeAdapter:
    """
    Real BigCode harness integration using custom Ollama model adapter.

    This adapter:
    - Creates a temporary BigCode model wrapper for Ollama
    - Executes real BigCode harness with the wrapped model
    - Parses actual BigCode results and metrics (Pass@1, Pass@10, etc.)
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()  # Convert to absolute path
        self.bigcode_dir = self.project_root / "harnesses" / "bigcode-evaluation-harness"
        self.venv_python = self.bigcode_dir / "venv" / "bin" / "python"

        # Verify venv exists
        if not self.venv_python.exists():
            raise RuntimeError(f"BigCode venv not found at {self.venv_python}")

        # Verify bigcode_eval module is available
        try:
            result = subprocess.run(
                [str(self.venv_python), "-c", "import bigcode_eval; print('BigCode-Eval available')"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"BigCode-Eval not available in venv: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Failed to verify BigCode-Eval installation: {e}")

    def run_evaluation(self, task: str, model_name: str, model_interface: OllamaInterface, **kwargs) -> RealBigCodeResult:
        """
        Run real BigCode evaluation using Ollama interface.

        This creates a custom model adapter that BigCode can use to evaluate Ollama models.
        """

        start_time = time.time()

        try:
            # Create temporary model adapter file
            adapter_file = self._create_ollama_adapter(model_name)

            # Run BigCode harness with the custom adapter
            result = self._execute_bigcode_harness(task, adapter_file, **kwargs)

            execution_time = time.time() - start_time

            if result["success"]:
                return RealBigCodeResult(
                    task=task,
                    model=model_name,
                    metrics=result["metrics"],
                    raw_output=result["output"],
                    execution_time=execution_time,
                    success=True
                )
            else:
                return RealBigCodeResult(
                    task=task,
                    model=model_name,
                    metrics={},
                    raw_output=result["output"],
                    execution_time=execution_time,
                    success=False,
                    error_message=result.get("error", "BigCode harness execution failed")
                )

        except Exception as e:
            return RealBigCodeResult(
                task=task,
                model=model_name,
                metrics={},
                raw_output="",
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _create_ollama_adapter(self, model_name: str) -> Path:
        """Create a temporary BigCode model adapter for Ollama"""

        # Create adapter directly in BigCode directory for proper import
        adapter_file = self.bigcode_dir / "ollama_bigcode_adapter.py"

        adapter_code = f'''
"""
Temporary Ollama adapter for BigCode evaluation harness
"""

import requests
import json
import logging
import warnings
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch

class OllamaBigCodeModel:
    """BigCode adapter for Ollama models"""

    def __init__(self, model_name: str = "{model_name}", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        # Create a dummy tokenizer for BigCode compatibility
        # This won't be used for actual tokenization, just for interface compatibility
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium",
                                                         padding_side="left")
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            # Fallback to a simple tokenizer if the above fails
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_ids: torch.Tensor, **generation_kwargs) -> torch.Tensor:
        """
        Generate completions for BigCode evaluation.

        This method converts tensors to text, calls Ollama, and converts back to tensors.
        """
        try:
            # Convert input_ids tensor to text
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)  # Add batch dimension

                batch_size = input_ids.shape[0]
                prompts = []

                for i in range(batch_size):
                    prompt = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    prompts.append(prompt)
            else:
                prompts = input_ids if isinstance(input_ids, list) else [str(input_ids)]
                batch_size = len(prompts)

            # Generate responses with Ollama
            responses = []
            for prompt in prompts:
                response = self._ollama_generate(prompt, generation_kwargs)
                responses.append(response)

            # Convert responses back to tensor format
            output_ids = []
            for i, response in enumerate(responses):
                # Combine original prompt with generated response
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
            logging.error(f"Ollama generation failed: {{e}}")
            # Return dummy output tensor with correct shape
            dummy_length = 50
            dummy_output = torch.full((batch_size, dummy_length),
                                    self.tokenizer.pad_token_id,
                                    dtype=torch.long)
            return dummy_output

    def _ollama_generate(self, prompt: str, generation_kwargs: Dict[str, Any]) -> str:
        """Generate text using Ollama API"""
        try:
            # Extract generation parameters
            max_new_tokens = generation_kwargs.get('max_new_tokens', 512)
            temperature = generation_kwargs.get('temperature', 0.1)
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
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            return generated_text

        except Exception as e:
            logging.error(f"Ollama generation failed: {{e}}")
            return ""

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        """Compatibility method for device placement"""
        return self

    def eval(self):
        """Compatibility method for eval mode"""
        return self

    def __call__(self, *args, **kwargs):
        """Make the model callable like a HuggingFace model"""
        return self.generate(*args, **kwargs)

# Create a global instance that can be imported
model = OllamaBigCodeModel("{model_name}")
tokenizer = model.tokenizer
'''

        with open(adapter_file, 'w') as f:
            f.write(adapter_code)

        return adapter_file

    def _execute_bigcode_harness(self, task: str, adapter_file: Path, **kwargs) -> Dict[str, Any]:
        """Execute BigCode harness with the Ollama adapter"""

        # Prepare BigCode command
        cmd = [
            str(self.venv_python.absolute()), "main.py",
            "--model", "ollama_bigcode_adapter",
            "--tasks", task,
            "--allow_code_execution",  # Required for actual code execution
            "--batch_size", "1",
            "--modeltype", "causal",
            "--trust_remote_code",
            "--save_generations",
            "--metric_output_path", "ollama_results.json"
        ]

        # Add limit if specified (default to 1 for Sprint 1.0)
        limit = kwargs.get('limit', 1)
        cmd.extend(["--limit", str(limit)])

        # Add other parameters
        if max_length := kwargs.get('max_length_generation', 512):
            cmd.extend(["--max_length_generation", str(max_length)])

        try:
            # Execute BigCode harness
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.bigcode_dir)

            result = subprocess.run(
                cmd,
                cwd=self.bigcode_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for code execution
                env=env
            )

            if result.returncode == 0:
                # Parse BigCode output for metrics
                metrics = self._parse_bigcode_output(result.stdout)
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
                    "error": f"BigCode harness failed with return code {result.returncode}"
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "metrics": {},
                "output": "",
                "error": "BigCode harness execution timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "metrics": {},
                "output": "",
                "error": str(e)
            }
        finally:
            # Cleanup adapter file
            try:
                adapter_file.unlink()
            except:
                pass

            # Cleanup results file
            try:
                results_file = self.bigcode_dir / "ollama_results.json"
                if results_file.exists():
                    results_file.unlink()
            except:
                pass

    def _parse_bigcode_output(self, output: str) -> Dict[str, Any]:
        """Parse metrics from BigCode harness output"""
        metrics = {}

        try:
            # Look for JSON results file first
            results_file = self.bigcode_dir / "ollama_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        metrics = data
                        return metrics

            # Parse output text for metrics
            lines = output.split('\n')
            for line in lines:
                # Look for pass@k metrics
                if "pass@" in line.lower():
                    try:
                        import re
                        # Extract pass@k metrics like "pass@1: 0.85"
                        match = re.search(r'pass@(\d+):\s*([\d.]+)', line.lower())
                        if match:
                            k = match.group(1)
                            score = float(match.group(2))
                            metrics[f"pass@{k}"] = score
                    except:
                        continue

                # Look for other BigCode metrics
                if any(keyword in line.lower() for keyword in ['accuracy', 'score', 'eval']):
                    try:
                        import re
                        numbers = re.findall(r'(\d+\.?\d*)', line)
                        if numbers:
                            metrics["extracted_score"] = float(numbers[-1])
                    except:
                        continue

            # If no metrics found, set default
            if not metrics:
                metrics = {"pass@1": 0.0, "note": "No metrics extracted from output"}

        except Exception as e:
            metrics = {"parse_error": str(e)}

        return metrics