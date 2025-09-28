#!/usr/bin/env python3
"""
Custom BigCode runner that directly integrates with Ollama
without trying to fake a HuggingFace model.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add BigCode harness to path
project_root = Path(__file__).parent.parent
bigcode_dir = project_root / "harnesses" / "bigcode-evaluation-harness"
sys.path.insert(0, str(bigcode_dir))

import requests
from bigcode_eval.tasks import get_task
from bigcode_eval.arguments import EvalArguments

class OllamaDirectEvaluator:
    """
    Direct BigCode evaluation using Ollama without HuggingFace model loading.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

    def _generate_with_ollama(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Ollama API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                }
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            print(f"Ollama generation failed: {e}")
            return ""

    def evaluate_humaneval(self, limit: int = 1) -> Dict[str, Any]:
        """
        Evaluate HumanEval task directly without BigCode harness model loading.
        """
        print(f"Evaluating HumanEval with {self.model_name} (limit: {limit})")

        # Create a minimal EvalArguments object
        args = type('Args', (), {
            'limit': limit,
            'limit_start': 0,
            'allow_code_execution': True,
            'postprocess': True,
            'save_generations': True,
            'save_generations_path': 'generations.json',
            'save_references': False
        })()

        try:
            # Get HumanEval task
            task = get_task("humaneval", args)
            dataset = task.get_dataset()

            print(f"Loaded HumanEval dataset with {len(dataset)} problems")

            # Limit dataset
            n_tasks = min(limit, len(dataset)) if limit else len(dataset)
            dataset = dataset.select(range(n_tasks))

            generations = []
            references = []

            print(f"Generating solutions for {n_tasks} problems...")

            for i, example in enumerate(dataset):
                print(f"  Problem {i+1}/{n_tasks}: {example.get('task_id', f'problem_{i}')}")

                # Get the prompt
                prompt = example.get('prompt', '')

                # Generate solution
                generated_code = self._generate_with_ollama(prompt, 512)

                generations.append(generated_code)
                references.append(example)

                print(f"    Generated {len(generated_code)} characters")

            # Save generations for potential BigCode evaluation
            generations_data = []
            for i, (gen, ref) in enumerate(zip(generations, references)):
                generations_data.append({
                    "task_id": ref.get('task_id', f'problem_{i}'),
                    "prompt": ref.get('prompt', ''),
                    "generation": gen,
                    "canonical_solution": ref.get('canonical_solution', ''),
                    "test": ref.get('test', '')
                })

            # Save to file
            output_file = bigcode_dir / "ollama_generations.json"
            with open(output_file, 'w') as f:
                json.dump(generations_data, f, indent=2)

            print(f"Saved generations to: {output_file}")

            # Now try to evaluate using BigCode's evaluation
            return self._evaluate_with_bigcode(output_file, n_tasks)

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "pass@1": 0.0}

    def _evaluate_with_bigcode(self, generations_file: Path, n_tasks: int) -> Dict[str, Any]:
        """
        Use BigCode harness to evaluate our generated solutions.
        """
        try:
            print("Evaluating generations with BigCode harness...")

            # Use BigCode harness evaluation-only mode
            venv_python = bigcode_dir / "venv" / "bin" / "python"

            cmd = [
                str(venv_python), "main.py",
                "--load_generations_path", str(generations_file),
                "--tasks", "humaneval",
                "--allow_code_execution",
                "--metric_output_path", "ollama_evaluation_results.json"
            ]

            print(f"Running BigCode evaluation: {' '.join(cmd[:4])}...")

            result = subprocess.run(
                cmd,
                cwd=bigcode_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✅ BigCode evaluation completed successfully")

                # Parse results
                results_file = bigcode_dir / "ollama_evaluation_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                    # Extract metrics
                    metrics = {}
                    if 'humaneval' in results:
                        humaneval_results = results['humaneval']
                        if 'pass@1' in humaneval_results:
                            metrics['pass@1'] = humaneval_results['pass@1']
                        metrics.update(humaneval_results)

                    print(f"Results: {metrics}")
                    return metrics
                else:
                    print("❌ Results file not found")
                    return {"error": "Results file not found", "pass@1": 0.0}
            else:
                print(f"❌ BigCode evaluation failed: {result.stderr}")
                return {"error": f"BigCode evaluation failed: {result.stderr}", "pass@1": 0.0}

        except subprocess.TimeoutExpired:
            return {"error": "BigCode evaluation timed out", "pass@1": 0.0}
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"error": str(e), "pass@1": 0.0}

def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="BigCode evaluation with Ollama")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--task", default="humaneval", help="Task to evaluate")
    parser.add_argument("--limit", type=int, default=1, help="Number of problems to evaluate")

    args = parser.parse_args()

    if args.task != "humaneval":
        print(f"Error: Task '{args.task}' not supported yet. Only 'humaneval' is supported.")
        return 1

    evaluator = OllamaDirectEvaluator(args.model)

    start_time = time.time()
    results = evaluator.evaluate_humaneval(args.limit)
    execution_time = time.time() - start_time

    print(f"\n🎯 EVALUATION COMPLETE")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Problems: {args.limit}")
    print(f"Time: {execution_time:.1f}s")
    print(f"Results: {results}")

    if "pass@1" in results:
        print(f"\n✅ SUCCESS: Pass@1 = {results['pass@1']}")
        return 0
    else:
        print(f"\n❌ FAILED: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())