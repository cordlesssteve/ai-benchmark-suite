#!/usr/bin/env python3
"""
Advanced Prompting Techniques for Code Completion

Implementation of research-backed strategies to adapt conversational models
to non-conversational code completion tasks.
"""

import json
import requests
import time
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class AdvancedPromptingEngine:
    """Advanced prompting strategies based on 2024-2025 research"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"

    def get_system_prompts(self) -> List[Dict[str, Any]]:
        """Research-backed system prompts for code completion"""
        return [
            {
                "name": "code_engine",
                "prompt": "You are a code completion engine. Output only executable code. No explanations, comments, or descriptions.",
                "temperature": 0.0,
                "stop_tokens": ["\n\n", "def ", "class ", "```", "Here", "The ", "To ", "This "]
            },
            {
                "name": "silent_generator",
                "prompt": "Role: Silent code generator. Input: partial code. Output: completion only.",
                "temperature": 0.1,
                "stop_tokens": ["\n\n", "# ", "def ", "class ", "```"]
            },
            {
                "name": "deterministic",
                "prompt": "Act as a deterministic code generator. Complete the code with no additional text.",
                "temperature": 0.0,
                "stop_tokens": ["\n", "def ", "class ", "# ", "```", "Here's", "To "]
            },
            {
                "name": "negative_prompt",
                "prompt": "Complete the code. Do NOT include explanations, markdown, commentary, or descriptions. Code only.",
                "temperature": 0.2,
                "stop_tokens": ["\n\n", "```", "Here", "To ", "The "]
            },
            {
                "name": "format_constraint",
                "prompt": "Output format: [code_only]. Complete the missing code without any explanations.",
                "temperature": 0.1,
                "stop_tokens": ["\n\n", "def ", "class ", "# ", "```"]
            },
            {
                "name": "role_based",
                "prompt": "You are an autocomplete engine for programmers. Generate only the missing code to complete the function.",
                "temperature": 0.0,
                "stop_tokens": ["\n\n", "def ", "class ", "```", "Note", "Here"]
            }
        ]

    def get_fim_prompts(self) -> List[Dict[str, Any]]:
        """Fill-in-Middle prompting strategies"""
        return [
            {
                "name": "standard_fim",
                "format": "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
                "temperature": 0.1
            },
            {
                "name": "codellama_fim",
                "format": "<PRE> {prefix} <SUF>{suffix} <MID>",
                "temperature": 0.1
            },
            {
                "name": "custom_fim",
                "format": "PREFIX:\n{prefix}\nSUFFIX:\n{suffix}\nCOMPLETE:",
                "temperature": 0.0
            }
        ]

    def get_instruction_prompts(self) -> List[Dict[str, Any]]:
        """Enhanced instruction-based prompts"""
        return [
            {
                "name": "direct_instruction",
                "template": "Complete this Python code:\n\n{code}\n\nCompletion:",
                "temperature": 0.1,
                "stop_tokens": ["\n\n", "def ", "class "]
            },
            {
                "name": "constrained_format",
                "template": "# Code completion task\n# Input: {code}\n# Output (code only):",
                "temperature": 0.0,
                "stop_tokens": ["\n\n", "# ", "def ", "class "]
            },
            {
                "name": "minimal_instruction",
                "template": "COMPLETE:\n{code}",
                "temperature": 0.0,
                "stop_tokens": ["\n", "def ", "class ", "# "]
            }
        ]

    def generate_with_prompt(self, prompt: str, strategy: Dict[str, Any]) -> Optional[str]:
        """Generate completion using specific prompting strategy"""

        try:
            options = {
                "temperature": strategy.get("temperature", 0.1),
                "num_predict": 50,
                "top_p": 0.9
            }

            if "stop_tokens" in strategy:
                options["stop"] = strategy["stop_tokens"]

            response = requests.post(f"{self.base_url}/api/generate", json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options
            }, timeout=30)

            if response.status_code == 200:
                generated = response.json().get('response', '').strip()

                # Post-process to remove conversational elements
                generated = self.clean_response(generated)

                return generated

        except Exception as e:
            print(f"    Error: {e}")

        return None

    def clean_response(self, response: str) -> str:
        """Clean conversational elements from response"""

        # Remove common conversational starters
        conversational_patterns = [
            "Here's", "Here is", "Certainly", "Sure", "I'll", "To complete",
            "The function", "This function", "You can", "Let me", "I can",
            "```python", "```", "# Explanation", "# This"
        ]

        lines = response.split('\n')
        cleaned_lines = []

        for line in lines:
            line_clean = line.strip()

            # Skip lines that start with conversational patterns
            if any(line_clean.startswith(pattern) for pattern in conversational_patterns):
                continue

            # Skip lines that are pure explanations
            if line_clean.startswith("# ") and any(word in line_clean.lower() for word in
                ["this", "the", "here", "explanation", "note", "example"]):
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def test_prompting_strategies(self, test_code: str, expected: str) -> List[Dict[str, Any]]:
        """Test various prompting strategies on a code completion task"""

        results = []

        print(f"🧪 Testing prompting strategies on {self.model_name}...")
        print(f"Test code: {test_code[:50]}...")
        print(f"Expected: {expected}")
        print()

        # Test system prompt strategies
        system_prompts = self.get_system_prompts()

        for strategy in system_prompts:
            print(f"  Testing: {strategy['name']}")

            # Create full prompt with system instruction
            full_prompt = f"{strategy['prompt']}\n\n{test_code}"

            start_time = time.time()
            generated = self.generate_with_prompt(full_prompt, strategy)
            execution_time = time.time() - start_time

            if generated:
                contains_expected = expected.lower() in generated.lower()
                is_conversational = any(phrase in generated.lower() for phrase in
                    ["here's", "certainly", "to complete", "this function"])

                result = {
                    "strategy": strategy['name'],
                    "type": "system_prompt",
                    "generated": generated,
                    "contains_expected": contains_expected,
                    "is_conversational": is_conversational,
                    "execution_time": execution_time,
                    "success": contains_expected and not is_conversational
                }

                results.append(result)

                status = "✅" if result['success'] else "❌"
                conv_indicator = "🗣️" if is_conversational else "🤖"
                print(f"    {status} {conv_indicator} '{generated[:40]}...'")
            else:
                print(f"    ❌ Failed to generate")

        # Test instruction-based strategies
        print(f"\n  Testing instruction-based prompts...")
        instruction_prompts = self.get_instruction_prompts()

        for strategy in instruction_prompts:
            print(f"  Testing: {strategy['name']}")

            prompt = strategy['template'].format(code=test_code)

            start_time = time.time()
            generated = self.generate_with_prompt(prompt, strategy)
            execution_time = time.time() - start_time

            if generated:
                contains_expected = expected.lower() in generated.lower()
                is_conversational = any(phrase in generated.lower() for phrase in
                    ["here's", "certainly", "to complete", "this function"])

                result = {
                    "strategy": strategy['name'],
                    "type": "instruction",
                    "generated": generated,
                    "contains_expected": contains_expected,
                    "is_conversational": is_conversational,
                    "execution_time": execution_time,
                    "success": contains_expected and not is_conversational
                }

                results.append(result)

                status = "✅" if result['success'] else "❌"
                conv_indicator = "🗣️" if is_conversational else "🤖"
                print(f"    {status} {conv_indicator} '{generated[:40]}...'")
            else:
                print(f"    ❌ Failed to generate")

        return results

def run_advanced_prompting_research():
    """Run comprehensive prompting strategy research"""

    # Get available models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not responding")
            return

        models = [m['name'] for m in response.json().get('models', [])]

        # Test on fastest models first
        test_models = ["phi3.5:latest", "mistral:7b-instruct"]
        available_test_models = [m for m in test_models if m in models]

        if not available_test_models:
            available_test_models = models[:2]

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Test cases based on our previous failures
    test_cases = [
        {
            "name": "simple_addition",
            "code": "def add(a, b):\n    return ",
            "expected": "a + b"
        },
        {
            "name": "even_check",
            "code": "def is_even(n):\n    return n ",
            "expected": "% 2 == 0"
        },
        {
            "name": "max_function",
            "code": "def max_three(a, b, c):\n    return ",
            "expected": "max("
        }
    ]

    Path("advanced_prompting_results").mkdir(exist_ok=True)

    all_results = []

    for model in available_test_models:
        print(f"\n{'='*60}")
        print(f"ADVANCED PROMPTING RESEARCH: {model}")
        print(f"{'='*60}")

        engine = AdvancedPromptingEngine(model)
        model_results = {
            "model": model,
            "timestamp": time.time(),
            "test_cases": []
        }

        for test_case in test_cases:
            print(f"\n--- Test Case: {test_case['name']} ---")

            results = engine.test_prompting_strategies(
                test_case['code'],
                test_case['expected']
            )

            test_case_result = {
                "test_name": test_case['name'],
                "test_code": test_case['code'],
                "expected": test_case['expected'],
                "results": results
            }

            model_results["test_cases"].append(test_case_result)

            # Print summary for this test case
            successful_strategies = [r for r in results if r['success']]
            print(f"\n  📊 Summary for {test_case['name']}:")
            print(f"    Successful strategies: {len(successful_strategies)}/{len(results)}")

            if successful_strategies:
                best = max(successful_strategies, key=lambda x: not x['is_conversational'])
                print(f"    Best strategy: {best['strategy']} ({best['type']})")
            else:
                print(f"    No successful strategies found")

        all_results.append(model_results)

        # Save individual model results
        safe_name = model.replace(":", "_").replace("/", "_")
        output_file = f"advanced_prompting_results/{safe_name}_advanced_prompting_{int(time.time())}.json"

        with open(output_file, 'w') as f:
            json.dump(model_results, f, indent=2)

        print(f"\n💾 Results saved: {output_file}")

    # Generate overall analysis
    print(f"\n{'='*60}")
    print("ADVANCED PROMPTING ANALYSIS")
    print(f"{'='*60}")

    # Find best strategies across all models and test cases
    all_strategy_results = []
    for model_result in all_results:
        for test_case in model_result["test_cases"]:
            for result in test_case["results"]:
                all_strategy_results.append({
                    "model": model_result["model"],
                    "test_case": test_case["test_name"],
                    "strategy": result["strategy"],
                    "type": result["type"],
                    "success": result["success"],
                    "conversational": result["is_conversational"],
                    "execution_time": result["execution_time"]
                })

    # Calculate strategy effectiveness
    strategy_stats = {}
    for result in all_strategy_results:
        strategy = result["strategy"]
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {"total": 0, "successful": 0, "conversational": 0}

        strategy_stats[strategy]["total"] += 1
        if result["success"]:
            strategy_stats[strategy]["successful"] += 1
        if result["conversational"]:
            strategy_stats[strategy]["conversational"] += 1

    print(f"\n📈 Strategy Effectiveness Rankings:")
    print(f"{'Strategy':<20} {'Success Rate':<12} {'Non-Conv Rate':<12}")
    print("-" * 50)

    sorted_strategies = sorted(strategy_stats.items(),
                             key=lambda x: x[1]["successful"] / x[1]["total"],
                             reverse=True)

    for strategy, stats in sorted_strategies:
        success_rate = stats["successful"] / stats["total"]
        non_conv_rate = (stats["total"] - stats["conversational"]) / stats["total"]
        print(f"{strategy:<20} {success_rate:.1%}       {non_conv_rate:.1%}")

    # Save comprehensive analysis
    analysis = {
        "timestamp": time.time(),
        "models_tested": [r["model"] for r in all_results],
        "total_test_cases": len(test_cases),
        "strategy_statistics": strategy_stats,
        "best_strategies": sorted_strategies[:3],
        "detailed_results": all_results
    }

    analysis_file = f"advanced_prompting_results/comprehensive_analysis_{int(time.time())}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n💾 Comprehensive analysis saved: {analysis_file}")
    print(f"\n✅ Advanced prompting research completed!")

if __name__ == "__main__":
    run_advanced_prompting_research()