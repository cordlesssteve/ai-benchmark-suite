#!/usr/bin/env python3
"""
Ollama Experimental Adapter

Connects Ollama models to the experimental framework for rigorous evaluation.
"""

import requests
import time
from typing import Dict, List, Optional
from experimental_framework import ExperimentalFramework, Problem, ModelType, PromptTemplate

class OllamaModelInterface:
    """Interface adapter for Ollama models to work with experimental framework"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
        # Verify model exists
        self._verify_model_exists()
    
    def _verify_model_exists(self):
        """Check if model exists in Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    raise ValueError(f"Model '{self.model_name}' not found. Available: {model_names}")
            else:
                raise ConnectionError("Cannot connect to Ollama server")
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama server error: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Generate completion from Ollama model"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["\ndef", "\nclass", "\n\n"]
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/generate", 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.RequestException as e:
            raise RuntimeError(f"Generation failed: {e}")

def create_humaneval_problems() -> List[Problem]:
    """Create simplified HumanEval problems for testing"""
    return [
        Problem(
            id="HumanEval/0",
            prompt="""from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    """,
            canonical_solution="""for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False""",
            test_cases=["assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False"],
            difficulty="easy"
        ),
        
        Problem(
            id="HumanEval/1",
            prompt="""from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group and return the list containing them. Each group is balanced (each open brace is properly closed) and each group is separated from other groups.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
    """,
            canonical_solution="""result = []
    current_string = []
    current_depth = 0
    
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
                
    return result""",
            test_cases=["assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']"],
            difficulty="medium"
        ),
        
        Problem(
            id="HumanEval/2",
            prompt="""def truncate_number(number: float) -> float:
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"
    """,
            canonical_solution="return number % 1.0",
            test_cases=["assert abs(truncate_number(3.5) - 0.5) < 1e-6"],
            difficulty="easy"
        ),

        Problem(
            id="Simple/Add",
            prompt="""def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers and return the result\"\"\"
    """,
            canonical_solution="return a + b",
            test_cases=["assert add_numbers(2, 3) == 5"],
            difficulty="trivial"
        ),
        
        Problem(
            id="Simple/Even",
            prompt="""def is_even(n: int) -> bool:
    \"\"\"Check if a number is even\"\"\"
    """,
            canonical_solution="return n % 2 == 0",
            test_cases=["assert is_even(4) == True", "assert is_even(3) == False"],
            difficulty="trivial"
        )
    ]

def run_ollama_experiment(model_name: str, num_problems: int = 5, samples_per_template: int = 5):
    """Run experimental evaluation on Ollama model"""
    print(f"🧪 Starting experimental evaluation: {model_name}")
    
    # Initialize components
    model_interface = OllamaModelInterface(model_name)
    framework = ExperimentalFramework()
    
    # Get problems
    all_problems = create_humaneval_problems()
    problems = all_problems[:num_problems]
    
    print(f"📝 Testing {len(problems)} problems with {samples_per_template} samples per template")
    
    # Run experiment
    result = framework.run_experiment(
        model_interface=model_interface,
        problems=problems,
        templates=None,  # Auto-select based on model type
        samples_per_template=samples_per_template
    )
    
    # Print results
    print(f"\n🎯 Results for {model_name}:")
    print(f"   Model Type: {result.model_type.value}")
    print(f"   Problems Tested: {result.problems_tested}")
    
    print(f"\n📊 Pass@K Metrics:")
    for k, score in result.pass_at_k.items():
        print(f"   Pass@{k}: {score:.3f} ({score*100:.1f}%)")
    
    print(f"\n🎯 Template Analysis:")
    print(f"   Sensitivity: {result.statistical_metrics['sensitivity']:.3f}")
    print(f"   Mean Score: {result.statistical_metrics['mean_score']:.3f}")
    print(f"   Std Dev: {result.statistical_metrics['std_dev']:.3f}")
    
    print(f"\n📋 Template Scores:")
    for template, score in result.statistical_metrics['template_scores'].items():
        print(f"   {template}: {score:.3f} ({score*100:.1f}%)")
    
    return result

def compare_ollama_models(model_names: List[str], num_problems: int = 3):
    """Compare multiple Ollama models experimentally"""
    print(f"🔬 Comparing {len(model_names)} models on {num_problems} problems")
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        try:
            result = run_ollama_experiment(model_name, num_problems, samples_per_template=3)
            results[model_name] = result
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Comparison summary
    if len(results) > 1:
        print(f"\n{'='*50}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*50}")
        
        print("\n🏆 Pass@1 Rankings:")
        pass1_scores = [(name, res.pass_at_k[1]) for name, res in results.items()]
        pass1_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(pass1_scores, 1):
            print(f"   {i}. {name}: {score:.3f} ({score*100:.1f}%)")
        
        print("\n🎯 Template Sensitivity:")
        for name, result in results.items():
            sensitivity = result.statistical_metrics['sensitivity']
            print(f"   {name}: {sensitivity:.3f} (lower is better)")
    
    return results

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experimental evaluation of Ollama models")
    parser.add_argument('--model', help='Single model to evaluate')
    parser.add_argument('--compare', nargs='+', help='Multiple models to compare')
    parser.add_argument('--problems', type=int, default=3, help='Number of problems to test')
    parser.add_argument('--samples', type=int, default=5, help='Samples per template')
    
    args = parser.parse_args()
    
    try:
        if args.model:
            run_ollama_experiment(args.model, args.problems, args.samples)
        elif args.compare:
            compare_ollama_models(args.compare, args.problems)
        else:
            # Default: compare available models
            available_models = ['phi3:latest', 'phi3.5:latest', 'qwen2.5-coder:3b']
            print("No model specified. Comparing default models...")
            compare_ollama_models(available_models, 2)
            
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()