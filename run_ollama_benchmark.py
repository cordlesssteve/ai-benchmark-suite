#!/usr/bin/env python3
"""
Simple Ollama Benchmark Runner

Direct integration with your local Ollama models for code generation benchmarking.
"""

import json
import requests
import time
import argparse
from pathlib import Path

def test_ollama_model(model_name: str, test_problems: int = 3):
    """Test an Ollama model with simple coding problems"""
    
    print(f"🚀 Testing {model_name} with {test_problems} problems...")
    
    # Simple test problems
    problems = [
        {
            "id": "test_1",
            "prompt": "def add_numbers(a, b):\n    \"\"\"Add two numbers and return the result\"\"\"\n    return",
            "expected_pattern": "a + b"
        },
        {
            "id": "test_2", 
            "prompt": "def is_even(n):\n    \"\"\"Check if a number is even\"\"\"\n    return",
            "expected_pattern": "% 2 == 0"
        },
        {
            "id": "test_3",
            "prompt": "def max_of_three(a, b, c):\n    \"\"\"Return the maximum of three numbers\"\"\"\n    return",
            "expected_pattern": "max("
        },
        {
            "id": "test_4",
            "prompt": "def reverse_string(s):\n    \"\"\"Reverse a string\"\"\"\n    return",
            "expected_pattern": "[::-1]"
        },
        {
            "id": "test_5",
            "prompt": "def count_vowels(text):\n    \"\"\"Count vowels in a string\"\"\"\n    count = 0\n    for char in text.lower():\n        if char in",
            "expected_pattern": "aeiou"
        }
    ]
    
    results = {
        "model": model_name,
        "timestamp": int(time.time()),
        "problems_tested": min(test_problems, len(problems)),
        "results": []
    }
    
    for i, problem in enumerate(problems[:test_problems]):
        print(f"  Problem {i+1}: {problem['id']}")
        
        # Call Ollama API
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model_name,
            "prompt": f"Complete this Python code:\n\n{problem['prompt']} ",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 100,
                "stop": ["\ndef", "\nclass", "\n\n"]
            }
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            generated = data.get('response', '')
            
            # Check if generation contains expected pattern
            passed = problem['expected_pattern'].lower() in generated.lower()
            
            result = {
                "problem_id": problem['id'],
                "prompt": problem['prompt'],
                "generated": generated,
                "expected_pattern": problem['expected_pattern'],
                "passed": passed
            }
            
            results["results"].append(result)
            
            status = "✅" if passed else "❌"
            print(f"    {status} Generated: {generated[:50]}...")
            
        else:
            print(f"    ❌ API Error: {response.status_code}")
            results["results"].append({
                "problem_id": problem['id'],
                "error": f"API Error: {response.status_code}",
                "passed": False
            })
    
    # Calculate score
    passed_count = sum(1 for r in results["results"] if r.get("passed", False))
    total_count = len(results["results"])
    score = passed_count / total_count if total_count > 0 else 0.0
    
    results["summary"] = {
        "passed": passed_count,
        "total": total_count,
        "score": score
    }
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    results_file = results_dir / f"ollama_{safe_model_name}_simple_test_{int(time.time())}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Results for {model_name}:")
    print(f"   Score: {passed_count}/{total_count} ({score:.2%})")
    print(f"   Results saved: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test Ollama models with simple coding problems")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., phi3:latest)")
    parser.add_argument("--problems", type=int, default=3, help="Number of test problems (1-5)")
    
    args = parser.parse_args()
    
    if args.problems < 1 or args.problems > 5:
        print("❌ Problems must be between 1 and 5")
        return
    
    try:
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama server not responding. Run 'ollama serve' first.")
            return
            
        # Check if model exists
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if args.model not in model_names:
            print(f"❌ Model '{args.model}' not found.")
            print(f"Available models: {', '.join(model_names)}")
            return
            
        # Run test
        test_ollama_model(args.model, args.problems)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure Ollama is running with 'ollama serve'")

if __name__ == "__main__":
    main()