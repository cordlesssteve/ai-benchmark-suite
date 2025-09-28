#!/usr/bin/env python3
"""
Quick Prompting Strategy Test
"""

import requests
import json

def test_quick_strategies():
    model = "mistral:7b-instruct"  # Best performing model

    strategies = [
        ("Direct", "def add(a, b):\n    return "),
        ("With instruction", "# Complete this code only:\ndef add(a, b):\n    return "),
        ("Code block", "```python\ndef add(a, b):\n    return "),
        ("Minimal", "def add(a, b): return "),
        ("Fill-in", "Complete: def add(a, b):\n    return "),
    ]

    print(f"🧪 Quick strategy test on {model}")

    for name, prompt in strategies:
        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 15,
                    "stop": ["\n", "def", "#"]
                }
            }, timeout=20)

            if response.status_code == 200:
                generated = response.json().get('response', '').strip()
                has_solution = 'a + b' in generated.lower() or 'a+b' in generated.lower()
                status = "✅" if has_solution else "❌"
                print(f"  {status} {name:15}: '{generated[:40]}...'")
            else:
                print(f"  ❌ {name:15}: API Error {response.status_code}")

        except Exception as e:
            print(f"  ❌ {name:15}: {e}")

if __name__ == "__main__":
    test_quick_strategies()