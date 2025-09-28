#!/usr/bin/env python3
"""Debug enhanced prompting API calls"""

import sys
from pathlib import Path
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_raw_api():
    """Test raw API call similar to enhanced interface"""
    model_name = 'phi3.5:latest'
    prompt = 'def add(a, b):\n    return '
    system_prompt = 'You are a code completion engine. Output only executable code. No explanations, comments, or descriptions.'

    enhanced_prompt = f'{system_prompt}\n\n{prompt}'

    payload = {
        'model': model_name,
        'prompt': enhanced_prompt,
        'stream': False,
        'options': {
            'temperature': 0.0,
            'top_p': 0.9,
            'num_predict': 50,
            'stop': ['\n\n', 'def ', 'class ', '```', 'Here', 'The ', 'To ', 'This ']
        }
    }

    print('Testing raw API call...')
    response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
    print(f'Status: {response.status_code}')
    result = response.json()
    raw_response = result.get('response', '')
    print(f'Raw API response: "{raw_response}"')
    print(f'Response length: {len(raw_response)}')

    return raw_response

def test_baseline_api():
    """Test baseline API call"""
    model_name = 'phi3.5:latest'
    prompt = 'def add(a, b):\n    return '

    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.2,
            'top_p': 0.9,
            'max_tokens': 2048,
        }
    }

    print('\nTesting baseline API call...')
    response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
    print(f'Status: {response.status_code}')
    result = response.json()
    raw_response = result.get('response', '')
    print(f'Baseline response: "{raw_response[:100]}..."')
    print(f'Response length: {len(raw_response)}')

    return raw_response

if __name__ == "__main__":
    enhanced_response = test_raw_api()
    baseline_response = test_baseline_api()

    print(f"\n=== COMPARISON ===")
    print(f"Enhanced response empty: {len(enhanced_response) == 0}")
    print(f"Baseline response empty: {len(baseline_response) == 0}")