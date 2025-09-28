#!/usr/bin/env python3
"""Debug stop token issue"""

import requests

def test_without_stop_tokens():
    """Test enhanced prompt without stop tokens"""
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
            # No stop tokens
        }
    }

    print('Testing without stop tokens...')
    response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
    result = response.json()
    raw_response = result.get('response', '')
    print(f'Response: "{raw_response}"')
    return raw_response

def test_with_minimal_stop_tokens():
    """Test with minimal stop tokens"""
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
            'stop': ['\n\n']  # Only double newline
        }
    }

    print('\nTesting with minimal stop tokens...')
    response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
    result = response.json()
    raw_response = result.get('response', '')
    print(f'Response: "{raw_response}"')
    return raw_response

if __name__ == "__main__":
    resp1 = test_without_stop_tokens()
    resp2 = test_with_minimal_stop_tokens()

    print(f"\n=== RESULTS ===")
    print(f"Without stop tokens: {len(resp1) > 0}")
    print(f"With minimal stop tokens: {len(resp2) > 0}")