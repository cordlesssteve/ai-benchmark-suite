#!/usr/bin/env python3
"""
Empty Response Debug - Trace exactly what happens with simple prompts
"""

import sys
from pathlib import Path
import requests
import json
import time

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_interfaces.fixed_enhanced_ollama_interface import FixedEnhancedOllamaInterface, PromptingStrategy

def trace_empty_responses():
    """Debug trace of empty response problem"""
    print("🔍 EMPTY RESPONSE DEBUG TRACE")
    print("="*60)

    # Test the problematic simple prompts
    test_prompts = [
        "def add(a, b):\n    return ",
        "def multiply(x, y):\n    return ",
    ]

    models_to_test = ["qwen2.5:0.5b", "phi3.5:latest"]

    for model in models_to_test:
        print(f"\n🧪 DEBUGGING: {model}")
        print("-" * 40)

        interface = FixedEnhancedOllamaInterface(model)
        if not interface.is_available():
            print("❌ Ollama not available")
            continue

        for prompt in test_prompts:
            print(f"\n  🎯 Testing prompt: {repr(prompt)}")

            # Get model config to see what we're sending
            model_config = interface.get_model_config()
            print(f"    Model config: {model_config}")

            # Test each strategy individually
            strategies = [PromptingStrategy.CODE_ENGINE, PromptingStrategy.ROLE_BASED, PromptingStrategy.DETERMINISTIC]

            for strategy in strategies:
                print(f"\n    📝 Strategy: {strategy.value}")

                try:
                    # Manual step-by-step trace
                    prompts = interface.get_standardized_prompts()
                    strategy_config = prompts[strategy]

                    # Show what we're actually sending
                    enhanced_prompt = f"{strategy_config['prompt']}\n\n{prompt}"
                    print(f"      Enhanced prompt: {repr(enhanced_prompt[:100])}...")

                    # Build payload
                    options = {
                        "temperature": model_config["temperature"],
                        "top_p": 0.9,
                        "num_predict": 50,
                    }

                    stop_tokens = model_config["stop_tokens"]
                    if stop_tokens:
                        options["stop"] = stop_tokens
                    print(f"      Options: {options}")

                    payload = {
                        "model": interface.model_name,
                        "prompt": enhanced_prompt,
                        "stream": False,
                        "options": options
                    }

                    # Make direct API call for debugging
                    print(f"      Making API call...")
                    start_time = time.time()

                    response = requests.post(
                        f"{interface.base_url}/api/generate",
                        json=payload,
                        timeout=15
                    )

                    execution_time = time.time() - start_time
                    print(f"      Response status: {response.status_code}")
                    print(f"      Execution time: {execution_time:.2f}s")

                    if response.status_code == 200:
                        result = response.json()
                        raw_response = result.get("response", "")
                        print(f"      Raw response: {repr(raw_response)}")
                        print(f"      Raw response length: {len(raw_response)}")

                        # Show cleaning process
                        cleaned_response = interface.clean_response(raw_response)
                        print(f"      Cleaned response: {repr(cleaned_response)}")
                        print(f"      Cleaned length: {len(cleaned_response)}")

                        # Check conversational detection
                        is_conv = interface.is_conversational(cleaned_response)
                        print(f"      Is conversational: {is_conv}")

                        # Check if this would be considered a success
                        has_content = len(cleaned_response.strip()) > 0
                        print(f"      Has content: {has_content}")

                        # Check if Ollama thinks it succeeded
                        ollama_done = result.get("done", False)
                        print(f"      Ollama 'done': {ollama_done}")

                        # Additional debug info
                        if "error" in result:
                            print(f"      Ollama error: {result['error']}")

                        print(f"      Full result keys: {list(result.keys())}")

                    else:
                        print(f"      ❌ HTTP Error: {response.status_code}")
                        print(f"      Response text: {response.text}")

                except Exception as e:
                    print(f"      ❌ Exception: {e}")

def direct_ollama_test():
    """Test Ollama directly without our interface"""
    print(f"\n🔬 DIRECT OLLAMA API TEST")
    print("="*60)

    simple_payload = {
        "model": "qwen2.5:0.5b",
        "prompt": "def add(a, b):\n    return ",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 10
        }
    }

    try:
        print(f"Direct API call with minimal payload:")
        print(f"Payload: {json.dumps(simple_payload, indent=2)}")

        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json=simple_payload,
            timeout=15
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

def main():
    trace_empty_responses()
    direct_ollama_test()

if __name__ == "__main__":
    main()