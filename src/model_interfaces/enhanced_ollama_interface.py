#!/usr/bin/env python3
"""
Enhanced Ollama Model Interface with Advanced Prompting

Integrates research-backed prompting strategies for optimal code completion
with conversational models. Based on the AdvancedPromptingEngine research.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

@dataclass
class EnhancedOllamaResponse:
    """Enhanced response from Ollama model with prompting metadata"""
    text: str
    execution_time: float
    success: bool
    prompting_strategy: str
    is_conversational: bool
    error_message: Optional[str] = None
    raw_response: Optional[str] = None

class PromptingStrategy(Enum):
    """Available prompting strategies based on research"""
    CODE_ENGINE = "code_engine"
    SILENT_GENERATOR = "silent_generator"
    DETERMINISTIC = "deterministic"
    NEGATIVE_PROMPT = "negative_prompt"
    FORMAT_CONSTRAINT = "format_constraint"
    ROLE_BASED = "role_based"
    AUTO_BEST = "auto_best"

class EnhancedOllamaInterface:
    """Enhanced Ollama interface with advanced prompting strategies"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        # Model-specific optimized strategies based on research
        self.model_strategies = {
            "phi3.5:latest": PromptingStrategy.ROLE_BASED,
            "mistral:7b-instruct": PromptingStrategy.CODE_ENGINE,
            "qwen2.5-coder:3b": PromptingStrategy.DETERMINISTIC,
            "codellama:13b-instruct": PromptingStrategy.NEGATIVE_PROMPT,
        }

    def get_system_prompts(self) -> Dict[PromptingStrategy, Dict[str, Any]]:
        """Research-backed system prompts for code completion"""
        return {
            PromptingStrategy.CODE_ENGINE: {
                "prompt": "You are a code completion engine. Output only executable code. No explanations, comments, or descriptions.",
                "temperature": 0.0,
                "stop_tokens": ["\n\n", "def ", "class ", "```", "Here", "The ", "To ", "This "]
            },
            PromptingStrategy.SILENT_GENERATOR: {
                "prompt": "Role: Silent code generator. Input: partial code. Output: completion only.",
                "temperature": 0.1,
                "stop_tokens": ["\n\n", "# ", "def ", "class ", "```"]
            },
            PromptingStrategy.DETERMINISTIC: {
                "prompt": "Act as a deterministic code generator. Complete the code with no additional text.",
                "temperature": 0.0,
                "stop_tokens": ["\n", "def ", "class ", "# ", "```", "Here's", "To "]
            },
            PromptingStrategy.NEGATIVE_PROMPT: {
                "prompt": "Complete the code. Do NOT include explanations, markdown, commentary, or descriptions. Code only.",
                "temperature": 0.2,
                "stop_tokens": ["\n\n", "```", "Here", "To ", "The "]
            },
            PromptingStrategy.FORMAT_CONSTRAINT: {
                "prompt": "Output format: [code_only]. Complete the missing code without any explanations.",
                "temperature": 0.1,
                "stop_tokens": ["\n\n", "def ", "class ", "# ", "```"]
            },
            PromptingStrategy.ROLE_BASED: {
                "prompt": "You are an autocomplete engine for programmers. Generate only the missing code to complete the function.",
                "temperature": 0.0,
                "stop_tokens": ["\n\n", "def ", "class ", "```", "Note", "Here"]
            }
        }

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

    def is_conversational(self, response: str) -> bool:
        """Check if response contains conversational elements"""
        conversational_indicators = [
            "here's", "certainly", "to complete", "this function", "i'll",
            "let me", "you can", "the code", "this code", "sure"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in conversational_indicators)

    def get_optimal_strategy(self, model_name: str = None) -> PromptingStrategy:
        """Get optimal prompting strategy for the model"""
        target_model = model_name or self.model_name

        # Use model-specific strategy if available
        if target_model in self.model_strategies:
            return self.model_strategies[target_model]

        # Default fallback based on model name patterns
        if "phi" in target_model.lower():
            return PromptingStrategy.ROLE_BASED
        elif "mistral" in target_model.lower():
            return PromptingStrategy.CODE_ENGINE
        elif "coder" in target_model.lower():
            return PromptingStrategy.DETERMINISTIC
        elif "llama" in target_model.lower():
            return PromptingStrategy.NEGATIVE_PROMPT
        else:
            return PromptingStrategy.CODE_ENGINE  # Safe default

    def generate_with_strategy(self, prompt: str, strategy: PromptingStrategy, **kwargs) -> EnhancedOllamaResponse:
        """Generate completion using specific prompting strategy"""
        start_time = time.time()

        strategies = self.get_system_prompts()
        strategy_config = strategies[strategy]

        # Create enhanced prompt with system instruction
        enhanced_prompt = f"{strategy_config['prompt']}\n\n{prompt}"

        # Configure options based on strategy
        options = {
            "temperature": strategy_config.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.9),
            "num_predict": kwargs.get("max_tokens", 100),
        }

        # Add stop tokens if specified
        if "stop_tokens" in strategy_config:
            options["stop"] = strategy_config["stop_tokens"]

        payload = {
            "model": self.model_name,
            "prompt": enhanced_prompt,
            "stream": False,
            "options": options
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", 30)
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            # Clean the response
            cleaned_response = self.clean_response(raw_response)

            # Check if response is conversational
            is_conv = self.is_conversational(cleaned_response)

            execution_time = time.time() - start_time

            return EnhancedOllamaResponse(
                text=cleaned_response,
                execution_time=execution_time,
                success=True,
                prompting_strategy=strategy.value,
                is_conversational=is_conv,
                raw_response=raw_response
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return EnhancedOllamaResponse(
                text="",
                execution_time=execution_time,
                success=False,
                prompting_strategy=strategy.value,
                is_conversational=False,
                error_message=str(e)
            )

    def generate_auto_best(self, prompt: str, **kwargs) -> EnhancedOllamaResponse:
        """Generate using the automatically selected best strategy for this model"""
        optimal_strategy = self.get_optimal_strategy()
        return self.generate_with_strategy(prompt, optimal_strategy, **kwargs)

    def generate_with_fallback(self, prompt: str, max_attempts: int = 3, **kwargs) -> EnhancedOllamaResponse:
        """Generate with fallback strategies if first attempt is conversational"""

        # Try strategies in order of effectiveness for this model
        strategies_to_try = [
            self.get_optimal_strategy(),
            PromptingStrategy.CODE_ENGINE,
            PromptingStrategy.DETERMINISTIC,
            PromptingStrategy.NEGATIVE_PROMPT
        ]

        # Remove duplicates while preserving order
        unique_strategies = []
        for strategy in strategies_to_try:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)

        for i, strategy in enumerate(unique_strategies[:max_attempts]):
            response = self.generate_with_strategy(prompt, strategy, **kwargs)

            # If successful and not conversational, return immediately
            if response.success and not response.is_conversational and response.text.strip():
                return response

            # If this is the last attempt, return even if not perfect
            if i == len(unique_strategies) - 1 or i == max_attempts - 1:
                return response

        # Fallback - should not reach here
        return self.generate_with_strategy(prompt, PromptingStrategy.CODE_ENGINE, **kwargs)

    def generate(self, prompt: str, strategy: Optional[PromptingStrategy] = None, **kwargs) -> EnhancedOllamaResponse:
        """Main generation method with enhanced prompting capabilities"""

        # Use specific strategy if provided
        if strategy:
            return self.generate_with_strategy(prompt, strategy, **kwargs)

        # Use fallback strategy for best results
        return self.generate_with_fallback(prompt, **kwargs)

    def benchmark_strategies(self, test_prompt: str, expected_completion: str) -> List[Dict[str, Any]]:
        """Benchmark all strategies on a test prompt"""
        results = []

        for strategy in PromptingStrategy:
            if strategy == PromptingStrategy.AUTO_BEST:
                continue  # Skip meta-strategy

            response = self.generate_with_strategy(test_prompt, strategy)

            # Evaluate response
            contains_expected = expected_completion.lower() in response.text.lower()

            results.append({
                "strategy": strategy.value,
                "success": response.success,
                "is_conversational": response.is_conversational,
                "contains_expected": contains_expected,
                "execution_time": response.execution_time,
                "response_length": len(response.text),
                "response": response.text[:100] + "..." if len(response.text) > 100 else response.text
            })

        return results

    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model.get("name", "").startswith(self.model_name):
                        return model
            return {}
        except:
            return {}