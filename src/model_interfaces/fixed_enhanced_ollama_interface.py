#!/usr/bin/env python3
"""
Fixed Enhanced Ollama Model Interface with Advanced Prompting

Addresses critical issues from remediation review:
1. Fixed stop token configuration
2. Improved error handling
3. Standardized prompting patterns
4. Model-specific tuning

Version: 2.0 (Fixed)
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

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

class FixedEnhancedOllamaInterface:
    """Fixed enhanced Ollama interface with advanced prompting strategies"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        # Standardized model-specific configurations based on research
        self.model_configs = {
            "phi3.5:latest": {
                "strategy": PromptingStrategy.ROLE_BASED,
                "temperature": 0.1,
                "stop_tokens": ["\n\n"],  # Minimal stop tokens
                "max_tokens": 100
            },
            "mistral:7b-instruct": {
                "strategy": PromptingStrategy.CODE_ENGINE,
                "temperature": 0.0,
                "stop_tokens": ["\n\n", "```"],  # Conservative stop tokens
                "max_tokens": 100
            },
            "qwen2.5-coder:3b": {
                "strategy": PromptingStrategy.DETERMINISTIC,
                "temperature": 0.0,
                "stop_tokens": ["\n\n"],
                "max_tokens": 100
            },
            "codellama:13b-instruct": {
                "strategy": PromptingStrategy.NEGATIVE_PROMPT,
                "temperature": 0.1,
                "stop_tokens": ["\n\n"],
                "max_tokens": 100
            },
            # Default configuration for unknown models
            "_default": {
                "strategy": PromptingStrategy.CODE_ENGINE,
                "temperature": 0.1,
                "stop_tokens": ["\n\n"],
                "max_tokens": 100
            }
        }

    def get_standardized_prompts(self) -> Dict[PromptingStrategy, Dict[str, Any]]:
        """Standardized system prompts with consistent patterns across models"""
        return {
            PromptingStrategy.CODE_ENGINE: {
                "prompt": "You are a code completion engine. Complete the code with only the missing parts. No explanations.",
                "description": "Direct code completion instruction"
            },
            PromptingStrategy.SILENT_GENERATOR: {
                "prompt": "Complete the code. Output only the completion.",
                "description": "Minimal completion instruction"
            },
            PromptingStrategy.DETERMINISTIC: {
                "prompt": "Complete this code with the most likely continuation:",
                "description": "Deterministic completion"
            },
            PromptingStrategy.NEGATIVE_PROMPT: {
                "prompt": "Complete the code. Do NOT include explanations, comments, or markdown. Code only.",
                "description": "Negative prompting to avoid conversational responses"
            },
            PromptingStrategy.FORMAT_CONSTRAINT: {
                "prompt": "Format: code_only. Complete: ",
                "description": "Format-constrained completion"
            },
            PromptingStrategy.ROLE_BASED: {
                "prompt": "As a code autocomplete tool, complete this code:",
                "description": "Role-based instruction"
            }
        }

    def get_model_config(self, model_name: str = None) -> Dict[str, Any]:
        """Get configuration for specific model with fallback to default"""
        target_model = model_name or self.model_name

        # Direct match
        if target_model in self.model_configs:
            return self.model_configs[target_model].copy()

        # Pattern matching for model families
        for model_key, config in self.model_configs.items():
            if model_key != "_default" and model_key.split(":")[0] in target_model:
                return config.copy()

        # Fallback to default
        return self.model_configs["_default"].copy()

    def clean_response(self, response: str) -> str:
        """Clean conversational elements from response with improved logic"""
        if not response:
            return response

        # Remove common conversational starters (case insensitive)
        conversational_patterns = [
            "here's", "here is", "certainly", "sure", "i'll", "to complete",
            "the function", "this function", "you can", "let me", "i can",
            "```python", "```", "# explanation", "# this", "# here"
        ]

        lines = response.split('\n')
        cleaned_lines = []

        for line in lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()

            # Skip empty lines at the start
            if not cleaned_lines and not line_clean:
                continue

            # Skip lines that start with conversational patterns
            if any(line_lower.startswith(pattern) for pattern in conversational_patterns):
                continue

            # Skip markdown code blocks
            if line_clean.startswith("```"):
                continue

            # Skip obvious explanation comments
            if line_clean.startswith("# ") and any(word in line_lower for word in
                ["this", "the", "here", "explanation", "note", "example", "above", "below"]):
                continue

            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()

        # Remove leading/trailing markdown if present
        if result.startswith('```') and result.endswith('```'):
            lines = result.split('\n')
            if len(lines) >= 2:
                result = '\n'.join(lines[1:-1]).strip()

        return result

    def is_conversational(self, response: str) -> bool:
        """Check if response contains conversational elements"""
        if not response:
            return False

        conversational_indicators = [
            "here's", "certainly", "to complete", "this function", "i'll",
            "let me", "you can", "the code", "this code", "sure", "here is",
            "i can help", "let's", "we can", "explanation", "note that"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in conversational_indicators)

    def generate_with_strategy(self, prompt: str, strategy: PromptingStrategy, **kwargs) -> EnhancedOllamaResponse:
        """Generate completion using specific prompting strategy with improved error handling"""
        start_time = time.time()

        try:
            prompts = self.get_standardized_prompts()
            if strategy not in prompts:
                raise ValueError(f"Unknown strategy: {strategy}")

            strategy_config = prompts[strategy]
            model_config = self.get_model_config()

            # Create enhanced prompt with consistent formatting
            enhanced_prompt = f"{strategy_config['prompt']}\n\n{prompt}"

            # Configure options based on model and strategy
            options = {
                "temperature": kwargs.get("temperature", model_config["temperature"]),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", model_config["max_tokens"]),
            }

            # Use model-specific stop tokens (fixed from aggressive approach)
            stop_tokens = kwargs.get("stop_tokens", model_config["stop_tokens"])
            if stop_tokens:
                options["stop"] = stop_tokens

            payload = {
                "model": self.model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": options
            }

            logger.debug(f"Sending request to Ollama: strategy={strategy.value}, model={self.model_name}")

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

        except requests.RequestException as e:
            execution_time = time.time() - start_time
            error_msg = f"HTTP request failed: {str(e)}"
            logger.error(error_msg)
            return EnhancedOllamaResponse(
                text="",
                execution_time=execution_time,
                success=False,
                prompting_strategy=strategy.value,
                is_conversational=False,
                error_message=error_msg
            )

        except ValueError as e:
            execution_time = time.time() - start_time
            error_msg = f"Configuration error: {str(e)}"
            logger.error(error_msg)
            return EnhancedOllamaResponse(
                text="",
                execution_time=execution_time,
                success=False,
                prompting_strategy=strategy.value,
                is_conversational=False,
                error_message=error_msg
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return EnhancedOllamaResponse(
                text="",
                execution_time=execution_time,
                success=False,
                prompting_strategy=strategy.value,
                is_conversational=False,
                error_message=error_msg
            )

    def generate_auto_best(self, prompt: str, **kwargs) -> EnhancedOllamaResponse:
        """Generate using the automatically selected best strategy for this model"""
        model_config = self.get_model_config()
        optimal_strategy = model_config["strategy"]
        return self.generate_with_strategy(prompt, optimal_strategy, **kwargs)

    def generate_with_fallback(self, prompt: str, max_attempts: int = 3, **kwargs) -> EnhancedOllamaResponse:
        """Generate with fallback strategies if first attempt fails or is conversational"""

        model_config = self.get_model_config()

        # Standardized fallback order based on effectiveness research
        primary_strategy = model_config["strategy"]
        fallback_strategies = [
            primary_strategy,
            PromptingStrategy.CODE_ENGINE,      # Most reliable
            PromptingStrategy.DETERMINISTIC,    # Fallback #1
            PromptingStrategy.NEGATIVE_PROMPT   # Fallback #2
        ]

        # Remove duplicates while preserving order
        unique_strategies = []
        for strategy in fallback_strategies:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)

        last_response = None
        for i, strategy in enumerate(unique_strategies[:max_attempts]):
            logger.debug(f"Trying strategy {i+1}/{max_attempts}: {strategy.value}")

            response = self.generate_with_strategy(prompt, strategy, **kwargs)

            # If successful and not conversational and has content, return immediately
            if response.success and not response.is_conversational and response.text.strip():
                logger.debug(f"Success with strategy: {strategy.value}")
                return response

            # Keep the last response for potential return
            last_response = response

            # If this is the last attempt, return what we have
            if i == len(unique_strategies) - 1 or i == max_attempts - 1:
                logger.warning(f"All fallback attempts exhausted, returning last response")
                return last_response or response

        # Should not reach here, but just in case
        return last_response or self.generate_with_strategy(prompt, PromptingStrategy.CODE_ENGINE, **kwargs)

    def generate(self, prompt: str, strategy: Optional[PromptingStrategy] = None, **kwargs) -> EnhancedOllamaResponse:
        """Main generation method with enhanced prompting capabilities"""

        # Use specific strategy if provided
        if strategy:
            return self.generate_with_strategy(prompt, strategy, **kwargs)

        # Use fallback strategy for best results
        return self.generate_with_fallback(prompt, **kwargs)

    def benchmark_strategies(self, test_prompt: str, expected_completion: str) -> List[Dict[str, Any]]:
        """Benchmark all strategies on a test prompt with standardized evaluation"""
        results = []

        logger.info(f"Benchmarking strategies for model: {self.model_name}")

        for strategy in PromptingStrategy:
            if strategy == PromptingStrategy.AUTO_BEST:
                continue  # Skip meta-strategy

            logger.debug(f"Testing strategy: {strategy.value}")
            response = self.generate_with_strategy(test_prompt, strategy, max_tokens=50)

            # Standardized evaluation
            contains_expected = any(
                expected.lower() in response.text.lower()
                for expected in expected_completion if expected.strip()
            ) if isinstance(expected_completion, list) else expected_completion.lower() in response.text.lower()

            results.append({
                "strategy": strategy.value,
                "success": response.success,
                "is_conversational": response.is_conversational,
                "contains_expected": contains_expected,
                "execution_time": response.execution_time,
                "response_length": len(response.text),
                "response": response.text[:100] + "..." if len(response.text) > 100 else response.text,
                "error": response.error_message
            })

        return results

    def is_available(self) -> bool:
        """Check if Ollama server is available with proper error handling"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama availability: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information with improved error handling"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            for model in models:
                if model.get("name", "").startswith(self.model_name):
                    return model

            logger.warning(f"Model {self.model_name} not found in available models")
            return {}

        except requests.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting model info: {e}")
            return {}

    def get_strategy_effectiveness(self) -> Dict[str, float]:
        """Get effectiveness scores for different strategies based on model type"""
        model_config = self.get_model_config()

        # Return effectiveness scores (can be used for adaptive strategy selection)
        effectiveness_scores = {
            PromptingStrategy.CODE_ENGINE.value: 0.8,
            PromptingStrategy.DETERMINISTIC.value: 0.7,
            PromptingStrategy.NEGATIVE_PROMPT.value: 0.6,
            PromptingStrategy.ROLE_BASED.value: 0.5,
            PromptingStrategy.SILENT_GENERATOR.value: 0.4,
            PromptingStrategy.FORMAT_CONSTRAINT.value: 0.3
        }

        # Boost primary strategy for this model
        primary_strategy = model_config["strategy"].value
        if primary_strategy in effectiveness_scores:
            effectiveness_scores[primary_strategy] += 0.2

        return effectiveness_scores