#!/usr/bin/env python3
"""
Phase 1 Enhanced Ollama Interface with Smart Response Processing

Integrates the new SmartResponseProcessor to address critical issues:
1. Smart response cleaning that preserves code content
2. Dual success flag system (HTTP + content quality)
3. Quality-based evaluation with detailed metrics

Version: 1.0 (Phase 1 Implementation)
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .smart_response_processor import SmartResponseProcessor, ProcessedResponse, ContentQuality

logger = logging.getLogger(__name__)

@dataclass
class Phase1OllamaResponse:
    """Phase 1 response with dual success flags and quality metrics"""
    text: str                           # Cleaned response text
    raw_text: str                      # Original response text
    execution_time: float              # Time taken for request
    http_success: bool                 # HTTP request succeeded
    content_quality_success: bool      # Content meets quality threshold
    overall_success: bool              # Combined success indicator
    quality_level: ContentQuality      # Detailed quality assessment
    quality_score: float               # Numerical quality score (0.0-1.0)
    prompting_strategy: str            # Strategy used for generation
    is_conversational: bool           # Contains conversational elements
    is_executable: bool               # Appears to be executable code
    has_syntax_errors: bool           # Contains syntax errors
    cleaning_applied: bool            # Whether cleaning was applied
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptingStrategy(Enum):
    """Available prompting strategies optimized for Phase 1"""
    CODE_ENGINE = "code_engine"
    SILENT_GENERATOR = "silent_generator"
    DETERMINISTIC = "deterministic"
    NEGATIVE_PROMPT = "negative_prompt"
    FORMAT_CONSTRAINT = "format_constraint"
    ROLE_BASED = "role_based"

class Phase1EnhancedOllamaInterface:
    """Phase 1 enhanced Ollama interface with smart response processing"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434",
                 quality_threshold: float = 0.3):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

        # Initialize smart response processor
        self.response_processor = SmartResponseProcessor(quality_threshold=quality_threshold)

        # Conservative model-specific configurations (Phase 1 focus: stability)
        self.model_configs = {
            "phi3.5:latest": {
                "strategy": PromptingStrategy.ROLE_BASED,
                "temperature": 0.0,
                "stop_tokens": ["\n\n"],  # Conservative - only paragraph breaks
                "max_tokens": 100
            },
            "mistral:7b-instruct": {
                "strategy": PromptingStrategy.CODE_ENGINE,
                "temperature": 0.0,
                "stop_tokens": ["\n\n"],  # Conservative
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
                "temperature": 0.0,
                "stop_tokens": ["\n\n"],
                "max_tokens": 100
            },
            # Conservative default
            "_default": {
                "strategy": PromptingStrategy.CODE_ENGINE,
                "temperature": 0.0,
                "stop_tokens": ["\n\n"],
                "max_tokens": 100
            }
        }

    def get_optimized_prompts(self) -> Dict[PromptingStrategy, Dict[str, Any]]:
        """Phase 1 optimized prompts focused on code preservation"""
        return {
            PromptingStrategy.CODE_ENGINE: {
                "prompt": "You are a code completion engine. Complete the code with only the missing parts. No explanations or comments.",
                "description": "Direct code completion"
            },
            PromptingStrategy.SILENT_GENERATOR: {
                "prompt": "Complete the code. Output only the completion.",
                "description": "Minimal instruction"
            },
            PromptingStrategy.DETERMINISTIC: {
                "prompt": "Complete this code with the most likely continuation:",
                "description": "Deterministic completion"
            },
            PromptingStrategy.NEGATIVE_PROMPT: {
                "prompt": "Complete the code. Do NOT include explanations, comments, or markdown. Code only.",
                "description": "Explicit negative prompting"
            },
            PromptingStrategy.FORMAT_CONSTRAINT: {
                "prompt": "Format: code_only. Complete: ",
                "description": "Format constraint"
            },
            PromptingStrategy.ROLE_BASED: {
                "prompt": "As a code autocomplete tool, complete this code:",
                "description": "Role-based instruction"
            }
        }

    def get_model_config(self, model_name: str = None) -> Dict[str, Any]:
        """Get conservative configuration for model"""
        target_model = model_name or self.model_name

        # Direct match
        if target_model in self.model_configs:
            return self.model_configs[target_model].copy()

        # Pattern matching for model families
        for model_key, config in self.model_configs.items():
            if model_key != "_default" and model_key.split(":")[0] in target_model:
                return config.copy()

        # Conservative default
        return self.model_configs["_default"].copy()

    def generate_with_strategy(self, prompt: str, strategy: PromptingStrategy, **kwargs) -> Phase1OllamaResponse:
        """Generate completion with smart response processing"""
        start_time = time.time()

        try:
            prompts = self.get_optimized_prompts()
            if strategy not in prompts:
                raise ValueError(f"Unknown strategy: {strategy}")

            strategy_config = prompts[strategy]
            model_config = self.get_model_config()

            # Create enhanced prompt
            enhanced_prompt = f"{strategy_config['prompt']}\n\n{prompt}"

            # Conservative options
            options = {
                "temperature": kwargs.get("temperature", model_config["temperature"]),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", model_config["max_tokens"]),
            }

            # Conservative stop tokens (Phase 1: avoid aggressive stopping)
            stop_tokens = kwargs.get("stop_tokens", model_config["stop_tokens"])
            if stop_tokens:
                options["stop"] = stop_tokens

            payload = {
                "model": self.model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": options
            }

            logger.debug(f"Phase 1 request: strategy={strategy.value}, model={self.model_name}")

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", 30)
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")
            execution_time = time.time() - start_time

            # Phase 1: Use smart response processor
            processed = self.response_processor.process_response(raw_response, http_success=True)

            return Phase1OllamaResponse(
                text=processed.text,
                raw_text=processed.raw_text,
                execution_time=execution_time,
                http_success=processed.http_success,
                content_quality_success=processed.content_quality_success,
                overall_success=processed.overall_success,
                quality_level=processed.quality_level,
                quality_score=processed.quality_score,
                prompting_strategy=strategy.value,
                is_conversational=processed.is_conversational,
                is_executable=processed.is_executable,
                has_syntax_errors=processed.has_syntax_errors,
                cleaning_applied=processed.cleaning_applied,
                metadata=processed.metadata
            )

        except requests.RequestException as e:
            execution_time = time.time() - start_time
            error_msg = f"HTTP request failed: {str(e)}"
            logger.error(error_msg)

            # Process empty response to get consistent structure
            processed = self.response_processor.process_response("", http_success=False)

            return Phase1OllamaResponse(
                text="",
                raw_text="",
                execution_time=execution_time,
                http_success=False,
                content_quality_success=False,
                overall_success=False,
                quality_level=ContentQuality.EMPTY,
                quality_score=0.0,
                prompting_strategy=strategy.value,
                is_conversational=False,
                is_executable=False,
                has_syntax_errors=False,
                cleaning_applied=False,
                error_message=error_msg
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)

            processed = self.response_processor.process_response("", http_success=False)

            return Phase1OllamaResponse(
                text="",
                raw_text="",
                execution_time=execution_time,
                http_success=False,
                content_quality_success=False,
                overall_success=False,
                quality_level=ContentQuality.EMPTY,
                quality_score=0.0,
                prompting_strategy=strategy.value,
                is_conversational=False,
                is_executable=False,
                has_syntax_errors=False,
                cleaning_applied=False,
                error_message=error_msg
            )

    def generate_with_fallback(self, prompt: str, max_attempts: int = 3, **kwargs) -> Phase1OllamaResponse:
        """Generate with fallback strategies prioritizing quality"""
        model_config = self.get_model_config()

        # Phase 1 fallback strategy: conservative and reliable
        fallback_strategies = [
            model_config["strategy"],
            PromptingStrategy.CODE_ENGINE,      # Most reliable
            PromptingStrategy.DETERMINISTIC,    # Conservative fallback
            PromptingStrategy.NEGATIVE_PROMPT   # Explicit instructions
        ]

        # Remove duplicates while preserving order
        unique_strategies = []
        for strategy in fallback_strategies:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)

        best_response = None
        for i, strategy in enumerate(unique_strategies[:max_attempts]):
            logger.debug(f"Phase 1 fallback attempt {i+1}/{max_attempts}: {strategy.value}")

            response = self.generate_with_strategy(prompt, strategy, **kwargs)

            # Phase 1 success criteria: overall_success (HTTP + content quality)
            if response.overall_success:
                logger.debug(f"Phase 1 success with strategy: {strategy.value}")
                return response

            # Keep the best response so far
            if best_response is None or response.quality_score > best_response.quality_score:
                best_response = response

        # Return the best response we found
        logger.warning(f"Phase 1 fallback exhausted, returning best response (quality: {best_response.quality_score:.3f})")
        return best_response

    def generate(self, prompt: str, strategy: Optional[PromptingStrategy] = None, **kwargs) -> Phase1OllamaResponse:
        """Main generation method with Phase 1 enhancements"""
        if strategy:
            return self.generate_with_strategy(prompt, strategy, **kwargs)

        return self.generate_with_fallback(prompt, **kwargs)

    def benchmark_strategies(self, test_prompt: str, expected_completion: str = None) -> List[Dict[str, Any]]:
        """Benchmark strategies with Phase 1 quality metrics"""
        results = []

        logger.info(f"Phase 1 benchmarking for model: {self.model_name}")

        for strategy in PromptingStrategy:
            logger.debug(f"Testing strategy: {strategy.value}")
            response = self.generate_with_strategy(test_prompt, strategy, max_tokens=50)

            # Phase 1 evaluation focuses on quality metrics
            result = {
                "strategy": strategy.value,
                "http_success": response.http_success,
                "content_quality_success": response.content_quality_success,
                "overall_success": response.overall_success,
                "quality_level": response.quality_level.value,
                "quality_score": response.quality_score,
                "is_conversational": response.is_conversational,
                "is_executable": response.is_executable,
                "has_syntax_errors": response.has_syntax_errors,
                "cleaning_applied": response.cleaning_applied,
                "execution_time": response.execution_time,
                "response_length": len(response.text),
                "response": response.text[:100] + "..." if len(response.text) > 100 else response.text,
                "error": response.error_message
            }

            # Optional expected completion check
            if expected_completion:
                contains_expected = expected_completion.lower() in response.text.lower()
                result["contains_expected"] = contains_expected

            results.append(result)

        return results

    def set_quality_threshold(self, threshold: float):
        """Update quality threshold for content evaluation"""
        self.response_processor.set_quality_threshold(threshold)

    def get_quality_report(self, response: Phase1OllamaResponse) -> Dict[str, Any]:
        """Generate detailed quality report"""
        # Create a ProcessedResponse for the report generator
        processed = ProcessedResponse(
            text=response.text,
            raw_text=response.raw_text,
            http_success=response.http_success,
            content_quality_success=response.content_quality_success,
            overall_success=response.overall_success,
            quality_level=response.quality_level,
            quality_score=response.quality_score,
            is_conversational=response.is_conversational,
            is_executable=response.is_executable,
            has_syntax_errors=response.has_syntax_errors,
            cleaning_applied=response.cleaning_applied,
            metadata=response.metadata or {}
        )

        return self.response_processor.get_quality_report(processed)

    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            for model in models:
                if model.get("name", "").startswith(self.model_name):
                    return model

            logger.warning(f"Model {self.model_name} not found")
            return {}

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

def create_phase1_interface(model_name: str, quality_threshold: float = 0.3) -> Phase1EnhancedOllamaInterface:
    """Create Phase 1 interface with default settings"""
    return Phase1EnhancedOllamaInterface(model_name, quality_threshold=quality_threshold)

def create_strict_interface(model_name: str) -> Phase1EnhancedOllamaInterface:
    """Create Phase 1 interface with strict quality requirements"""
    return Phase1EnhancedOllamaInterface(model_name, quality_threshold=0.6)

def create_research_interface(model_name: str) -> Phase1EnhancedOllamaInterface:
    """Create Phase 1 interface with lenient quality for research"""
    return Phase1EnhancedOllamaInterface(model_name, quality_threshold=0.1)