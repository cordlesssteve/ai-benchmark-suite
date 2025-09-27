#!/usr/bin/env python3
"""
Ollama Model Interface

Provides unified interface for local Ollama models in the benchmarking suite.
"""

import requests
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class OllamaResponse:
    """Response from Ollama model"""
    text: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class OllamaInterface:
    """Interface for Ollama local models"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()

    def generate(self, prompt: str, **kwargs) -> OllamaResponse:
        """Generate text using Ollama model"""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.2),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 2048),
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", 60)
            )
            response.raise_for_status()

            result = response.json()
            execution_time = time.time() - start_time

            return OllamaResponse(
                text=result.get("response", ""),
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return OllamaResponse(
                text="",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

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