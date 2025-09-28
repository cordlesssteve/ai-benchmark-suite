#!/usr/bin/env python3
"""
HuggingFace Model Interface (Sprint 4.0)

Enterprise-grade HuggingFace Transformers integration for AI Benchmark Suite with:
- Automatic model loading and caching
- GPU/CPU optimization and device management
- Batch processing and memory optimization
- Support for popular code generation models
- Integration with Sprint 3.0 performance optimizations
"""

import torch
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# HuggingFace ecosystem
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        PreTrainedModel, PreTrainedTokenizer,
        GenerationConfig, BitsAndBytesConfig
    )
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    import torch.nn.functional as F
    HF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HuggingFace transformers not available: {e}")
    HF_AVAILABLE = False

# Base interface
try:
    from .base_interface import ModelInterface, GenerationResult
except ImportError:
    # For standalone testing, create minimal base classes
    @dataclass
    class GenerationResult:
        text: str
        success: bool
        execution_time: float
        error_message: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    class ModelInterface:
        def generate(self, prompt: str, **kwargs) -> GenerationResult:
            raise NotImplementedError


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace model interface"""
    model_name: str
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "auto"   # "auto", "float16", "bfloat16", "int8", "int4"
    max_memory: Optional[Dict[str, str]] = None
    use_cache: bool = True
    trust_remote_code: bool = False
    revision: Optional[str] = None

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1

    # Performance optimizations
    use_flash_attention: bool = True
    torch_compile: bool = False
    offload_folder: Optional[str] = None


class ModelSize(Enum):
    """Model size categories for optimization"""
    SMALL = "small"      # < 1B parameters
    MEDIUM = "medium"    # 1B - 7B parameters
    LARGE = "large"      # 7B - 30B parameters
    XLARGE = "xlarge"    # > 30B parameters


class HuggingFaceInterface(ModelInterface):
    """
    Enterprise HuggingFace model interface with advanced optimizations.

    Features:
    - Automatic device placement and memory optimization
    - Support for quantization (int8, int4) for large models
    - Batch processing with dynamic batching
    - Model caching and efficient loading
    - Integration with Sprint 3.0 performance systems
    """

    def __init__(self, config: HuggingFaceConfig):
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace transformers not available. Install with: pip install transformers torch")

        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device = None
        self.model_size = None
        self.generation_config = None

        # Performance tracking
        self.load_time = 0.0
        self.total_generations = 0
        self.total_generation_time = 0.0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize tokenizer and model with optimizations"""
        self.logger.info(f"Loading HuggingFace model: {self.config.model_name}")
        start_time = time.time()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.revision
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device and dtype
            self.device = self._get_optimal_device()
            dtype = self._get_optimal_dtype()

            # Configure model loading based on size
            self.model_size = self._estimate_model_size()
            model_kwargs = self._get_model_loading_kwargs(dtype)

            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )

            # Apply post-loading optimizations
            self._apply_model_optimizations()

            # Setup generation config
            self._setup_generation_config()

            self.load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            self.logger.info(f"Device: {self.device}, Model size: {self.model_size.value}")

        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise

    def _get_optimal_device(self) -> torch.device:
        """Determine optimal device for model"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                # Choose GPU with most memory
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory.append((i, memory))

                if gpu_memory:
                    best_gpu = max(gpu_memory, key=lambda x: x[1])[0]
                    device = torch.device(f"cuda:{best_gpu}")
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        return device

    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine optimal dtype for model"""
        if self.config.dtype == "auto":
            if self.device.type == "cuda":
                # Use bfloat16 if supported, otherwise float16
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32
        else:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            return dtype_map.get(self.config.dtype, torch.float32)

    def _estimate_model_size(self) -> ModelSize:
        """Estimate model size category based on model name"""
        model_name_lower = self.config.model_name.lower()

        # Common patterns for model sizes
        if any(x in model_name_lower for x in ["125m", "350m", "760m", "small"]):
            return ModelSize.SMALL
        elif any(x in model_name_lower for x in ["1b", "1.3b", "2.7b", "6.7b", "7b", "medium"]):
            return ModelSize.MEDIUM
        elif any(x in model_name_lower for x in ["13b", "15b", "20b", "30b", "large"]):
            return ModelSize.LARGE
        else:
            # Default to large for unknown sizes to be safe
            return ModelSize.LARGE

    def _get_model_loading_kwargs(self, dtype: torch.dtype) -> Dict[str, Any]:
        """Get model loading kwargs based on configuration and size"""
        kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.config.trust_remote_code,
            "revision": self.config.revision,
        }

        # Device placement
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"

            # Memory optimization for large models
            if self.model_size in [ModelSize.LARGE, ModelSize.XLARGE]:
                if self.config.max_memory:
                    kwargs["max_memory"] = self.config.max_memory

                # Quantization for very large models
                if self.config.dtype == "int8":
                    kwargs["load_in_8bit"] = True
                elif self.config.dtype == "int4":
                    kwargs["load_in_4bit"] = True
                    kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
                    kwargs["bnb_4bit_use_double_quant"] = True
        else:
            # CPU loading
            kwargs["device_map"] = "cpu"
            kwargs["torch_dtype"] = torch.float32  # CPU works best with float32

        # Offloading for large models
        if self.config.offload_folder and self.model_size == ModelSize.XLARGE:
            kwargs["offload_folder"] = self.config.offload_folder

        return kwargs

    def _apply_model_optimizations(self):
        """Apply post-loading model optimizations"""
        if self.model is None:
            return

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Set model to eval mode
        self.model.eval()

        # Torch compile for supported models and PyTorch versions
        if self.config.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Applied torch.compile optimization")
            except Exception as e:
                self.logger.warning(f"Failed to apply torch.compile: {e}")

    def _setup_generation_config(self):
        """Setup generation configuration"""
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            num_return_sequences=self.config.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.config.use_cache,
        )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using the HuggingFace model"""
        if not self.is_available():
            return GenerationResult(
                text="",
                success=False,
                execution_time=0.0,
                error_message="Model not available"
            )

        start_time = time.time()

        try:
            # Override generation config with kwargs
            generation_config = self._update_generation_config(kwargs)

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Generate with memory and error handling
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode generated text
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            execution_time = time.time() - start_time

            # Update statistics
            self.total_generations += 1
            self.total_generation_time += execution_time

            # Create metadata
            metadata = {
                "model_name": self.config.model_name,
                "prompt_tokens": inputs['input_ids'].shape[1],
                "generated_tokens": len(generated_ids),
                "device": str(self.device),
                "generation_config": generation_config.to_dict(),
                "avg_generation_time": self.total_generation_time / self.total_generations
            }

            return GenerationResult(
                text=generated_text,
                success=True,
                execution_time=execution_time,
                metadata=metadata
            )

        except torch.cuda.OutOfMemoryError as e:
            # Handle CUDA OOM gracefully
            self.logger.error("CUDA out of memory during generation")
            torch.cuda.empty_cache()
            gc.collect()

            return GenerationResult(
                text="",
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"CUDA out of memory: {e}"
            )

        except Exception as e:
            return GenerationResult(
                text="",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def generate_batch(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate text for multiple prompts in batch"""
        if not self.is_available():
            return [GenerationResult("", False, 0.0, "Model not available") for _ in prompts]

        results = []
        batch_size = kwargs.get('batch_size', 4)

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self._generate_batch_internal(batch, **kwargs)
            results.extend(batch_results)

        return results

    def _generate_batch_internal(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Internal batch generation with padding"""
        start_time = time.time()

        try:
            generation_config = self._update_generation_config(kwargs)

            # Tokenize all prompts with padding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Reasonable limit
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True
                )

            # Decode each generated sequence
            results = []
            for i, sequence in enumerate(outputs.sequences):
                # Extract only the newly generated tokens
                prompt_length = inputs['attention_mask'][i].sum().item()
                generated_ids = sequence[prompt_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                results.append(GenerationResult(
                    text=generated_text,
                    success=True,
                    execution_time=(time.time() - start_time) / len(prompts),  # Amortized time
                    metadata={
                        "model_name": self.config.model_name,
                        "batch_size": len(prompts),
                        "prompt_index": i
                    }
                ))

            return results

        except Exception as e:
            # Return error for all prompts in batch
            execution_time = time.time() - start_time
            return [
                GenerationResult("", False, execution_time / len(prompts), str(e))
                for _ in prompts
            ]

    def _update_generation_config(self, kwargs: Dict[str, Any]) -> GenerationConfig:
        """Update generation config with provided kwargs"""
        config_dict = self.generation_config.to_dict()

        # Update with provided parameters
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                config_dict[key] = value

        return GenerationConfig(**config_dict)

    def is_available(self) -> bool:
        """Check if the model is loaded and available"""
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        if not self.is_available():
            return {"available": False}

        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Memory usage
        model_memory = 0
        if self.device.type == "cuda":
            model_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB

        return {
            "available": True,
            "model_name": self.config.model_name,
            "model_size": self.model_size.value,
            "device": str(self.device),
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_usage_gb": model_memory,
            "load_time": self.load_time,
            "total_generations": self.total_generations,
            "avg_generation_time": self.total_generation_time / max(self.total_generations, 1),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown')
        }

    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self.logger.info("Model resources cleaned up")


# Predefined configurations for popular models
POPULAR_MODELS = {
    "codellama-7b": HuggingFaceConfig(
        model_name="codellama/CodeLlama-7b-Python-hf",
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9
    ),
    "codellama-13b": HuggingFaceConfig(
        model_name="codellama/CodeLlama-13b-Python-hf",
        dtype="int8",  # Use quantization for larger model
        max_new_tokens=512,
        temperature=0.1
    ),
    "starcoder": HuggingFaceConfig(
        model_name="bigcode/starcoder",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95
    ),
    "phi-1": HuggingFaceConfig(
        model_name="microsoft/phi-1",
        max_new_tokens=256,
        temperature=0.1,
        trust_remote_code=True
    ),
    "phi-1.5": HuggingFaceConfig(
        model_name="microsoft/phi-1_5",
        max_new_tokens=512,
        temperature=0.1,
        trust_remote_code=True
    ),
    "deepseek-coder-1b": HuggingFaceConfig(
        model_name="deepseek-ai/deepseek-coder-1.3b-base",
        max_new_tokens=512,
        temperature=0.2
    ),
    "deepseek-coder-6b": HuggingFaceConfig(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        max_new_tokens=512,
        temperature=0.2
    )
}


def create_huggingface_interface(model_name: str, **kwargs) -> HuggingFaceInterface:
    """Factory function to create HuggingFace interface with optimal settings"""

    # Use predefined config if available
    if model_name in POPULAR_MODELS:
        config = POPULAR_MODELS[model_name]
        # Override with provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        # Create default config
        config = HuggingFaceConfig(model_name=model_name, **kwargs)

    return HuggingFaceInterface(config)


# Testing and demonstration
if __name__ == "__main__":
    print("🤗 HuggingFace Interface Demo")
    print("=" * 50)

    if not HF_AVAILABLE:
        print("❌ HuggingFace transformers not available")
        print("Install with: pip install transformers torch")
        exit(1)

    # Test with a small model for demonstration
    model_name = "microsoft/DialoGPT-small"  # Small model for testing

    try:
        config = HuggingFaceConfig(
            model_name=model_name,
            max_new_tokens=50,
            temperature=0.7
        )

        print(f"Loading model: {model_name}")
        interface = HuggingFaceInterface(config)

        if interface.is_available():
            print("✅ Model loaded successfully")

            # Test generation
            test_prompt = "def fibonacci(n):"
            print(f"\nTest prompt: {test_prompt}")

            result = interface.generate(test_prompt)
            if result.success:
                print(f"Generated: {result.text}")
                print(f"Time: {result.execution_time:.2f}s")
            else:
                print(f"Generation failed: {result.error_message}")

            # Show model info
            info = interface.get_model_info()
            print(f"\nModel info:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            # Cleanup
            interface.cleanup()
            print("\n✅ Demo completed!")

        else:
            print("❌ Model not available")

    except Exception as e:
        print(f"❌ Error: {e}")