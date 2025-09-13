#!/usr/bin/env python3
"""
Local Model Handler for local Hugging Face causal LMs (GPT-2, LLaMA, etc.)
-------------------------------------------------------------------------
Provides a local model interface compatible with the MSIT API test system.
Supports GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl) and other
Hugging Face causal LMs such as Meta LLaMA (e.g., "meta-llama/Llama-3.2-1B").
"""

import logging
import torch
import psutil
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MPS memory management for Apple Silicon
if torch.backends.mps.is_available():
    # Disable MPS memory limit to prevent memory allocation issues
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    logger.info("Configured MPS memory management for Apple Silicon")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not found. Install with: pip install transformers torch")


class LocalModelHandler:
    """Handler for local models using direct loading (memory-efficient)."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
    
    def cleanup_models(self):
        """Clear all loaded models and free memory."""
        logger.info("Cleaning up loaded models...")
        
        # Clear model references
        self._models.clear()
        self._tokenizers.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear device caches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    
    @classmethod
    def cleanup_all_models(cls):
        """Static method to cleanup models without instance."""
        import gc
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleared device caches")
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device for model inference."""
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU via Metal Performance Shaders
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"  # CPU fallback
    
    def _load_hf_causal_model(self, model_name: str) -> bool:
        """Load a generic HF causal LM (e.g., LLaMA) and tokenizer."""
        if not HAS_TRANSFORMERS:
            logger.error("transformers library required for local models")
            return False

        try:
            device = self._get_optimal_device()
            logger.info(f"Loading {model_name} model and tokenizer on {device.upper()}...")

            # Clear any existing cache first
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

            # dtype selection
            torch_dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

            # Optional HF token for gated models (e.g., Meta LLaMA)
            hf_token = (
                os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                token=hf_token,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                token=hf_token,
            ).to(device).eval()

            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer

            logger.info(f"Successfully loaded {model_name} on {device.upper()}")

            # Log memory
            memory_stats = self.get_system_memory_usage()
            if device == "mps":
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                logger.info(f"MPS GPU memory: {allocated:.2f} GB | System RAM: {memory_stats['process_rss_gb']:.2f} GB")
            else:
                logger.info(f"System RAM after model load: {memory_stats['process_rss_gb']:.2f} GB")

            return True

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False

    def _load_gpt2_model(self, model_name: str) -> bool:
        """Load a GPT2 model and tokenizer using direct loading (memory-efficient)."""
        if not HAS_TRANSFORMERS:
            logger.error("transformers library required for local models")
            return False
        
        try:
            device = self._get_optimal_device()
            logger.info(f"Loading {model_name} model and tokenizer on {device.upper()}...")
            
            # Clear any existing cache first
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            
            # Device-specific dtype selection (following example code pattern)
            torch_dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
            
            # Direct model loading (memory-efficient approach)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Memory optimization options with 8-bit quantization
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            
            # For large models, use 8-bit quantization to reduce memory usage
            if "8B" in model_name or "7B" in model_name or "13B" in model_name:
                try:
                    # Import bitsandbytes for quantization
                    import bitsandbytes as bnb
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"
                    logger.info(f"Loading {model_name} with 8-bit quantization to reduce memory usage")
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to device mapping")
                    model_kwargs["device_map"] = "auto"
                except Exception as e:
                    logger.warning(f"8-bit quantization failed: {e}, falling back to device mapping")
                    model_kwargs["device_map"] = "auto"
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Only move to device if device_map wasn't used
            if "device_map" not in model_kwargs:
                model = model.to(device)
            
            model = model.eval()
            
            # Store model and tokenizer
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name} on {device.upper()}")
            
            # Log memory usage after model load
            memory_stats = self.get_system_memory_usage()
            if device == "mps":
                allocated = torch.mps.current_allocated_memory() / (1024**3)  # GB
                logger.info(f"MPS GPU memory: {allocated:.2f} GB | System RAM: {memory_stats['process_rss_gb']:.2f} GB")
            else:
                logger.info(f"System RAM after model load: {memory_stats['process_rss_gb']:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False
    
    def is_gpt2_model(self, model_name: str) -> bool:
        """Check if model name corresponds to a GPT2 model."""
        gpt2_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        return model_name.lower() in gpt2_models

    def is_hf_causal_model(self, model_name: str) -> bool:
        """Heuristic to decide if a name looks like an HF causal LM (e.g., LLaMA)."""
        name = model_name.lower()
        return (
            self.is_gpt2_model(name)
            or "llama" in name
            or "/" in name  # repo IDs like meta-llama/Llama-3.2-1B
        )
    
    def get_digit_logits_info(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """
        Get logits information for digits 0-9 at the actual digit token position.
        This method generates tokens and finds the first digit token to analyze.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            
        Returns:
            Dictionary containing logits, ranks, and probabilities for digits 0-9
        """
        if not HAS_TRANSFORMERS:
            return {"error": "transformers library not installed"}
        
        # Ensure model is loaded
        model_key = model_name
        if self.is_gpt2_model(model_name.lower()):
            if model_key not in self._models:
                if not self._load_gpt2_model(model_name.lower()):
                    return {"error": f"Failed to load model {model_name}"}
        elif self.is_hf_causal_model(model_name):
            if model_key not in self._models:
                if not self._load_hf_causal_model(model_name):
                    return {"error": f"Failed to load model {model_name}"}
        else:
            return {"error": f"Unsupported local model: {model_name}"}
        
        try:
            model = self._models[model_key]
            tokenizer = self._tokenizers[model_key]
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            # Generate a few tokens to find the first digit position
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,  # Generate up to 10 tokens to find first digit
                    do_sample=False,  # temperature=0
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True  # Get logits for each generated position
                )
            
            # Find the first digit token in generated sequence
            input_length = input_ids.shape[1]
            generated_tokens = generated.sequences[0][input_length:]
            digit_token_ids = set(range(15, 25))  # Token IDs for digits 0-9
            
            digit_position = None
            digit_token_id = None
            all_generated_tokens = []
            
            for i, token_id in enumerate(generated_tokens):
                token_text = tokenizer.decode([token_id.item()])
                all_generated_tokens.append({
                    "position": i,
                    "token_id": token_id.item(),
                    "token_text": token_text
                })
                
                if token_id.item() in digit_token_ids and digit_position is None:
                    digit_position = i
                    digit_token_id = token_id.item()
                    break
            
            # If no digit found in first 10 tokens, fall back to next token position
            if digit_position is None:
                digit_position = 0
                # Get logits for the immediate next token position
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0, -1, :]
            else:
                # Use the logits from the position just before the digit
                # generated.scores[i] contains logits for position i (0-indexed)
                if digit_position < len(generated.scores):
                    logits = generated.scores[digit_position][0]  # [0] for batch dimension
                else:
                    # Fallback to next token position if scores not available
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits[0, -1, :]
            
            # Get token IDs for digits 0-9
            digit_tokens = {}
            digit_logits = {}
            
            for digit in range(10):
                digit_str = str(digit)
                # Try different tokenizations of the digit
                possible_tokens = [
                    tokenizer.encode(digit_str, add_special_tokens=False),
                    tokenizer.encode(f" {digit_str}", add_special_tokens=False),
                    tokenizer.encode(f"{digit_str} ", add_special_tokens=False)
                ]
                
                # Use the first valid single-token encoding
                token_id = None
                for tokens in possible_tokens:
                    if len(tokens) == 1:
                        token_id = tokens[0]
                        break
                
                if token_id is not None:
                    digit_tokens[digit] = token_id
                    digit_logits[digit] = float(logits[token_id].cpu())
            
            # Convert all logits to numpy for ranking
            all_logits = logits.cpu().numpy()
            
            # Get global ranks for each digit
            global_ranks = {}
            sorted_indices = np.argsort(-all_logits)  # Sort in descending order
            rank_map = {idx: rank for rank, idx in enumerate(sorted_indices)}
            
            for digit, token_id in digit_tokens.items():
                if token_id in rank_map:
                    global_ranks[digit] = int(rank_map[token_id])  # 0-indexed rank
                else:
                    global_ranks[digit] = -1
            
            # Calculate softmax probabilities among digits 0-9
            digit_logits_array = np.array([digit_logits.get(i, -np.inf) for i in range(10)])
            # Handle case where some digits might not have valid tokens
            valid_mask = digit_logits_array != -np.inf
            if np.any(valid_mask):
                softmax_probs = np.full(10, 0.0)
                valid_logits = digit_logits_array[valid_mask]
                valid_softmax = np.exp(valid_logits - np.max(valid_logits))
                valid_softmax = valid_softmax / np.sum(valid_softmax)
                softmax_probs[valid_mask] = valid_softmax
            else:
                softmax_probs = np.full(10, 0.1)  # Uniform if no valid tokens
            
            # Clean up tensors
            del inputs, input_ids, attention_mask, generated
            if model.device.type == "mps":
                torch.mps.empty_cache()
            elif model.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "digit_logits": {str(k): v for k, v in digit_logits.items()},
                "digit_global_ranks": {str(k): v for k, v in global_ranks.items()},
                "digit_softmax_probs": {str(k): float(softmax_probs[k]) for k in range(10)},
                "digit_token_ids": {str(k): v for k, v in digit_tokens.items()},
                "analysis_position": digit_position,
                "generated_tokens_sequence": all_generated_tokens,
                "first_digit_found": digit_token_id is not None,
                "first_digit_token_id": digit_token_id
            }
            
        except Exception as e:
            logger.error(f"Error getting digit logits for {model_name}: {e}")
            return {"error": f"Failed to get digit logits: {str(e)}"}
    
    def generate(self, model_name: str, prompt: str, max_tokens: int = 100, 
                temperature: float = 0.0) -> str:
        """
        Generate text using direct model generation (memory-efficient).
        
        Args:
            model_name: Name of the model (e.g., 'gpt2-large')
            prompt: Input prompt
            max_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation (0.0 = deterministic)
        
        Returns:
            Generated text (continuation of prompt)
        """
        # Auto-optimize max_tokens for MSIT tasks (responses are typically short)
        prompt_len = len(prompt)
        logger.info(f"Input: prompt_len={prompt_len}, max_tokens={max_tokens}")
        if max_tokens > 200 and prompt_len > 500:
            logger.info(f"Auto-optimizing max_tokens from {max_tokens} to 50 for long prompt")
            max_tokens = 50  # MSIT responses are typically 1-2 lines
        elif max_tokens > 500:  # Always cap high token counts
            logger.info(f"Capping max_tokens from {max_tokens} to 100 for memory safety")
            max_tokens = 100
        if not HAS_TRANSFORMERS:
            return "ERROR: transformers library not installed. Install with: pip install transformers torch"
        
        # Decide model family and ensure loaded
        model_key = model_name  # preserve case for HF repo IDs
        if self.is_gpt2_model(model_name.lower()):
            if model_key not in self._models:
                if not self._load_gpt2_model(model_name.lower()):
                    return f"ERROR: Failed to load model {model_name}"
        elif self.is_hf_causal_model(model_name):
            if model_key not in self._models:
                if not self._load_hf_causal_model(model_name):
                    return f"ERROR: Failed to load model {model_name}"
        else:
            return f"ERROR: Unsupported local model: {model_name}"
        
        try:
            model = self._models[model_key]
            tokenizer = self._tokenizers[model_key]
            
            # Truncate very long prompts to prevent memory explosion
            # Use model_max_length if available; apply a safety cap
            try:
                model_max_len = int(getattr(tokenizer, 'model_max_length', 1024))
            except Exception:
                model_max_len = 1024
            safety_cap = min(max(model_max_len, 512), 2048)

            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=safety_cap,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            # Generation settings
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            
            # Generate with memory management
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)
            
            # Decode only the new tokens (exclude input)
            input_length = input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clear intermediate tensors to free memory
            del inputs, input_ids, attention_mask, outputs
            if model.device.type == "mps":
                torch.mps.empty_cache()
                allocated_after = torch.mps.current_allocated_memory() / (1024**3)
                memory_after = self.get_system_memory_usage()
                logger.info(f"After generation - MPS: {allocated_after:.2f} GB | System RAM: {memory_after['process_rss_gb']:.2f} GB")
            elif model.device.type == "cuda":
                torch.cuda.empty_cache()
                memory_after = self.get_system_memory_usage()
                logger.info(f"After generation - System RAM: {memory_after['process_rss_gb']:.2f} GB")
            
            return generated_text  # Preserve complete raw output including leading/trailing whitespace
                
        except Exception as e:
            logger.error(f"Error generating with {model_name}: {e}")
            return f"ERROR: Generation failed: {str(e)}"
    
    def clear_cache(self):
        """Clear loaded models to free memory."""
        self._models.clear()
        self._tokenizers.clear()
        
        # Clear device-specific caches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("MPS cache cleared")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        logger.info("Model cache cleared")
    
    def clear_generation_cache(self):
        """Clear only generation cache, keep models loaded to avoid fragmentation."""
        # Clear device-specific caches only
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("MPS generation cache cleared")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA generation cache cleared")
        
        memory_after_clear = self.get_system_memory_usage()
        logger.info(f"Generation cache cleared (models kept loaded) - System RAM: {memory_after_clear['process_rss_gb']:.2f} GB")
    
    def get_system_memory_usage(self) -> Dict[str, float]:
        """Get current system memory usage in GB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
            'process_vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
            'system_used_gb': system_memory.used / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_percent': system_memory.percent
        }


# Global handler instance
_local_handler = LocalModelHandler()


def call_local_model(model_name: str, prompt: str, max_tokens: int, 
                    temperature: float = 0.0) -> str:
    """
    Call local model API (compatible interface with other API calls).
    
    Args:
        model_name: Name of the local model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
    
    Returns:
        Generated response text
    """
    return _local_handler.generate(model_name, prompt, max_tokens, temperature)


def get_local_model_digit_logits(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    Get logits information for digits 0-9 from local model.
    
    Args:
        model_name: Name of the local model
        prompt: Input prompt
        
    Returns:
        Dictionary containing logits, ranks, and probabilities for digits 0-9
    """
    return _local_handler.get_digit_logits_info(model_name, prompt)


def is_local_model(model_name: str) -> bool:
    """Check if model name corresponds to a local model."""
    return _local_handler.is_hf_causal_model(model_name)


def clear_local_cache():
    """Clear local model cache."""
    _local_handler.clear_cache()

def clear_local_generation_cache():
    """Clear only generation cache, keep models loaded."""
    _local_handler.clear_generation_cache()

def get_memory_usage():
    """Get current system memory usage."""
    return _local_handler.get_system_memory_usage()


if __name__ == "__main__":
    # Simple test
    if HAS_TRANSFORMERS:
        print("Testing GPT2 local model...")
        response = call_local_model("gpt2", "The meaning of life is", 50)
        print(f"Response: {response}")
    else:
        print("transformers library not installed")
