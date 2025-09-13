#!/usr/bin/env python3
"""
Debug Digit Token Mappings for Llama Models
-------------------------------------------
This script helps debug the correct token mappings for digits 0-9 in Llama models
and compares them with actual model generation to identify discrepancies.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import json

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers library not installed. Install with: pip install transformers torch")
    exit(1)


def get_optimal_device() -> str:
    """Determine the best available device for model inference."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    device = get_optimal_device()
    print(f"Loading {model_name} on {device.upper()}...")
    
    # Clear cache first
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    
    torch_dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    
    # Get HF token
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
    
    print(f"Successfully loaded {model_name}")
    return model, tokenizer, device


def analyze_digit_tokenization(tokenizer) -> Dict[str, Any]:
    """Analyze how digits 0-9 are tokenized in different contexts."""
    print("\n=== Digit Tokenization Analysis ===")
    
    results = {}
    
    for digit in range(10):
        digit_str = str(digit)
        
        # Test different tokenization contexts
        contexts = {
            "bare": digit_str,
            "space_before": f" {digit_str}",
            "space_after": f"{digit_str} ",
            "space_both": f" {digit_str} ",
            "newline_before": f"\n{digit_str}",
            "answer_context": f"Answer: {digit_str}",
            "standalone_answer": f"{digit_str}\n"
        }
        
        digit_results = {}
        for context_name, context_text in contexts.items():
            tokens = tokenizer.encode(context_text, add_special_tokens=False)
            decoded = [tokenizer.decode([t]) for t in tokens]
            
            digit_results[context_name] = {
                "tokens": tokens,
                "decoded": decoded,
                "num_tokens": len(tokens)
            }
        
        results[digit_str] = digit_results
        
        # Print summary for this digit
        print(f"\nDigit {digit}:")
        for context_name, info in digit_results.items():
            if info["num_tokens"] == 1:
                print(f"  {context_name:15}: token_id={info['tokens'][0]:3d}, decoded='{info['decoded'][0]}'")
            else:
                print(f"  {context_name:15}: {info['num_tokens']} tokens: {info['tokens']} -> {info['decoded']}")
    
    return results


def test_actual_generation(model, tokenizer, device, prompt: str) -> Dict[str, Any]:
    """Test actual model generation and capture the first token details."""
    print(f"\n=== Testing Actual Generation ===")
    print(f"Prompt: {prompt}")
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get logits for next token position
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[0, -1, :]  # Last position logits
    
    # Generate actual response
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False,  # temperature=0
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract first generated token
    input_length = input_ids.shape[1]
    first_token_id = generated[0][input_length].item()
    first_token_text = tokenizer.decode([first_token_id])
    first_token_logit = next_token_logits[first_token_id].item()
    
    # Get full generated text
    new_tokens = generated[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Find top 10 tokens by logit
    top_logits, top_indices = torch.topk(next_token_logits, 10)
    top_tokens = []
    for i, (logit, token_id) in enumerate(zip(top_logits, top_indices)):
        token_text = tokenizer.decode([token_id.item()])
        top_tokens.append({
            "rank": i + 1,
            "token_id": token_id.item(),
            "token_text": repr(token_text),
            "logit": logit.item()
        })
    
    print(f"Generated text: {repr(generated_text)}")
    print(f"First token: ID={first_token_id}, text={repr(first_token_text)}, logit={first_token_logit:.6f}")
    print(f"\nTop 10 tokens by logit:")
    for token_info in top_tokens:
        print(f"  Rank {token_info['rank']:2d}: ID={token_info['token_id']:5d}, logit={token_info['logit']:8.4f}, text={token_info['token_text']}")
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "first_token_id": first_token_id,
        "first_token_text": first_token_text,
        "first_token_logit": first_token_logit,
        "top_tokens": top_tokens,
        "input_length": input_length
    }


def get_digit_logits_detailed(model, tokenizer, device, prompt: str, tokenization_results: Dict) -> Dict[str, Any]:
    """Get detailed logits information for digits 0-9 using different tokenization strategies."""
    print(f"\n=== Detailed Digit Logits Analysis ===")
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get logits for next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]
    
    # Convert to numpy for ranking
    all_logits = logits.cpu().numpy()
    sorted_indices = np.argsort(-all_logits)  # Descending order
    rank_map = {idx: rank for rank, idx in enumerate(sorted_indices)}
    
    # Analyze each digit using different tokenization strategies
    digit_analysis = {}
    
    for digit in range(10):
        digit_str = str(digit)
        digit_info = {"digit": digit_str}
        
        # Get tokenization info for this digit
        digit_tokenization = tokenization_results[digit_str]
        
        # Find the best single-token representation
        single_token_candidates = []
        for context_name, token_info in digit_tokenization.items():
            if token_info["num_tokens"] == 1:
                token_id = token_info["tokens"][0]
                single_token_candidates.append({
                    "context": context_name,
                    "token_id": token_id,
                    "decoded": token_info["decoded"][0],
                    "logit": float(logits[token_id].cpu()),
                    "rank": rank_map.get(token_id, -1)
                })
        
        # Sort by logit (highest first)
        single_token_candidates.sort(key=lambda x: x["logit"], reverse=True)
        
        digit_info["single_token_candidates"] = single_token_candidates
        
        # Use the highest-logit single token as the "best" representation
        if single_token_candidates:
            best_token = single_token_candidates[0]
            digit_info["best_token_id"] = best_token["token_id"]
            digit_info["best_logit"] = best_token["logit"]
            digit_info["best_rank"] = best_token["rank"]
            digit_info["best_context"] = best_token["context"]
        else:
            digit_info["best_token_id"] = None
            digit_info["best_logit"] = -float('inf')
            digit_info["best_rank"] = -1
            digit_info["best_context"] = None
        
        digit_analysis[digit_str] = digit_info
        
        print(f"\nDigit {digit}:")
        if single_token_candidates:
            for i, candidate in enumerate(single_token_candidates):
                marker = "★" if i == 0 else " "
                print(f"  {marker} {candidate['context']:15}: ID={candidate['token_id']:5d}, logit={candidate['logit']:8.4f}, rank={candidate['rank']:3d}, text={repr(candidate['decoded'])}")
        else:
            print(f"    No single-token representations found")
    
    # Calculate softmax probabilities among best digit tokens
    valid_digits = [d for d in digit_analysis.values() if d["best_token_id"] is not None]
    if valid_digits:
        best_logits = np.array([d["best_logit"] for d in valid_digits])
        softmax_probs = np.exp(best_logits - np.max(best_logits))
        softmax_probs = softmax_probs / np.sum(softmax_probs)
        
        for i, digit_info in enumerate(valid_digits):
            digit_info["softmax_prob"] = float(softmax_probs[i])
    
    return digit_analysis


def main():
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Test prompt (similar to MSIT task)
    test_prompt = "MSIT Task Instruction: \n\n    Goal:, there's a number that only appears once; report its position.\n    - Counting from left to right, the candidate positions are 1, 2, 3, 4. \n    \n    Important:\n    - Answer with a single number only.\n\n\ntask\n1 0 0 0 -> Answer:"
    
    print(f"Debug Digit Token Mappings for {model_name}")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    # Analyze digit tokenization
    tokenization_results = analyze_digit_tokenization(tokenizer)
    
    # Test actual generation
    generation_results = test_actual_generation(model, tokenizer, device, test_prompt)
    
    # Get detailed digit logits
    digit_logits_results = get_digit_logits_detailed(model, tokenizer, device, test_prompt, tokenization_results)
    
    # Summary comparison
    print(f"\n=== Summary Comparison ===")
    print(f"Model generated: {repr(generation_results['first_token_text'])}")
    print(f"First token ID: {generation_results['first_token_id']}")
    print(f"First token logit: {generation_results['first_token_logit']:.6f}")
    
    # Check if first token matches any digit
    first_token_id = generation_results['first_token_id']
    matching_digit = None
    for digit_str, info in digit_logits_results.items():
        if info['best_token_id'] == first_token_id:
            matching_digit = digit_str
            break
    
    if matching_digit:
        print(f"✓ First token matches digit {matching_digit}")
        print(f"  Digit {matching_digit} logit: {digit_logits_results[matching_digit]['best_logit']:.6f}")
        print(f"  Digit {matching_digit} rank: {digit_logits_results[matching_digit]['best_rank']}")
    else:
        print(f"✗ First token does not match any digit 0-9")
        print(f"  Checking if any digit has higher logit than first token...")
        for digit_str, info in digit_logits_results.items():
            if info['best_logit'] > generation_results['first_token_logit']:
                print(f"    Digit {digit_str}: logit={info['best_logit']:.6f} > {generation_results['first_token_logit']:.6f}")
    
    # Save results to JSON
    output_data = {
        "model_name": model_name,
        "test_prompt": test_prompt,
        "tokenization_analysis": tokenization_results,
        "generation_results": generation_results,
        "digit_logits_analysis": digit_logits_results
    }
    
    output_file = f"debug_digit_tokens_{model_name.replace('/', '_').replace('-', '_')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Clean up
    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
