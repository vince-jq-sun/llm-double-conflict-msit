#!/usr/bin/env python3
"""
MSIT API Test Script
-------------------
Test script for MSIT generation using various LLM APIs.
Generates stimuli, calls API, and saves results for analysis.
"""

import os
import json
import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

def save_json_compact(data: Dict[str, Any], filepath: Path):
    """Save JSON with compact array formatting."""
    # First save with standard formatting
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    
    # Post-process to make numeric arrays compact
    import re
    
    # Pattern to match arrays with mixed content (numbers, strings, etc.)
    # This matches: [\n    1,\n    "t",\n    3\n  ] or [\n    1,\n    2,\n    3\n  ]
    pattern = r'\[\s*\n(\s*(?:-?\d+|"[^"]*")(?:,\s*\n\s*(?:-?\d+|"[^"]*"))*)\s*\n\s*\]'
    
    def compact_array(match):
        content = match.group(1)
        # Extract all values (numbers and quoted strings)
        values = re.findall(r'-?\d+|"[^"]*"', content)
        # Create compact format
        return '[' + ', '.join(values) + ']'
    
    # Apply compacting to arrays
    json_str = re.sub(pattern, compact_array, json_str)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)

# Import MSIT generation functions
from msit_gen import gen_msit_rep_sequence, msit_instruciton, as_prompt_line

# Import local model handler
try:
    from local_model_handler import call_local_model, is_local_model, get_local_model_digit_logits
    HAS_LOCAL_MODELS = True
except ImportError:
    HAS_LOCAL_MODELS = False
    call_local_model = None
    is_local_model = None
    get_local_model_digit_logits = None

# Import ACDC circuit discovery
try:
    from acdc_circuit_discovery import (
        ACDCCircuitDiscovery, ACDCConfig, run_msit_acdc_analysis, 
        create_msit_corrupted_data, HAS_ACDC
    )
except ImportError:
    HAS_ACDC = False
    ACDCCircuitDiscovery = None
    ACDCConfig = None
    run_msit_acdc_analysis = None

# API client imports
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None
    

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

try:
    import openai as grok_openai  # Grok uses OpenAI-compatible API
    HAS_GROK = True
except ImportError:
    HAS_GROK = False
    grok_openai = None

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

try:
    import requests
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    requests = None


def load_api_keys(api_file_path: str = "API.json") -> Dict[str, str]:
    """Load API keys from JSON file."""
    try:
        with open(api_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"API file {api_file_path} not found. Please ensure it exists.")
        return {}


def get_openai_client(api_key: str):
    """Initialize OpenAI client."""
    return openai.OpenAI(api_key=api_key)


def get_gemini_client(api_key: str):
    """Initialize Gemini client."""
    if not HAS_GEMINI:
        return None
    client = genai.Client(api_key=api_key)
    return client


def get_anthropic_client(api_key: str):
    """Initialize Anthropic client."""
    return anthropic.Anthropic(api_key=api_key)


def get_grok_client(api_key: str, base_url: str):
    """Initialize Grok client using OpenAI-compatible interface."""
    return grok_openai.OpenAI(api_key=api_key, base_url=base_url)


def call_ollama_api(model_name: str, prompt: str, max_tokens: int, 
                   temperature: float = 0.0, base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API."""
    if not HAS_OLLAMA:
        return "ERROR: requests library not installed (required for Ollama API)"
    
    try:
        # Ollama API endpoint
        url = f"{base_url}/api/generate"
        
        # Prepare request data
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Make request
        response = requests.post(url, json=data, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result.get("response", "ERROR: No response field in Ollama output")
        
    except requests.exceptions.RequestException as e:
        return f"ERROR: Ollama API request failed: {str(e)}"
    except Exception as e:
        return f"ERROR: Ollama API error: {str(e)}"


def call_openai_api(client, model_name: str, prompt: str, 
                   max_tokens: int, temperature: float = 0.0) -> str:
    """Call OpenAI API."""
    if "gpt-5" in model_name:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"ERROR: {str(e)}"
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return f"ERROR: {str(e)}"



def call_gemini_api(client, model_name: str, prompt: str, 
                   max_tokens: int, temperature: float = 0.0) -> str:
    """Call Gemini API."""
    if not HAS_GEMINI:
        return "ERROR: Google Generative AI library not installed"
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens  # 设置最大输出 token 数
            }
        )
        return response.text

    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"ERROR: {str(e)}"


def call_anthropic_api(client, model_name: str, prompt: str,
                      max_tokens: int, temperature: float = 0.0) -> str:
    """Call Anthropic API."""
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return f"ERROR: {str(e)}"


def call_grok_api(client, model_name: str, prompt: str,
                 max_tokens: int, temperature: float = 0.0) -> str:
    """Call Grok API using OpenAI-compatible interface."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Grok API error: {e}")
        return f"ERROR: {str(e)}"


def determine_api_type(model_name: str) -> str:
    """Determine API type based on model name."""
    model_name_lower = model_name.lower()
    
    # Check for Ollama models first (before local models)
    # Ollama models typically have format like "llama3.1:8b-instruct-q4_0"
    if any(x in model_name_lower for x in ['llama3.1', 'llama3.2', 'ollama']) or ':' in model_name:
        return 'ollama'
    # Check for local models
    elif HAS_LOCAL_MODELS and is_local_model and is_local_model(model_name):
        return 'local'
    elif any(x in model_name_lower for x in ['gpt', 'o1', 'openai']):
        return 'openai'
    elif any(x in model_name_lower for x in ['gemini', 'google']):
        return 'gemini'
    elif any(x in model_name_lower for x in ['claude', 'anthropic']):
        return 'anthropic'
    elif any(x in model_name_lower for x in ['grok']):
        return 'grok'
    else:
        # Default to OpenAI for unknown models
        return 'openai'


def call_api(model_name: str, prompt: str, max_tokens: int, 
            temperature: float = 0.0, api_keys: Dict[str, str] = None) -> str:
    """Generic API caller that routes to appropriate service."""
    if api_keys is None:
        api_keys = load_api_keys()
    
    api_type = determine_api_type(model_name)
    
    if api_type == 'local':
        if not HAS_LOCAL_MODELS:
            return "ERROR: Local model handler not available. Install transformers: pip install transformers torch"
        return call_local_model(model_name, prompt, max_tokens, temperature)
    
    elif api_type == 'openai':
        if not HAS_OPENAI:
            return "ERROR: OpenAI library not installed"
        client = get_openai_client(api_keys.get('OPENAI_API_KEY', ''))
        return call_openai_api(client, model_name, prompt, max_tokens, temperature)
    
    elif api_type == 'gemini':
        if not HAS_GEMINI:
            return "ERROR: Google Generative AI library not installed"
        client = get_gemini_client(api_keys.get('GEMINI_API_KEY_vince', ''))
        return call_gemini_api(client, model_name, prompt, max_tokens, temperature)
    
    elif api_type == 'anthropic':
        if not HAS_ANTHROPIC:
            return "ERROR: Anthropic library not installed"
        client = get_anthropic_client(api_keys.get('Claude_API_KEY', ''))
        return call_anthropic_api(client, model_name, prompt, max_tokens, temperature)
    
    elif api_type == 'grok':
        if not HAS_GROK:
            return "ERROR: OpenAI library not installed (required for Grok API)"
        client = get_grok_client(api_keys.get('GROK_API_KEY', ''), api_keys.get('GROK_ENDPOINT', 'https://api.x.ai/v1'))
        return call_grok_api(client, model_name, prompt, max_tokens, temperature)
    
    elif api_type == 'ollama':
        if not HAS_OLLAMA:
            return "ERROR: requests library not installed (required for Ollama API)"
        ollama_base_url = api_keys.get('OLLAMA_BASE_URL', 'http://localhost:11434') if api_keys else 'http://localhost:11434'
        return call_ollama_api(model_name, prompt, max_tokens, temperature, ollama_base_url)
    
    else:
        return f"ERROR: Unknown API type for model {model_name}"


def extract_answers_from_response(response: str, expected_count: int) -> List[int]:
    """
    Extract numerical answers from model response.
    
    Args:
        response: Model's response text
        expected_count: Expected number of answers
        
    Returns:
        List of extracted answers (integers)
    """
    # Strip whitespace from response for parsing, but preserve original
    response_stripped = response.strip()
    
    # Find all numbers in the stripped response
    numbers = re.findall(r'\b\d+\b', response_stripped)
    
    # Convert to integers
    try:
        answers = [int(num) for num in numbers]
    except ValueError:
        answers = []
    
    # If we have exactly the expected count, return as is
    if len(answers) == expected_count:
        return answers
    
    # If we have more numbers, try to find a sequence of the right length
    if len(answers) > expected_count:
        # Look for consecutive sequences that might be the answers
        for i in range(len(answers) - expected_count + 1):
            candidate = answers[i:i + expected_count]
            # Simple heuristic: answers should be reasonable position values
            if all(1 <= ans <= 10 for ans in candidate):  # Assuming max 10 positions
                return candidate
        
        # If no good sequence found, take the first N numbers
        return answers[:expected_count]
    
    # If we have fewer numbers, pad with -1 (indicating missing/error)
    while len(answers) < expected_count:
        answers.append(-1)
    
    return answers


def run_single_session(session_id: int, ndigits: int, stim_types_str: str, nrep: int,
                      restriction: str, model_name: str, max_tokens: int,
                      output_dir: Path, api_keys: Dict[str, str], temperature: float = 0.0, 
                      is_random: bool = True, seed_arg: str = None, with_examples: bool = False,
                      exhaustive_stimulus = None) -> Dict[str, Any]:
    """
    Run a single test session.
    
    Returns:
        Dictionary containing session results
    """
    print(f"Running session {session_id}...")
    
    # Determine seed for stimulus generation
    if seed_arg == "session":
        stimulus_seed = session_id
    elif seed_arg is not None:
        try:
            stimulus_seed = int(seed_arg)
        except ValueError:
            print(f"Warning: Invalid seed '{seed_arg}', using None")
            stimulus_seed = None
    else:
        stimulus_seed = None
    
    # Generate MSIT stimuli
    try:
        if exhaustive_stimulus is not None:
            # Use the provided exhaustive stimulus - only store the digits
            stimuli = " ".join(str(x) for x in exhaustive_stimulus.digits)
            answers = [exhaustive_stimulus.target_pos_index + 1]  # 1-based position
            identities = [exhaustive_stimulus.target_identity]
            final_types = [int(stim_types_str.split(',')[0].strip())]  # Single condition type
            flanker_values = [exhaustive_stimulus.flanker_value]
        else:
            # Generate stimuli normally
            stimuli, answers, identities, final_types, flanker_values = gen_msit_rep_sequence(
                ndigits=ndigits,
                stim_types_str=stim_types_str,
                nrep=nrep,
                is_random=is_random,  # Use the randomized parameter
                seed=stimulus_seed  # Use determined seed
            )
    except Exception as e:
        print(f"Error generating stimuli for session {session_id}: {e}")
        return {"error": str(e)}
    
    # Create instruction and full input  
    instruction = msit_instruciton(ndigits, by_image=False, restriction=restriction, 
                                  with_examples=with_examples, stim_types_str=stim_types_str,
                                  nstims=len(answers))
    
    # Keep original stimuli format - all stimuli at once, model responds with all answers
    full_input = instruction + "\n" + stimuli + " -> Answer:"
    
    # Call API
    start_time = time.time()
    try:
        response = call_api(model_name, full_input, max_tokens, temperature=temperature, api_keys=api_keys)
        api_time = time.time() - start_time
        
        # Clear memory for local models after each session (not needed for Ollama)
        api_type = determine_api_type(model_name)
        if api_type == 'local':
            try:
                from local_model_handler import clear_local_generation_cache
                clear_local_generation_cache()
                print(f"Cleared local generation cache after session {session_id}")
            except ImportError:
                pass  # local_model_handler not available
        elif api_type == 'ollama':
            # Ollama manages its own memory, no need to clear cache
            pass
                
    except Exception as e:
        print(f"API call failed for session {session_id}: {e}")
        return {"error": str(e)}
    
    # Extract answers from response
    extracted_answers = extract_answers_from_response(response, len(answers))
    
    # Get digit logits information for local models (not available for Ollama)
    digit_logits_info = None
    api_type = determine_api_type(model_name)
    if api_type == 'local' and HAS_LOCAL_MODELS and get_local_model_digit_logits:
        try:
            print(f"Collecting digit logits information for session {session_id}...")
            digit_logits_info = get_local_model_digit_logits(model_name, full_input)
            if "error" in digit_logits_info:
                print(f"Warning: Failed to get digit logits: {digit_logits_info['error']}")
                digit_logits_info = None
        except Exception as e:
            print(f"Warning: Error collecting digit logits: {e}")
            digit_logits_info = None
    elif api_type == 'ollama':
        print(f"Note: Digit logits not available for Ollama models (session {session_id})")
    
    # Prepare session data
    session_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "ndigits": ndigits,
            "stim_types_str": stim_types_str,
            "nrep": nrep,
            "restriction": restriction,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "exhaustive_mode": exhaustive_stimulus is not None
        },
        "stimuli": stimuli,
        "correct_answers": answers,
        "identities": identities,
        "final_types": final_types,
        "flanker_values": flanker_values,
        "full_input": full_input,
        "model_response": response,
        "extracted_answers": extracted_answers,
        "api_time_seconds": api_time,
        "accuracy": sum(1 for i, (correct, extracted) in enumerate(zip(answers, extracted_answers)) 
                       if correct == extracted) / len(answers) if answers else 0
    }
    
    # Add digit logits information if available (for local models)
    if digit_logits_info is not None:
        session_data["digit_logits_info"] = digit_logits_info
    
    # Add exhaustive stimulus information if applicable
    if exhaustive_stimulus is not None:
        session_data["exhaustive_stimulus_info"] = {
            "digits": exhaustive_stimulus.digits,
            "target_identity": exhaustive_stimulus.target_identity,
            "target_pos_index": exhaustive_stimulus.target_pos_index,
            "is_simon": exhaustive_stimulus.is_simon,
            "is_flanker": exhaustive_stimulus.is_flanker,
            "condition": exhaustive_stimulus.condition,
            "flanker_value": exhaustive_stimulus.flanker_value
        }
    
    # Save individual session files
    session_file = output_dir / f"session_{session_id:03d}.json"
    save_json_compact(session_data, session_file)
    
    print(f"Session {session_id} completed. Accuracy: {session_data['accuracy']:.2%}")
    
    return session_data


def run_acdc_analysis(sessions_data: List[Dict[str, Any]], args, output_dir: Path) -> Dict[str, Any]:
    """
    Run ACDC circuit discovery analysis on the collected session data.
    
    Args:
        sessions_data: List of session results
        args: Command line arguments
        output_dir: Output directory for ACDC results
        
    Returns:
        Dictionary containing ACDC analysis results
    """
    if not HAS_ACDC:
        print("WARNING: ACDC dependencies not installed. Skipping ACDC analysis.")
        print("Install with: pip install -r requirements_acdc.txt")
        return {"error": "ACDC dependencies not available"}
    
    if not args.enable_acdc:
        return {"skipped": "ACDC analysis not enabled"}
    
    # Check if model is supported for ACDC (local models only for now)
    api_type = determine_api_type(args.model)
    if api_type != 'local':
        print(f"WARNING: ACDC currently only supports local models. Model '{args.model}' is {api_type}.")
        return {"error": f"ACDC not supported for {api_type} models"}
    
    print("\n=== Starting ACDC Circuit Discovery ===")
    print(f"Model: {args.model}")
    print(f"Parameters: tau={args.acdc_tau}, k_edges={args.acdc_k_edges}")
    
    # Extract clean prompts from session data
    clean_prompts = []
    for session in sessions_data:
        if "error" not in session and "full_input" in session:
            clean_prompts.append(session["full_input"])
    
    if len(clean_prompts) < args.acdc_train_samples + args.acdc_test_samples:
        print(f"WARNING: Only {len(clean_prompts)} prompts available, need at least {args.acdc_train_samples + args.acdc_test_samples}")
        print("Adjusting sample sizes...")
        args.acdc_train_samples = min(args.acdc_train_samples, len(clean_prompts) // 2)
        args.acdc_test_samples = min(args.acdc_test_samples, len(clean_prompts) - args.acdc_train_samples)
    
    # Create ACDC configuration
    acdc_config = ACDCConfig(
        tau=args.acdc_tau,
        k_edges=args.acdc_k_edges,
        faithfulness_target=args.acdc_faithfulness_target,
        seq_len=args.acdc_seq_len,
        batch_size=args.acdc_batch_size,
        n_train_samples=args.acdc_train_samples,
        n_test_samples=args.acdc_test_samples,
        ablation_type=args.acdc_ablation_type,
        save_visualizations=args.acdc_save_viz
    )
    
    # Create ACDC output directory
    acdc_output_dir = output_dir / "acdc_analysis"
    acdc_output_dir.mkdir(exist_ok=True)
    
    try:
        print("Creating ACDC analyzer...")
        
        # Clear HF model to free memory before ACDC
        print("Clearing HF model cache to free memory...")
        import gc
        import torch
        
        # Clear device caches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("Memory cleared successfully")
        
        # Run ACDC analysis
        acdc_results = run_msit_acdc_analysis(
            clean_prompts=clean_prompts,
            model_name=args.model,
            config=acdc_config,
            output_dir=str(acdc_output_dir)
        )
        
        print(f"ACDC Analysis Results:")
        print(f"  - Discovered {acdc_results['circuit_discovery']['n_edges']} edges")
        print(f"  - Found {len(acdc_results['circuit_discovery']['heads'])} attention heads")
        print(f"  - Found {len(acdc_results['circuit_discovery']['mlps'])} MLP layers")
        print(f"  - Circuit faithfulness: {acdc_results['evaluation'].get('circuit_faithfulness', 'N/A')}")
        
        # Save ACDC summary
        acdc_summary_file = output_dir / "acdc_summary.json"
        with open(acdc_summary_file, 'w') as f:
            json.dump(acdc_results, f, indent=2)
        
        print(f"ACDC results saved to: {acdc_output_dir}")
        return acdc_results
        
    except Exception as e:
        import traceback
        print(f"ACDC analysis failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="MSIT API Test Script")
    
    # Required arguments
    parser.add_argument("--sessions", type=int, required=True,
                       help="Number of sessions to run")
    parser.add_argument("--ndigits", type=int, required=True,
                       help="Number of digit positions")
    parser.add_argument("--stim_types", type=str, required=True,
                       help="Stimulus types (e.g., '0,1,2,3')")
    parser.add_argument("--nrep", type=int, required=True,
                       help="Number of repetitions")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'gemini-pro', 'claude-3-sonnet')")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum output tokens")
    
    # Optional arguments
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for model generation (default: 0.0)")
    parser.add_argument("--restriction", type=str, default="strict",
                       choices=["none", "not1by1", "strict", "strict-rev"],
                       help="Restriction type")
    parser.add_argument("--output_dir", type=str, default="data/msit_pilot_outputs",
                       help="Output directory for results")
    parser.add_argument("--api_file", type=str, default="API.json",
                       help="Path to API keys file")
    parser.add_argument("--nickname", type=str, default=None,
                       help="Optional nickname to add to output folder name")
    parser.add_argument("--randomized", action="store_true", default=False,
                       help="Whether to randomize the stimulus sequence (default: False)")
    parser.add_argument("--seed", type=str, default=None,
                       help="Random seed for stimulus generation. Use 'session' to use session ID as seed, or specify a number (default: None for random)")
    parser.add_argument("--with_examples", action="store_true", default=False,
                       help="Include examples in instruction to help non-instructed models")
    
    # ACDC Circuit Discovery arguments
    parser.add_argument("--enable_acdc", action="store_true", default=False,
                       help="Enable ACDC circuit discovery analysis")
    parser.add_argument("--acdc_tau", type=float, default=1e-3,
                       help="ACDC tolerance for faithfulness degradation (default: 1e-3)")
    parser.add_argument("--acdc_k_edges", type=int, default=80,
                       help="Number of top edges to keep in ACDC (default: 80)")
    parser.add_argument("--acdc_faithfulness_target", type=str, default="kl_div",
                       choices=["kl_div", "logit_diff"],
                       help="ACDC faithfulness target metric (default: kl_div)")
    parser.add_argument("--acdc_seq_len", type=int, default=64,
                       help="Sequence length for ACDC patchable model (default: 64)")
    parser.add_argument("--acdc_batch_size", type=int, default=8,
                       help="Batch size for ACDC computation (default: 8)")
    parser.add_argument("--acdc_train_samples", type=int, default=500,
                       help="Number of training samples for ACDC (default: 500)")
    parser.add_argument("--acdc_test_samples", type=int, default=100,
                       help="Number of test samples for ACDC evaluation (default: 100)")
    parser.add_argument("--acdc_ablation_type", type=str, default="tokenwise_mean_corrupt",
                       choices=["tokenwise_mean_corrupt", "zero", "mean_corrupt"],
                       help="Type of ablation for ACDC (default: tokenwise_mean_corrupt)")
    parser.add_argument("--acdc_save_viz", action="store_true", default=False,
                       help="Save ACDC circuit visualizations")
    parser.add_argument("--acdc_head_ablation", action="store_true", default=False,
                       help="Run individual head ablation analysis on discovered circuit")
    parser.add_argument("--exhaustive", action="store_true", default=False,
                       help="Run exhaustive mode: generate all possible stimuli for single condition and run each individually")
    
    args = parser.parse_args()
    
    # Validate exhaustive mode requirements
    if args.exhaustive:
        # Check if stim_types has only one value
        stim_types_list = [int(x.strip()) for x in args.stim_types.split(',')]
        if len(set(stim_types_list)) != 1:
            print("ERROR: Exhaustive mode requires stim_types to have only one unique value.")
            print(f"Current stim_types: {args.stim_types} (unique values: {set(stim_types_list)})")
            return
        
        # Check if nrep is 1
        if args.nrep != 1:
            print("ERROR: Exhaustive mode requires nrep=1.")
            print(f"Current nrep: {args.nrep}")
            return
        
        print(f"Exhaustive mode enabled for condition {stim_types_list[0]} with {args.ndigits} digits")
    
    # Load API keys (not required for local models)
    api_type_for_model = determine_api_type(args.model)
    api_keys = load_api_keys(args.api_file)
    if api_type_for_model != 'local' and not api_keys:
        print("Failed to load API keys. Exiting (required for non-local models).")
        return
    if api_type_for_model == 'local' and not api_keys:
        print("Running local model without API.json (no API keys needed).")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create condition codes suffix by removing commas from stim_types
    ## remove space and comma from stim_types
    str_codes = args.stim_types.replace(' ', '').replace(',', '')
    run_name = f"{timestamp}_{args.model.replace('/', '-')}_ssn-{args.sessions}_nrep-{args.nrep}_dgi-{args.ndigits}_cond-{str_codes}"
    if args.nickname:
        run_name = f"{run_name}_{args.nickname}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting MSIT API test with {args.sessions} sessions")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    
    # Handle exhaustive mode
    exhaustive_stimuli = None
    if args.exhaustive:
        # Import the exhaustive stimulus generation function
        from msit_gen import generate_exhastive_stims_by_type
        
        # Get the single condition type
        condition = int(args.stim_types.split(',')[0].strip())
        
        # Generate all possible stimuli for this condition
        exhaustive_stimuli = generate_exhastive_stims_by_type(args.ndigits, cond=condition)
        
        # Update session count to match number of stimuli
        original_sessions = args.sessions
        args.sessions = len(exhaustive_stimuli)
        
        print(f"Generated {len(exhaustive_stimuli)} exhaustive stimuli for condition {condition}")
        print(f"Updated session count from {original_sessions} to {args.sessions}")
    
    # Save run metadata
    metadata = {
        "run_name": run_name,
        "timestamp": timestamp,
        "parameters": vars(args),
        "total_sessions": args.sessions,
        "exhaustive_mode": args.exhaustive
    }
    
    if args.exhaustive:
        metadata["exhaustive_info"] = {
            "condition": int(args.stim_types.split(',')[0].strip()),
            "total_stimuli": len(exhaustive_stimuli),
            "original_sessions": original_sessions
        }
    
    metadata_file = output_dir / "run_metadata.json"
    save_json_compact(metadata, metadata_file)
    
    # Run sessions
    all_sessions = []
    successful_sessions = 0
    
    for session_id in range(1, args.sessions + 1):
        try:
            # Pass exhaustive stimulus if in exhaustive mode
            exhaustive_stimulus = exhaustive_stimuli[session_id - 1] if exhaustive_stimuli else None
            
            session_data = run_single_session(
                session_id=session_id,
                ndigits=args.ndigits,
                stim_types_str=args.stim_types,
                nrep=args.nrep,
                restriction=args.restriction,
                model_name=args.model,
                max_tokens=args.max_tokens,
                output_dir=output_dir,
                api_keys=api_keys,
                temperature=args.temperature,
                is_random=args.randomized,
                seed_arg=args.seed,
                with_examples=args.with_examples,
                exhaustive_stimulus=exhaustive_stimulus
            )
            
            if "error" not in session_data:
                successful_sessions += 1
            
            all_sessions.append(session_data)
            
        except Exception as e:
            print(f"Unexpected error in session {session_id}: {e}")
            all_sessions.append({
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Save summary results
    summary = {
        "run_metadata": metadata,
        "total_sessions": args.sessions,
        "successful_sessions": successful_sessions,
        "failed_sessions": args.sessions - successful_sessions,
        "overall_accuracy": sum(s.get("accuracy", 0) for s in all_sessions if "accuracy" in s) / max(successful_sessions, 1),
        "sessions": all_sessions
    }
    
    summary_file = output_dir / "summary.json"
    save_json_compact(summary, summary_file)
    
    # Run ACDC analysis if enabled
    acdc_results = None
    if args.enable_acdc and successful_sessions > 0:
        acdc_results = run_acdc_analysis(all_sessions, args, output_dir)
        # Add ACDC results to summary
        summary["acdc_analysis"] = acdc_results
        # Re-save summary with ACDC results
        save_json_compact(summary, summary_file)
    
    print(f"\n=== Test Complete ===")
    print(f"Total sessions: {args.sessions}")
    print(f"Successful sessions: {successful_sessions}")
    print(f"Failed sessions: {args.sessions - successful_sessions}")
    if successful_sessions > 0:
        print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
    if acdc_results and "error" not in acdc_results and "skipped" not in acdc_results:
        print(f"ACDC circuit discovery: {acdc_results['circuit_discovery']['n_heads']} heads, {acdc_results['circuit_discovery']['n_edges']} edges")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
