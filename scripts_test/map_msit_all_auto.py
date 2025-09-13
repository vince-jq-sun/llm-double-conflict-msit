
from msit_gen_word import MSITTrial, msit_instruciton, as_prompt_line
from msit_api_test_word import determine_api_type, load_api_keys, extract_answers_from_response, get_local_model_digit_logits, save_json_compact, call_api
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Import local model handler for optimization
try:
    from local_model_handler import call_local_model, is_local_model, clear_local_generation_cache
    HAS_LOCAL_MODELS = True
except ImportError:
    HAS_LOCAL_MODELS = False
    call_local_model = None
    is_local_model = None
    clear_local_generation_cache = None



ndigits = 4

condition_code={
    "0": "Cg", ## congruent
    "1": "Sm", ## Simon-only
    "2": "Fk", ## Flanker-only
    "3": "SmFk", ## Simon+Flanker
    "5": "CgExN", ## Congruent trials with extra number as target, sth like 600, the variants are 400, 500, 600, 700, 800, 900
    "7": "SmExNFk", ## example, 115
    "8": "SmFkExN", ## example, 552
    "9": "SmFkIdPos", ## example, 331
    "10": "CgFkExN", ## example, 525
    "99": "none"
}


def tell_condition(identity_value, pos_index_0based, flanker_value):
    pos_index = pos_index_0based + 1
    if flanker_value == identity_value:
        return "99"
    if pos_index == identity_value:
        if flanker_value == 0:
            return "0"
        elif flanker_value in range(1,ndigits+1):
            return "2"
        elif flanker_value in range(ndigits+1,10):
            return "10"
    elif pos_index != identity_value:
        if flanker_value == 0:
            return "1"
        elif flanker_value in range(1,ndigits+1):
            return "3"
        elif flanker_value in range(ndigits+1,10):
            return "8"

def generate_trials(ndigits,formality="english"):
    trials = []
    for identity_value in range(1, ndigits + 1):
        for pos_index in range(0, ndigits):
            for flanker_value in range(0,10):
                digits = [flanker_value] * ndigits
                digits[pos_index] = identity_value
                condition_code = tell_condition(identity_value, pos_index, flanker_value)
                trial = MSITTrial(
                    digits=digits,
                    target_identity=identity_value,
                    target_pos_index=pos_index,
                    is_simon=False,
                    is_flanker=False,
                    condition=condition_code,
                    flanker_value=flanker_value
                )

                if formality == "arabic":
                    pass
                elif formality == "english":
                    trial.map_to_english_word()

                trials.append(trial)
    return trials

def preview_trials_row_and_condition(trials):
    for trial in trials:
        if trial.condition != "99":
            print(trial.digits, trial.condition)

def generate_sessions(ndigits=4, formality="english"):
    """
    Generate all possible MSIT trials (sessions) for the given parameters.
    Each trial will be run as a separate session.
    
    Args:
        ndigits: Number of digit positions
        formality: "arabic" or "english"
        
    Returns:
        List of MSITTrial objects (excluding condition "99")
    """
    trials = generate_trials(ndigits, formality)
    # Filter out condition "99" (flanker_value == identity_value)
    valid_trials = [trial for trial in trials if trial.condition != "99"]
    print(f"Generated {len(valid_trials)} valid trials (excluding condition 99)")
    return valid_trials

def preload_local_model(model_name: str) -> bool:
    """
    Preload local model to avoid reloading for each session.
    Based on memory optimization patterns from previous work.
    
    Returns:
        True if model was preloaded successfully, False otherwise
    """
    if not HAS_LOCAL_MODELS or not is_local_model(model_name):
        return False
        
    try:
        print(f"Preloading local model: {model_name}...")
        # Make a dummy call to load the model
        dummy_response = call_local_model(model_name, "Test", 1, 0.0)
        if "ERROR" not in dummy_response:
            print(f"Model {model_name} preloaded successfully")
            return True
        else:
            print(f"Failed to preload model {model_name}: {dummy_response}")
            return False
    except Exception as e:
        print(f"Error preloading model {model_name}: {e}")
        return False

def run_single_trial_session(session_id: int, trial: MSITTrial, model_name: str, 
                           max_tokens: int, restriction: str, formality: str,
                           output_dir: Path, api_keys: Dict[str, str], 
                           temperature: float = 0.0, with_examples: bool = False,
                           add_head_tail_space: bool = True) -> Dict[str, Any]:
    """
    Run a single MSIT trial as a session.
    
    Args:
        session_id: Session identifier
        trial: MSITTrial object to run
        model_name: Name of the model to use
        max_tokens: Maximum tokens for response
        restriction: Restriction type for instruction
        formality: "arabic" or "english"
        output_dir: Directory to save results
        api_keys: API keys dictionary
        temperature: Temperature for generation
        with_examples: Include examples in instruction
        add_head_tail_space: Add spaces around stimulus
        
    Returns:
        Dictionary containing session results
    """
    print(f"Running session {session_id} - Condition: {trial.condition} ({condition_code.get(trial.condition, 'Unknown')})")
    
    # Prepare stimulus string
    if formality == "english":
        stimulus_str = " ".join(trial.digits)
    else:  # arabic
        stimulus_str = " ".join(str(d) for d in trial.digits)
    
    if add_head_tail_space:
        stimulus_str = " " + stimulus_str + " "
    
    # Create instruction for single stimulus
    instruction = msit_instruciton(
        ndigits=ndigits, 
        by_image=False, 
        restriction=restriction,
        with_examples=with_examples, 
        stim_types_str=trial.condition,
        formality=formality,
        nstims=1
    )
    
    # Create full input
    full_input = instruction + "\n" + stimulus_str + " -> Answer:"
    
    # Call API
    start_time = time.time()
    try:
        response = call_api(model_name, full_input, max_tokens, temperature=temperature, api_keys=api_keys)
        api_time = time.time() - start_time
        
        # Clear memory for local models after each session (only generation cache, not model)
        api_type = determine_api_type(model_name)
        if api_type == 'local' and session_id % 10 == 0:  # Clear every 10 sessions to reduce logging
            try:
                if clear_local_generation_cache:
                    clear_local_generation_cache()
                    print(f"Cleared local generation cache after session {session_id}")
            except Exception as e:
                print(f"Warning: Failed to clear generation cache: {e}")
                
    except Exception as e:
        print(f"API call failed for session {session_id}: {e}")
        return {"session_id": session_id, "error": str(e), "timestamp": datetime.now().isoformat()}
    
    # Extract answer from response
    extracted_answers = extract_answers_from_response(response, 1)
    extracted_answer = extracted_answers[0] if extracted_answers else -1
    
    # Calculate accuracy (target_pos_index is 0-based, but answer should be 1-based)
    correct_answer = trial.target_pos_index
    is_correct = extracted_answer == correct_answer
    
    # Get digit logits information for local models
    digit_logits_info = None
    api_type = determine_api_type(model_name)
    if api_type == 'local' and HAS_LOCAL_MODELS and get_local_model_digit_logits:
        try:
            digit_logits_info = get_local_model_digit_logits(model_name, full_input)
            if "error" in digit_logits_info:
                print(f"Warning: Failed to get digit logits: {digit_logits_info['error']}")
                digit_logits_info = None
        except Exception as e:
            print(f"Warning: Error collecting digit logits: {e}")
            digit_logits_info = None
    
    # Prepare session data
    session_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "ndigits": ndigits,
            "condition": trial.condition,
            "condition_name": condition_code.get(trial.condition, "Unknown"),
            "model": model_name,
            "max_tokens": max_tokens,
            "restriction": restriction,
            "formality": formality
        },
        "trial_info": {
            "digits": trial.digits,
            "target_identity": trial.target_identity,
            "target_pos_index": trial.target_pos_index,
            "flanker_value": trial.flanker_value,
            "condition": trial.condition,
            "is_simon": trial.is_simon,
            "is_flanker": trial.is_flanker
        },
        "stimulus": stimulus_str.strip(),
        "correct_answer": correct_answer,
        "full_input": full_input,
        "model_response": response,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
        "accuracy": 1.0 if is_correct else 0.0,
        "api_time_seconds": api_time
    }
    
    # Add digit logits information if available
    if digit_logits_info is not None:
        session_data["digit_logits_info"] = digit_logits_info
    
    # Save individual session file
    session_file = output_dir / f"session_{session_id:03d}.json"
    save_json_compact(session_data, session_file)
    
    print(f"Session {session_id} completed. Correct: {is_correct} (Expected: {correct_answer}, Got: {extracted_answer})")
    
    return session_data

def main():
    parser = argparse.ArgumentParser(description="MSIT Auto Test Script - Run all possible trials")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'meta-llama/Llama-3.2-3B-Instruct')")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum output tokens (default: 50)")
    
    # Optional arguments
    parser.add_argument("--ndigits", type=int, default=4,
                       help="Number of digit positions (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for model generation (default: 0.0)")
    parser.add_argument("--restriction", type=str, default="strict",
                       choices=["none", "not1by1", "strict", "strict-rev"],
                       help="Restriction type (default: strict)")
    parser.add_argument("--formality", type=str, default="arabic",
                       choices=["arabic", "english"],
                       help="Formality of the stimulus (default: arabic)")
    parser.add_argument("--output_dir", type=str, default="data/msit_auto_outputs",
                       help="Output directory for results (default: data/msit_auto_outputs)")
    parser.add_argument("--api_file", type=str, default="API.json",
                       help="Path to API keys file (default: API.json)")
    parser.add_argument("--nickname", type=str, default=None,
                       help="Optional nickname to add to output folder name")
    parser.add_argument("--with_examples", action="store_true", default=False,
                       help="Include examples in instruction")
    parser.add_argument("--preview_only", action="store_true", default=False,
                       help="Only preview trials without running them")
    
    args = parser.parse_args()
    
    # Load API keys (not required for local models)
    api_type_for_model = determine_api_type(args.model)
    api_keys = load_api_keys(args.api_file)
    if api_type_for_model != 'local' and not api_keys:
        print("Failed to load API keys. Exiting (required for non-local models).")
        return
    if api_type_for_model == 'local' and not api_keys:
        print("Running local model without API.json (no API keys needed).")
    
    # Generate all sessions (trials)
    print(f"Generating sessions for {args.ndigits} digits with {args.formality} formality...")
    sessions = generate_sessions(args.ndigits, args.formality)
    
    if args.preview_only:
        print("\nPreview of trials (first 10):")
        for i, trial in enumerate(sessions[:10]):
            print(f"Session {i+1}: {trial.digits} -> Position {trial.target_pos_index + 1}, Condition: {trial.condition} ({condition_code.get(trial.condition, 'Unknown')})")
        print(f"\nTotal sessions to run: {len(sessions)}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.model.replace('/', '-')}_auto_ndigits-{args.ndigits}_{args.formality}"
    if args.nickname:
        run_name = f"{run_name}_{args.nickname}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting MSIT auto test with {len(sessions)} sessions")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    
    # Preload local model for optimization
    if api_type_for_model == 'local':
        preload_local_model(args.model)
    
    # Save run metadata
    metadata = {
        "run_name": run_name,
        "timestamp": timestamp,
        "parameters": vars(args),
        "total_sessions": len(sessions),
        "ndigits": args.ndigits,
        "formality": args.formality,
        "auto_mode": True
    }
    
    metadata_file = output_dir / "run_metadata.json"
    save_json_compact(metadata, metadata_file)
    
    # Run all sessions
    all_sessions = []
    successful_sessions = 0
    condition_stats = {}
    
    for session_id, trial in enumerate(sessions, 1):
        try:
            session_data = run_single_trial_session(
                session_id=session_id,
                trial=trial,
                model_name=args.model,
                max_tokens=args.max_tokens,
                restriction=args.restriction,
                formality=args.formality,
                output_dir=output_dir,
                api_keys=api_keys,
                temperature=args.temperature,
                with_examples=args.with_examples
            )
            
            if "error" not in session_data:
                successful_sessions += 1
                
                # Track condition statistics
                condition = trial.condition
                if condition not in condition_stats:
                    condition_stats[condition] = {"total": 0, "correct": 0}
                condition_stats[condition]["total"] += 1
                if session_data["is_correct"]:
                    condition_stats[condition]["correct"] += 1
            
            all_sessions.append(session_data)
            
        except Exception as e:
            print(f"Unexpected error in session {session_id}: {e}")
            all_sessions.append({
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Calculate condition accuracies
    for condition in condition_stats:
        stats = condition_stats[condition]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["condition_name"] = condition_code.get(condition, "Unknown")
    
    # Save summary results
    summary = {
        "run_metadata": metadata,
        "total_sessions": len(sessions),
        "successful_sessions": successful_sessions,
        "failed_sessions": len(sessions) - successful_sessions,
        "overall_accuracy": sum(s.get("accuracy", 0) for s in all_sessions if "accuracy" in s) / max(successful_sessions, 1),
        "condition_statistics": condition_stats,
        "sessions": all_sessions
    }
    
    summary_file = output_dir / "summary.json"
    save_json_compact(summary, summary_file)
    
    print(f"\n=== Test Complete ===")
    print(f"Total sessions: {len(sessions)}")
    print(f"Successful sessions: {successful_sessions}")
    print(f"Failed sessions: {len(sessions) - successful_sessions}")
    if successful_sessions > 0:
        print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
        
        print("\nAccuracy by condition:")
        for condition, stats in condition_stats.items():
            print(f"  {condition} ({stats['condition_name']}): {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
