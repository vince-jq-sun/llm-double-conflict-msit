
from msit_gen_word import MSITTrial,msit_instruciton
from msit_api_test_word import determine_api_type,load_api_keys,extract_answers_from_response,get_local_model_digit_logits,save_json_compact,call_api
import argparse
import time
from datetime import datetime
from pathlib import Path



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



def preload_local_model(model_name):
    """Preload local model to avoid reloading between sessions."""
    api_type = determine_api_type(model_name)
    if api_type == 'local':
        try:
            from local_model_handler import _local_handler
            # Load the model by making a dummy call
            print(f"Preloading local model: {model_name}...")
            dummy_response = _local_handler.generate(model_name, "test", 1, 0.0)
            print(f"Model {model_name} preloaded successfully")
            return True
        except Exception as e:
            print(f"Failed to preload model {model_name}: {e}")
            return False
    return True  # Non-local models don't need preloading


def run_single_session(trial,session_id,model_name,max_tokens,api_keys,temperature=0.0,restriction="strict",formality="english",output_dir=None):
    stimuli = " ".join(str(x) for x in trial.digits)
    stimuli = " " + stimuli + " "
    answers = [trial.target_pos_index]  # Make it a list for consistency
    
    # Extract trial information
    identities = [trial.target_identity]
    final_types = [trial.condition]
    final_conditions = [condition_code[trial.condition]]
    flanker_values = [trial.flanker_value]

    instruction = msit_instruciton(ndigits, by_image=False, restriction=restriction,formality=formality, 
                                with_examples=False, stim_types_str="0",nstims=1)
    full_input = instruction + "\n" + stimuli + "-> Answer:"
                     
    # print(full_input)

    # Call API
    start_time = time.time()
    try:
        response = call_api(model_name, full_input, max_tokens, temperature, api_keys)

        api_time = time.time() - start_time
        
        # Clear memory for local models after each session (not needed for Ollama)
        api_type = determine_api_type(model_name)
        if api_type == 'local':
            try:
                from local_model_handler import clear_local_generation_cache
                clear_local_generation_cache()
                # Reduced logging frequency for cache clearing operations
                if session_id % 10 == 0:  # Log every 10 sessions instead of every session
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
    
    # Check if local models are available
    HAS_LOCAL_MODELS = True
    try:
        from local_model_handler import get_local_model_digit_logits
    except ImportError:
        HAS_LOCAL_MODELS = False
        get_local_model_digit_logits = None
    
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
            "restriction": restriction,
            "model_name": model_name,
            "max_tokens": max_tokens,
        },
        "formality": formality,
        "stimuli": stimuli,
        "correct_answers": answers,
        "identities": identities,
        "final_types": final_types,
        "final_conditions": final_conditions,
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
    
    
    # Save individual session files
    session_file = output_dir / f"session_{session_id:03d}.json"
    save_json_compact(session_data, session_file)
    
    print(f"Session {session_id} completed. Accuracy: {session_data['accuracy']:.2%}")
    
    return session_data



def main():
    run_name = "msit-full-run-v1"
    parser = argparse.ArgumentParser(description=run_name)
    
    # Required arguments
    parser.add_argument("--ndigits", type=int, default=4,
                       help="Number of digit positions")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="Model name (e.g., 'gpt-4', 'gemini-pro', 'claude-3-sonnet')")
    parser.add_argument("--max_tokens", type=int, default=400,
                       help="Maximum output tokens")
    
    # Optional arguments
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for model generation (default: 0.0)")
    parser.add_argument("--restriction", type=str, default="strict",
                       choices=["none", "not1by1", "strict", "strict-rev"],
                       help="Restriction type")
    parser.add_argument("--formality", type=str, default="english",
                    choices=["arabic", "english"],
                    help="Formality of the stimuli")
    parser.add_argument("--output_dir", type=str, default="data/msit_pilot_outputs_mapall",
                       help="Output directory for results")
    parser.add_argument("--api_file", type=str, default="API.json",
                       help="Path to API keys file")
    parser.add_argument("--nickname", type=str, default=None,
                       help="Optional nickname to add to output folder name")

    args = parser.parse_args()
    
    trials = generate_trials(args.ndigits,args.formality)
    args.sessions = len(trials)

    # Load API keys (not required for local models)
    api_type_for_model = determine_api_type(args.model)
    api_keys = load_api_keys(args.api_file)
    if api_type_for_model != 'local' and not api_keys:
        print("Failed to load API keys. Exiting (required for non-local models).")
        return
    if api_type_for_model == 'local' and not api_keys:
        print("Running local model without API.json (no API keys needed).")
    
    # Preload local model to avoid reloading between sessions
    if not preload_local_model(args.model):
        print(f"Failed to preload model {args.model}. Continuing anyway...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.model.replace('/', '-')}_ssn-{args.sessions}_dgi-{args.ndigits}_{run_name}"
    if args.nickname:
        run_name = f"{run_name}_{args.nickname}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting MSIT API test with {args.sessions} sessions")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
        
    # Save run metadata
    metadata = {
        "run_name": run_name,
        "timestamp": timestamp,
        "parameters": vars(args),
        "total_sessions": args.sessions,
    }
        
    metadata_file = output_dir / "run_metadata.json"
    save_json_compact(metadata, metadata_file)
    
    # Run sessions
    all_sessions = []
    successful_sessions = 0
    
    for session_id in range(1, args.sessions + 1):
        trial = trials[session_id-1]
        try:
            session_data = run_single_session(
                trial,
                session_id,
                args.model,
                args.max_tokens,
                api_keys,
                args.temperature,
                args.restriction,
                args.formality,
                output_dir
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

    
    print(f"\n=== Test Complete ===")
    print(f"Total sessions: {args.sessions}")
    print(f"Successful sessions: {successful_sessions}")
    print(f"Failed sessions: {args.sessions - successful_sessions}")
    if successful_sessions > 0:
        print(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # trials = generate_trials(ndigits)
    # preview_trials_row_and_condition(trials)
    main()
    
