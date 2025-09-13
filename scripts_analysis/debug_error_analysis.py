#!/usr/bin/env python3
"""
Debug script to analyze specific trials and understand why SmErr and Otr don't appear
"""

import json
import os
from pathlib import Path

def classify_error_type(stimulus_type, correct_answer, extracted_answer, identity_value, flanker_value, stimulus_row):
    """Same classification logic as main script"""
    if extracted_answer == -1:
        return 'Empty'
    
    if correct_answer == extracted_answer:
        return 'Corr'
    
    if 1 <= extracted_answer <= len(stimulus_row):
        chosen_value = stimulus_row[extracted_answer - 1]
    else:
        return 'Otr'
    
    if chosen_value == identity_value:
        return 'SmErr'
    elif chosen_value == flanker_value:
        return 'FkErr'
    else:
        return 'Otr'

def debug_single_session(session_file_path, max_trials=10):
    """Debug a single session file to see detailed trial analysis"""
    try:
        with open(session_file_path, 'r') as f:
            session_data = json.load(f)
    except:
        print(f"Could not load {session_file_path}")
        return
    
    final_types = session_data.get('final_types', [])
    correct_answers = session_data.get('correct_answers', [])
    extracted_answers = session_data.get('extracted_answers', [])
    identities = session_data.get('identities', [])
    flanker_values = session_data.get('flanker_values', [])
    stimuli_str = session_data.get('stimuli', '')
    
    # Parse stimuli
    stimulus_rows = []
    if stimuli_str:
        for line in stimuli_str.strip().split('\\n'):
            row = [int(x) for x in line.split()]
            stimulus_rows.append(row)
    
    print(f"\nDebugging session: {session_data.get('session_id', 'unknown')}")
    print("=" * 80)
    
    error_counts = {'Corr': 0, 'SmErr': 0, 'FkErr': 0, 'Otr': 0, 'Empty': 0}
    
    for i in range(min(max_trials, len(final_types))):
        stimulus_type = final_types[i]
        correct_answer = correct_answers[i]
        extracted_answer = extracted_answers[i]
        identity_value = identities[i]
        flanker_value = flanker_values[i]
        stimulus_row = stimulus_rows[i] if i < len(stimulus_rows) else None
        
        if stimulus_row is None:
            continue
            
        error_type = classify_error_type(stimulus_type, correct_answer, extracted_answer, 
                                       identity_value, flanker_value, stimulus_row)
        error_counts[error_type] += 1
        
        chosen_value = stimulus_row[extracted_answer - 1] if 1 <= extracted_answer <= len(stimulus_row) else "Invalid"
        
        print(f"Trial {i+1}:")
        print(f"  Stimulus: {stimulus_row} (type {stimulus_type})")
        print(f"  Identity: {identity_value}, Flanker: {flanker_value}")
        print(f"  Correct answer: {correct_answer}, Extracted: {extracted_answer}")
        print(f"  Chosen value: {chosen_value}")
        print(f"  Error type: {error_type}")
        
        # Explain the classification
        if error_type == 'Corr':
            print(f"  -> Correct: chose position {extracted_answer} which has identity value {identity_value}")
        elif error_type == 'SmErr':
            print(f"  -> Simon Error: chose identity value {identity_value} but wrong position")
        elif error_type == 'FkErr':
            print(f"  -> Flanker Error: chose flanker value {flanker_value}")
        elif error_type == 'Otr':
            print(f"  -> Other Error: chose value {chosen_value} (not identity {identity_value} or flanker {flanker_value})")
        
        print()
    
    print("Error counts in this sample:")
    for error_type, count in error_counts.items():
        if count > 0:
            print(f"  {error_type}: {count}")
    
    return error_counts

def main():
    """Debug error analysis on sample sessions"""
    
    # Test one session from each folder
    folder_paths = [
        "data/msit_pilot_outputs_smallnrep/20250907_023509_model-gpt-4.1-nano_sessions-100_Sm",
        "data/msit_pilot_outputs_smallnrep/20250907_022558_model-gpt-4.1-nano_sessions-100_Fk",
        "data/msit_pilot_outputs_smallnrep/20250907_022159_model-gpt-4.1-nano_sessions-100_SmFk",
    ]
    
    for folder_path in folder_paths:
        print(f"\n{'='*100}")
        print(f"DEBUGGING FOLDER: {folder_path}")
        print('='*100)
        
        test_folder = Path(folder_path)
        if not test_folder.exists():
            print(f"Folder not found: {folder_path}")
            continue
            
        # Find a session file to debug
        session_files = sorted(test_folder.glob("session_*.json"))
        if session_files:
            debug_single_session(session_files[0], max_trials=20)  # Debug first 20 trials
        else:
            print("No session files found")

if __name__ == "__main__":
    main()
