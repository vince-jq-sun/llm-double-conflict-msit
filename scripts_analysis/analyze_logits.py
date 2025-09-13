#!/usr/bin/env python3
"""
MSIT Logits Analysis

This script analyzes MSIT test results based on logits information for different value types.
It extracts digit_softmax_probs from digit_logits_info and averages them across sessions
for different value types:
- Corr: Correct answer (identity position) digit logits
- IdVal: Simon value digit logits
- FkVal: Flanker value digit logits
- OtrTop: Best other candidate digit logits (max for logits/logprobs, min for rank)

Usage:
    python analyze_logits.py
"""

import json
import os
import sys
import math
from pathlib import Path
from collections import defaultdict
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    print("Will perform analysis without visualization.")
    PLOTTING_AVAILABLE = False


def load_session_data(session_file_path):
    """Load data from a session JSON file."""
    try:
        with open(session_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Session file not found: {session_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {session_file_path}")
        return None


def get_value_types_for_trial(stimulus_type, correct_answer, identity_value, flanker_value, stimulus_row, ndigits=4):
    """
    Get the different value types for a trial.
    
    Args:
        stimulus_type: The stimulus condition (0, 1, 2, 3, etc.)
        correct_answer: The correct answer position (1, 2, or 3)
        identity_value: The identity digit value
        flanker_value: The flanker digit value  
        stimulus_row: The actual stimulus row (e.g., [3, 2, 2])
        ndigits: Total number of possible digits (default: 4)
    
    Returns:
        dict: {'Corr': [digits], 'IdVal': [digits], 'FkVal': [digits], 'OtrTop': [digits]}
    """
    value_types = {
        'Corr': [],    # Correct answer (identity position) digit
        'IdVal': [],   # Simon value digit
        'FkVal': [],   # Flanker value digit
        'OtrTop': []   # Other candidate digits (all remaining digits 1-ndigits)
    }
    
    # Corr: The correct answer digit (identity position)
    value_types['Corr'].append(correct_answer)
    
    # IdVal: The simon value digit (identity value)
    value_types['IdVal'].append(identity_value)
    
    # FkVal: The flanker value digit
    value_types['FkVal'].append(flanker_value)
    
    # OtrTop: All remaining candidate digits (1 to ndigits) not already used
    # Only include digits that are valid candidate answers (1 to ndigits)
    # Flanker and identity values might be outside this range
    used_digits = {correct_answer, identity_value, flanker_value}
    for digit in range(1, ndigits + 1):
        if digit not in used_digits:
            value_types['OtrTop'].append(digit)
    
    return value_types


def extract_logits_analysis(session_data, variable_x="digit_softmax_probs"):
    """
    Extract logits analysis from a session based on digit_logits_info.
    
    Args:
        session_data: Session data dictionary
        variable_x: Variable to extract from digit_logits_info (default: "digit_softmax_probs")
    
    Returns:
        dict: {stimulus_type: {'Corr': [values], 'IdVal': [values], 'FkVal': [values], 'OtrTop': [values]}}
    """
    final_types = session_data.get('final_types', [])
    correct_answers = session_data.get('correct_answers', [])
    extracted_answers = session_data.get('extracted_answers', [])
    identities = session_data.get('identities', [])
    flanker_values = session_data.get('flanker_values', [])
    stimuli_str = session_data.get('stimuli', '')
    digit_logits_info = session_data.get('digit_logits_info', {})
    
    if len(final_types) != len(correct_answers) or len(final_types) != len(extracted_answers):
        print(f"Warning: Mismatched array lengths in session {session_data.get('session_id', 'unknown')}")
        return {}
    
    # Check if digit_logits_info exists and has the required variable
    if not isinstance(digit_logits_info, dict) or variable_x not in digit_logits_info:
        print(f"Warning: {variable_x} not found in digit_logits_info in session {session_data.get('session_id', 'unknown')}")
        return {}
    
    # Parse stimuli string to get individual rows
    stimulus_rows = []
    if stimuli_str:
        for line in stimuli_str.strip().split('\\n'):
            try:
                row = [int(x) for x in line.split()]
            except ValueError:
                row = [x for x in line.split()]
            stimulus_rows.append(row)
    
    # Group logits values by stimulus condition and value type
    logits_stats = defaultdict(lambda: {'Corr': [], 'IdVal': [], 'FkVal': [], 'OtrTop': []})
    
    # Get the logits values for all digits (this is session-level, not per trial)
    logits_values = digit_logits_info[variable_x]
    
    # Process all trials in this session
    for i in range(len(final_types)):
        stimulus_type = final_types[i]
        correct_answer = correct_answers[i]
        extracted_answer = extracted_answers[i]
        
        # Get identity and flanker values for this trial
        identity_value = identities[i] if i < len(identities) else None
        flanker_value = flanker_values[i] if i < len(flanker_values) else None
        stimulus_row = stimulus_rows[i] if i < len(stimulus_rows) else None
        
        if identity_value is None or flanker_value is None or stimulus_row is None:
            print(f"Warning: Missing data for trial {i} in session {session_data.get('session_id', 'unknown')}")
            continue
        
        # Get value types for this trial
        value_types = get_value_types_for_trial(stimulus_type, correct_answer, 
                                              identity_value, flanker_value, stimulus_row)
        
        # Extract logits values for each value type
        for value_type, digits in value_types.items():
            if value_type == 'OtrTop':
                # For OtrTop, find the best value among all other candidate digits
                candidate_values = []
                for digit in digits:
                    digit_logits = None
                    if isinstance(logits_values, dict):
                        if str(digit) in logits_values:
                            digit_logits = logits_values[str(digit)]
                        elif digit in logits_values:
                            digit_logits = logits_values[digit]
                    elif isinstance(logits_values, list):
                        if 1 <= digit <= len(logits_values):
                            digit_logits = logits_values[digit - 1]
                        elif 0 <= digit < len(logits_values):
                            digit_logits = logits_values[digit]
                    
                    if digit_logits is not None:
                        candidate_values.append(digit_logits)
                
                if candidate_values:
                    # Choose max for logits/logprobs, min for rank
                    if 'rank' in variable_x.lower():
                        best_value = min(candidate_values)
                    else:
                        best_value = max(candidate_values)
                    logits_stats[stimulus_type][value_type].append(best_value)
            else:
                # For other value types, extract single digit value
                for digit in digits:
                    digit_logits = None
                    if isinstance(logits_values, dict):
                        if str(digit) in logits_values:
                            digit_logits = logits_values[str(digit)]
                        elif digit in logits_values:
                            digit_logits = logits_values[digit]
                        else:
                            # Skip digits that don't exist in logits (should be rare since digits 0-9 are typically available)
                            print(f"Info: Digit {digit} not found in logits dict for {value_type} in trial {i} (available: {list(logits_values.keys())})")
                            continue
                    elif isinstance(logits_values, list):
                        if 1 <= digit <= len(logits_values):
                            digit_logits = logits_values[digit - 1]
                        elif 0 <= digit < len(logits_values):
                            digit_logits = logits_values[digit]
                        else:
                            print(f"Info: Digit {digit} out of range for {value_type} in trial {i} (list length: {len(logits_values)})")
                            continue
                    else:
                        print(f"Warning: Unexpected logits_values format: {type(logits_values)} for trial {i}")
                        continue
                    
                    if digit_logits is not None:
                        logits_stats[stimulus_type][value_type].append(digit_logits)
    
    return dict(logits_stats)


def analyze_folder_logits(test_folder_path, variable_x="digit_softmax_probs"):
    """
    Analyze logits for all sessions in a test result folder.
    
    Args:
        test_folder_path: Path to the test result folder
        variable_x: Variable to extract from digit_logits_info
        
    Returns:
        tuple: (logits_results dict, metadata dict)
    """
    test_folder = Path(test_folder_path)
    
    if not test_folder.exists():
        print(f"Error: Test folder not found: {test_folder_path}")
        return None, None
    
    # Load run metadata
    metadata_file = test_folder / "run_metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Aggregate logits values across all sessions
    aggregated_logits = defaultdict(lambda: {'Corr': [], 'IdVal': [], 'FkVal': [], 'OtrTop': []})
    session_count = 0
    
    # Find all session files
    session_files = sorted(test_folder.glob("session_*.json"))
    
    if not session_files:
        print("Error: No session files found in the test folder")
        return None, None
    
    for session_file in session_files:
        session_data = load_session_data(session_file)
        if session_data is None:
            continue
        
        # Sanity check: skip sessions with "error" in model output
        model_response = session_data.get('model_response', '')
        if 'error' in model_response.lower():
            print(f"Skipping session {session_data.get('session_id', 'unknown')} due to error in model output")
            continue
            
        session_count += 1
        session_logits = extract_logits_analysis(session_data, variable_x)
        
        # Aggregate logits values
        for stim_type, value_logits in session_logits.items():
            for value_type, values in value_logits.items():
                aggregated_logits[stim_type][value_type].extend(values)
    
    print(f"Processed {session_count} sessions from {test_folder_path}")
    return dict(aggregated_logits), metadata


def create_logits_analysis_plot(logits_results_list, metadata_list, folder_paths, output_path, figure_name, variable_x="digit_softmax_probs", cols_per_row=3):
    """
    Create nested bar plots showing average logits for each condition across folders.
    
    Args:
        logits_results_list: List of logits analysis results for each folder
        metadata_list: List of metadata for each folder
        folder_paths: List of folder paths
        output_path: Path to save the figure
        figure_name: Name of the figure file
        variable_x: Variable name being analyzed
        cols_per_row: Number of columns (subplots) per row in the figure layout (default: 3)
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping visualization.")
        return
        
    # Color scheme for value types
    value_colors = {
        'Corr': '#2ca02c',    # Green - Correct answer
        'IdVal': '#ff7f0e',   # Orange - Simon value
        'FkVal': '#d62728',   # Red - Flanker value
        'OtrTop': '#808080',  # Gray - Best other value
    }
    
    # Stimulus type name mapping
    stim_type_names = {
        0: "Cg (eg 1000)",
        1: "Sm (eg 0100)", 
        2: "Fk (eg 1222)",
        3: "Sm+Fk (eg 3133)",
        4: "CgLtr (eg u000)",
        5: "CgExN (eg 7000)",
        6: "CgExN-R (eg 5000)",
        7: "SmExN+Fk (eg 1711)",
        8: "Sm+FkExN (eg 7177)",
        9: "Sm+FkIdPos (eg 2122)",
        10: "Cg+FkExN (eg 1777)",
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    n_folders = len(logits_results_list)
    
    # Create figure with subplots
    if n_folders == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    else:
        cols = min(cols_per_row, n_folders)
        rows = (n_folders + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 4*rows))
        if n_folders == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
    
    for i, (logits_results, metadata, folder_path) in enumerate(zip(logits_results_list, metadata_list, folder_paths)):
        ax = axes[i]
        
        # Handle empty folder paths (None placeholders)
        if logits_results is None or not folder_path or folder_path.strip() == "":
            ax.text(0.5, 0.5, 'Reserved\n(Empty folder path)', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='gray', style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            continue
        
        # Get all stimulus types present in the data
        all_stim_types = sorted(logits_results.keys())
        
        if not all_stim_types:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Prepare data for nested bar chart
        value_types = ['Corr', 'IdVal', 'FkVal', 'OtrTop']
        n_conditions = len(all_stim_types)
        n_value_types = len(value_types)
        
        # Calculate bar positions
        bar_width = 0.8 / n_value_types
        x_positions = np.arange(n_conditions)
        
        # Plot bars for each value type
        for j, value_type in enumerate(value_types):
            means = []
            stds = []
            counts = []
            for stim_type in all_stim_types:
                values = logits_results[stim_type][value_type]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    count = len(values)
                else:
                    mean_val = 0
                    std_val = 0
                    count = 0
                means.append(mean_val)
                stds.append(std_val)
                counts.append(count)
            
            x_pos = x_positions + (j - n_value_types/2 + 0.5) * bar_width
            bars = ax.bar(x_pos, means, bar_width, yerr=stds, label=value_type, 
                         color=value_colors[value_type], alpha=0.8, capsize=3)
            
            # Add mean and count labels on bars
            for k, (bar, mean_val, count) in enumerate(zip(bars, means, counts)):
                if mean_val > 0.001:  # Show label if mean > 0.001 (lowered threshold to show small FkVal)
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + stds[k] + 0.01,
                           f'{mean_val:.4f}\n(n={count})', ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        # Extract nrep from metadata parameters or folder name
        nrep_value = 'unknown'
        if 'parameters' in metadata and 'nrep' in metadata['parameters']:
            nrep_value = metadata['parameters']['nrep']
        else:
            # Try to extract from folder name as fallback
            folder_name = os.path.basename(folder_path)
            if 'nrep-' in folder_name:
                nrep_part = [part for part in folder_name.split('_') if 'nrep-' in part]
                if nrep_part:
                    nrep_value = nrep_part[0].split('-')[1]
        
        ax.set_ylabel(f'Mean {variable_x} ({nrep_value}rep)')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([stim_type_names.get(st, f'Type{st}') for st in all_stim_types])
        ax.set_xlabel(f'Experimental Conditions ({nrep_value}rep)')

        if "prob" in variable_x:
            ax.set_ylim(0, 1)
        elif "logit" in variable_x:
            ax.set_ylim(0, 23)
        
        # Extract information from metadata parameters
        model_name = 'unknown'
        sessions = 'unknown'
        condition_part = 'unknown'
        
        if 'parameters' in metadata:
            params = metadata['parameters']
            # Extract model name
            if 'model' in params:
                model_name = params['model'].replace('-', '.')
            # Extract session count
            if 'sessions' in params:
                sessions = params['sessions']
            # Extract condition info - try multiple possible keys
            for key in ['condition', 'cond', 'test_condition']:
                if key in params:
                    condition_part = params[key]
                    break
        
        # Fallback to folder name parsing if metadata doesn't have the info
        if model_name == 'unknown' or condition_part == 'unknown':
            folder_name = os.path.basename(folder_path)
            parts = folder_name.split('_')
            if model_name == 'unknown':
                model_part = next((p for p in parts if 'model-' in p), 'unknown-model')
                model_name = model_part.replace('model-', '').replace('-', '.')
            if condition_part == 'unknown':
                condition_part = parts[-1] if parts else 'unknown'
        
        # Create title with metadata information
        title_parts = [model_name, '\n']
        if sessions != 'unknown':
            title_parts.append(f'ssn={sessions}')
        if condition_part != 'unknown':
            title_parts.append(str(condition_part))
        if nrep_value != 'unknown':
            title_parts.append(f'{nrep_value}rep')
        
        ax.set_title(' - '.join(title_parts), fontsize=10, pad=10)
        
        # Add legend (only for first subplot)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
    
    # Hide unused subplots
    for i in range(n_folders, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(figure_name)
    plt.tight_layout()
    
    # Save figure
    full_path = os.path.join(output_path, figure_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nLogits analysis figure saved to: {full_path}")
    plt.show()


def analyze_multiple_folders_logits(folder_paths, output_path, figure_name, variable_x="digit_softmax_probs", cols_per_row=3):
    """
    Analyze logits for multiple result folders and create combined visualization.
    
    Args:
        folder_paths: List of paths to test result folders
        output_path: Path to save the output figure
        figure_name: Name of the figure file
        variable_x: Variable to extract from digit_logits_info
        cols_per_row: Number of columns (subplots) per row in the figure layout (default: 3)
    """
    logits_results_list = []
    metadata_list = []
    
    print("=" * 80)
    print(f"ANALYZING {variable_x.upper()} IN MULTIPLE MSIT TEST RESULT FOLDERS")
    print("=" * 80)
    
    for i, folder_path in enumerate(folder_paths, 1):
        # Skip empty folder paths but add placeholder for subplot positioning
        if not folder_path or folder_path.strip() == "":
            print(f"\n[{i}/{len(folder_paths)}] Skipping empty folder path (reserving subplot position)")
            print("-" * 60)
            logits_results_list.append(None)  # Add None as placeholder
            metadata_list.append({})  # Add empty metadata as placeholder
            continue
            
        print(f"\n[{i}/{len(folder_paths)}] Processing folder: {folder_path}")
        print("-" * 60)
        
        # If relative path provided, make it relative to current working directory
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(os.getcwd(), folder_path)
        
        logits_results, metadata = analyze_folder_logits(folder_path, variable_x)
        
        if logits_results is None:
            print(f"Skipping folder due to errors: {folder_path}")
            logits_results_list.append(None)  # Add None as placeholder
            metadata_list.append({})  # Add empty metadata as placeholder
            continue
            
        logits_results_list.append(logits_results)
        metadata_list.append(metadata)
        
        # Print summary for this folder
        print(f"{variable_x} summary:")
        for stim_type in sorted(logits_results.keys()):
            print(f"  Condition {stim_type}:")
            for value_type, values in logits_results[stim_type].items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    count = len(values)
                    print(f"    {value_type}: mean={mean_val:.4f}, std={std_val:.4f}, n={count}")
    
    if not logits_results_list:
        print("\nError: No valid results found from any folder.")
        return
    
    print("\n" + "=" * 80)
    print("CREATING LOGITS VISUALIZATION...")
    print("=" * 80)
    
    # Create the logits analysis plot
    if PLOTTING_AVAILABLE:
        create_logits_analysis_plot(logits_results_list, metadata_list, folder_paths, output_path, figure_name, variable_x, cols_per_row)
    else:
        print("Plotting not available. Analysis complete - check the printed summaries above.")


def main(variable_x="digit_softmax_probs", cols_per_row=3):
    """Main function to run the logits analysis.
    
    Args:
        variable_x: Variable to extract from digit_logits_info (default: "digit_softmax_probs")
        cols_per_row: Number of columns (subplots) per row in the figure layout (default: 3)
    """

    ### Configuration - using the same folder paths as specified by user
    folder_paths = [        
        "data/msit_pilot_outputs_word/20250912-200940_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-0",
        "data/msit_pilot_outputs_word/20250912-201033_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-1",
        "data/msit_pilot_outputs_word/20250912-201218_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-2",
        "data/msit_pilot_outputs_word/20250912-201401_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-3",
        "data/msit_pilot_outputs_word/20250912-202454_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-10",
        "data/msit_pilot_outputs_word/20250912-201716_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-8",
    ]
    output_path = "results/msit_pilot_figures"
    figure_name = f"logits_{variable_x}_llama3_3b_dg4_word-stims_digit-resps.png"
    cols_per_row = 4

    # ## Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250911-024926_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250911-025019_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250911-025201_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250911-025342_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_smallnrep/20250911-151453_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_smallnrep/20250911-152755_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-8",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = f"logits_{variable_x}_llama3_3b_dg4.png"
    # cols_per_row = 4



    # ## Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250912-004836_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250912-004927_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250912-005108_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250912-005251_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_smallnrep/20250912-010322_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_smallnrep/20250912-005600_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-8",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = f"logits_{variable_x}_llama3_3b_dg4_instruct-numbers.png"
    # cols_per_row = 4

    
    
    if not folder_paths:
        print("Please specify folder paths in the script configuration.")
        sys.exit(1)
    
    # Analyze multiple folders and create visualization
    analyze_multiple_folders_logits(folder_paths, output_path, figure_name, variable_x, cols_per_row)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze logits from MSIT test results')
    parser.add_argument('--variable', '-v', default='digit_softmax_probs', choices=['digit_softmax_probs', 'digit_global_ranks', 'digit_logits'],
                       help='Variable to extract from digit_logits_info (default: digit_softmax_probs)')
    parser.add_argument('--cols', '-c', type=int, default=4,
                       help='Number of columns per row in the figure layout (default: 4)')
    
    args = parser.parse_args()
    
    # args.variable = 'digit_logits'
    args.variable = 'digit_global_ranks'

    main(variable_x=args.variable, cols_per_row=args.cols)
