#!/usr/bin/env python3
"""
MSIT Error Type Analysis

This script analyzes MSIT test results and categorizes errors into different types:
- Corr: Correct response (chose identity position)
- SmErr: Simon error (chose identity when should choose flanker)
- FkErr: Flanker error (chose flanker when should choose identity)
- Otr: Other error (chose neither identity nor flanker position)

Usage:
    python analyze_error_types.py
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



def classify_error_type(stimulus_type, correct_answer, extracted_answer, identity_value, flanker_value, stimulus_row):
    """
    Classify the error type based on the response.
    
    Args:
        stimulus_type: The stimulus condition (0, 1, 2, 3, etc.)
        correct_answer: The correct answer position (1, 2, or 3)
        extracted_answer: The actual response given (1, 2, or 3)
        identity_value: The identity value (can be digit, letter, or symbol)
        flanker_value: The flanker value (can be digit, letter, or symbol)
        stimulus_row: The actual stimulus row (e.g., [3, 2, 2] or ['a', 'b', 'b'])
    
    Returns:
        str: Error type ('Corr', 'SmErr', 'FkErr', 'Otr')
    """
    # Handle empty responses
    if extracted_answer == -1:
        return 'Empty'
    
    # Correct response - chose the correct identity position
    if correct_answer == extracted_answer:
        return 'Corr'
    
    # Convert extracted_answer to string for comparison with identity/flanker values
    # This handles cases where extracted_answer is numeric but identity/flanker are strings
    extracted_str = str(extracted_answer)
    identity_str = str(identity_value)
    flanker_str = str(flanker_value)
    
    # Simon error: chose identity value when should choose flanker
    if extracted_str == identity_str:
        return 'SmErr'
    
    # Flanker error: chose flanker value when should choose identity  
    if extracted_str == flanker_str:
        return 'FkErr'
    
    return 'Otr'


def extract_error_analysis(session_data):
    """
    Extract error type analysis from a session.
    
    Returns:
        dict: {stimulus_type: {'Corr': count, 'SmErr': count, 'FkErr': count, 'Otr': count, 'Empty': count}}
    """
    final_types = session_data.get('final_types', [])
    correct_answers = session_data.get('correct_answers', [])
    extracted_answers = session_data.get('extracted_answers', [])
    identities = session_data.get('identities', [])
    flanker_values = session_data.get('flanker_values', [])
    stimuli_str = session_data.get('stimuli', '')
    
    if len(final_types) != len(correct_answers) or len(final_types) != len(extracted_answers):
        print(f"Warning: Mismatched array lengths in session {session_data.get('session_id', 'unknown')}")
        return {}
    
    # Parse stimuli string to get individual rows
    stimulus_rows = []
    if stimuli_str:
        for line in stimuli_str.strip().split('\\n'):
            # Handle both numeric and non-numeric characters (letters, symbols)
            row = []
            for x in line.split():
                # Try to convert to int first, if that fails keep as string
                try:
                    row.append(int(x))
                except ValueError:
                    row.append(x)
            stimulus_rows.append(row)
    
    # Group error types by stimulus condition
    error_stats = defaultdict(lambda: {'Corr': 0, 'SmErr': 0, 'FkErr': 0, 'Otr': 0, 'Empty': 0})
    
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
        
        # Classify error type
        error_type = classify_error_type(stimulus_type, correct_answer, extracted_answer, 
                                       identity_value, flanker_value, stimulus_row)
        error_stats[stimulus_type][error_type] += 1
    
    return dict(error_stats)


def analyze_folder_errors(test_folder_path):
    """
    Analyze error types for all sessions in a test result folder.
    
    Args:
        test_folder_path: Path to the test result folder
        
    Returns:
        tuple: (error_results dict, metadata dict)
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
    
    # Aggregate error counts across all sessions
    aggregated_errors = defaultdict(lambda: {'Corr': 0, 'SmErr': 0, 'FkErr': 0, 'Otr': 0, 'Empty': 0})
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
        session_errors = extract_error_analysis(session_data)
        
        # Aggregate error counts
        for stim_type, error_counts in session_errors.items():
            for error_type, count in error_counts.items():
                aggregated_errors[stim_type][error_type] += count
    
    print(f"Processed {session_count} sessions from {test_folder_path}")
    return dict(aggregated_errors), metadata


def create_error_analysis_plot(error_results_list, metadata_list, folder_paths, output_path, figure_name, cols_per_row=3):
    """
    Create nested bar plots showing error types for each condition across folders.
    
    Args:
        error_results_list: List of error analysis results for each folder
        metadata_list: List of metadata for each folder
        folder_paths: List of folder paths
        output_path: Path to save the figure
        figure_name: Name of the figure file
        cols_per_row: Number of columns (subplots) per row in the figure layout (default: 3)
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping visualization.")
        return
        
    # Color scheme for error types
    error_colors = {
        'Corr': '#2ca02c',    # Green
        'SmErr': '#ff7f0e',   # Orange  
        'FkErr': '#d62728',   # Red
        'Otr': '#808080',     # Gray
        'Empty': '#c7c7c7'    # Light gray
    }
    
    # Stimulus type name mapping
    stim_type_names = {
        0: "No conflict\n(eg 1 0 0 0)",
        1: "Simon only\n(eg 0 0 1 0)", 
        2: "Flanker only\n(eg 1 2 2 2)",
        3: "Simon + Flanker\n(eg 2 2 1 2)",
        4: "CgLtr\n(eg u 0 0 0)",
        5: "CgExN\n(eg 7 0 0 0)",
        6: "CgExN-R\n(eg 5 0 0 0)",
        7: "SmExN+Fk\n(eg 1 7 1 1)",
        8: "Simon + Flanker extended\n(eg 7 7 1 7)",
        9: "Sm+FkIdPos\n(eg 2 1 2 2)",
        10: "Flanker extended\n(eg 1 7 7 7)",
        11: "All letters\n(eg t u t t)",
        12: "Cg+FkLtr\n(eg 1 u u u)"
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    n_folders = len(error_results_list)
    
    # Create figure with subplots
    if n_folders == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    else:
        cols = min(cols_per_row, n_folders)
        rows = (n_folders + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 4*rows),sharey=True)
        if n_folders == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
    
    for i, (error_results, metadata, folder_path) in enumerate(zip(error_results_list, metadata_list, folder_paths)):
        ax = axes[i]
        
        # Handle empty folder paths (None placeholders)
        if error_results is None or not folder_path or folder_path.strip() == "":
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
        all_stim_types = sorted(error_results.keys())
        
        if not all_stim_types:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Prepare data for nested bar chart
        error_types = ['Corr', 'SmErr', 'FkErr', 'Otr']  # Exclude 'Empty' from main plot
        n_conditions = len(all_stim_types)
        n_error_types = len(error_types)
        
        # Calculate bar positions
        bar_width = 0.8 / n_error_types
        x_positions = np.arange(n_conditions)
        
        # Plot bars for each error type
        for j, error_type in enumerate(error_types):
            counts = []
            for stim_type in all_stim_types:
                total_trials = sum(error_results[stim_type].values()) - error_results[stim_type]['Empty']
                count = error_results[stim_type][error_type]
                proportion = count / total_trials if total_trials > 0 else 0
                counts.append(proportion)
            
            x_pos = x_positions + (j - n_error_types/2 + 0.5) * bar_width
            bars = ax.bar(x_pos, counts, bar_width, label=error_type, 
                         color=error_colors[error_type], alpha=0.8)
            
            # Add accuracy and count labels on bars
            for k, (bar, count) in enumerate(zip(bars, counts)):
                if count > 0.01:  # Only show label if proportion > 1%
                    stim_type = all_stim_types[k]
                    actual_count = error_results[stim_type][error_type]
                    accuracy = count  # count is already the proportion/accuracy
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{accuracy:.2f}\n({actual_count})', ha='center', va='bottom', fontsize=8)
        
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
        
        ## only plot the ylabel for the leftmost subplot
        if i%cols_per_row == 0:
            ax.set_ylabel(f'Choice proportion')
        ax.set_ylim(0, 1.)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([stim_type_names.get(st, f'Type{st}') for st in all_stim_types])
        # ax.set_xlabel(f'Experimental Conditions {nrep_value}rep')
        
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
        title_parts = [model_name,'\n']
        if sessions != 'unknown':
            title_parts.append(f'ssn={sessions}')
        if condition_part != 'unknown':
            title_parts.append(str(condition_part))
        if nrep_value != 'unknown':
            title_parts.append(f'{nrep_value}rep')
        
        # ax.set_title(' - '.join(title_parts), fontsize=10, pad=10)
        
        # Add legend (only for first subplot)
        if i == 0:
            ax.legend(frameon=False)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
    
    # Hide unused subplots
    for i in range(n_folders, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(model_name)
    plt.tight_layout()
    
    # Save figure
    full_path = os.path.join(output_path, figure_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nError analysis figure saved to: {full_path}")
    plt.show()


def analyze_multiple_folders_errors(folder_paths, output_path, figure_name, cols_per_row=3):
    """
    Analyze error types for multiple result folders and create combined visualization.
    
    Args:
        folder_paths: List of paths to test result folders
        output_path: Path to save the output figure
        figure_name: Name of the figure file
        cols_per_row: Number of columns (subplots) per row in the figure layout (default: 3)
    """
    error_results_list = []
    metadata_list = []
    
    print("=" * 80)
    print("ANALYZING ERROR TYPES IN MULTIPLE MSIT TEST RESULT FOLDERS")
    print("=" * 80)
    
    for i, folder_path in enumerate(folder_paths, 1):
        # Skip empty folder paths but add placeholder for subplot positioning
        if not folder_path or folder_path.strip() == "":
            print(f"\n[{i}/{len(folder_paths)}] Skipping empty folder path (reserving subplot position)")
            print("-" * 60)
            error_results_list.append(None)  # Add None as placeholder
            metadata_list.append({})  # Add empty metadata as placeholder
            continue
            
        print(f"\n[{i}/{len(folder_paths)}] Processing folder: {folder_path}")
        print("-" * 60)
        
        # If relative path provided, make it relative to current working directory
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(os.getcwd(), folder_path)
        
        error_results, metadata = analyze_folder_errors(folder_path)
        
        if error_results is None:
            print(f"Skipping folder due to errors: {folder_path}")
            error_results_list.append(None)  # Add None as placeholder
            metadata_list.append({})  # Add empty metadata as placeholder
            continue
            
        error_results_list.append(error_results)
        metadata_list.append(metadata)
        
        # Print summary for this folder
        print("Error type summary:")
        for stim_type in sorted(error_results.keys()):
            total = sum(error_results[stim_type].values())
            print(f"  Condition {stim_type}: {total} total trials")
            for error_type, count in error_results[stim_type].items():
                if count > 0:
                    proportion = count / total if total > 0 else 0
                    print(f"    {error_type}: {count} ({proportion:.3f})")
    
    if not error_results_list:
        print("\nError: No valid results found from any folder.")
        return
    
    print("\n" + "=" * 80)
    print("CREATING ERROR TYPE VISUALIZATION...")
    print("=" * 80)
    
    # Create the error analysis plot
    if PLOTTING_AVAILABLE:
        create_error_analysis_plot(error_results_list, metadata_list, folder_paths, output_path, figure_name, cols_per_row)
    else:
        print("Plotting not available. Analysis complete - check the printed summaries above.")


def main(cols_per_row=3):
    """Main function to run the error type analysis.
    
    Args:
        cols_per_row: Number of columns (subplots) per row in the figure layout (default: 3)
    """

    # ## Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_word/20250912-223716_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_word/20250912-224037_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_word/20250912-224531_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_word/20250912-223851_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_word/20250912-224221_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_word/20250912-224357_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-8",
    #     # "data/msit_pilot_outputs_word/20250912-230818_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-10_noFk5",
    #     # "data/msit_pilot_outputs_word/20250912-231032_gpt-4.1-nano_ssn-100_nrep-4_dgi-4_cond-8_noFk5",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_gpt4.1nano3b_dg4_word-stims_digit-resps.png"
    # cols_per_row = 3

    ### Configuration - using the same folder paths as specified by user
    folder_paths = [        
        "data/msit_pilot_outputs_word/20250913-094943_meta-llama-Llama-3.2-3B-Instruct_ssn-100_nrep-1_dgi-4_cond-0",
        "data/msit_pilot_outputs_word/20250913-101558_meta-llama-Llama-3.2-3B-Instruct_ssn-100_nrep-1_dgi-4_cond-2",
        "data/msit_pilot_outputs_word/20250913-105512_meta-llama-Llama-3.2-3B-Instruct_ssn-100_nrep-1_dgi-4_cond-10",
        "data/msit_pilot_outputs_word/20250913-100245_meta-llama-Llama-3.2-3B-Instruct_ssn-100_nrep-1_dgi-4_cond-1",
        "data/msit_pilot_outputs_word/20250913-102911_meta-llama-Llama-3.2-3B-Instruct_ssn-100_nrep-1_dgi-4_cond-3",
        "data/msit_pilot_outputs_word/20250913-104211_meta-llama-Llama-3.2-3B-Instruct_ssn-100_nrep-1_dgi-4_cond-8",
        # "data/msit_pilot_outputs_word/20250912-205115_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-11",
        # "data/msit_pilot_outputs_word/20250912-211711_meta-llama-Llama-3.2-3B-Instruct_ssn-1_nrep-1_dgi-4_cond-12",
    ]
    output_path = "results/msit_pilot_figures"
    figure_name = "error_types_llama3_3b_dg4_word-stims_digit-resps.png"
    cols_per_row = 3

    
    # ### Configuration - using the same folder paths as specified by user
    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250907-163637_gemini-1.5-flash-8b_ssn-20_nrep-3_dgi-4_cond-3_wtoLtrPrmpt",
    #     "data/msit_pilot_outputs_smallnrep/20250907-163737_gemini-1.5-flash-8b_ssn-20_nrep-3_dgi-4_cond-3_withLtrPrmpt",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_cond3_geni8b_compare-instruction.png"
    # cols_per_row = 4
    # test_between = {}


    # ### Configuration - using the same folder paths as specified by user
    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250907-154052_gemini-1.5-flash-8b_ssn-200_nrep-1_dgi-3_cond-0123",
    #     "data/msit_pilot_outputs_smallnrep/20250907-160035_gemini-1.5-flash-8b_ssn-200_nrep-1_dgi-3_cond-0123_wtoLtrPrmpt",
    #     "data/msit_pilot_outputs_smallnrep/20250907-154955_gemini-1.5-flash-8b_ssn-200_nrep-1_dgi-3_cond-3210",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_analysis_cond0123_geni8b_mixedrun.png"



        # Configuration - using the same folder paths as specified by user
    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250906_025212_model-gemini-1.5-flash-8b_sessions-40_Sm",
    #     "data/msit_pilot_outputs_smallnrep/20250907_021155_model-gemini-1.5-flash-8b_sessions-100_Fk",
    #     "data/msit_pilot_outputs_smallnrep/20250907_021004_model-gemini-1.5-flash-8b_sessions-100_SmFk",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_analysis_3conditions_geni8b.png"



    # ### Configuration - using the same folder paths as specified by user
    # folder_paths = [        

    #     "data/msit_pilot_outputs_smallnrep/20250909-235215_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250909-235727_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250910-000223_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250910-000732_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-3",
    #     "",
    #     "data/msit_pilot_outputs_smallnrep/20250910-002623_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-7",
    #     "data/msit_pilot_outputs_smallnrep/20250910-002003_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-8",
    #     "data/msit_pilot_outputs_smallnrep/20250910-001228_meta-llama-Llama-3.2-1B-instruct_ssn-100_nrep-1_dgi-4_cond-9",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_llama3_1b_dg4.png"
    # cols_per_row = 4


    # ### Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250911-024926_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250911-025019_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250911-025201_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250911-025342_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_smallnrep/20250911-151453_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_smallnrep/20250911-152755_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-8",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_llama3_3b_dg4.png"
    # cols_per_row = 4


    ## Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250912-004836_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250912-004927_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250912-005108_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250912-005251_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_smallnrep/20250912-010322_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_smallnrep/20250912-005600_meta-llama-Llama-3.2-3B-Instruct_ssn-0_nrep-1_dgi-4_cond-8",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_llama3_3b_dg4_instruct-numbers.png"
    # cols_per_row = 4


    # ## Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250912-003937_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003945_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003953_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250912-004001_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_smallnrep/20250912-004032_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_smallnrep/20250912-004012_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-8",
    #     "data/msit_pilot_outputs_smallnrep/20250912-004042_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-11",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_llama3_8b_dg4_instruct-numbers.png"
    # cols_per_row = 4


    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250912-003349_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-0",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003354_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-1",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003359_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-2",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003404_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-3",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003509_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-10",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003415_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-8",
    #     "data/msit_pilot_outputs_smallnrep/20250912-003535_llama3.1:8b-instruct-q4_0_ssn-0_nrep-1_dgi-4_cond-11",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_llama3_8b_dg4_instruct-characters.png"
    # cols_per_row = 4


    # ### Configuration - using the same folder paths as specified by user
    # folder_paths = [        
    #     "data/msit_pilot_outputs_smallnrep/20250907_023509_model-gpt-4.1-nano_sessions-100_SmDgi3",
    #     "data/msit_pilot_outputs_smallnrep/20250907_022558_model-gpt-4.1-nano_sessions-100_FkDgi3",
    #     "data/msit_pilot_outputs_smallnrep/20250907_022159_model-gpt-4.1-nano_sessions-100_SmFkDgi3",
    #     "data/msit_pilot_outputs_smallnrep/20250907-173400_gpt-4.1-nano_ssn-100_nrep-10_dgi-3_cond-1_SmDgi3",
    #     "data/msit_pilot_outputs_smallnrep/20250907-173731_gpt-4.1-nano_ssn-100_nrep-10_dgi-3_cond-2_FkDgi3",
    #     "data/msit_pilot_outputs_smallnrep/20250907-174034_gpt-4.1-nano_ssn-100_nrep-10_dgi-3_cond-3_SmFkDgi3",
    #     "",
    #     "",
    #     "",

    #     "data/msit_pilot_outputs_smallnrep/20250907_031715_model-gpt-4.1-nano_sessions-100_SmDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907_032208_model-gpt-4.1-nano_sessions-100_FkDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907_032421_model-gpt-4.1-nano_sessions-100_SmFkDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-175917_gpt-4.1-nano_ssn-100_nrep-10_dgi-4_cond-1_SmDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-180849_gpt-4.1-nano_ssn-100_nrep-10_dgi-4_cond-2_FkDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-181209_gpt-4.1-nano_ssn-100_nrep-10_dgi-4_cond-3_SmFkDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-184943_gpt-4.1-nano_ssn-100_nrep-5_dgi-4_cond-1_SmDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-185245_gpt-4.1-nano_ssn-100_nrep-5_dgi-4_cond-2_FkDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-185600_gpt-4.1-nano_ssn-100_nrep-5_dgi-4_cond-3_SmFkDgi4",

    #     "data/msit_pilot_outputs_smallnrep/20250907_122835_model-gpt-4.1-nano_sessions-100_SmDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907_123251_model-gpt-4.1-nano_sessions-100_FkDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907_123831_model-gpt-4.1-nano_sessions-100_SmFkDgi4",
    #     "data/msit_pilot_outputs_smallnrep/20250907-182357_gpt-4.1-nano_ssn-100_nrep-10_dgi-5_cond-1_SmDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907-182703_gpt-4.1-nano_ssn-100_nrep-10_dgi-5_cond-2_FkDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907-183036_gpt-4.1-nano_ssn-100_nrep-10_dgi-5_cond-3_SmFkDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907-183620_gpt-4.1-nano_ssn-100_nrep-5_dgi-5_cond-1_SmDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907-183929_gpt-4.1-nano_ssn-100_nrep-5_dgi-5_cond-2_FkDgi5",
    #     "data/msit_pilot_outputs_smallnrep/20250907-184235_gpt-4.1-nano_ssn-100_nrep-5_dgi-5_cond-3_SmFkDgi5",

    #     "data/msit_pilot_outputs_smallnrep/20250907-164735_gpt-4.1-nano_ssn-100_nrep-20_dgi-6_cond-1_SmDgi6",
    #     "data/msit_pilot_outputs_smallnrep/20250907-165202_gpt-4.1-nano_ssn-100_nrep-20_dgi-6_cond-2_FkDgi6",
    #     "data/msit_pilot_outputs_smallnrep/20250907-165540_gpt-4.1-nano_ssn-100_nrep-20_dgi-6_cond-3_SmFkDgi6",
    #     "",
    #     "",
    #     "",
    #     "",
    #     "",
    #     "",        
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "error_types_analysis_3conditions.png"
    # cols_per_row = 9
    
    if not folder_paths:
        print("Please specify folder paths in the script configuration.")
        sys.exit(1)
    
    # Analyze multiple folders and create visualization
    analyze_multiple_folders_errors(folder_paths, output_path, figure_name, cols_per_row)


if __name__ == "__main__":
    main()
