#!/usr/bin/env python3
"""
MSIT Test Results Analyzer

This script analyzes MSIT test results and computes the grand average accuracy 
for different stimulus types (0, 1, 2, 3) across all sessions in a test run.

Usage:
    python analyze_msit_results.py <test_result_folder_name>

Example:
    python analyze_msit_results.py msit_test_results/20250904_121638_model-gpt-4.1-nano_sessions-2
"""

import json
import os
import sys
import math
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random


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


def extract_trial_results(session_data):
    """
    Extract individual trial results from a session.
    
    Returns:
        tuple: (stimulus_trials dict, empty_counts dict)
        - stimulus_trials: {stimulus_type: [list of 1s and 0s for correct/incorrect]}
        - empty_counts: {stimulus_type: count of empty (-1) responses}
    """
    final_types = session_data.get('final_types', [])
    correct_answers = session_data.get('correct_answers', [])
    extracted_answers = session_data.get('extracted_answers', [])
    
    if len(final_types) != len(correct_answers) or len(final_types) != len(extracted_answers):
        print(f"Warning: Mismatched array lengths in session {session_data.get('session_id', 'unknown')}")
        return {}, {}
    
    # Group trial results by stimulus type
    stimulus_trials = defaultdict(list)
    empty_counts = defaultdict(int)
    
    for i in range(len(final_types)):
        stimulus_type = final_types[i]
        correct_answer = correct_answers[i]
        extracted_answer = extracted_answers[i]
        
        # Count empty responses (-1 indicates parsing failure)
        if extracted_answer == -1:
            empty_counts[stimulus_type] += 1
            continue
            
        # Record 1 for correct, 0 for incorrect
        is_correct = 1 if correct_answer == extracted_answer else 0
        stimulus_trials[stimulus_type].append(is_correct)
    
    return stimulus_trials, empty_counts


def compute_accuracy_by_stimulus_type(session_data):
    """
    Compute accuracy for each stimulus type in a session.
    
    Returns:
        dict: {stimulus_type: {'correct': count, 'total': count, 'empty': count}}
    """
    stimulus_trials, empty_counts = extract_trial_results(session_data)
    
    stimulus_stats = {}
    for stim_type, trials in stimulus_trials.items():
        correct = sum(trials)
        total = len(trials)
        empty = empty_counts.get(stim_type, 0)
        stimulus_stats[stim_type] = {'correct': correct, 'total': total, 'empty': empty}
    
    # Also include stimulus types that only had empty responses
    for stim_type, empty in empty_counts.items():
        if stim_type not in stimulus_stats:
            stimulus_stats[stim_type] = {'correct': 0, 'total': 0, 'empty': empty}
    
    return stimulus_stats


def compute_confidence_interval(trials, confidence=0.95):
    """
    Compute confidence interval for accuracy using Wilson score interval.
    
    Args:
        trials: List of 1s and 0s (correct/incorrect)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (mean_accuracy, lower_bound, upper_bound)
    """
    if not trials:
        return 0, 0, 0
    
    n = len(trials)
    p = sum(trials) / n  # proportion of successes
    
    # Wilson score interval
    # For 95% CI, z = 1.96
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * math.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return p, lower, upper


def analyze_test_results(test_folder_path):
    """
    Analyze all sessions in a test result folder.
    
    Args:
        test_folder_path: Path to the test result folder
        
    Returns:
        tuple: (grand_results dict, metadata dict)
    """
    test_folder = Path(test_folder_path)
    
    if not test_folder.exists():
        print(f"Error: Test folder not found: {test_folder_path}")
        return None
    
    # Load run metadata
    metadata_file = test_folder / "run_metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Analyzing test run: {metadata.get('run_name', 'Unknown')}")
        print(f"Model: {metadata.get('parameters', {}).get('model', 'Unknown')}")
        print(f"Total sessions: {metadata.get('total_sessions', 'Unknown')}")
        print("-" * 50)
    
    # Pool all individual trial results across sessions
    pooled_trials = defaultdict(list)
    pooled_empty = defaultdict(int)
    session_count = 0
    
    # Find all session files
    session_files = sorted(test_folder.glob("session_*.json"))
    
    if not session_files:
        print("Error: No session files found in the test folder")
        return None
    
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
        session_id = session_data.get('session_id', session_count)
        print(f"Processing Session {session_id}...")
        
        # Extract individual trial results for this session
        session_trials, session_empty = extract_trial_results(session_data)
        
        # Compute accuracy for this session (for display)
        session_stats = compute_accuracy_by_stimulus_type(session_data)
        
        # Print session-level results
        print(f"  Session {session_id} accuracy by stimulus type:")
        for stim_type in sorted(session_stats.keys()):
            correct = session_stats[stim_type]['correct']
            total = session_stats[stim_type]['total']
            empty = session_stats[stim_type]['empty']
            accuracy = correct / total if total > 0 else 0
            print(f"    Type {stim_type}: {correct}/{total} = {accuracy:.3f} (empty: {empty})")
        
        # Add individual trials to pooled results
        for stim_type, trials in session_trials.items():
            pooled_trials[stim_type].extend(trials)
        
        # Add empty counts to pooled results
        for stim_type, count in session_empty.items():
            pooled_empty[stim_type] += count
    
    print("-" * 50)
    print(f"GRAND AVERAGE ACCURACY WITH 95% CI (POOLED FROM {session_count} SESSIONS):")
    print("-" * 50)
    
    # Compute grand averages and confidence intervals from pooled data
    grand_results = {}
    all_trials = []
    
    for stim_type in sorted(pooled_trials.keys()):
        trials = pooled_trials[stim_type]
        mean_acc, ci_lower, ci_upper = compute_confidence_interval(trials)
        
        grand_results[stim_type] = {
            'accuracy': mean_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_trials': len(trials),
            'n_empty': pooled_empty.get(stim_type, 0)
        }
        
        all_trials.extend(trials)
        
        empty_count = pooled_empty.get(stim_type, 0)
        print(f"Stimulus Type {stim_type}: {mean_acc:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}] (n={len(trials)}, empty={empty_count})")
    
    # Overall accuracy with confidence interval
    if all_trials:
        overall_acc, overall_ci_lower, overall_ci_upper = compute_confidence_interval(all_trials)
        print(f"Overall Accuracy: {overall_acc:.3f} [95% CI: {overall_ci_lower:.3f}, {overall_ci_upper:.3f}] (n={len(all_trials)})")
        
        grand_results['overall'] = {
            'accuracy': overall_acc,
            'ci_lower': overall_ci_lower,
            'ci_upper': overall_ci_upper,
            'n_trials': len(all_trials)
        }
    
    return grand_results, metadata, pooled_trials


def permutation_test(data1, data2, n_permutations=10000):
    """
    Perform a permutation test to compare two groups of binary data.
    
    Args:
        data1: List of 1s and 0s (correct/incorrect) for group 1
        data2: List of 1s and 0s (correct/incorrect) for group 2
        n_permutations: Number of permutations to perform
    
    Returns:
        float: p-value from the permutation test
    """
    if not data1 or not data2:
        return 1.0  # No data to compare
    
    # Calculate observed difference in means
    mean1 = sum(data1) / len(data1)
    mean2 = sum(data2) / len(data2)
    observed_diff = abs(mean1 - mean2)
    
    # Combine all data
    combined_data = data1 + data2
    n1, n2 = len(data1), len(data2)
    
    # Perform permutations
    extreme_count = 0
    
    for _ in range(n_permutations):
        # Randomly shuffle and split
        random.shuffle(combined_data)
        perm_group1 = combined_data[:n1]
        perm_group2 = combined_data[n1:n1+n2]
        
        # Calculate difference in means for this permutation
        perm_diff = abs(sum(perm_group1)/len(perm_group1) - sum(perm_group2)/len(perm_group2))
        
        # Count if this difference is as extreme or more extreme
        if perm_diff >= observed_diff:
            extreme_count += 1
    
    # Calculate p-value
    p_value = extreme_count / n_permutations
    return p_value


def create_bar_plot(results_list, metadata_list, output_path, figure_name, test_between=None, pooled_trials_list=None, cols_per_row=4):
    """
    Create bar plots with CI95 error bars for multiple result folders.
    
    Args:
        results_list: List of grand_results dictionaries
        metadata_list: List of metadata dictionaries
        output_path: Path to save the figure
        figure_name: Name of the figure file
        test_between: Dictionary specifying permutation tests {condition: folder_index}
        pooled_trials_list: List of pooled trials data for each folder
        cols_per_row: Number of columns per row in subplot layout (default: 4)
    """
    # Stimulus type name mapping
    stim_type_names = {
        0: "Cg",
        1: "Sm", 
        2: "Fk",
        3: "Sm+Fk",
        4: "CgLtr",
        5: "CgExN",
        6: "CgExN-R"
    }
    
    # Fixed color scheme for all stimulus types
    stim_type_colors = {
        0: '#1f77b4',  # blue
        1: '#ff7f0e',  # orange
        2: '#2ca02c',  # green
        3: '#d62728',  # red
        4: '#9467bd',  # purple
        5: '#8c564b',  # brown
        6: '#808000',  # sandy brown
    }
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    n_folders = len(results_list)
    
    # Create figure with subplots
    if n_folders == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        # Calculate subplot layout
        cols = min(cols_per_row, n_folders)  # Use user-defined cols_per_row
        rows = (n_folders + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 4*rows))
        if n_folders == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
    
    for i, (results, metadata) in enumerate(zip(results_list, metadata_list)):
        ax = axes[i]
        
        # Extract stimulus types (excluding 'overall')
        stim_types = sorted([k for k in results.keys() if k != 'overall'])
        
        if not stim_types:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Prepare data for plotting
        accuracies = [results[stim_type]['accuracy'] for stim_type in stim_types]
        ci_lowers = [results[stim_type]['ci_lower'] for stim_type in stim_types]
        ci_uppers = [results[stim_type]['ci_upper'] for stim_type in stim_types]
        
        # Calculate error bars (distance from mean to CI bounds)
        yerr_lower = [acc - ci_low for acc, ci_low in zip(accuracies, ci_lowers)]
        yerr_upper = [ci_up - acc for acc, ci_up in zip(accuracies, ci_uppers)]
        yerr = [yerr_lower, yerr_upper]
        
        # Create bar plot with consistent colors
        x_pos = np.arange(len(stim_types))
        colors = [stim_type_colors[st] for st in stim_types]
        bars = ax.bar(x_pos, accuracies, yerr=yerr, capsize=5, alpha=0.7, color=colors)
        
        # Customize plot
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([stim_type_names[st] for st in stim_types])
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove grid
        ax.grid(False)
        
        # Create title with model name, nickname, nrep, and overall accuracy
        model_name = metadata.get('parameters', {}).get('model', 'Unknown Model')
        nrep = metadata.get('parameters', {}).get('nrep', 'Unknown')
        # Extract nickname from the run_name (usually at the end after the last underscore)
        run_name = metadata.get('run_name', '')
        if run_name:
            # Split by underscore and take the last part as nickname
            nickname = run_name.split('_')[-1]
        else:
            nickname = 'Unknown'
        
        # Get overall accuracy if available
        overall_acc_text = ""
        if 'overall' in results:
            overall_acc = results['overall']['accuracy']
            overall_acc_text = f"\n(Acc={overall_acc:.3f})"
        
        title = f'{model_name} {nickname} (nrep={nrep}){overall_acc_text}'
        ax.set_xlabel(f'Stimulus Type\n|{title}')
        # ax.set_title(title, fontsize=10, pad=10)
        
        # Add value labels on bars (moved higher to avoid overlap with error bars)
        for j, (bar, acc, n_trials, n_empty, ci_upper) in enumerate(zip(bars, accuracies, 
                                                                             [results[st]['n_trials'] for st in stim_types],
                                                                             [results[st]['n_empty'] for st in stim_types], 
                                                                             ci_uppers)):
            height = bar.get_height()
            # Position text above the upper confidence interval
            text_y = ci_upper + 0.03
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{acc:.3f}\n(n={n_trials})\nempty={n_empty}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_folders, len(axes)):
        axes[i].set_visible(False)
    
    # Add permutation test p-value to figure suptitle if specified
    if test_between and pooled_trials_list:
        p_value_text = ""
        print(f"DEBUG: test_between = {test_between}")
        print(f"DEBUG: pooled_trials_list length = {len(pooled_trials_list)}")
        for i, trials in enumerate(pooled_trials_list):
            print(f"DEBUG: Folder {i} has conditions: {list(trials.keys())}")
        
        # 收集所有需要比较的数据
        comparison_data = []
        for condition, folder_idx in test_between.items():
            print(f"DEBUG: Getting condition {condition} from folder {folder_idx}")
            if (len(pooled_trials_list) > folder_idx and 
                condition in pooled_trials_list[folder_idx]):
                data = pooled_trials_list[folder_idx][condition]
                comparison_data.append((condition, folder_idx, data))
                print(f"DEBUG: Found condition {condition} in folder {folder_idx}, data length: {len(data)}")
            else:
                print(f"DEBUG: Condition {condition} not found in folder {folder_idx}")
        
        # 如果有两组数据，进行比较
        if len(comparison_data) == 2:
            cond1, folder1, data1 = comparison_data[0]
            cond2, folder2, data2 = comparison_data[1]
            print(f"DEBUG: Comparing condition {cond1} (folder {folder1}) vs condition {cond2} (folder {folder2})")
            print(f"DEBUG: Data1 length: {len(data1)}, Data2 length: {len(data2)}")
            p_value = permutation_test(data1, data2)
            print(f"DEBUG: P-value: {p_value}")
            cond_name_dict = {0: 'Cg', 1: 'Sm', 2: 'Fk', 3: 'SmFk'}
            cond1_name = cond_name_dict[cond1]
            cond2_name = cond_name_dict[cond2]
            p_value_text = f"\n{cond1_name}(F{folder1+1}) vs {cond2_name}(F{folder2+1}): p={p_value:.4f}"
        else:
            print(f"DEBUG: Expected 2 conditions for comparison, got {len(comparison_data)}")
        
        if p_value_text:
            fig.suptitle(f"Permutation Test Results:{p_value_text}", fontsize=12, y=0.98)
        else:
            print("DEBUG: No p-value text generated")
    
    plt.tight_layout()
    
    # Save figure
    full_path = os.path.join(output_path, figure_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {full_path}")
    plt.show()


def analyze_multiple_folders(folder_paths, output_path, figure_name, test_between=None, cols_per_row=4):
    """
    Analyze multiple result folders and create combined visualization.
    
    Args:
        folder_paths: List of paths to test result folders
        output_path: Path to save the output figure
        figure_name: Name of the figure file
        test_between: Dictionary specifying permutation tests {condition: folder_index}
        cols_per_row: Number of columns per row in subplot layout (default: 4)
    """
    results_list = []
    metadata_list = []
    pooled_trials_list = []
    
    print("=" * 80)
    print("ANALYZING MULTIPLE MSIT TEST RESULT FOLDERS")
    print("=" * 80)
    
    for i, folder_path in enumerate(folder_paths, 1):
        print(f"\n[{i}/{len(folder_paths)}] Processing folder: {folder_path}")
        print("-" * 60)
        
        # If relative path provided, make it relative to current working directory
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(os.getcwd(), folder_path)
        
        result = analyze_test_results(folder_path)
        
        if result is None:
            print(f"Skipping folder due to errors: {folder_path}")
            continue
            
        grand_results, metadata, pooled_trials = result
        results_list.append(grand_results)
        metadata_list.append(metadata)
        pooled_trials_list.append(pooled_trials)
    
    if not results_list:
        print("\nError: No valid results found from any folder.")
        return
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION...")
    print("=" * 80)
    
    # Create the bar plot
    create_bar_plot(results_list, metadata_list, output_path, figure_name, test_between, pooled_trials_list, cols_per_row)


def main():
    # For backward compatibility with command line usage
    if len(sys.argv) == 2:
        test_folder_path = sys.argv[1]
        
        # If relative path provided, make it relative to current working directory
        if not os.path.isabs(test_folder_path):
            test_folder_path = os.path.join(os.getcwd(), test_folder_path)
        
        result = analyze_test_results(test_folder_path)
        
        if result is None:
            sys.exit(1)
        
        grand_results, metadata, pooled_trials = result
        print("\nGrand Results:")
        print(grand_results)
        return
    
    # ============================================================================
    # USER CONFIGURATION SECTION
    # ============================================================================
    ### Configuration - using the same folder paths as specified by user
    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250907-163637_gemini-1.5-flash-8b_ssn-20_nrep-3_dgi-4_cond-3_wtoLtrPrmpt",
    #     "data/msit_pilot_outputs_smallnrep/20250907-163737_gemini-1.5-flash-8b_ssn-20_nrep-3_dgi-4_cond-3_withLtrPrmpt",
    # ]
    # output_path = "results/msit_pilot_figures"
    # figure_name = "acc_cond3_geni8b_compare-instruction.png"
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
    # figure_name = "acc_cond0123_geni8b_mixedrun.png"
    # cols_per_row = 4
    # test_between = {}


    # folder_paths = [
    # # Add your folder paths here, one per line. Examples:
    # "data/msit_pilot_outputs_smallnrep/20250907-152437_gemini-1.5-flash-8b_ssn-200_nrep-1_dgi-3_cond-1234",
    # ]
    # figure_name = "cond1234_geni8b_mixedrun_1rep.png"  # Name of the output figure file
    # cols_per_row = 4
    # test_between = {}

        
    # folder_paths = [
    # # Add your folder paths here, one per line. Examples:
    # "data/msit_pilot_outputs_smallnrep/20250907_004514_model-gemini-1.5-flash-8b_sessions-50_all5-rm",
    # ]
    # figure_name = "5conds_seprate_50rep.png"  # Name of the output figure file
    # test_between = {}
    
    # folder_paths = [
    # # Add your folder paths here, one per line. Examples:
    # "data/msit_pilot_outputs_smallnrep/20250905_201222_model-gemini-1.5-flash-8b_sessions-100_all-alter",
    # "data/msit_pilot_outputs_smallnrep/20250906_024110_model-gpt-4.1-nano_sessions-20_all-alter-rm",
    # ]
    # figure_name = "2conds_seprate_6rep.png"  # Name of the output figure file
    # test_between = {2:0,3:0}


    folder_paths = [
        "data/msit_pilot_outputs_smallnrep/20250907_023509_model-gpt-4.1-nano_sessions-100_SmDgi3",
        "data/msit_pilot_outputs_smallnrep/20250907_022558_model-gpt-4.1-nano_sessions-100_FkDgi3",
        "data/msit_pilot_outputs_smallnrep/20250907_022159_model-gpt-4.1-nano_sessions-100_SmFkDgi3",
        "data/msit_pilot_outputs_smallnrep/20250907_031715_model-gpt-4.1-nano_sessions-100_SmDgi4",
        "data/msit_pilot_outputs_smallnrep/20250907_032208_model-gpt-4.1-nano_sessions-100_FkDgi4",
        "data/msit_pilot_outputs_smallnrep/20250907_032421_model-gpt-4.1-nano_sessions-100_SmFkDgi4",
        "data/msit_pilot_outputs_smallnrep/20250907_122835_model-gpt-4.1-nano_sessions-100_SmDgi5",
        "data/msit_pilot_outputs_smallnrep/20250907_123251_model-gpt-4.1-nano_sessions-100_FkDgi5",
        "data/msit_pilot_outputs_smallnrep/20250907_123831_model-gpt-4.1-nano_sessions-100_SmFkDgi4",
        "data/msit_pilot_outputs_smallnrep/20250907-164735_gpt-4.1-nano_ssn-100_nrep-20_dgi-6_cond-1_SmDgi6",
        "data/msit_pilot_outputs_smallnrep/20250907-165202_gpt-4.1-nano_ssn-100_nrep-20_dgi-6_cond-2_FkDgi6",
        "data/msit_pilot_outputs_smallnrep/20250907-165540_gpt-4.1-nano_ssn-100_nrep-20_dgi-6_cond-3_SmFkDgi6",
    ]
    figure_name = "3conds_seprate_20rep_gpt-nano.png"  # Name of the output figure file
    test_between = {}
    cols_per_row = 3  # Number of columns per row in subplot layout
    

    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250907_010104_model-gemini-1.5-flash-8b_sessions-40_CgLtr",
    #     "data/msit_pilot_outputs_smallnrep/20250907_020048_model-gemini-1.5-flash-8b_sessions-100_CgExN",
    #     "data/msit_pilot_outputs_smallnrep/20250905_185802_model-gemini-1.5-flash-8b_sessions-10_Cg",
    #     "data/msit_pilot_outputs_smallnrep/20250906_025212_model-gemini-1.5-flash-8b_sessions-40_Sm",
    #     "data/msit_pilot_outputs_smallnrep/20250907_021155_model-gemini-1.5-flash-8b_sessions-100_Fk",
    #     "data/msit_pilot_outputs_smallnrep/20250907_021004_model-gemini-1.5-flash-8b_sessions-100_SmFk",
    #     # "data/msit_pilot_outputs_smallnrep/20250907_005324_model-gemini-1.5-flash-8b_sessions-100_CgLtr",
    #     "data/msit_pilot_outputs_smallnrep/20250907_020257_model-gemini-1.5-flash-8b_sessions-100_CgExN-R",
    # ]
    # figure_name = "6conds_seprate_8rep.png"  # Name of the output figure file
    # # test_between = {2:2,3:3}
    # test_between = {}
    # cols_per_row = 6  # Number of columns per row in subplot layout
    

    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250904_220711_model-gemini-1.5-flash-8b_sessions-100_all-alter-rm",
    #     "data/msit_pilot_outputs_smallnrep/20250904_215525_model-gemini-2.0-flash_sessions-100_all-alter-rm",
    #     "data/msit_pilot_outputs_smallnrep/20250904_205052_model-gpt-4.1-nano_sessions-60_all-alter-rm",
    #     "data/msit_pilot_outputs_smallnrep/20250904_221523_model-gpt-4.1-mini_sessions-10_all-alter-rm",
    # ]
    # figure_name = "all-alter_1rep.png"  # Name of the output figure file


    # SPECIFY RESULT FOLDER PATHS HERE (can be relative or absolute paths)
    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250904_170833_model-gpt-4.1-nano_sessions-10_Sm-only",
    #     "data/msit_pilot_outputs_smallnrep/20250904_171020_model-gpt-4.1-nano_sessions-10_mix-Sm-Fk",
    #     "data/msit_pilot_outputs_smallnrep/20250904_170935_model-gpt-4.1-nano_sessions-10_Fk-only",        
    # ]
    # figure_name = "Sm_Fk_mix&sep_10reps.png"  # Name of the output figure file


    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs_smallnrep/20250904_165244_model-gpt-4.1-nano_sessions-10_SmFk-only",
    #     "data/msit_pilot_outputs_smallnrep/20250904_165826_model-gpt-4.1-nano_sessions-10_SmFk-only",
    #     "data/msit_pilot_outputs_smallnrep/20250904_165415_model-gpt-4.1-nano_sessions-10_mix-Cg-SmFk",        
    # ]
    # figure_name = "Cg_SmFk_mix&sep_10reps.png"  # Name of the output figure file

    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs/20250904_143424_model-gpt-4.1-nano_sessions-5_CgOnly",
    #     "data/msit_pilot_outputs/20250904_133658_model-gpt-4.1-nano_sessions-5_onlyC",        
    #     "data/msit_pilot_outputs/20250904_145111_model-gpt-4.1-nano_sessions-20_onlyC",
    #     "data/msit_pilot_outputs/20250904_144951_model-gpt-4.1-nano_sessions-20_mixed",
    # ]
    # figure_name = "onlyC_vs_mixed.png"  # Name of the output figure file

    # folder_paths = [
    #     # Add your folder paths here, one per line. Examples:
    #     "data/msit_pilot_outputs/20250904_140834_model-gpt-4.1-nano_sessions-5_SmOnly",
    #     "data/msit_pilot_outputs/20250904_140935_model-gpt-4.1-nano_sessions-5_FkOnly",
    #     "data/msit_pilot_outputs/20250904_141529_model-gpt-4.1-nano_sessions-5_Sm-Fk-alter",
    #     "data/msit_pilot_outputs/20250904_142203_model-gpt-4.1-nano_sessions-5_Cg-Sm-alter",
    #     "data/msit_pilot_outputs/20250904_142230_model-gpt-4.1-nano_sessions-5_Cg-Fk-alter",
    #     # "path/to/another/result/folder",
    # ]
    # figure_name = "SmOnly_vs_FkOnly_vs_Sm-Fk-alter.png"  # Name of the output figure file
    
    # SPECIFY OUTPUT CONFIGURATION HERE
    output_path = "results/msit_pilot_figures"  # Directory to save the figure
    
    
    # ============================================================================
    # END USER CONFIGURATION
    # ============================================================================
    
    if not folder_paths:
        print("Please specify folder paths in the USER CONFIGURATION SECTION of the script.")
        print("Edit the 'folder_paths' list in the main() function.")
        sys.exit(1)
    
    # Analyze multiple folders and create visualization
    analyze_multiple_folders(folder_paths, output_path, figure_name, test_between, cols_per_row)


if __name__ == "__main__":
    main()
    """
    Usage examples:
    
    1. Command line (single folder, backward compatibility):
       python scripts_analysis/analyze_msit_results.py "data/msit_pilot_outputs/20250904_133658_model-gpt-4.1-nano_sessions-5_onlyC"
    
    2. Multiple folders (edit the script's main() function):
       - Edit the folder_paths list in the USER CONFIGURATION SECTION
       - Edit output_path and figure_name as needed
       - Run: python scripts_analysis/analyze_msit_results.py
    """
