#!/usr/bin/env python3
"""
MSIT Type 3 Context Analysis

This script analyzes MSIT test results specifically for type 3 stimuli, 
classifying them based on their preceding stimulus type (0 or 3) and 
computing grand accuracy for each category.

Usage:
    python analyze_type3_context.py <test_result_folder_name>

Example:
    python analyze_type3_context.py data/msit_pilot_outputs/20250904_144951_model-gpt-4.1-nano_sessions-20_mixed
"""

import json
import os
import sys
import math
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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


def detect_available_stimulus_types(session_files):
    """
    Detect available stimulus types from session data.
    
    Args:
        session_files: List of session file paths
        
    Returns:
        set: Set of available stimulus types
    """
    available_types = set()
    
    for session_file in session_files[:5]:  # Check first 5 sessions for efficiency
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            final_types = session_data.get('final_types', [])
            available_types.update(final_types)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    
    return available_types


def extract_context_results(session_data, target_type, available_types):
    """
    Extract trial results for a target stimulus type classified by their preceding stimulus type.
    
    Args:
        session_data: Session data dictionary
        target_type: The stimulus type to analyze (e.g., 2, 3)
        available_types: Set of available stimulus types in the data
    
    Returns:
        tuple: (context_trials dict, minus_one_counts dict)
    """
    final_types = session_data.get('final_types', [])
    correct_answers = session_data.get('correct_answers', [])
    extracted_answers = session_data.get('extracted_answers', [])
    
    if len(final_types) != len(correct_answers) or len(final_types) != len(extracted_answers):
        print(f"Warning: Mismatched array lengths in session {session_data.get('session_id', 'unknown')}")
        return {}, {}
    
    # Initialize context trials dictionary and -1 counts based on available types
    context_trials = {}
    minus_one_counts = {}
    for prev_type in sorted(available_types):
        key = f'type{target_type}_after_{prev_type}'
        context_trials[key] = []
        minus_one_counts[key] = 0
    
    for i in range(1, len(final_types)):  # Start from index 1 to have a preceding trial
        current_type = final_types[i]
        previous_type = final_types[i-1]
        
        # Only analyze target type trials
        if current_type == target_type:
            correct_answer = correct_answers[i]
            extracted_answer = extracted_answers[i]
            
            # Track -1 values (indicates parsing failure) but exclude from accuracy calculation
            if extracted_answer == -1:
                if previous_type in available_types:
                    key = f'type{target_type}_after_{previous_type}'
                    if key in minus_one_counts:
                        minus_one_counts[key] += 1
                continue
                
            # Record 1 for correct, 0 for incorrect
            is_correct = 1 if correct_answer == extracted_answer else 0
            
            # Classify based on preceding type
            if previous_type in available_types:
                key = f'type{target_type}_after_{previous_type}'
                if key in context_trials:
                    context_trials[key].append(is_correct)
    
    return context_trials, minus_one_counts


def permutation_test(group1, group2, n_permutations=10000):
    """
    Perform a permutation test to compare two groups.
    
    Args:
        group1: List of 1s and 0s for group 1
        group2: List of 1s and 0s for group 2
        n_permutations: Number of permutations to perform
    
    Returns:
        float: p-value from the permutation test
    """
    if not group1 or not group2:
        return 1.0  # No difference if one group is empty
    
    # Calculate observed difference in means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    observed_diff = abs(mean1 - mean2)
    
    # Combine all data
    combined = group1 + group2
    n1 = len(group1)
    n2 = len(group2)
    
    # Perform permutations
    extreme_count = 0
    
    for _ in range(n_permutations):
        # Randomly shuffle and split
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:n1+n2]
        
        # Calculate difference for this permutation
        perm_diff = abs(np.mean(perm_group1) - np.mean(perm_group2))
        
        # Count if this difference is as extreme or more extreme
        if perm_diff >= observed_diff:
            extreme_count += 1
    
    # Calculate p-value
    p_value = extreme_count / n_permutations
    return p_value


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


def analyze_context_effects(test_folder_path):
    """
    Analyze context effects in a test result folder, automatically detecting available stimulus types.
    
    Args:
        test_folder_path: Path to the test result folder
        
    Returns:
        tuple: (context_results dict, metadata dict, target_type int)
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
    
    # Find all session files
    session_files = sorted(test_folder.glob("session_*.json"))
    
    if not session_files:
        print("Error: No session files found in the test folder")
        return None
    
    # Detect available stimulus types
    available_types = detect_available_stimulus_types(session_files)
    print(f"Detected stimulus types: {sorted(available_types)}")
    
    # Determine target type for analysis (prefer 3, then 2, then highest available)
    if 3 in available_types:
        target_type = 3
        analysis_name = "Type 3"
    elif 2 in available_types:
        target_type = 2
        analysis_name = "Type 2"
    else:
        target_type = max(available_types) if available_types else None
        analysis_name = f"Type {target_type}" if target_type is not None else "Unknown"
    
    if target_type is None:
        print("Error: No valid stimulus types found")
        return None
    
    print(f"Analyzing {analysis_name} context effects")
    print("-" * 50)
    
    # Initialize pooled context trials and -1 counts based on available types
    pooled_context_trials = {}
    pooled_minus_one_counts = {}
    for prev_type in sorted(available_types):
        key = f'type{target_type}_after_{prev_type}'
        pooled_context_trials[key] = []
        pooled_minus_one_counts[key] = 0
    
    session_count = 0
    
    for session_file in session_files:
        session_data = load_session_data(session_file)
        if session_data is None:
            continue
            
        session_count += 1
        session_id = session_data.get('session_id', session_count)
        print(f"Processing Session {session_id}...")
        
        # Extract context results for this session
        session_context_trials, session_minus_one_counts = extract_context_results(session_data, target_type, available_types)
        
        # Print session-level results
        print(f"  Session {session_id} {analysis_name.lower()} context results:")
        for context_type, trials in session_context_trials.items():
            minus_one_count = session_minus_one_counts.get(context_type, 0)
            if trials:
                correct = sum(trials)
                total = len(trials)
                accuracy = correct / total if total > 0 else 0
                print(f"    {context_type}: {correct}/{total} = {accuracy:.3f} (-1 count: {minus_one_count})")
            else:
                print(f"    {context_type}: No trials (-1 count: {minus_one_count})")
        
        # Add individual trials and -1 counts to pooled results
        for context_type, trials in session_context_trials.items():
            if context_type in pooled_context_trials:
                pooled_context_trials[context_type].extend(trials)
        for context_type, minus_one_count in session_minus_one_counts.items():
            if context_type in pooled_minus_one_counts:
                pooled_minus_one_counts[context_type] += minus_one_count
    
    print("-" * 50)
    print(f"GRAND AVERAGE ACCURACY FOR {analysis_name.upper()} CONTEXT EFFECTS WITH 95% CI (POOLED FROM {session_count} SESSIONS):")
    print("-" * 50)
    
    # Compute grand averages and confidence intervals from pooled data
    context_results = {}
    all_target_trials = []
    
    for context_type in sorted(pooled_context_trials.keys()):
        trials = pooled_context_trials[context_type]
        minus_one_count = pooled_minus_one_counts.get(context_type, 0)
        if trials:
            mean_acc, ci_lower, ci_upper = compute_confidence_interval(trials)
            
            context_results[context_type] = {
                'accuracy': mean_acc,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_trials': len(trials),
                'minus_one_count': minus_one_count
            }
            
            all_target_trials.extend(trials)
            
            print(f"{context_type}: {mean_acc:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}] (n={len(trials)}, -1 count: {minus_one_count})")
        else:
            context_results[context_type] = {
                'accuracy': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'n_trials': 0,
                'minus_one_count': minus_one_count
            }
            print(f"{context_type}: No trials found (-1 count: {minus_one_count})")
    
    # Overall target type accuracy with confidence interval
    total_minus_one_count = sum(pooled_minus_one_counts.values())
    if all_target_trials:
        overall_acc, overall_ci_lower, overall_ci_upper = compute_confidence_interval(all_target_trials)
        print(f"Overall {analysis_name} Accuracy: {overall_acc:.3f} [95% CI: {overall_ci_lower:.3f}, {overall_ci_upper:.3f}] (n={len(all_target_trials)}, total -1 count: {total_minus_one_count})")
        
        context_results[f'overall_type{target_type}'] = {
            'accuracy': overall_acc,
            'ci_lower': overall_ci_lower,
            'ci_upper': overall_ci_upper,
            'n_trials': len(all_target_trials),
            'minus_one_count': total_minus_one_count
        }
    
    # Perform permutation test if we have at least two groups with data
    valid_groups = [trials for trials in pooled_context_trials.values() if trials]
    if len(valid_groups) >= 2:
        # Compare the two largest groups
        valid_groups.sort(key=len, reverse=True)
        p_value = permutation_test(valid_groups[0], valid_groups[1])
        context_results['permutation_p_value'] = p_value
        print(f"Permutation test p-value: {p_value:.4f}")
    else:
        context_results['permutation_p_value'] = None
        print("Permutation test: Not enough data for comparison")
    
    return context_results, metadata, target_type


def create_context_bar_plot(results_list, metadata_list, target_types_list, output_path, figure_name):
    """
    Create bar plots with CI95 error bars for context effects.
    
    Args:
        results_list: List of context_results dictionaries
        metadata_list: List of metadata dictionaries
        target_types_list: List of target types for each result
        output_path: Path to save the figure
        figure_name: Name of the figure file
    """
    # Generate dynamic context type names and colors
    def get_context_type_names_and_colors(results_list, target_types_list):
        all_context_types = set()
        for results in results_list:
            for key in results.keys():
                if key.startswith('type') and '_after_' in key and not key.startswith('overall_'):
                    all_context_types.add(key)
        
        # Create name mapping
        context_type_names = {}
        for context_type in all_context_types:
            # Parse context_type like "type2_after_0" -> "Type 2 after Type 0"
            parts = context_type.split('_')
            if len(parts) >= 3:
                target = parts[0].replace('type', 'Type ')
                prev = parts[2]
                context_type_names[context_type] = f"{target} after Type {prev}"
        
        # Create color mapping with a color palette
        colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        context_type_colors = {}
        for i, context_type in enumerate(sorted(all_context_types)):
            context_type_colors[context_type] = colors[i % len(colors)]
        
        return context_type_names, context_type_colors
    
    context_type_names, context_type_colors = get_context_type_names_and_colors(results_list, target_types_list)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    n_folders = len(results_list)
    
    # Create figure with subplots
    if n_folders == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    else:
        # Calculate subplot layout
        cols = min(4, n_folders)  # Max 4 columns
        rows = (n_folders + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_folders == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
    
    for i, (results, metadata) in enumerate(zip(results_list, metadata_list)):
        ax = axes[i]
        
        # Extract context types (excluding overall results)
        context_types = [key for key in results.keys() 
                        if key.startswith('type') and '_after_' in key and not key.startswith('overall_')]
        context_types = sorted(context_types)
        
        # Check if we have any data
        has_data = any(results.get(ct, {}).get('n_trials', 0) > 0 for ct in context_types)
        
        if not has_data:
            target_type = target_types_list[i] if i < len(target_types_list) else 'Unknown'
            ax.text(0.5, 0.5, f'No Type {target_type} context data available', ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Prepare data for plotting
        accuracies = []
        ci_lowers = []
        ci_uppers = []
        valid_types = []
        
        for context_type in context_types:
            if results.get(context_type, {}).get('n_trials', 0) > 0:
                accuracies.append(results[context_type]['accuracy'])
                ci_lowers.append(results[context_type]['ci_lower'])
                ci_uppers.append(results[context_type]['ci_upper'])
                valid_types.append(context_type)
        
        if not valid_types:
            target_type = target_types_list[i] if i < len(target_types_list) else 'Unknown'
            ax.text(0.5, 0.5, f'No valid Type {target_type} context data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate error bars (distance from mean to CI bounds)
        yerr_lower = [acc - ci_low for acc, ci_low in zip(accuracies, ci_lowers)]
        yerr_upper = [ci_up - acc for acc, ci_up in zip(accuracies, ci_uppers)]
        yerr = [yerr_lower, yerr_upper]
        
        # Create bar plot with consistent colors
        x_pos = np.arange(len(valid_types))
        colors = [context_type_colors[ct] for ct in valid_types]
        bars = ax.bar(x_pos, accuracies, yerr=yerr, capsize=5, alpha=0.7, color=colors)
        
        # Customize plot
        ax.set_xlabel('Context Type')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([context_type_names[ct] for ct in valid_types], rotation=15, ha='right')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove grid
        ax.grid(False)
        
        # Create title with model name and nickname (extracted from folder path)
        model_name = metadata.get('parameters', {}).get('model', 'Unknown Model')
        # Extract nickname from the run_name (usually at the end after the last underscore)
        run_name = metadata.get('run_name', '')
        if run_name:
            # Split by underscore and take the last part as nickname
            nickname = run_name.split('_')[-1]
        else:
            nickname = 'Unknown'
        
        target_type = target_types_list[i] if i < len(target_types_list) else 'Unknown'
        title = f'{model_name} {nickname}\nType {target_type} Context Effects'
        ax.set_title(title, fontsize=10, pad=15)
        
        # Add value labels on bars (moved higher to avoid overlap with error bars)
        for j, (bar, acc, n_trials, ci_upper) in enumerate(zip(bars, accuracies, [results[ct]['n_trials'] for ct in valid_types], ci_uppers)):
            height = bar.get_height()
            # Position text above the upper confidence interval
            text_y = ci_upper + 0.03
            minus_one_count = results[valid_types[j]].get('minus_one_count', 0)
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{acc:.3f}\n(n={n_trials})\n(-1: {minus_one_count})', ha='center', va='bottom', fontsize=8)
        
        # Add permutation test p-value if available
        p_value = results.get('permutation_p_value')
        if p_value is not None:
            ax.text(0.5, 0.95, f'Permutation test p = {p_value:.4f}', 
                   transform=ax.transAxes, ha='center', va='top', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7),
                   fontsize=9)
    
    # Hide unused subplots
    for i in range(n_folders, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    full_path = os.path.join(output_path, figure_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {full_path}")
    plt.show()


def analyze_multiple_folders(folder_paths, output_path, figure_name):
    """
    Analyze multiple result folders for context effects and create combined visualization.
    
    Args:
        folder_paths: List of paths to test result folders
        output_path: Path to save the output figure
        figure_name: Name of the figure file
    """
    results_list = []
    metadata_list = []
    target_types_list = []
    
    print("=" * 80)
    print("ANALYZING CONTEXT EFFECTS IN MULTIPLE MSIT TEST RESULT FOLDERS")
    print("=" * 80)
    
    for i, folder_path in enumerate(folder_paths, 1):
        print(f"\n[{i}/{len(folder_paths)}] Processing folder: {folder_path}")
        print("-" * 60)
        
        # If relative path provided, make it relative to current working directory
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(os.getcwd(), folder_path)
        
        result = analyze_context_effects(folder_path)
        
        if result is None:
            print(f"Skipping folder due to errors: {folder_path}")
            continue
            
        context_results, metadata, target_type = result
        results_list.append(context_results)
        metadata_list.append(metadata)
        target_types_list.append(target_type)
    
    if not results_list:
        print("\nError: No valid results found from any folder.")
        return
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION...")
    print("=" * 80)
    
    # Create the bar plot
    create_context_bar_plot(results_list, metadata_list, target_types_list, output_path, figure_name)


def main():
    # For backward compatibility with command line usage
    if len(sys.argv) == 2:
        test_folder_path = sys.argv[1]
        
        # If relative path provided, make it relative to current working directory
        if not os.path.isabs(test_folder_path):
            test_folder_path = os.path.join(os.getcwd(), test_folder_path)
        
        result = analyze_context_effects(test_folder_path)
        
        if result is None:
            sys.exit(1)
        
        context_results, metadata, target_type = result
        print(f"\nType {target_type} Context Results:")
        print(context_results)
        return
    
    # ============================================================================
    # USER CONFIGURATION SECTION
    # ============================================================================
    
    # SPECIFY RESULT FOLDER PATHS HERE (can be relative or absolute paths)
    folder_paths = [
        # Add your folder paths here, one per line. Examples:
        "data/msit_pilot_outputs/20250904_152123_model-gpt-4.1-nano_sessions-120_mixed-Rm-merged",
        "data/msit_pilot_outputs/20250904_160006_model-gpt-4.1-nano_sessions-60_mixed-Rm-CgFk",
    ]
    
    # SPECIFY OUTPUT CONFIGURATION HERE
    output_path = "results/msit_pilot_figures"  # Directory to save the figure
    figure_name = "type3_context_effects.png"  # Name of the output figure file
    
    # ============================================================================
    # END USER CONFIGURATION
    # ============================================================================
    
    if not folder_paths:
        print("Please specify folder paths in the USER CONFIGURATION SECTION of the script.")
        print("Edit the 'folder_paths' list in the main() function.")
        print("\nExample usage:")
        print("python analyze_type3_context.py data/msit_pilot_outputs/20250904_144951_model-gpt-4.1-nano_sessions-20_mixed")
        sys.exit(1)
    
    # Analyze multiple folders and create visualization
    analyze_multiple_folders(folder_paths, output_path, figure_name)


if __name__ == "__main__":
    main()
    """
    Usage examples:
    
    1. Command line (single folder):
       python scripts_analysis/analyze_type3_context.py "data/msit_pilot_outputs/20250904_144951_model-gpt-4.1-nano_sessions-20_mixed"
    
    2. Multiple folders (edit the script's main() function):
       - Edit the folder_paths list in the USER CONFIGURATION SECTION
       - Edit output_path and figure_name as needed
       - Run: python scripts_analysis/analyze_type3_context.py
    """
