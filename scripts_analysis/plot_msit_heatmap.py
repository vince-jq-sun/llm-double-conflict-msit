#!/usr/bin/env python3
"""
MSIT Heatmap Plotting Script

Creates a heatmap visualization of digit logits/probabilities from MSIT experiment results.
- X-axis: Flanker values (0-9)
- Y-axis: Target identity @ Target position combinations
- Values: Specified variable (digit_softmax_probs, digit_logits, digit_token_ids) for correct answer
- NaN values for combinations where target_identity == flanker_value (invalid conditions)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import glob
import argparse


def load_meta_data(result_folder):
    """Load all session JSON files from the result folder."""
    meta_file = os.path.join(result_folder, "run_metadata.json")
    ## load the json file as a dictionary
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    return meta_data

def load_session_data(result_folder):
    """Load all session JSON files from the result folder."""
    session_files = glob.glob(os.path.join(result_folder, "session_*.json"))
    session_data = []
    
    print(f"Found {len(session_files)} session files")
    
    for file_path in sorted(session_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                session_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return session_data

def extract_heatmap_data(ndigits,target_id_ranges,session_data, variable_x, trace_mode="target-position", filter="all_valid"):
    """Extract data for heatmap construction."""

    link_symbol = "<|"

    # Initialize data structures
    flanker_values = list(range(10))  # 0-9
    target_identities = list(range(target_id_ranges[0],target_id_ranges[1]))  # 0-9 (assuming ndigits=4 means digits 0-3, but keeping flexible)
    target_positions = list(range(1, ndigits+1))  # 1-4 positions
    
    # Create all possible combinations
    y_labels = []
    
    if filter == "match-first":
        # Add matching pairs first (id_val == id_pos)
        for target_pos in target_positions:
            for target_id in target_identities:
                if target_id == target_pos:
                    y_labels.append(f"{target_pos}{link_symbol}{target_id}")
        
        # Then add non-matching pairs (id_val != id_pos)
        for target_pos in target_positions:
            for target_id in target_identities:
                if target_id != target_pos:
                    y_labels.append(f"{target_pos}{link_symbol}{target_id}")
    else:
        # Default ordering: iterate by position first, then identity
        for target_pos in target_positions:
            for target_id in target_identities:
                y_labels.append(f"{target_pos}{link_symbol}{target_id}")
    
    # Initialize heatmap matrix with NaN
    heatmap_data = np.full((len(y_labels), len(flanker_values)), np.nan)
     
    # Process each session
    for session in session_data:
        try:
            trial_info = session['trial_info']
            target_identity = trial_info['target_identity']
            target_pos_index = trial_info['target_pos_index']
            flanker_value = trial_info['flanker_value']
            correct_answer = session['correct_answer']
            
            # Skip if target_identity == flanker_value (invalid condition)
            if target_identity == flanker_value:
                continue
            
            # Get the variable value based on trace mode
            digit_logits_info = session.get('digit_logits_info', {})
            variable_data = digit_logits_info.get(variable_x, {})
            
            # Determine which digit to trace based on trace_mode
            if trace_mode == "target-position":
                target_digit = str(correct_answer)  # target-position position digit (current behavior)
            elif trace_mode == "flanker":
                target_digit = str(flanker_value)  # Flanker value digit
            else:
                raise ValueError(f"Invalid trace_mode: {trace_mode}. Must be 'target-position' or 'flanker'")
            
            if target_digit in variable_data:
                value = variable_data[target_digit]

                if filter in ["all_valid", "match-first"]:
                    pass
                elif filter == "idval-is-idpos":
                    if target_identity == target_pos_index:
                        value = np.nan
                elif filter == "idval-isnt-idpos":
                    if target_identity != target_pos_index:
                        value = np.nan
                if filter == "@1":
                    if target_pos_index != 1:
                        value = np.nan
                elif filter == "not@1":
                    if target_pos_index == 1:
                        value = np.nan
                elif filter == "idval-1":
                    if target_identity != 1:
                        value = np.nan
                elif filter == "idval-not1":
                    if target_identity == 1:
                        value = np.nan
                
                # Find the corresponding row and column indices
                y_label = f"{target_pos_index}{link_symbol}{target_identity}"
                if y_label in y_labels and flanker_value in flanker_values:
                    row_idx = y_labels.index(y_label)
                    col_idx = flanker_values.index(flanker_value)
                    heatmap_data[row_idx, col_idx] = value
            
        except Exception as e:
            print(f"Error processing session {session.get('session_id', 'unknown')}: {e}")
    
    return heatmap_data, y_labels, flanker_values

def create_heatmap(heatmap_data, y_labels, x_labels, variable_x, trace_mode, colormap, ndigits, filter, output_path, show_text=False):
    """Create and save the heatmap visualization with line plot below."""

    upper_color = 'b'
    lower_color = 'r'
    first_divider_color = 'black'
    second_divider_color = 'black'
    divider_lw = 1.5

    # Map variable name for display
    display_variable = "softmax for digits" if variable_x == "digit_softmax_probs" else variable_x
    trace_description = "Target Position" if trace_mode == "target-position" else "Flanker Value"
    # Create figure with manual subplot positioning
    fig = plt.figure(figsize=(6, ndigits*3 + 2))
    
    # Define positions manually: [left, bottom, width, height]
    if filter == "match-first":
        # Add space for upper line plot
        heatmap_pos = [0.1, 0.3, 0.65, 0.5]  # Main heatmap (reduced height)
        colorbar_pos = [0.78, 0.3, 0.03, 0.5]  # Colorbar on the right
        upper_lineplot_pos = [0.1, 0.82, 0.65, 0.10]  # Upper line plot (matching pairs)
        lineplot_pos = [0.1, 0.18, 0.65, 0.10]  # Lower line plot (non-matching pairs)
    else:
        heatmap_pos = [0.1, 0.3, 0.65, 0.6]  # Main heatmap
        colorbar_pos = [0.78, 0.3, 0.03, 0.6]  # Colorbar on the right
        lineplot_pos = [0.1, 0.18, 0.65, 0.10]  # Line plot below, same width as heatmap
    
    # Create heatmap axis
    ax1 = fig.add_axes(heatmap_pos)
    
    # Create heatmap without colorbar first
    im = ax1.imshow(heatmap_data, cmap=colormap, aspect='auto', 
                    interpolation='nearest')
    
    # Add text annotations (conditional based on show_text parameter)
    if show_text:
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax1.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                   ha="center", va="center", color="white" if heatmap_data[i, j] < 0.5 else "black")
    
    # Set ticks and labels
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax1.set_yticks(range(len(y_labels)))
    ax1.set_yticklabels(y_labels)
    
    # Add vertical lines
    ax1.axhline(ndigits-0.5, color='white', linestyle='-', linewidth=6)
    ax1.axvline(0.5, color=first_divider_color, linestyle='--', linewidth=divider_lw)
    ax1.axvline(ndigits+0.5, color=second_divider_color, linestyle='--', linewidth=divider_lw)

    ## plot a vertical line at x=0, from y=0 to y=ndigits
    ax1.vlines(-0.5, -0.5, -0.6 + ndigits, colors=upper_color, linestyles='-', linewidth=7)
    ax1.vlines(9.5, -0.5, -0.6 + ndigits, colors=upper_color, linestyles='-', linewidth=7)
    
    ## plot a vertical line at x=0, from y=ndigits to ylabel length
    ax1.vlines(-0.5, -0.4+ndigits, len(y_labels)-0.5, colors=lower_color, linestyles='-', linewidth=7)
    ax1.vlines(9.5, -0.4+ndigits, len(y_labels)-0.5, colors=lower_color, linestyles='-', linewidth=7)
    
    # Remove heatmap borders
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Customize heatmap
    # ax1.set_title(f'MSIT Heatmap: {display_variable} for {trace_description} Digit\n',
    #               fontsize=14, pad=20)
    ax1.set_xlabel('')  # Remove xlabel from heatmap since line plot will have it
    ax1.set_ylabel(f'Target Position <| Target Identity', fontsize=12)
    ax1.tick_params(axis='y', rotation=0)
    ax1.tick_params(axis='x', rotation=0)
    
    # Create separate colorbar axis
    cax = fig.add_axes(colorbar_pos)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(display_variable, rotation=270, labelpad=15)
    
    # Calculate marginal averages across y-axis
    if filter == "match-first":
        # Calculate marginals for matching pairs (first ndigits rows)
        matching_marginal_means = []
        for col_idx in range(heatmap_data.shape[1]):  # For each flanker value
            col_data = heatmap_data[:ndigits, col_idx]  # First ndigits rows
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                matching_marginal_means.append(np.mean(valid_data))
            else: 
                matching_marginal_means.append(np.nan)
        
        # Calculate marginals for non-matching pairs (remaining rows)
        nonmatching_marginal_means = []
        for col_idx in range(heatmap_data.shape[1]):  # For each flanker value
            col_data = heatmap_data[ndigits:, col_idx]  # Remaining rows
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                nonmatching_marginal_means.append(np.mean(valid_data))
            else:
                nonmatching_marginal_means.append(np.nan)
    else:
        # Calculate marginals for all rows (original behavior)
        marginal_means = []
        for col_idx in range(heatmap_data.shape[1]):  # For each flanker value
            col_data = heatmap_data[:, col_idx]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                marginal_means.append(np.mean(valid_data))
            else:
                marginal_means.append(np.nan)
    
    if filter == "match-first":
        # Create upper line plot for matching pairs
        ax_upper = fig.add_axes(upper_lineplot_pos)
        marker_style = 'D' if trace_mode == 'flanker' else 'o'
        ax_upper.plot(x_labels, matching_marginal_means, f'{upper_color}-{marker_style}', linewidth=2, markersize=6, 
                     markerfacecolor=upper_color, markeredgecolor=upper_color, label='identity pos. = identity val.')
        
        # Add vertical lines matching heatmap
        ax_upper.axvline(0.5, color=first_divider_color, linestyle='--', linewidth=divider_lw)
        ax_upper.axvline(ndigits+0.5, color=second_divider_color, linestyle='--', linewidth=divider_lw)
        
        # Add legend to upper plot
        ax_upper.legend(fontsize=9, frameon=False)
        
        # Customize upper line plot
        ax_upper.set_ylabel(f'{trace_mode}-digits\nmean softmax', fontsize=10)
        ax_upper.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
        ax_upper.spines['top'].set_visible(False)
        ax_upper.spines['right'].set_visible(False)
        ax_upper.spines['bottom'].set_visible(False)
        ax_upper.set_xlim(-0.5, len(x_labels)-0.5)
        ax_upper.set_xticks(range(len(x_labels)))
        
        # Create lower line plot for non-matching pairs
        ax2 = fig.add_axes(lineplot_pos)
        marker_style = 'D' if trace_mode == 'flanker' else 'o'
        ax2.plot(x_labels, nonmatching_marginal_means, f'{lower_color}-{marker_style}', linewidth=2, markersize=6, 
                markerfacecolor=lower_color, markeredgecolor=lower_color, label='identity pos. != identity val.')
        
        # Add vertical lines matching heatmap
        ax2.axvline(0.5, color=first_divider_color, linestyle='--', linewidth=divider_lw)
        ax2.axvline(ndigits+0.5, color=second_divider_color, linestyle='--', linewidth=divider_lw)
        
        # Add legend to lower plot
        ax2.legend(fontsize=9, frameon=False)
        
        # Customize lower line plot
        ax2.set_xlabel('Flanker Value', fontsize=12)
        ax2.set_ylabel(f'{trace_mode}-digits\nmean softmax', fontsize=10)
        ax2.tick_params(axis='x', rotation=0)
        
        # Remove top and right spines from line plot
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Set x-axis to match heatmap exactly
        ax2.set_xlim(-0.5, len(x_labels)-0.5)
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels)
    else:
        # Create single line plot axis with exact same width as heatmap
        ax2 = fig.add_axes(lineplot_pos)
        
        # Create line plot
        marker_style = 'D' if trace_mode == 'flanker' else 'o'
        ax2.plot(x_labels, marginal_means, f'k-{marker_style}', linewidth=2, markersize=6, 
                 markerfacecolor='black', markeredgecolor='black')
        
        # Add vertical lines matching heatmap
        ax2.axvline(0.5, color=first_divider_color, linestyle='--', linewidth=divider_lw)
        ax2.axvline(ndigits+0.5, color=second_divider_color, linestyle='--', linewidth=divider_lw)
        
        # Customize line plot
        ax2.set_xlabel('Flanker Value', fontsize=12)
        ax2.set_ylabel(f'{trace_mode}-digits mean softmax', fontsize=12)
        ax2.tick_params(axis='x', rotation=0)
        
        # Remove top and right spines from line plot
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Set x-axis to match heatmap exactly
        ax2.set_xlim(-0.5, len(x_labels)-0.5)
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels)
        
    # Save the plot
    output_filename = f"msit_heatmap_{ndigits}digts_{variable_x}_{trace_mode}_{filter}.png"
    full_output_path = os.path.join(output_path, output_filename)
    
    # Adjust layout
    # plt.suptitle(output_filename)
    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap with line plot saved to: {full_output_path}")
    
    # Show plot
    plt.close()
    
    return full_output_path

def print_data_summary(heatmap_data, y_labels, x_labels):
    """Print summary statistics of the heatmap data."""
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    
    print(f"\nData Summary:")
    print(f"Total cells: {heatmap_data.size}")
    print(f"Valid data points: {len(valid_data)}")
    print(f"NaN values: {np.sum(np.isnan(heatmap_data))}")
    
    if len(valid_data) > 0:
        print(f"Min value: {np.min(valid_data):.6f}")
        print(f"Max value: {np.max(valid_data):.6f}")
        print(f"Mean value: {np.mean(valid_data):.6f}")
        print(f"Std value: {np.std(valid_data):.6f}")
    
    # Print which combinations have data
    print(f"\nCombinations with data:")
    for i, y_label in enumerate(y_labels):
        for j, x_label in enumerate(x_labels):
            if not np.isnan(heatmap_data[i, j]):
                print(f"  {y_label} x Flanker {x_label}: {heatmap_data[i, j]:.6f}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MSIT Heatmap Plotting Script')
    
    parser.add_argument('--result_folder', '-r', type=str, required=True,
                        help='Path to the result folder containing session JSON files')
    
    parser.add_argument('--variable', '-v', type=str, default='digit_softmax_probs',
                        choices=['digit_softmax_probs', 'digit_logits', 'digit_token_ids'],
                        help='Variable to plot (default: digit_softmax_probs)')
    
    parser.add_argument('--trace_mode', '-t', type=str, default='flanker',
                        choices=['target-position', 'flanker'],
                        help='Trace mode: target-position (trace identity position digit) or flanker (trace flanker value digit) (default: flanker)')
    
    parser.add_argument('--colormap', '-c', type=str, default='viridis',
                        help='Colormap for heatmap (default: viridis)')
    
    parser.add_argument('--output_path', '-o', type=str, default='results/msit_pilot_figures',
                        help='Output directory for plots (default: results/msit_pilot_figures)')
    
    parser.add_argument('--filters', '-f', type=str, nargs='+', default=['match-first'],
                        choices=['idval-is-idpos', 'idval-isnt-idpos', 'all_valid', '@1', 'not@1', 'match-first'],
                        help='List of filters to apply (default: match-first)')
    
    parser.add_argument('--show_text', '-s', action='store_true', default=False,
                        help='Display text annotations on heatmap cells (default: False)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Dynamic colormap selection based on trace mode
    if args.trace_mode == "flanker":
        colormap = "plasma"
    else:
        colormap = args.colormap
    
    print(f"MSIT Heatmap Analysis")
    print(f"Result folder: {args.result_folder}")
    print(f"Variable: {args.variable}")
    print(f"Trace mode: {args.trace_mode}")
    print(f"Colormap: {colormap} (auto-selected for {args.trace_mode})")
    print(f"Filters: {args.filters}")
    print("-" * 50)
    
    # Check if result folder exists
    if not os.path.exists(args.result_folder):
        print(f"Error: Result folder does not exist: {args.result_folder}")
        return
    
    # Load session data
    session_data = load_session_data(args.result_folder)
    if not session_data:
        print("No session data found!")
        return
    
    meta_data = load_meta_data(args.result_folder)
    ndigits = meta_data['ndigits']

    # Extract heatmap data
    for filter_name in args.filters:
        heatmap_data, y_labels, x_labels = extract_heatmap_data( 
                                                ndigits,
                                                (1,ndigits+1),
                                                session_data,
                                                args.variable,
                                                args.trace_mode,
                                                filter_name)
    
        # Print summary
        # print_data_summary(heatmap_data, y_labels, x_labels)
        
        # Create and save heatmap
        output_file = create_heatmap(heatmap_data, y_labels, x_labels, args.variable, args.trace_mode, colormap, ndigits, filter_name, args.output_path, args.show_text)
        
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
