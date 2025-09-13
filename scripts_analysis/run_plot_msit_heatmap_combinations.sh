#!/bin/bash

# MSIT Heatmap Plotting Script - Run Combinations
# This script runs plot_msit_heatmap.py for combinations of result folders and trace modes

# Configuration
PYTHON_PATH="/opt/homebrew/Caskroom/miniforge/base/envs/llm-local/bin/python"
SCRIPT_PATH="scripts_analysis/plot_msit_heatmap.py"

# Result folders (4 different ndigits experiments)
RESULT_FOLDERS=(
    "data/msit_pilot_outputs_mapall/20250913-040452_meta-llama-Llama-3.2-3B-Instruct_auto_ndigits-3_english_try"
    "data/msit_pilot_outputs_mapall/20250913-031246_meta-llama-Llama-3.2-3B-Instruct_auto_ndigits-4_english_try"
    "data/msit_pilot_outputs_mapall/20250913-044114_meta-llama-Llama-3.2-3B-Instruct_auto_ndigits-5_english_try"
    "data/msit_pilot_outputs_mapall/20250913-051650_meta-llama-Llama-3.2-3B-Instruct_auto_ndigits-6_english_try"
)

# Trace modes (2 options)
TRACE_MODES=(
    "target-position"
    "flanker"
)

# Other configuration options
VARIABLE="digit_softmax_probs"
COLORMAP="viridis"
OUTPUT_PATH="results/msit_pilot_figures"
FILTERS="match-first"

echo "Starting MSIT Heatmap Combination Analysis"
echo "=========================================="
echo "Python path: $PYTHON_PATH"
echo "Script path: $SCRIPT_PATH"
echo "Variable: $VARIABLE"
echo "Colormap: $COLORMAP"
echo "Output path: $OUTPUT_PATH"
echo "Filters: $FILTERS"
echo ""

# Counter for tracking progress
total_combinations=$((${#RESULT_FOLDERS[@]} * ${#TRACE_MODES[@]}))
current_combination=0

# Loop through all combinations
for result_folder in "${RESULT_FOLDERS[@]}"; do
    for trace_mode in "${TRACE_MODES[@]}"; do
        current_combination=$((current_combination + 1))
        
        echo "[$current_combination/$total_combinations] Processing:"
        echo "  Result folder: $result_folder"
        echo "  Trace mode: $trace_mode"
        
        # Check if result folder exists
        if [ ! -d "$result_folder" ]; then
            echo "  WARNING: Result folder does not exist, skipping..."
            echo ""
            continue
        fi
        
        # Run the plotting script
        echo "  Running heatmap generation..."
        $PYTHON_PATH $SCRIPT_PATH \
            --result_folder "$result_folder" \
            --variable "$VARIABLE" \
            --trace_mode "$trace_mode" \
            --colormap "$COLORMAP" \
            --output_path "$OUTPUT_PATH" \
            --filters "$FILTERS"
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully generated heatmap"
        else
            echo "  ✗ Error generating heatmap"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "Analysis complete! Generated $total_combinations heatmaps."
echo "Output directory: $OUTPUT_PATH"
echo ""
echo "Generated files should include:"
for result_folder in "${RESULT_FOLDERS[@]}"; do
    # Extract ndigits from folder name
    ndigits=$(echo "$result_folder" | grep -o 'ndigits-[0-9]' | cut -d'-' -f2)
    if [ -n "$ndigits" ]; then
        for trace_mode in "${TRACE_MODES[@]}"; do
            echo "  - msit_heatmap_${ndigits}digts_${VARIABLE}_${trace_mode}_${FILTERS}.png"
        done
    fi
done
