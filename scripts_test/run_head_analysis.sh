#!/bin/bash

# Head Ablation and Swap Analysis Runner
# Comprehensive script to run head ablation/swap experiments and generate impact reports

set -e  # Exit on error

# Default parameters
MODEL="gpt2"
SAMPLES=20
MODE="both"  # Options: ablate, swap, both
TOP_N=15
OUTPUT_DIR=""
LAYER=""
HEAD=""
WITH_EXAMPLES="true"
VERBOSE="false"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_usage() {
    cat << EOF
Head Ablation and Swap Analysis Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -m, --model MODEL        Model name (default: gpt2)
                            Options: gpt2, gpt2-large, gpt2-medium, gpt2-xl
    
    -s, --samples N          Number of test samples (default: 20)
    
    -M, --mode MODE          Analysis mode (default: both)
                            Options: ablate, swap, both
                            Note: 'swap' mode automatically runs multi-head test with top 15 heads
    
    -t, --top_n N           Number of top heads to show in report (default: 15)
    
    -o, --output DIR        Output directory (default: auto-generated)
    
    -l, --layer N           Single layer to test (optional, for quick tests)
    
    -h, --head N            Single head to test (requires --layer, for quick tests)
    
    --no-examples           Disable few-shot examples in prompts
    
    -v, --verbose           Enable verbose output
    
    --help                  Show this help message

EXAMPLES:
    # Run both ablation and swap with default settings
    $0
    
    # Run only ablation mode with gpt2-large, 50 samples
    $0 --model gpt2-large --mode ablate --samples 50
    
    # Run swap mode with custom top_n for report
    $0 --mode swap --top_n 25
    
    # Quick test of single head L5H8
    $0 --layer 5 --head 8 --mode ablate --samples 10
    
    # Run both modes with specific output directory
    $0 --mode both --output ~/my_analysis_results --top_n 20

OUTPUT:
    - Raw JSON results in data/msit_pilot_outputs_smallnrep/[timestamp]_[model]_[mode]/
    - Human-readable impact reports with "_impact_report.txt" suffix
    - Summary printed to console with key findings

EOF
}

print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

log_info() {
    print_colored "$BLUE" "ℹ️  $1"
}

log_success() {
    print_colored "$GREEN" "✅ $1"
}

log_warning() {
    print_colored "$YELLOW" "⚠️  $1"
}

log_error() {
    print_colored "$RED" "❌ $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -M|--mode)
            MODE="$2"
            shift 2
            ;;
        -t|--top_n)
            TOP_N="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--layer)
            LAYER="$2"
            shift 2
            ;;
        -h|--head)
            HEAD="$2"
            shift 2
            ;;
        --no-examples)
            WITH_EXAMPLES="false"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate parameters
if [[ ! "$MODE" =~ ^(ablate|swap|both)$ ]]; then
    log_error "Invalid mode: $MODE. Must be 'ablate', 'swap', or 'both'"
    exit 1
fi

if [[ ! "$MODEL" =~ ^(gpt2|gpt2-large|gpt2-medium|gpt2-xl)$ ]]; then
    log_warning "Unusual model name: $MODEL. Common options: gpt2, gpt2-large, gpt2-medium, gpt2-xl"
fi

if [[ -n "$HEAD" && -z "$LAYER" ]]; then
    log_error "Head specified without layer. Use --layer N --head M for single head testing"
    exit 1
fi

# Set up output directory if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    OUTPUT_DIR="${REPO_ROOT}/data/msit_pilot_outputs_smallnrep"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
log_info "Starting Head Analysis with configuration:"
echo "  Model: $MODEL"
echo "  Samples: $SAMPLES" 
echo "  Mode: $MODE"
echo "  Top N: $TOP_N"
echo "  Output: $OUTPUT_DIR"
if [[ -n "$LAYER" ]]; then
    echo "  Target: Layer $LAYER$(if [[ -n "$HEAD" ]]; then echo ", Head $HEAD"; fi)"
fi
echo "  Examples: $WITH_EXAMPLES"
echo ""

# Function to run ablation/swap experiment
run_experiment() {
    local mode=$1
    local experiment_name="${mode}"
    
    log_info "Running $mode mode experiment..."
    
    # Build command
    local cmd="python ${SCRIPT_DIR}/head_ablation_sweep.py"
    cmd="$cmd --model $MODEL"
    cmd="$cmd --samples $SAMPLES" 
    cmd="$cmd --mode $mode"
    
    if [[ "$WITH_EXAMPLES" == "false" ]]; then
        cmd="$cmd --no_examples"
    fi
    
    if [[ -n "$OUTPUT_DIR" ]]; then
        cmd="$cmd --output_dir $OUTPUT_DIR"
    fi
    
    if [[ -n "$LAYER" ]]; then
        cmd="$cmd --layer $LAYER"
        if [[ -n "$HEAD" ]]; then
            cmd="$cmd --head $HEAD"
        fi
    fi
    
    # Execute experiment
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Executing: $cmd"
    fi
    
    cd "$SCRIPT_DIR"
    
    if eval "$cmd"; then
        log_success "$mode experiment completed successfully"
        return 0
    else
        log_error "$mode experiment failed"
        return 1
    fi
}

# Function to find latest results file
find_latest_results() {
    local mode=$1
    local pattern="*${MODEL}*head-${mode}*"
    
    # Look for most recent results directory
    local latest_dir=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort -r | head -1)
    
    if [[ -z "$latest_dir" ]]; then
        log_warning "No results directory found for pattern: $pattern"
        return 1
    fi
    
    # Look for JSON results file
    local results_file=$(find "$latest_dir" -name "*.json" -type f | head -1)
    
    if [[ -z "$results_file" ]]; then
        log_warning "No JSON results file found in: $latest_dir"
        return 1
    fi
    
    echo "$results_file"
    return 0
}

# Function to generate impact report
generate_report() {
    local results_file=$1
    local mode=$2
    
    if [[ ! -f "$results_file" ]]; then
        log_error "Results file not found: $results_file"
        return 1
    fi
    
    log_info "Generating impact analysis report for $mode mode..."
    
    local report_cmd="python ${SCRIPT_DIR}/analyze_head_impact.py"
    report_cmd="$report_cmd \"$results_file\""
    report_cmd="$report_cmd --top_n $TOP_N"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Executing: $report_cmd"
    fi
    
    cd "$SCRIPT_DIR"
    
    if eval "$report_cmd"; then
        # Find the generated report file
        local report_file="${results_file%.*}_impact_report.txt"
        if [[ -f "$report_file" ]]; then
            log_success "Impact report generated: $report_file"
            
            # Show preview of key findings
            log_info "Key findings preview:"
            echo "----------------------------------------"
            grep -A 10 "KEY OBSERVATIONS:" "$report_file" 2>/dev/null || echo "No key observations section found"
            echo "----------------------------------------"
            
            return 0
        else
            log_warning "Report file not found at expected location: $report_file"
            return 1
        fi
    else
        log_error "Failed to generate impact report"
        return 1
    fi
}

# Function to run multi-head swap analysis
run_multi_head_swap() {
    local results_file=$1
    local exp_mode=$2
    
    if [[ "$exp_mode" != "swap" ]]; then
        log_info "Skipping multi-head swap (only applicable to swap mode)"
        return 0
    fi
    
    log_info "Running multi-head swap analysis..."
    
    # Extract top heads from results JSON
    if [[ ! -f "$results_file" ]]; then
        log_error "Results file not found: $results_file"
        return 1
    fi
    
    # Use Python to extract top 10 heads from results
    local head_list
    head_list=$(python3 -c "
import json
import sys

try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
    
    important_heads = data.get('important_heads', [])
    if len(important_heads) < 2:
        print('SKIP:Not enough important heads found')
        sys.exit(0)
    
    # Get top 10 heads (or fewer if available)
    top_heads = important_heads[:min(10, len(important_heads))]
    head_strings = [f\"L{head['layer']}H{head['head']}\" for head in top_heads]
    print(','.join(head_strings))
    
except Exception as e:
    print(f'ERROR:{e}')
    sys.exit(1)
" 2>/dev/null)
    
    if [[ -z "$head_list" ]]; then
        log_error "Failed to extract heads from results"
        return 1
    fi
    
    if [[ "$head_list" == "SKIP:"* ]]; then
        log_info "${head_list#SKIP:}"
        return 0
    fi
    
    if [[ "$head_list" == "ERROR:"* ]]; then
        log_error "Error extracting heads: ${head_list#ERROR:}"
        return 1
    fi
    
    log_info "Extracted top heads for multi-head swap: $head_list"
    
    # Build multi-head swap command
    local multi_cmd="python ${SCRIPT_DIR}/multi_head_swap_tool.py"
    multi_cmd="$multi_cmd --heads \"$head_list\""
    multi_cmd="$multi_cmd --model $MODEL"
    multi_cmd="$multi_cmd --samples $((SAMPLES * 2))"  # Use 2x samples for multi-head
    
    if [[ "$WITH_EXAMPLES" == "false" ]]; then
        multi_cmd="$multi_cmd --no_examples"
    fi
    
    # Use same output directory as main experiments
    local output_dir
    if [[ -n "$OUTPUT_DIR" ]]; then
        output_dir="$OUTPUT_DIR"
    else
        output_dir="${REPO_ROOT}/data/msit_pilot_outputs_smallnrep"
    fi
    multi_cmd="$multi_cmd --output_dir \"$output_dir\""
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Executing multi-head swap: $multi_cmd"
    fi
    
    cd "$SCRIPT_DIR"
    
    if eval "$multi_cmd"; then
        log_success "Multi-head swap analysis completed"
        return 0
    else
        log_error "Multi-head swap analysis failed"
        return 1
    fi
}

# Main execution
log_info "Starting head analysis pipeline..."

# Track results files for analysis
RESULTS_FILES=()

# Run experiments based on mode
if [[ "$MODE" == "both" ]]; then
    # Run both ablation and swap
    for exp_mode in "ablate" "swap"; do
        if run_experiment "$exp_mode"; then
            if results_file=$(find_latest_results "$exp_mode"); then
                RESULTS_FILES+=("$results_file:$exp_mode")
            fi
        fi
    done
else
    # Run single mode
    if run_experiment "$MODE"; then
        if results_file=$(find_latest_results "$MODE"); then
            RESULTS_FILES+=("$results_file:$MODE")
        fi
    fi
fi

# Run multi-head swap analysis for swap results
log_info "Running multi-head swap analysis on swap results..."
for result_entry in "${RESULTS_FILES[@]}"; do
    IFS=':' read -r results_file exp_mode <<< "$result_entry"
    run_multi_head_swap "$results_file" "$exp_mode"
done

# Generate reports for all results
if [[ ${#RESULTS_FILES[@]} -eq 0 ]]; then
    log_error "No results files found to analyze"
    exit 1
fi

log_info "Generating impact analysis reports..."

ALL_REPORTS=()
for result_entry in "${RESULTS_FILES[@]}"; do
    IFS=':' read -r results_file exp_mode <<< "$result_entry"
    if generate_report "$results_file" "$exp_mode"; then
        report_file="${results_file%.*}_impact_report.txt"
        ALL_REPORTS+=("$report_file")
    fi
done

# Final summary
echo ""
log_success "Head analysis pipeline completed!"
echo ""
echo "📊 ANALYSIS SUMMARY:"
echo "===================="
echo "Experiments run: $([[ "$MODE" == "both" ]] && echo "ablate, swap" || echo "$MODE")"
echo "Model: $MODEL"
echo "Samples: $SAMPLES"
echo "Top heads analyzed: $TOP_N"

if [[ ${#ALL_REPORTS[@]} -gt 0 ]]; then
    echo ""
    echo "📄 Generated Reports:"
    for report in "${ALL_REPORTS[@]}"; do
        echo "  • $(basename "$report")"
        echo "    Full path: $report"
    done
    
    echo ""
    log_info "To view detailed results:"
    for report in "${ALL_REPORTS[@]}"; do
        echo "  cat \"$report\""
    done
else
    log_warning "No impact reports were generated"
fi

echo ""
log_info "Pipeline completed successfully! 🎉"
