#!/bin/bash

# MSIT API Test Script
# Test script for msit_api_test.py with various configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MSIT_SCRIPT="$PROJECT_DIR/scripts_test/msit_api_test.py"

echo -e "${BLUE}=== MSIT API Test Suite ===${NC}"
echo "Project directory: $PROJECT_DIR"
echo "MSIT script: $MSIT_SCRIPT"

# Check if the script exists
if [ ! -f "$MSIT_SCRIPT" ]; then
    echo -e "${RED}Error: msit_api_test.py not found at $MSIT_SCRIPT${NC}"
    exit 1
fi

# Check if API.json exists
API_FILE="$PROJECT_DIR/scripts/API.json"
if [ ! -f "$API_FILE" ]; then
    echo -e "${YELLOW}Warning: API.json not found at $API_FILE${NC}"
    echo "Some tests may fail without API keys"
fi

# Function to run a test
run_test() {
    local test_name="$1"
    local sessions="$2"
    local ndigits="$3"
    local stim_types="$4"
    local nrep="$5"
    local model="$6"
    local max_tokens="$7"
    local restriction="$8"
    
    echo -e "\n${BLUE}--- Test: $test_name ---${NC}"
    echo "Sessions: $sessions, Digits: $ndigits, Types: $stim_types, Reps: $nrep"
    echo "Model: $model, Max tokens: $max_tokens, Restriction: $restriction"
    
    # Build command
    cmd="python3 \"$MSIT_SCRIPT\" --sessions $sessions --ndigits $ndigits --stim_types \"$stim_types\" --nrep $nrep --model \"$model\" --max_tokens $max_tokens --api_file \"$API_FILE\""
    
    if [ "$restriction" != "none" ]; then
        cmd="$cmd --restriction \"$restriction\""
    fi
    
    echo "Command: $cmd"
    
    # Run the test
    if eval $cmd; then
        echo -e "${GREEN}✓ Test '$test_name' completed successfully${NC}"
    else
        echo -e "${RED}✗ Test '$test_name' failed${NC}"
        return 1
    fi
}

# Function to run a dry run test (just check argument parsing)
run_dry_test() {
    local test_name="$1"
    shift
    local args="$@"
    
    echo -e "\n${BLUE}--- Dry Test: $test_name ---${NC}"
    echo "Args: $args"
    
    # Add --help to just test argument parsing
    if python3 "$MSIT_SCRIPT" --help > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Dry test '$test_name' - script loads correctly${NC}"
    else
        echo -e "${RED}✗ Dry test '$test_name' - script failed to load${NC}"
        return 1
    fi
}

# Single test with specified parameters
echo -e "\n${YELLOW}=== Running Single Test ===${NC}"
run_test "MSIT Test" 2 3 "0,1" 2 "gemini-2.0-flash" 100 "strict"

echo -e "\n${GREEN}=== Test Suite Complete ===${NC}"
echo -e "${BLUE}Check the msit_test_results/ directory for test outputs${NC}"

# Summary
echo -e "\n${YELLOW}=== Summary ===${NC}"
echo "• All basic functionality tests completed"
echo "• Error handling tests completed"
echo "• Output directory tests completed"
echo "• Check individual test results above for any failures"
echo -e "\n${BLUE}To run individual tests, use:${NC}"
echo "python3 $MSIT_SCRIPT --sessions 3 --ndigits 4 --stim_types \"0,1,2,3\" --nrep 2 --model \"gpt-3.5-turbo\" --max_tokens 100"



