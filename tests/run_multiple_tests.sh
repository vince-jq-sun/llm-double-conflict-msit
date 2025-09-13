#!/bin/bash

# Meta script to run multiple test scripts sequentially
# Usage: ./run_multiple_tests.sh test1.sh test2.sh test3.sh ...
# Example: ./run_multiple_tests.sh msit_test_2-1.sh msit_test_2-2.sh msit_test_2-3.sh

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    print_error "No test scripts specified!"
    echo "Usage: $0 <test_script1.sh> <test_script2.sh> ..."
    echo "Example: $0 msit_test_2-1.sh msit_test_2-2.sh msit_test_2-3.sh"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Initialize counters
total_tests=$#
passed_tests=0
failed_tests=0
failed_test_names=()

print_status "Starting execution of $total_tests test scripts..."
print_status "Project root: $PROJECT_ROOT"
echo ""

# Loop through all provided test script names
for test_script in "$@"; do
    # Construct full path to test script
    test_path="$SCRIPT_DIR/$test_script"
    
    # Check if test script exists
    if [ ! -f "$test_path" ]; then
        print_error "Test script not found: $test_path"
        ((failed_tests++))
        failed_test_names+=("$test_script (not found)")
        continue
    fi
    
    # Check if test script is executable
    if [ ! -x "$test_path" ]; then
        print_warning "Making $test_script executable..."
        chmod +x "$test_path"
    fi
    
    print_status "Running: $test_script"
    echo "----------------------------------------"
    
    # Record start time
    start_time=$(date +%s)
    
    # Execute the test script
    if bash "$test_path"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "$test_script completed successfully (${duration}s)"
        ((passed_tests++))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_error "$test_script failed (${duration}s)"
        ((failed_tests++))
        failed_test_names+=("$test_script")
    fi
    
    echo ""
done

# Print summary
echo "========================================"
print_status "Test execution summary:"
echo "Total tests: $total_tests"
print_success "Passed: $passed_tests"

if [ $failed_tests -gt 0 ]; then
    print_error "Failed: $failed_tests"
    echo "Failed tests:"
    for failed_test in "${failed_test_names[@]}"; do
        echo "  - $failed_test"
    done
    echo ""
    exit 1
else
    print_success "All tests passed!"
    exit 0
fi
