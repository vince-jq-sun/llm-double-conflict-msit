#!/bin/bash

# Convenience script to run test series 2 (msit_test_2-1, 2-2, 2-3)
# Usage: ./run_test_series_2.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the meta script with predefined test files
"$SCRIPT_DIR/run_multiple_tests.sh" msit_gpt_1-0.sh msit_gpt_1-1.sh msit_gpt_1-2.sh msit_gpt_1-3.sh msit_gpt_1-8.sh msit_gpt_1-10.sh
