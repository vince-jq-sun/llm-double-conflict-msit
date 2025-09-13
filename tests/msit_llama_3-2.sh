#!/bin/bash
# Run MSIT with Ollama model: llama3.1:8b-instruct-q4_0
# Prereqs:
#   - ollama installed and running
#   - ollama pull llama3.1:8b-instruct-q4_0

python scripts_test/msit_api_test_word.py \
    --sessions 100 \
    --ndigits 4 \
    --stim_types "2" \
    --nrep 1 \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --restriction "none"\
    --nickname "" \
    --output_dir "data/msit_pilot_outputs_word" \
    --max_tokens 100 \
    --temperature 1.0 \
    --formality "english"