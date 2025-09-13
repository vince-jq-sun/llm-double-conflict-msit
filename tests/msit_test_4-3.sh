#!/bin/bash
# --model "gpt-4.1-nano" \
# --model "gemini-1.5-flash-8b" \

python scripts_test/msit_api_test.py \
    --sessions 40 \
    --ndigits 4 \
    --stim_types "3" \
    --nrep 6 \
    --model "gemini-1.5-flash-8b" \
    --restriction "strict" \
    --nickname "" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --temperature 0.0 \