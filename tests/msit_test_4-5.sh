#!/bin/bash
# --model "gpt-4.1-nano" \
# --model "gemini-1.5-flash-8b" \

python scripts_test/msit_api_test.py \
    --sessions 40 \
    --ndigits 3 \
    --stim_types "4" \
    --nrep 6 \
    --model "gemini-1.5-flash-8b" \
    --restriction "strict" \
    --nickname "CgLtr" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --temperature 0.0 \