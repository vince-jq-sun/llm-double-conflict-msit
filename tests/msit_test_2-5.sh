#!/bin/bash
# --model "gpt-4.1-nano" \
# --model "gemini-1.5-flash-8b" \

python scripts_test/msit_api_test.py \
    --sessions 100 \
    --ndigits 4 \
    --stim_types "8" \
    --nrep 10 \
    --model "gpt-4.1-nano" \
    --restriction "strict" \
    --nickname "SmFkExNDigi4" \
    --output_dir "data/msit_pilot_outputs_smallnrep"
