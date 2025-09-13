#!/bin/bash
# --model "gpt-4.1-nano" \
# --model "gemini-1.5-flash-8b" \

python scripts_test/msit_api_test.py \
    --sessions 20 \
    --ndigits 4 \
    --stim_types "3" \
    --nrep 3 \
    --model "gemini-1.5-flash-8b" \
    --restriction "strict" \
    --nickname "withLtrPrmpt" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --temperature 0.0 \
    --seed "session"