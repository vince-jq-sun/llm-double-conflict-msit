#!/bin/bash
# --model "gpt-4.1-nano" \
# --model "gemini-1.5-flash-8b" \

python scripts_test/msit_api_test.py \
    --sessions 100 \
    --ndigits 3 \
    --stim_types "0,1,2,3" \
    --nrep 4 \
    --model "gemini-2.0-flash" \
    --restriction "strict" \
    --nickname "all-alter-rm" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --temperature 0.0   \
    --randomized
