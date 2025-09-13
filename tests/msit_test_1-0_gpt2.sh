#!/bin/bash
python scripts_test/msit_api_test.py \
    --sessions 10 \
    --ndigits 4 \
    --stim_types "2" \
    --nrep 1 \
    --model "gpt2-large" \
    --restriction "none" \
    --nickname "" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --randomized \
