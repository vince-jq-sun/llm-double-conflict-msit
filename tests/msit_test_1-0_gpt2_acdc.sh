#!/bin/bash
# MSIT Test with ACDC Circuit Discovery for GPT2
# This script runs the standard MSIT test and then performs ACDC circuit discovery

python scripts_test/msit_api_test.py \
    --sessions 10 \
    --ndigits 3 \
    --stim_types "0,1,2,3" \
    --nrep 10 \
    --model "gpt2-large" \
    --restriction "strict" \
    --nickname "gpt2-acdc" \
    --output_dir "data/msit_pilot_outputs_acdc" \
    --randomized \
    --enable_acdc \
    --acdc_tau 1e-3 \
    --acdc_k_edges 50 \
    --acdc_faithfulness_target "kl_div" \
    --acdc_seq_len 64 \
    --acdc_batch_size 4 \
    --acdc_train_samples 50 \
    --acdc_test_samples 20 \
    --acdc_save_viz \
    --acdc_head_ablation
