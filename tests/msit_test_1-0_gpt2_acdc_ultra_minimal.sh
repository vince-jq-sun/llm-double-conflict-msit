#!/bin/bash
# Ultra-minimal ACDC test - last 16 tokens only, maximum search space reduction
# Focus on final reasoning steps where MSIT circuits likely operate

python scripts_test/msit_api_test.py \
    --sessions 6 \
    --ndigits 3 \
    --stim_types "2" \
    --nrep 1 \
    --model "gpt2" \
    --restriction "strict" \
    --nickname "gpt2-acdc-ultra" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --max_tokens 15 \
    --temperature 0.0 \
    --randomized \
    --enable_acdc \
    --acdc_tau 0.1 \
    --acdc_k_edges 3 \
    --acdc_seq_len 16 \
    --acdc_batch_size 2 \
    --acdc_train_samples 4 \
    --acdc_test_samples 2 \
    --acdc_ablation_type tokenwise_mean_corrupt
