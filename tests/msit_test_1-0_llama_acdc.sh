#!/bin/bash
# LLaMA ACDC-ready MSIT run (harder setting, no examples)
# - Higher difficulty to avoid ceiling performance and give ACDC signal
# - Small probe run to gauge accuracy range first
# Prereqs:
#   - pip install -r llm_control/requirements_local.txt
#   - huggingface-cli login (access to Meta LLaMA 3.2 models)
#   - export HUGGING_FACE_HUB_TOKEN=hf_xxx (optional)

python scripts_test/msit_api_test.py \
    --sessions 20 \
    --ndigits 4 \
    --stim_types "0,1,2,3" \
    --nrep 1 \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --restriction "strict" \
    --nickname "llama-3.2-1b-acdc-probe" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --max_tokens 60 \
    --temperature 0.0 \
    --randomized \
    --enable_acdc \
    --acdc_tau 1e-3 \
    --acdc_k_edges 80 \
    --acdc_seq_len 160 \
    --acdc_batch_size 8 \
    --acdc_train_samples 80 \
    --acdc_test_samples 20 \
    --acdc_ablation_type tokenwise_mean_corrupt
