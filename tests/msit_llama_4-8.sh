#!/bin/bash
# Run MSIT with local Hugging Face model: meta-llama/Llama-3.2-1B
# Prereqs:
#   - pip install -r llm_control/requirements_local.txt
#   - huggingface-cli login  (and get access to Meta LLaMA 3.2 models)
#   - Optionally export HF token in env: export HUGGING_FACE_HUB_TOKEN=hf_xxx

python scripts_test/msit_api_test_word.py \
    --sessions 1 \
    --ndigits 4 \
    --stim_types "8" \
    --nrep 1 \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --restriction "none"\
    --nickname "" \
    --output_dir "data/msit_pilot_outputs_word" \
    --max_tokens 100 \
    --temperature 0.0 \
    --formality "english" \
    --exhaustive    


