#!/bin/bash

# MSIT Auto Test Script Configuration
# This script runs map_msit_all_auto.py with a pre-configured model

# Set the Python path for llm-local environment
PYTHON_PATH="/opt/homebrew/Caskroom/miniforge/base/envs/llm-local/bin/python"

# Base script path
SCRIPT_PATH="scripts_test/map_msit_all_auto.py"

# --- 在这里直接设置模型和参数 ---
# 你只需要修改下面这两行来更换模型
MODEL="meta-llama/Llama-3.2-3B-Instruct"
NICKNAME="try"
# PREVIEW_FLAG="--preview_only" # 如果需要预览模式，取消这一行的注释

# --- 通用参数 (根据需要修改) ---
NDIGITS=6
MAX_TOKENS=50
TEMPERATURE=0.0
RESTRICTION="strict"
FORMALITY="english"
OUTPUT_DIR="data/msit_pilot_outputs_mapall"

# --- 脚本执行逻辑 (无需修改) ---

echo "=== MSIT Auto Test Script Runner ==="
echo "Model set to: $MODEL"

# Build command
CMD="$PYTHON_PATH $SCRIPT_PATH --model \"$MODEL\" --max_tokens $MAX_TOKENS --ndigits $NDIGITS --temperature $TEMPERATURE --restriction $RESTRICTION --formality $FORMALITY --output_dir $OUTPUT_DIR"

if [ ! -z "$NICKNAME" ]; then
    CMD="$CMD --nickname $NICKNAME"
fi

# 如果设置了预览模式，则添加到命令中
if [ ! -z "$PREVIEW_FLAG" ]; then
    CMD="$CMD $PREVIEW_FLAG"
fi


echo ""
echo "Command to execute:"
echo "$CMD"
echo ""

echo "Starting execution..."
eval $CMD

echo "Execution finished."