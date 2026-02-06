#!/bin/bash

# Code Alpaca 推理脚本

# 设置参数
BASE_MODEL="TinyPixel/Llama-2-7B-bf16-sharded"
MODEL_PATH=${1:-"outputs/llama2_lora"}
INPUT_FILE="data/code_alpaca_eval.json"
OUTPUT_FILE="outputs/predictions.json"

echo "=== Code Alpaca 推理 ==="
echo "基础模型: $BASE_MODEL"
echo "微调模型: $MODEL_PATH"
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"

# 对比微调前后效果
python inference.py \
    --base_model $BASE_MODEL \
    --model_path $MODEL_PATH \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --compare_base \
    --num_samples 10

echo "推理完成！"
