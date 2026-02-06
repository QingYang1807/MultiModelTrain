#!/bin/bash

# Code Alpaca LoRA 训练脚本

# 设置参数
MODEL_NAME="TinyPixel/Llama-2-7B-bf16-sharded"
DATA_PATH="data/code_alpaca_train.json"
OUTPUT_DIR="outputs/llama2_lora"

# 可选参数
USE_QLORA=${USE_QLORA:-false}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-3e-4}

echo "=== Code Alpaca LoRA 训练 ==="
echo "模型: $MODEL_NAME"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT_DIR"
echo "QLoRA: $USE_QLORA"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "学习率: $LEARNING_RATE"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行训练
python train.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --use_qlora $USE_QLORA \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE

echo "训练完成！"
