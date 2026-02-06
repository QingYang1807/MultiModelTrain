# Coder 模型 - LoRA/QLoRA 微调

## 项目说明

本项目使用 Code Alpaca 数据集，通过 LoRA/QLoRA 技术微调小型语言模型，实现代码生成能力。

## 环境配置

```bash
pip install torch transformers
pip install datasets accelerate
pip install peft bitsandbytes
pip install loralib
pip install wandb
```

## 数据集

使用 Code Alpaca 数据集：
- 仓库: https://github.com/sahil280114/codealpaca
- 数据格式: instruction, input, output

## 模型

支持以下模型进行 LoRA 微调：
- LLaMA-2 7B/13B
- CodeLlama 7B/13B
- StarCoder 7B
- Qwen 1.5 7B

## 使用方法

### 1. 准备数据
```bash
# 下载 Code Alpaca 数据
python data/prepare_data.py
```

### 2. 训练
```bash
# 单 GPU 训练
python train.py --model_name meta-llama/Llama-2-7b-hf \
    --data_path data/code_alpaca_train.json \
    --output_dir outputs/llama2_lora

# QLoRA 训练（4-bit 量化）
python train.py --model_name meta-llama/Llama-2-7b-hf \
    --use_qlora \
    --data_path data/code_alpaca_train.json \
    --output_dir outputs/llama2_qlora
```

### 3. 推理对比
```bash
# 对比微调前后效果
python inference.py --model_path outputs/llama2_lora \
    --compare_base
```

## 项目结构

```
coder_model/
├── data/
│   ├── prepare_data.py     # 数据准备
│   └── code_alpaca_*.json  # 数据集
├── models/
│   ├── lora_model.py       # LoRA 模型
│   └── qlora_model.py      # QLoRA 模型
├── scripts/
│   ├── train.sh           # 训练脚本
│   └── inference.sh        # 推理脚本
├── train.py               # 训练主脚本
├── inference.py           # 推理脚本
├── config.py              # 配置文件
├── requirements.txt       # 依赖
└── README.md
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --lora_r | 16 | LoRA 秩 |
| --lora_alpha | 32 | LoRA 缩放因子 |
| --lora_dropout | 0.05 | LoRA Dropout |
| --learning_rate | 3e-4 | 学习率 |
| --batch_size | 4 | 批次大小 |
| --num_epochs | 3 | 训练轮数 |
| --max_seq_len | 512 | 最大序列长度 |

## LoRA vs QLoRA 对比

| 特性 | LoRA | QLoRA |
|------|------|-------|
| 显存占用 | 较高 | 低（4-bit 量化） |
| 训练速度 | 快 | 稍慢 |
| 模型精度 | 原始精度 | 略有损失 |
| 最低显存 | ~16GB | ~8GB |

## 参考

- [Code Alpaca](https://github.com/sahil280114/codealpaca)
- [PEFT](https://github.com/huggingface/peft)
- [QLoRA](https://github.com/artidoro/qlora)
