"""
Coder 模型配置文件
"""

import os

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# 数据集配置
CODE_ALPACA_CONFIG = {
    "data_url": "https://raw.githubusercontent.com/sahil280114/codealpaca/main/data/code_alpaca_20k.json",
    "train_split": 0.9,
    "max_samples": None,  # None 表示使用全部数据
}

# 模型配置
MODEL_CONFIG = {
    # 支持的模型列表
    "supported_models": [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-13b-hf",
        "bigcode/starcoder",
        "Qwen/Qwen1.5-7B",
    ],
    "default_model": "TinyPixel/Llama-2-7B-bf16-sharded",  # 轻量级替代
}

# LoRA 配置
LORA_CONFIG = {
    "r": 16,  # LoRA 秩
    "lora_alpha": 32,  # LoRA 缩放因子
    "lora_dropout": 0.05,  # Dropout 比例
    "bias": "none",  # bias 类型
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # 目标模块
    "task_type": "CAUSAL_LM",  # 任务类型
}

# QLoRA 配置
QLORA_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
}

# 训练配置
TRAIN_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_len": 512,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,
    "num_epochs": 3,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "seed": 42,
    "fp16": False,  # 混合精度训练
    "bf16": True,   # BF16 训练（如果支持）
}

# 推理配置
INFERENCE_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.95,
    "do_sample": True,
    "num_return_sequences": 1,
}

# 数据预处理配置
DATA_CONFIG = {
    "max_source_length": 256,
    "max_target_length": 256,
    "max_examples": 20000,
    "template": {
        "prompt": "Below is an instruction that describes a task. "
                  "Write a response that appropriately completes the request.\n\n"
                  "### Instruction:\n{instruction}\n\n"
                  "### Input:\n{input}\n\n"
                  "### Response:",
        "response": "{output}",
    },
}

# 评估配置
EVAL_CONFIG = {
    "metrics": ["accuracy", "perplexity", "code_bleu"],
    "test_samples": 100,
    "compare_before_after": True,
}
