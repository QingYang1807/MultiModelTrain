"""
家居机器人配置文件
"""

import os

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# NYU Depth V2 数据集配置
NYU_CONFIG = {
    "data_path": os.path.join(DATA_DIR, "nyu_depth_v2"),
    "min_depth": 0.5,
    "max_depth": 10.0,
    "categories": [
        "bed", "chair", "sofa", "table", "tv",
        "lamp", "bookshelf", "cabinet", "mirror",
        "picture", "curtain", "pillow", "rug"
    ],
    "train_split": 0.8,
}

# Grounding DINO 配置
GROUNDING_DINO_CONFIG = {
    "model_type": "groundingdino",
    "device": "cuda",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
    "iou_threshold": 0.5,
}

# BLIP-2 配置
BLIP2_CONFIG = {
    "model_name": "Salesforce/blip2-flan-t5-xl",
    "device": "cuda",
    "max_length": 128,
    "num_beams": 5,
}

# LLaVA-NeXT 配置 (备选)
LLAVA_CONFIG = {
    "model_name": "llava-hf/llava-next-7b",
    "device": "cuda",
    "conv_type": "llava_next",
    "temperature": 0.2,
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 4,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "eval_steps": 1000,
    "save_steps": 2000,
    "seed": 42,
}

# 日志配置
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": os.path.join(OUTPUT_DIR, "train.log"),
    "wandb_project": "home_robot",
    "wandb_name": None,
}
