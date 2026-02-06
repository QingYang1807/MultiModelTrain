# 家居机器人 - Grounding + Counting + VQA

## 项目说明

本项目实现一个家居环境识别机器人，支持：
- **Grounding**: 根据自然语言描述定位图像中的物体
- **Counting**: 统计图像中特定物体的数量
- **VQA**: 针对家居图像进行视觉问答

## 环境配置

```bash
pip install torch torchvision
pip install transformers diffusers
pip install gradio
pip install opencv-python pillow
pip install groundingdino-py
```

## 数据集

使用 NYU Depth V2 数据集：
- 下载地址: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- 或使用预处理的数据集版本

## 模型架构

```
├── Grounding DINO: 开放词汇物体检测
├── BLIP-2: 视觉语言模型 (VQA)
└── Counting Module: 物体统计模块
```

## 使用方法

```python
from home_robot import HomeRobot

# 初始化机器人
robot = HomeRobot()

# Grounding: 定位物体
result = robot.ground("沙发", "data/room1.jpg")
# 返回: bounding boxes, scores

# Counting: 统计物体数量
count = robot.count("椅子", "data/room2.jpg")
# 返回: 5

# VQA: 视觉问答
answer = robot.vqa("这个房间里有几张床？", "data/room3.jpg")
# 返回: "这个房间里有2张床"
```

## 目录结构

```
home_robot/
├── config.py          # 配置文件
├── dataset.py         # 数据集加载
├── grounding.py        # Grounding DINO 模块
├── counting.py         # 计数模块
├── vqa.py             # VQA 模块
├── model.py           # 主模型集成
└── train.py           # 训练脚本
└── inference.py        # 推理脚本
└── requirements.txt    # 依赖
```
