"""
数据集加载模块 - NYU Depth V2
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path


class NYUDataset(Dataset):
    """NYU Depth V2 数据集"""

    def __init__(self, config, split="train", transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.data_path = config["data_path"]

        # 加载数据列表
        self.annotations = self._load_annotations()
        self.categories = config["categories"]

    def _load_annotations(self):
        """加载标注文件"""
        split_file = os.path.join(
            self.data_path,
            f"annotations_{self.split}.json"
        )

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return json.load(f)
        else:
            # 如果没有标注文件，创建示例数据
            return self._create_sample_data()

    def _create_sample_data(self):
        """创建示例数据用于测试"""
        samples = []
        image_dir = os.path.join(self.data_path, "images")

        if os.path.exists(image_dir):
            for img_file in os.listdir(image_dir)[:100]:
                img_path = os.path.join(image_dir, img_file)
                base_name = os.path.splitext(img_file)[0]

                # 查找对应的标签文件
                label_file = os.path.join(
                    self.data_path, "labels", f"{base_name}.txt"
                )

                boxes = []
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                category = parts[0]
                                x1, y1, x2, y2 = map(int, parts[1:5])
                                boxes.append({
                                    "category": category,
                                    "bbox": [x1, y1, x2, y2],
                                })

                samples.append({
                    "image_path": img_path,
                    "boxes": boxes,
                    "question": f"How many objects are in this scene?",
                    "answer": str(len(boxes)),
                })

        return samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # 加载图像
        image = Image.open(ann["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 返回数据
        return {
            "image": image,
            "boxes": ann["boxes"],
            "question": ann["question"],
            "answer": ann["answer"],
            "image_path": ann["image_path"],
        }


class Transform:
    """图像变换"""

    def __init__(self, size=(640, 480)):
        self.size = size

    def __call__(self, image):
        # 调整大小
        image = image.resize(self.size)
        # 转换为张量
        image = torch.tensor(np.array(image)).float() / 255.0
        # HWC -> CHW
        image = image.permute(2, 0, 1)
        return image


def create_dataloader(config, split="train", batch_size=4, num_workers=0):
    """创建数据加载器"""
    transform = Transform(size=(640, 480))

    dataset = NYUDataset(
        config=config["nyu_config"],
        split=split,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return dataloader


def create_sample_dataset():
    """创建示例数据集用于测试"""
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)

    # 创建示例标注
    annotations = {
        "train": [
            {
                "image_path": "data/images/room_001.jpg",
                "boxes": [
                    {"category": "chair", "bbox": [100, 200, 200, 400]},
                    {"category": "table", "bbox": [250, 250, 450, 400]},
                    {"category": "tv", "bbox": [500, 100, 600, 200]},
                ],
                "question": "How many chairs are in the room?",
                "answer": "1",
            },
            {
                "image_path": "data/images/room_002.jpg",
                "boxes": [
                    {"category": "bed", "bbox": [50, 100, 300, 400]},
                    {"category": "lamp", "bbox": [350, 50, 380, 150]},
                    {"category": "cabinet", "bbox": [450, 200, 600, 400]},
                ],
                "question": "What furniture is in the bedroom?",
                "answer": "bed, lamp, cabinet",
            },
        ],
        "test": [
            {
                "image_path": "data/images/room_003.jpg",
                "boxes": [
                    {"category": "sofa", "bbox": [100, 150, 400, 350]},
                    {"category": "table", "bbox": [450, 280, 580, 400]},
                    {"category": "lamp", "bbox": [50, 50, 100, 150]},
                ],
                "question": "Count all furniture items.",
                "answer": "3",
            },
        ],
    }

    # 保存标注
    for split, data in annotations.items():
        with open(f"data/annotations_{split}.json", 'w') as f:
            json.dump(data, f, indent=2)

    print("示例数据集已创建完成！")
    return annotations


if __name__ == "__main__":
    # 创建示例数据
    create_sample_dataset()

    # 测试数据加载
    from config import NYU_CONFIG

    config = {"nyu_config": NYU_CONFIG}
    train_loader = create_dataloader(config, split="train", batch_size=2)

    for batch in train_loader:
        print(f"图像形状: {batch['image'].shape}")
        print(f"问题: {batch['question']}")
        print(f"答案: {batch['answer']}")
        break
