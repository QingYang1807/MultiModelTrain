"""
计数模块 - 统计图像中特定物体的数量
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from collections import Counter


class CountingModule:
    """物体计数模块"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.category_counters = {}
        self.grounding_module = None

    def set_grounding_module(self, grounding_module):
        """设置 Grounding 模块"""
        self.grounding_module = grounding_module

    def count(
        self,
        image: Image.Image,
        category: str,
        grounding_module=None,
    ) -> int:
        """
        统计图像中特定类别的物体数量

        Args:
            image: PIL 图像
            category: 物体类别，如 "chair", "bed", "table"
            grounding_module: Grounding 检测模块

        Returns:
            物体数量
        """
        # 使用提供的模块或内置模块
        gm = grounding_module or self.grounding_module

        if gm is None:
            # 使用模拟计数
            return self._simulate_count(image, category)

        # 进行检测
        text_prompt = f"{category} ."
        detections = gm.detect(image, text_prompt)

        # 统计检测到的物体
        count = len(detections)

        # 更新计数记录
        self._update_counter(category, count)

        return count

    def count_multiple(
        self,
        image: Image.Image,
        categories: List[str],
        grounding_module=None,
    ) -> Dict[str, int]:
        """
        统计多种物体数量

        Args:
            image: PIL 图像
            categories: 物体类别列表
            grounding_module: Grounding 检测模块

        Returns:
            各类别的数量字典
        """
        gm = grounding_module or self.grounding_module

        results = {}
        for category in categories:
            results[category] = self.count(
                image, category, grounding_module=gm
            )

        return results

    def count_all(
        self,
        image: Image.Image,
        grounding_module=None,
        min_score: float = 0.5,
    ) -> Dict[str, int]:
        """
        统计所有检测到的物体数量

        Args:
            image: PIL 图像
            grounding_module: Grounding 检测模块
            min_score: 最小置信度阈值

        Returns:
            各类别的数量字典
        """
        gm = grounding_module or self.grounding_module

        if gm is None:
            return {"object": 1}

        # 检测所有常见家居物体
        all_categories = [
            "bed", "chair", "sofa", "table", "tv",
            "lamp", "bookshelf", "cabinet", "mirror",
            "picture", "curtain", "pillow", "rug"
        ]

        # 构建文本提示
        text_prompt = " . ".join(all_categories) + " ."

        # 执行检测
        detections = gm.detect(image, text_prompt)

        # 统计各类别数量
        category_counts = Counter()
        for det in detections:
            if det.get("score", 0) >= min_score:
                category = det.get("category", "unknown")
                category_counts[category] += 1

        return dict(category_counts)

    def _update_counter(self, category: str, count: int):
        """更新计数记录"""
        if category not in self.category_counters:
            self.category_counters[category] = []
        self.category_counters[category].append(count)

    def _simulate_count(self, image: Image.Image, category: str) -> int:
        """模拟计数结果"""
        import random

        # 根据类别返回模拟计数
        category_counts = {
            "bed": 1,
            "chair": random.randint(2, 6),
            "sofa": 1,
            "table": 1,
            "tv": 1,
            "lamp": random.randint(1, 3),
            "cabinet": random.randint(1, 2),
            "bookshelf": random.randint(0, 2),
        }

        return category_counts.get(category, random.randint(0, 3))

    def get_statistics(self) -> Dict:
        """获取计数统计信息"""
        stats = {}
        for category, counts in self.category_counters.items():
            if counts:
                stats[category] = {
                    "total": sum(counts),
                    "mean": sum(counts) / len(counts),
                    "max": max(counts),
                    "min": min(counts),
                }
        return stats


class CountingVisualizer:
    """计数结果可视化"""

    def __init__(self):
        pass

    def draw(
        self,
        image: Image.Image,
        counts: Dict[str, int],
    ) -> Image.Image:
        """绘制计数结果"""
        import cv2
        import numpy as np

        # 转换为 OpenCV 格式
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # 在图像上绘制计数信息
        y_offset = 30
        for category, count in counts.items():
            text = f"{category}: {count}"
            cv2.putText(
                image_cv,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += 30

        # 转回 PIL 格式
        result_image = Image.fromarray(
            cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        )

        return result_image


if __name__ == "__main__":
    # 测试计数模块
    from grounding import GroundingModule
    from config import GROUNDING_DINO_CONFIG

    # 初始化模块
    grounding_module = GroundingModule(GROUNDING_DINO_CONFIG)
    counting_module = CountingModule()
    counting_module.set_grounding_module(grounding_module)

    # 创建测试图像
    test_image = Image.new("RGB", (640, 480), color=(200, 200, 200))

    # 测试计数
    chair_count = counting_module.count(test_image, "chair")
    print(f"检测到 {chair_count} 把椅子")

    # 统计多种物体
    results = counting_module.count_multiple(
        test_image,
        ["chair", "table", "sofa"]
    )
    print(f"多种物体计数: {results}")

    # 统计所有物体
    all_counts = counting_module.count_all(test_image)
    print(f"所有物体计数: {all_counts}")

    # 获取统计信息
    stats = counting_module.get_statistics()
    print(f"统计信息: {stats}")
