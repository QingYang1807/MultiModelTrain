"""
Grounding DINO 模块 - 开放词汇物体检测
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import warnings

# 抑制警告
warnings.filterwarnings("ignore")


class GroundingModule:
    """Grounding DINO 开放词汇检测模块"""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")
        self.box_threshold = config.get("box_threshold", 0.35)
        self.text_threshold = config.get("text_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.5)

        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 Grounding DINO 模型"""
        try:
            from groundingdino.models import build_model
            from groundingdino.util import box_ops, get_tokenlizer
            from groundingdino.util.inference import Model

            # 尝试加载预训练模型
            checkpoint_path = self.config.get("checkpoint", None)

            if checkpoint_path and os.path.exists(checkpoint_path):
                self.model = Model.build_model(checkground=checkpoint_path)
                self.tokenizer = get_tokenlizer.get_tokenlizer(self.model.config_file)
            else:
                # 使用 HuggingFace 版本的 Grounding DINO
                from transformers import GroundingDinoForSegmentation

                self.model = GroundingDinoForSegmentation.from_pretrained(
                    "IDEA-Research/grounding-dino-base"
                )
                self.model.to(self.device)
                self.model.eval()

            print(f"[GroundingModule] 模型加载成功，设备: {self.device}")

        except ImportError as e:
            print(f"[GroundingModule] 警告: 无法加载 Grounding DINO ({e})")
            print("[GroundingModule] 将使用模拟模式进行演示")
            self.model = None

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> List[Dict]:
        """
        检测图像中的目标

        Args:
            image: PIL 图像
            text_prompt: 文本提示，如 "chair . sofa . bed"

        Returns:
            检测结果列表，每个包含 bbox, score, category
        """
        if self.model is None:
            # 模拟模式
            return self._simulate_detection(image, text_prompt)

        # 预处理图像
        image_tensor = self._preprocess_image(image)

        # 编码文本
        text_inputs = self._encode_text(text_prompt)

        # 执行检测
        outputs = self.model(image_tensor, text_inputs)

        # 后处理
        results = self._postprocess_outputs(outputs, image.size)

        return results

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        image = image.resize((640, 480))
        # 转换为数组
        image_array = np.array(image) / 255.0
        # 转换为张量 (B, C, H, W)
        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        # 使用模型的分词器
        if hasattr(self, 'tokenizer'):
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            return inputs["input_ids"].to(self.device)
        else:
            # 简单编码
            tokens = text.lower().split()
            return torch.tensor([tokens], dtype=torch.long).to(self.device)

    def _postprocess_outputs(
        self,
        outputs: Tuple,
        image_size: Tuple[int, int],
    ) -> List[Dict]:
        """后处理检测输出"""
        # 提取边界框和分数
        logits, boxes = outputs

        # 阈值过滤
        scores = torch.sigmoid(logits)
        valid_mask = scores > self.box_threshold

        results = []
        for i in range(len(valid_mask)):
            for j in range(len(valid_mask[i])):
                if valid_mask[i][j]:
                    box = boxes[i][j].cpu().numpy()
                    score = scores[i][j].item()

                    # 转换坐标
                    cx, cy, w, h = box
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    # 限制在图像范围内
                    H, W = image_size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)

                    results.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                    })

        # NMS
        results = self._nms(results)

        return results

    def _nms(self, boxes: List[Dict], iou_threshold: float = None) -> List[Dict]:
        """非极大值抑制"""
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        if len(boxes) == 0:
            return []

        # 按分数排序
        boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)

        keep = []
        while len(boxes) > 0:
            best = boxes.pop(0)
            keep.append(best)

            remaining = []
            for box in boxes:
                iou = self._calculate_iou(best["bbox"], box["bbox"])
                if iou < iou_threshold:
                    remaining.append(box)
            boxes = remaining

        return keep

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算 IoU"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _simulate_detection(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> List[Dict]:
        """
        模拟检测结果（当模型不可用时）
        用于测试和演示
        """
        import random

        # 解析文本提示
        categories = [c.strip() for c in text_prompt.split(".")]

        results = []
        H, W = image.size

        for category in categories:
            if category and random.random() > 0.3:
                # 随机生成边界框
                x1 = random.randint(0, W // 2)
                y1 = random.randint(0, H // 2)
                x2 = x1 + random.randint(W // 4, W // 2)
                y2 = y1 + random.randint(H // 4, H // 2)

                results.append({
                    "category": category,
                    "bbox": [x1, y1, min(x2, W), min(y2, H)],
                    "score": round(random.uniform(0.5, 0.95), 2),
                })

        return results


class GroundingVisualizer:
    """检测结果可视化"""

    def __init__(self, colors=None):
        self.colors = colors or {
            "bed": (255, 0, 0),
            "chair": (0, 255, 0),
            "sofa": (0, 0, 255),
            "table": (255, 255, 0),
            "tv": (255, 0, 255),
            "lamp": (0, 255, 255),
            "default": (128, 128, 128),
        }

    def draw(
        self,
        image: Image.Image,
        detections: List[Dict],
        show_scores: bool = True,
    ) -> Image.Image:
        """绘制检测结果"""
        # 转换为 OpenCV 格式
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        for det in detections:
            bbox = det["bbox"]
            category = det.get("category", "object")
            score = det.get("score", 1.0)

            # 获取颜色
            color = self.colors.get(category, self.colors["default"])

            # 绘制边界框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{category}: {score:.2f}" if show_scores else category
            cv2.putText(
                image_cv,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # 转回 PIL 格式
        result_image = Image.fromarray(
            cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        )

        return result_image


# 导入 os 用于路径检查
import os


if __name__ == "__main__":
    # 测试 Grounding 模块
    from config import GROUNDING_DINO_CONFIG

    module = GroundingModule(GROUNDING_DINO_CONFIG)

    # 创建测试图像
    test_image = Image.new("RGB", (640, 480), color=(200, 200, 200))

    # 测试检测
    detections = module.detect(
        test_image,
        "chair . sofa . table . bed"
    )

    print(f"检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"  - {det}")

    # 可视化
    visualizer = GroundingVisualizer()
    result_image = visualizer.draw(test_image, detections)
    result_image.save("grounding_result.jpg")
    print("结果已保存到 grounding_result.jpg")
