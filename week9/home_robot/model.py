"""
主模型集成模块 - 家居机器人
"""

import os
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class HomeRobot:
    """家居机器人主模型"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # 初始化各模块
        self.grounding_module = None
        self.counting_module = None
        self.vqa_module = None

        # 加载配置
        grounding_config = self.config.get("grounding", {})
        vqa_config = self.config.get("vqa", {})

        self._load_modules(grounding_config, vqa_config)

    def _load_modules(self, grounding_config, vqa_config):
        """加载各功能模块"""
        try:
            from grounding import GroundingModule
            self.grounding_module = GroundingModule(grounding_config)
            print("[HomeRobot] Grounding 模块加载成功")
        except Exception as e:
            print(f"[HomeRobot] 警告: Grounding 模块加载失败 ({e})")

        try:
            from counting import CountingModule
            self.counting_module = CountingModule()
            if self.grounding_module:
                self.counting_module.set_grounding_module(self.grounding_module)
            print("[HomeRobot] Counting 模块加载成功")
        except Exception as e:
            print(f"[HomeRobot] 警告: Counting 模块加载失败 ({e})")

        try:
            from vqa import VQAModule
            self.vqa_module = VQAModule(vqa_config)
            print("[HomeRobot] VQA 模块加载成功")
        except Exception as e:
            print(f"[HomeRobot] 警告: VQA 模块加载失败 ({e})")

    def ground(
        self,
        text_prompt: str,
        image_path: str,
    ) -> List[Dict]:
        """
        定位图像中的目标物体

        Args:
            text_prompt: 文本描述，如 "chair . sofa . bed"
            image_path: 图像路径

        Returns:
            检测结果列表
        """
        if self.grounding_module is None:
            print("[HomeRobot] 警告: Grounding 模块不可用")
            return []

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 执行检测
        results = self.grounding_module.detect(image, text_prompt)

        return results

    def count(
        self,
        category: str,
        image_path: str,
    ) -> int:
        """
        统计图像中特定物体的数量

        Args:
            category: 物体类别
            image_path: 图像路径

        Returns:
            物体数量
        """
        if self.counting_module is None:
            print("[HomeRobot] 警告: Counting 模块不可用")
            return 0

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 执行计数
        count = self.counting_module.count(
            image,
            category,
            grounding_module=self.grounding_module,
        )

        return count

    def count_all(
        self,
        image_path: str,
    ) -> Dict[str, int]:
        """
        统计图像中所有物体的数量

        Args:
            image_path: 图像路径

        Returns:
            各类别数量字典
        """
        if self.counting_module is None:
            print("[HomeRobot] 警告: Counting 模块不可用")
            return {}

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 执行计数
        counts = self.counting_module.count_all(
            image,
            grounding_module=self.grounding_module,
        )

        return counts

    def vqa(
        self,
        question: str,
        image_path: str,
    ) -> str:
        """
        视觉问答

        Args:
            question: 问题
            image_path: 图像路径

        Returns:
            回答
        """
        if self.vqa_module is None:
            print("[HomeRobot] 警告: VQA 模块不可用")
            return "抱歉，VQA 功能当前不可用。"

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 执行问答
        answer = self.vqa_module.answer(image, question)

        return answer

    def vqa_with_context(
        self,
        question: str,
        image_path: str,
        context: str = None,
    ) -> Tuple[str, str]:
        """
        带上下文的视觉问答

        Args:
            question: 问题
            image_path: 图像路径
            context: 上下文信息

        Returns:
            (回答, 增强后的问题)
        """
        from vqa import VQAWithContext

        if not hasattr(self, '_contextual_vqa'):
            self._contextual_vqa = VQAWithContext(self.vqa_module)

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 执行问答
        answer, enhanced_question = self._contextual_vqa.ask(
            image, question, context=context
        )

        return answer, enhanced_question

    def analyze(
        self,
        image_path: str,
    ) -> Dict:
        """
        完整分析图像

        Args:
            image_path: 图像路径

        Returns:
            包含检测、计数、问答结果的字典
        """
        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 常见家居物体
        categories = [
            "bed", "chair", "sofa", "table", "tv",
            "lamp", "bookshelf", "cabinet",
        ]

        results = {
            "detections": [],
            "counts": {},
            "answers": [],
        }

        # 执行检测
        text_prompt = " . ".join(categories) + " ."
        detections = self.grounding_module.detect(image, text_prompt)
        results["detections"] = detections

        # 执行计数
        if self.counting_module:
            counts = self.counting_module.count_all(
                image,
                grounding_module=self.grounding_module,
            )
            results["counts"] = counts

        # 生成问答
        if self.vqa_module:
            questions = [
                "Describe this room.",
                "What is the main furniture in this room?",
                "How would you rate the brightness of this room?",
            ]
            for q in questions:
                answer = self.vqa_module.answer(image, q)
                results["answers"].append({
                    "question": q,
                    "answer": answer,
                })

        return results

    def clear_context(self):
        """清空上下文历史"""
        if hasattr(self, '_contextual_vqa'):
            self._contextual_vqa.clear_history()


def create_home_robot(
    use_grounding: bool = True,
    use_counting: bool = True,
    use_vqa: bool = True,
    grounding_config: dict = None,
    vqa_config: dict = None,
) -> HomeRobot:
    """
    创建家居机器人实例

    Args:
        use_grounding: 是否使用 Grounding
        use_counting: 是否使用 Counting
        use_vqa: 是否使用 VQA
        grounding_config: Grounding 配置
        vqa_config: VQA 配置

    Returns:
        HomeRobot 实例
    """
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if use_grounding:
        config["grounding"] = grounding_config or {
            "device": config["device"],
            "box_threshold": 0.35,
            "text_threshold": 0.25,
        }

    if use_vqa:
        config["vqa"] = vqa_config or {
            "device": config["device"],
            "model_type": "blip2",
            "model_name": "Salesforce/blip2-flan-t5-xl",
        }

    return HomeRobot(config)


if __name__ == "__main__":
    # 创建家居机器人
    robot = create_home_robot()

    # 检查各模块状态
    print("\n=== 家居机器人状态 ===")
    print(f"Grounding: {'可用' if robot.grounding_module else '不可用'}")
    print(f"Counting: {'可用' if robot.counting_module else '不可用'}")
    print(f"VQA: {'可用' if robot.vqa_module else '不可用'}")

    # 测试图像（使用占位图）
    test_image = "test_image.jpg"

    # 创建测试图像
    from PIL import Image
    test_img = Image.new("RGB", (640, 480), color=(180, 180, 180))
    test_img.save(test_image)

    print(f"\n测试图像已保存到: {test_image}")

    # 功能测试
    print("\n=== 功能测试 ===")

    # Grounding 测试
    if robot.grounding_module:
        results = robot.ground("chair . sofa . table", test_image)
        print(f"Grounding 检测到 {len(results)} 个目标")

    # Counting 测试
    if robot.counting_module:
        count = robot.count("chair", test_image)
        print(f"Counting: 检测到 {count} 把椅子")

    # VQA 测试
    if robot.vqa_module:
        answer = robot.vqa("What furniture do you see?", test_image)
        print(f"VQA: {answer}")

    print("\n测试完成！")
