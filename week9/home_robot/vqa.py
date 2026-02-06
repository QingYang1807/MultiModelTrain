"""
VQA 模块 - 视觉问答
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class VQAModule:
    """视觉问答模块 - 使用 BLIP-2 或 LLaVA"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        self.model = None
        self.processor = None
        self.model_type = self.config.get("model_type", "blip2")

        self._load_model()

    def _load_model(self):
        """加载 VQA 模型"""
        try:
            if self.model_type == "blip2":
                self._load_blip2()
            elif self.model_type == "llava":
                self._load_llava()
            else:
                self._load_blip2()

        except Exception as e:
            print(f"[VQAModule] 警告: 无法加载 {self.model_type} 模型 ({e})")
            print("[VQAModule] 将使用模拟模式进行演示")
            self.model = None

    def _load_blip2(self):
        """加载 BLIP-2 模型"""
        try:
            from transformers import (
                Blip2ForConditionalGeneration,
                Blip2Processor,
            )

            model_name = self.config.get(
                "model_name",
                "Salesforce/blip2-flan-t5-xl"
            )

            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

            print(f"[VQAModule] BLIP-2 模型加载成功")

        except ImportError as e:
            print(f"[VQAModule] BLIP-2 依赖缺失: {e}")
            raise

    def _load_llava(self):
        """加载 LLaVA 模型"""
        try:
            from transformers import LlavaForConditionalGeneration
            from llava.model import LlavaConfig

            model_name = self.config.get(
                "model_name",
                "llava-hf/llava-next-7b"
            )

            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

            print(f"[VQAModule] LLaVA 模型加载成功")

        except ImportError as e:
            print(f"[VQAModule] LLaVA 依赖缺失: {e}")
            raise

    @torch.no_grad()
    def answer(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 128,
    ) -> str:
        """
        回答关于图像的问题

        Args:
            image: PIL 图像
            question: 问题
            max_length: 最大生成长度

        Returns:
            回答文本
        """
        if self.model is None:
            return self._simulate_answer(image, question)

        # 预处理
        if self.model_type == "blip2":
            return self._answer_blip2(image, question, max_length)
        elif self.model_type == "llava":
            return self._answer_llava(image, question, max_length)
        else:
            return self._answer_blip2(image, question, max_length)

    def _answer_blip2(
        self,
        image: Image.Image,
        question: str,
        max_length: int,
    ) -> str:
        """使用 BLIP-2 回答问题"""
        # 预处理图像和文本
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成回答
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=5,
            do_sample=False,
        )

        # 解码回答
        answer = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer

    def _answer_llava(
        self,
        image: Image.Image,
        question: str,
        max_length: int,
    ) -> str:
        """使用 LLaVA 回答问题"""
        # 构建对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # 预处理
        inputs = self.processor(
            conversations=conversation,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成回答
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
        )

        # 解码回答
        answer = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer

    def _simulate_answer(
        self,
        image: Image.Image,
        question: str,
    ) -> str:
        """模拟回答（当模型不可用时）"""
        import random

        # 基于问题类型生成模拟回答
        question_lower = question.lower()

        if "how many" in question_lower or "数量" in question:
            # 计数问题
            return random.choice([
                "There are 3 items in this scene.",
                "I can see 2 objects matching your description.",
                "The image contains one main item.",
            ])

        elif "what" in question_lower or "什么" in question:
            # 描述问题
            return random.choice([
                "This appears to be a living room scene with furniture.",
                "I can see various household items in the image.",
                "The scene contains typical home furniture.",
            ])

        elif "where" in question_lower or "哪里" in question:
            # 位置问题
            return random.choice([
                "The item is located in the center of the image.",
                "I can see it on the left side of the room.",
                "It's positioned near the window area.",
            ])

        else:
            return random.choice([
                "Yes, I can see that in the image.",
                "The answer is in the lower part of the image.",
                "I believe this is a standard household scene.",
            ])

    def batch_answer(
        self,
        image: Image.Image,
        questions: List[str],
    ) -> List[str]:
        """批量回答多个问题"""
        return [self.answer(image, q) for q in questions]

    def generate_caption(self, image: Image.Image) -> str:
        """生成图像描述"""
        if self.model is None:
            return self._simulate_answer(
                image,
                "Describe this image in detail."
            )

        # 使用 BLIP-2 生成分描述
        return self._answer_blip2(
            image,
            "Generate a detailed caption for this image.",
            max_length=256,
        )


class VQAWithContext:
    """带上下文的 VQA 模块"""

    def __init__(self, vqa_module: VQAModule = None):
        self.vqa_module = vqa_module or VQAModule()
        self.context_history = []
        self.max_history = 5

    def ask(
        self,
        image: Image.Image,
        question: str,
        context: str = None,
    ) -> Tuple[str, str]:
        """
        提问（带上下文）

        Args:
            image: 图像
            question: 问题
            context: 额外上下文信息

        Returns:
            (回答, 增强后的问题)
        """
        # 构建增强问题
        enhanced_question = question

        if context:
            enhanced_question = f"Context: {context}\nQuestion: {question}"

        # 添加历史信息
        if self.context_history:
            history = "\n".join(self.context_history[-3:])
            enhanced_question = f"{history}\n{enhanced_question}"

        # 限制历史长度
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]

        # 获取回答
        answer = self.vqa_module.answer(image, enhanced_question)

        # 更新历史
        self.context_history.append(f"Q: {question}\nA: {answer}")

        return answer, enhanced_question

    def clear_history(self):
        """清空历史"""
        self.context_history = []


if __name__ == "__main__":
    from config import BLIP2_CONFIG

    # 初始化 VQA 模块
    vqa = VQAModule(BLIP2_CONFIG)

    # 创建测试图像
    test_image = Image.new("RGB", (640, 480), color=(200, 200, 200))

    # 测试问答
    questions = [
        "What furniture do you see in this room?",
        "How many chairs are there?",
        "What color is the sofa?",
    ]

    print("=== VQA 测试 ===")
    for q in questions:
        answer = vqa.answer(test_image, q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    # 测试带上下文的问答
    contextual_vqa = VQAWithContext(vqa)
    answer, _ = contextual_vqa.ask(
        test_image,
        "How many items are there?",
        context="This is a living room with furniture."
    )
    print(f"带上下文问答: {answer}")

    # 测试图像描述
    caption = vqa.generate_caption(test_image)
    print(f"图像描述: {caption}")
