"""
LoRA 模型定义
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from typing import Optional, Dict, Any


class LoRAModel:
    """LoRA 微调模型"""

    def __init__(
        self,
        model_name: str,
        config: dict = None,
        use_qlora: bool = False,
    ):
        self.model_name = model_name
        self.config = config or {}
        self.use_qlora = use_qlora

        self.tokenizer = None
        self.model = None
        self.lora_config = None

        self._load_model()

    def _load_model(self):
        """加载模型和分词器"""
        print(f"加载模型: {self.model_name}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置量化（如果是 QLoRA）
        if self.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.get("load_in_4bit", True),
                bnb_4bit_quant_type=self.config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=self.config.get(
                    "bnb_4bit_compute_dtype", "float16"
                ),
                bnb_4bit_use_double_quant=self.config.get(
                    "bnb_4bit_use_double_quant", True
                ),
            )
        else:
            quantization_config = None

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.use_qlora else None,
            torch_dtype=(
                torch.float16 if self.use_qlora
                else torch.bfloat16
            ),
            trust_remote_code=True,
        )

        # 如果不是量化模型，移动到设备
        if not self.use_qlora:
            self.model = self.model.to("cuda")

        # 配置 LoRA
        self._setup_lora()

        # 打印可训练参数
        self._print_trainable_params()

    def _setup_lora(self):
        """配置 LoRA"""
        lora_config_dict = self.config.get("lora", {})

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("lora_alpha", 32),
            lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
            bias=lora_config_dict.get("bias", "none"),
            target_modules=lora_config_dict.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        )

        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.lora_config = lora_config

        print(f"LoRA 配置已应用: r={lora_config.r}, alpha={lora_config.lora_alpha}")

    def _print_trainable_params(self):
        """打印可训练参数"""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(f"\n可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"总参数: {all_params:,}")

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """获取可训练参数"""
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def save_pretrained(self, output_dir: str):
        """
        保存模型

        Args:
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存 LoRA 权重
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"模型已保存到: {output_dir}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        生成代码

        Args:
            prompt: 提示
            max_new_tokens: 最大生成长度
            temperature: 温度
            do_sample: 是否采样

        Returns:
            生成的代码
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # 提取生成部分
        if prompt in generated_text:
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()

        return response

    def merge_and_unload(self):
        """合并 LoRA 权重并卸载"""
        self.model = self.model.merge_and_unload()
        print("LoRA 权重已合并")


class CoderModel:
    """代码生成模型包装器"""

    def __init__(
        self,
        model_path: str = None,
        base_model: str = None,
        config: dict = None,
    ):
        """
        初始化代码生成模型

        Args:
            model_path: 微调后的模型路径
            base_model: 基础模型名称
            config: 配置
        """
        self.config = config or {}
        self.lora_model = None

        if model_path and os.path.exists(model_path):
            # 加载微调后的模型
            self.lora_model = LoRAModel(
                model_name=model_path,
                config=self.config,
            )
            self.is_finetuned = True
        elif base_model:
            # 加载基础模型（用于对比）
            self.lora_model = LoRAModel(
                model_name=base_model,
                config=self.config,
            )
            self.is_finetuned = False
        else:
            raise ValueError("必须指定 model_path 或 base_model")

    def generate_code(self, instruction: str, input_text: str = "") -> str:
        """
        生成代码

        Args:
            instruction: 指令
            input_text: 输入

        Returns:
            生成的代码
        """
        # 构建提示
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""

        # 生成
        response = self.lora_model.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.2,
        )

        return response

    def compare_generation(
        self,
        instruction: str,
        input_text: str = "",
    ) -> Dict[str, str]:
        """
        对比生成（如果模型是基础模型）

        Args:
            instruction: 指令
            input_text: 输入

        Returns:
            生成结果
        """
        if self.is_finetuned:
            return {
                "finetuned": self.generate_code(instruction, input_text),
            }

        # 如果是基础模型，直接返回生成结果
        return {
            "base": self.generate_code(instruction, input_text),
        }


# 导入 os
import os


if __name__ == "__main__":
    # 测试 LoRA 模型
    from config import LORA_CONFIG, QLORA_CONFIG

    config = {
        "lora": LORA_CONFIG,
        **QLORA_CONFIG,
    }

    # 使用轻量级模型测试
    model = LoRAModel(
        model_name="TinyPixel/Llama-2-7B-bf16-sharded",
        config=config,
        use_qlora=False,  # 设置为 True 以使用 QLoRA
    )

    # 测试生成
    test_instruction = "Write a Python function to calculate the sum of a list."
    response = model.generate(test_instruction)
    print(f"\n测试生成:\n{response}")
