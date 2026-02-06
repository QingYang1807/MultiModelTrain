"""
训练脚本 - Code Alpaca LoRA 微调
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from accelerate import Accelerator
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """代码数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

        # 格式化模板
        self.template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建完整文本
        text = self.template.format(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
            output=item.get("output", ""),
        )

        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        # 复制为 labels
        encoding["labels"] = encoding["input_ids"].copy()

        return encoding


class CoderTrainer:
    """代码生成模型训练器"""

    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str,
        config: dict = None,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.config = config or {}

        # 训练参数
        self.batch_size = self.config.get("batch_size", 4)
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.num_epochs = self.config.get("num_epochs", 3)
        self.max_seq_len = self.config.get("max_seq_len", 512)
        self.warmup_steps = self.config.get("warmup_steps", 100)
        self.gradient_accumulation_steps = self.config.get(
            "gradient_accumulation_steps", 4
        )
        self.eval_steps = self.config.get("eval_steps", 500)
        self.save_steps = self.config.get("save_steps", 500)
        self.seed = self.config.get("seed", 42)
        self.use_qlora = self.config.get("use_qlora", False)

        # 初始化加速器
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        # 设置随机种子
        self._set_seed()

        # 初始化
        self.tokenizer = None
        self.model = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None

        self._initialize()

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _initialize(self):
        """初始化所有组件"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载分词器
        logger.info("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载数据集
        logger.info("加载数据集...")
        self._load_datasets()

        # 加载模型
        logger.info("加载模型...")
        self._load_model()

        # 打印参数数量
        self._print_params()

    def _load_datasets(self):
        """加载数据集"""
        # 划分训练和验证集
        with open(self.data_path, 'r') as f:
            full_data = json.load(f)

        split_idx = int(len(full_data) * 0.9)
        train_data = full_data[:split_idx]
        eval_data = full_data[split_idx:]

        # 临时保存
        train_path = os.path.join(self.output_dir, "train_temp.json")
        eval_path = os.path.join(self.output_dir, "eval_temp.json")

        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f)

        # 创建数据集
        self.train_dataset = CodeDataset(
            train_path, self.tokenizer, self.max_seq_len
        )
        self.eval_dataset = CodeDataset(
            eval_path, self.tokenizer, self.max_seq_len
        )

        # 数据收集器
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model if hasattr(self, 'model') else None,
            padding=True,
        )

    def _load_model(self):
        """加载模型并配置 LoRA"""
        # 量化配置
        if self.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        # 新版本peft不需要prepare_model_for_int8_training
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        self.model = model

    def _print_params(self):
        """打印参数数量"""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        logger.info(f"可训练参数: {trainable_params:,} ({100*trainable_params/all_params:.2f}%)")
        logger.info(f"总参数: {all_params:,}")

    def train(self):
        """开始训练"""
        logger.info("开始训练...")

        # 准备优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # 准备数据
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        # 使用加速器
        model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )

        # 训练循环
        global_step = 0
        total_loss = 0

        model.train()

        progress_bar = tqdm(
            train_dataloader,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")

            for step, batch in enumerate(progress_bar):
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss

                # 反向传播
                self.accelerator.backward(loss)

                # 优化器步骤
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (step + 1):.4f}",
                })

                # 保存检查点
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step)

        logger.info("训练完成！")

        # 保存最终模型
        self._save_final_model()

    def _save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = os.path.join(
            self.output_dir, f"checkpoint-{step}"
        )
        self.accelerator.save_state(checkpoint_dir)
        logger.info(f"检查点已保存: {checkpoint_dir}")

    def _save_final_model(self):
        """保存最终模型"""
        # 获取基础模型和 LoRA 权重
        base_model = self.accelerator.unwrap_model(self.model)
        base_model.save_pretrained(
            self.output_dir,
            save_embedding_layers=True,
        )
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"模型已保存到: {self.output_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Code Alpaca LoRA 训练")
    parser.add_argument(
        "--model_name", type=str,
        default="TinyPixel/Llama-2-7B-bf16-sharded",
        help="模型名称"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/code_alpaca_train.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/llama2_lora",
        help="输出目录"
    )
    parser.add_argument(
        "--use_qlora", action="store_true",
        help="使用 QLoRA"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="批次大小"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="学习率"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512,
        help="最大序列长度"
    )

    args = parser.parse_args()

    # 配置
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "use_qlora": args.use_qlora,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 4,
        "save_steps": 500,
        "eval_steps": 500,
        "seed": 42,
    }

    # 创建训练器
    trainer = CoderTrainer(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config=config,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
