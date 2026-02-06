"""
训练脚本 - 家居机器人
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HomeRobotTrainer:
    """家居机器人训练器"""

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
        config: dict = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}

        # 训练参数
        self.batch_size = self.config.get("batch_size", 4)
        self.num_epochs = self.config.get("num_epochs", 10)
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.warmup_steps = self.config.get("warmup_steps", 500)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.gradient_accumulation_steps = self.config.get(
            "gradient_accumulation_steps", 4
        )
        self.eval_steps = self.config.get("eval_steps", 1000)
        self.save_steps = self.config.get("save_steps", 2000)
        self.seed = self.config.get("seed", 42)

        # 设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.lr_scheduler = None

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 设置随机种子
        self._set_seed()

        # 输出目录
        self.output_dir = Path(self.config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _create_optimizer(self):
        """创建优化器"""
        # 分离不同模块的参数
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )

    def train(self):
        """开始训练"""
        logger.info("开始训练...")

        total_steps = 0
        global_step = 0
        epochs_trained = 0

        # 训练循环
        for epoch in range(epochs_trained, self.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")

            # 训练一个 epoch
            train_loss = self._train_epoch(epoch)

            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # 评估
            if self.eval_dataloader:
                eval_results = self._evaluate()
                logger.info(f"Eval Results: {eval_results}")

            # 保存检查点
            self._save_checkpoint(epoch)

        logger.info("训练完成！")

    def _train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}"
        )

        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.get("loss", torch.tensor(0.0))

            # 梯度累积
            loss = loss / self.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            # 优化器步骤
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
            })

            global_step += 1

        return total_loss / num_batches

    @torch.no_grad()
    def _evaluate(self) -> dict:
        """评估模型"""
        if not self.eval_dataloader:
            return {}

        self.model.eval()
        total_eval_loss = 0.0
        num_eval_batches = 0

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating"
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.get("loss", torch.tensor(0.0))

            total_eval_loss += loss.item()
            num_eval_batches += 1

        self.model.train()

        return {
            "eval_loss": total_eval_loss / num_eval_batches,
        }

    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"checkpoint-epoch-{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        logger.info(f"检查点已保存到: {checkpoint_path}")


def main():
    """主函数"""
    from config import NYU_CONFIG, TRAIN_CONFIG
    from dataset import create_dataloader, create_sample_dataset
    from model import create_home_robot

    # 创建示例数据集
    logger.info("创建示例数据集...")
    create_sample_dataset()

    # 创建数据加载器
    config = {"nyu_config": NYU_CONFIG}

    train_loader = create_dataloader(
        config,
        split="train",
        batch_size=TRAIN_CONFIG["batch_size"],
    )

    eval_loader = create_dataloader(
        config,
        split="test",
        batch_size=TRAIN_CONFIG["batch_size"],
    )

    # 创建模型
    logger.info("创建模型...")
    model = create_home_robot(
        use_grounding=True,
        use_counting=True,
        use_vqa=True,
    )

    # 创建训练器
    trainer = HomeRobotTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=TRAIN_CONFIG,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
