import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from transformers import TrainingArguments, Trainer

from data import MnistDataset, data_collator
from model import VIT, DiT
from utils import DrawResults


@dataclass
class DitConfig:
    dataset_path: str = "/home/wsj/Desktop/data/dataset_origin/mnist"
    output_dir: str = "outputs/dit"

    img_size: int = 28
    patch_size: int = 4
    num_classes: int = 10
    img_channels: int = 1
    hidden_dim: int = 256
    num_heads: int = 16
    num_layers: int = 12

    batch_size: int = 50
    max_steps: int = 5000
    learning_rate: float = 1e-3
    save_steps = 1000


@dataclass
class DriftConfig:
    dataset_path: str = "/home/wsj/Desktop/data/dataset_origin/mnist"
    output_dir: str = "outputs/drift"

    img_size: int = 28
    patch_size: int = 4
    num_classes: int = 10
    img_channels: int = 1
    hidden_dim: int = 256
    num_heads: int = 16
    num_layers: int = 6

    batch_size: int = 50
    max_steps: int = 5000
    learning_rate: float = 1e-3
    save_steps = 1000

def run_train(config: dataclass):
    seed = 64
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    dataset = MnistDataset(data_dir=config.dataset_path)
    model = DiT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        img_channels=config.img_channels,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    train_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=1e-5,
        warmup_ratio=0.05,
        max_grad_norm=2.0,
        logging_steps=100,
        save_steps=config.save_steps,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        tf32=False,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=1,
        report_to="tensorboard",
        seed=64,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        eval_strategy="no",
        eval_steps=None,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        ignore_data_skip=True,
    )

    callback = DrawResults(
        model=model,
        input_channels=config.img_channels,
        img_size=config.img_size,
        draw_step=config.save_steps,
        plot_dir=os.path.join(config.output_dir, 'plots'),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_classes=config.num_classes,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[callback],
    )
    trainer.train()


if __name__ == "__main__":
    cfg = DitConfig()
    run_train(cfg)