"""
Contrastive encoder fine-tuning with MultipleNegativesRankingLoss (InfoNCE).
"""

from typing import Optional
import os
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers


def train_contrastive_encoder(
    train_dataset: Dataset,
    base_model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "enc_out",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    fp16: Optional[bool] = None,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """
    Fine-tunes a SentenceTransformer model using MultipleNegativesRankingLoss
    with mined hard negatives as described in Section 3.4 of Farouqe et al. (2026).

    Args:
        train_dataset: HuggingFace Dataset containing 'anchor', 'positive', 'neg1', etc.
        base_model_name: Pretrained SentenceTransformer checkpoint (default: all-MiniLM-L6-v2).
        output_dir: Output directory for checkpoints.
        num_epochs: Training epochs (default: 3).
        batch_size: Per-device batch size (default: 32).
        learning_rate: AdamW learning rate (default: 2e-5).
        warmup_ratio: Proportion of steps for linear warmup (default: 0.1).
        fp16: Whether to use mixed precision (defaults to True if CUDA available).
        device: Device to place the model on ('cuda' or 'cpu').

    Returns:
        Fine-tuned SentenceTransformer model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    model = SentenceTransformer(base_model_name, device=device)
    loss = losses.MultipleNegativesRankingLoss(model)

    os.makedirs(output_dir, exist_ok=True)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    return model
