#!/usr/bin/env python3
"""
CLI script to fine-tune the SentenceTransformer encoder using hard negative distractors.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tool_retrieval.data import load_toolalpaca, extract_tools_and_queries, split_by_api
from tool_retrieval.distractors import mine_distractors, build_triplets_dataset
from tool_retrieval.encoder import train_contrastive_encoder
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer on ToolAlpaca")
    parser.add_argument("--data_path", type=str, default="ToolAlpaca/data/train_data.json", help="Path to ToolAlpaca train_data.json")
    parser.add_argument("--base_model", type=str, default="all-MiniLM-L6-v2", help="Pretrained model")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetuned-all-MiniLM-L6-v2", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="API split random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading ToolAlpaca data from {args.data_path}...")
    raw = load_toolalpaca(args.data_path)
    tools, data, skipped = extract_tools_and_queries(raw)
    print(f"Extracted {len(tools)} tools, {len(data)} single-tool queries ({skipped} skipped).")

    print(f"Splitting data by API (seed={args.seed})...")
    splits, pool = split_by_api(data, tools, seed=args.seed)
    print(f"Train APIs queries: {len(splits['train'])}, Tools in pool: {len(pool['train'])}")

    print("Mining nearest-neighbor distractors using base encoder...")
    base_encoder = SentenceTransformer(args.base_model)
    train_neighbors = mine_distractors(pool["train"], tools, base_encoder, k=30)

    print("Constructing contrastive triplets dataset (1 anchor, 1 positive, 3 hard negatives)...")
    train_ds = build_triplets_dataset(splits["train"], tools, train_neighbors, top_n_candidates=10, num_hard_negatives=3, seed=args.seed)

    print(f"Fine-tuning encoder for {args.epochs} epochs with MultipleNegativesRankingLoss...")
    model = train_contrastive_encoder(
        train_dataset=train_ds,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    model.save(args.output_dir)
    print(f"Model saved successfully to {args.output_dir}")


if __name__ == "__main__":
    main()
