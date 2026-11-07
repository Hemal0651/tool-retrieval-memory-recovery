#!/usr/bin/env python3
"""
CLI script to run 5-split pooled evaluation and statistical hypothesis testing (reproducing Table 1 & Table 2).
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tool_retrieval.data import load_toolalpaca, extract_tools_and_queries, split_by_api
from tool_retrieval.distractors import mine_distractors, build_triplets_dataset
from tool_retrieval.encoder import train_contrastive_encoder
from tool_retrieval.pipeline import RetrieveSelectExecuteLoop
from tool_retrieval.stats import compute_paired_mcnemar_and_ci
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-split evaluation of baseline and experience memory")
    parser.add_argument("--data_path", type=str, default="ToolAlpaca/data/train_data.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4], help="Random seeds for API splits")
    parser.add_argument("--distractors", type=int, nargs="+", default=[10, 20, 30], help="Distractor pool sizes")
    parser.add_argument("--lam", type=float, default=0.3, help="Penalty lambda")
    parser.add_argument("--tau", type=float, default=0.6, help="Similarity threshold tau")
    parser.add_argument("--retrain", action="store_true", help="Retrain encoder per seed (default: use base if false)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_toolalpaca(args.data_path)
    tools, data, _ = extract_tools_and_queries(raw)
    base_model = SentenceTransformer("all-MiniLM-L6-v2")

    all_results = {}  # (seed, k, mode) -> list of row tuples

    for seed in args.seeds:
        print(f"\n=================== PROCESSING SPLIT SEED {seed} ===================")
        splits, pool = split_by_api(data, tools, seed=seed)
        neighbors = mine_distractors(pool["test"], tools, base_model, k=max(args.distractors))

        if args.retrain:
            print(f"Fine-tuning encoder for seed {seed}...")
            train_nb = mine_distractors(pool["train"], tools, base_model, k=30)
            train_ds = build_triplets_dataset(splits["train"], tools, train_nb, seed=seed)
            encoder = train_contrastive_encoder(train_ds, num_epochs=3, output_dir=f"checkpoints/enc_seed_{seed}")
        else:
            encoder = base_model

        # Precompute embeddings
        Q = encoder.encode([x["query"] for x in splits["test"]], convert_to_tensor=True, normalize_embeddings=True)
        T = encoder.encode([tools[t] for t in pool["test"]], convert_to_tensor=True, normalize_embeddings=True)

        loop = RetrieveSelectExecuteLoop(lam=args.lam, tau=args.tau, R=3)

        for k in args.distractors:
            for mode in ["none", "E1", "E12"]:
                rows = loop.evaluate_split(
                    queries=splits["test"],
                    pool_ids=pool["test"],
                    neighbors=neighbors,
                    Q=Q,
                    T=T,
                    k=k,
                    mode=mode,
                    seed=seed,
                )
                all_results[(seed, k, mode)] = rows

                fam_rows = [r for r in rows if r[2]]
                top1 = np.mean([r[3] for r in rows]) if rows else 0.0
                fam_top1 = np.mean([r[3] for r in fam_rows]) if fam_rows else 0.0
                succ3 = np.mean([r[4] for r in rows]) if rows else 0.0
                avg_att = np.mean([r[5] for r in rows]) if rows else 0.0

                print(f"[Seed {seed} | +{k} | Mode {mode:4s}] Top-1: {top1:.3f} | Fam Top-1: {fam_top1:.3f} (n={len(fam_rows)}) | Succ<=3: {succ3:.3f} | Avg Att: {avg_att:.3f}")

    # Save raw row evaluation results
    with open(args.output_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved raw multi-split results to {args.output_file}")

    # Summary Paired Comparison (Pooled across seeds)
    print("\n" + "="*70)
    print("POOLED PAIRED STATISTICAL RESULTS (Baseline vs E1+E2)")
    print("="*70)
    for k in args.distractors:
        pooled_baseline = []
        pooled_memory = []
        for seed in args.seeds:
            pooled_baseline.extend(all_results.get((seed, k, "none"), []))
            pooled_memory.extend(all_results.get((seed, k, "E12"), []))

        if not pooled_baseline:
            continue

        # Overall Top-1
        stats_overall = compute_paired_mcnemar_and_ci(pooled_baseline, pooled_memory, col=3)
        # Familiar Top-1
        mask_fam = np.array([r[2] for r in pooled_baseline])
        stats_fam = compute_paired_mcnemar_and_ci(pooled_baseline, pooled_memory, col=3, mask=mask_fam)

        print(f"\n--- Distractor Count: +{k} ---")
        print(f"Overall Top-1:  {stats_overall['baseline_mean']:.3f} -> {stats_overall['memory_mean']:.3f} "
              f"(diff: {stats_overall['diff']:+.3f}, 95% CI [{stats_overall['ci_lower']:+.3f}, {stats_overall['ci_upper']:+.3f}], p={stats_overall['p_value']:.4f})")
        print(f"Familiar Top-1: {stats_fam['baseline_mean']:.3f} -> {stats_fam['memory_mean']:.3f} "
              f"(diff: {stats_fam['diff']:+.3f}, 95% CI [{stats_fam['ci_lower']:+.3f}, {stats_fam['ci_upper']:+.3f}], p={stats_fam['p_value']:.4f})")


if __name__ == "__main__":
    main()
