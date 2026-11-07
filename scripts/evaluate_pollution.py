#!/usr/bin/env python3
"""
CLI script to evaluate vulnerability to memory pollution (simulating noise/permission errors, reproducing Table 4).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tool_retrieval.data import load_toolalpaca, extract_tools_and_queries, split_by_api
from tool_retrieval.distractors import mine_distractors
from tool_retrieval.pipeline import RetrieveSelectExecuteLoop
from tool_retrieval.visualization import plot_paper_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate memory corruption / pollution impact")
    parser.add_argument("--data_path", type=str, default="ToolAlpaca/data/train_data.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractors", type=int, default=30)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6], help="Corruption proportions")
    parser.add_argument("--output_csv", type=str, default="pollution_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_toolalpaca(args.data_path)
    tools, data, _ = extract_tools_and_queries(raw)
    splits, pool = split_by_api(data, tools, seed=args.seed)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    neighbors = mine_distractors(pool["test"], tools, encoder, k=args.distractors)

    Q = encoder.encode([x["query"] for x in splits["test"]], convert_to_tensor=True, normalize_embeddings=True)
    T = encoder.encode([tools[t] for t in pool["test"]], convert_to_tensor=True, normalize_embeddings=True)

    rows_summary = []
    print(f"Running Memory Pollution Evaluation on Split {args.seed} with +{args.distractors} Distractors...")

    for rate in args.rates:
        loop = RetrieveSelectExecuteLoop(lam=0.3, tau=0.6, R=3, pollution_rate=rate)
        eval_rows = loop.evaluate_split(
            queries=splits["test"],
            pool_ids=pool["test"],
            neighbors=neighbors,
            Q=Q,
            T=T,
            k=args.distractors,
            mode="E12",
            seed=args.seed,
        )

        fam_rows = [r for r in eval_rows if r[2]]
        top1 = float(np.mean([r[3] for r in eval_rows])) if eval_rows else 0.0
        fam_top1 = float(np.mean([r[3] for r in fam_rows])) if fam_rows else 0.0
        avg_att = float(np.mean([r[5] for r in eval_rows])) if eval_rows else 0.0

        print(f"Corruption: {rate*100:4.1f}% | Overall Top-1: {top1:.3f} | Familiar Top-1: {fam_top1:.3f} | Avg Attempts: {avg_att:.3f}")
        rows_summary.append({
            "pollution": rate,
            "distractors": args.distractors,
            "top1": top1,
            "fam_top1": fam_top1,
            "avg_attempts": avg_att,
            "n": len(eval_rows),
            "n_fam": len(fam_rows),
        })

    df = pd.DataFrame(rows_summary)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
