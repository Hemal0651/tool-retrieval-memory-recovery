#!/usr/bin/env python3
"""
CLI script to evaluate LLM Selector (openai/gpt-oss-20b via Groq, reproducing Table 3).
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tool_retrieval.data import load_toolalpaca, extract_tools_and_queries, split_by_api
from tool_retrieval.distractors import mine_distractors
from tool_retrieval.memory import ExperienceMemory
from tool_retrieval.selector import GroqLLMSelector
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM Selector with Groq API")
    parser.add_argument("--data_path", type=str, default="ToolAlpaca/data/train_data.json")
    parser.add_argument("--api_key", type=str, default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractors", type=int, default=30)
    parser.add_argument("--shortlist_size", type=int, default=5)
    parser.add_argument("--cache_file", type=str, default="llm_cache.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        print("Warning: GROQ_API_KEY environment variable not set. Please provide --api_key or set GROQ_API_KEY.")

    cache = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} cached responses from {args.cache_file}")

    raw = load_toolalpaca(args.data_path)
    tools, data, _ = extract_tools_and_queries(raw)
    splits, pool = split_by_api(data, tools, seed=args.seed)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    neighbors = mine_distractors(pool["test"], tools, encoder, k=args.distractors)

    Q = encoder.encode([x["query"] for x in splits["test"]], convert_to_tensor=True, normalize_embeddings=True)
    T = encoder.encode([tools[t] for t in pool["test"]], convert_to_tensor=True, normalize_embeddings=True)
    tool_idx = {t: i for i, t in enumerate(pool["test"])}

    selector = GroqLLMSelector(api_key=args.api_key, model_name=args.model, cache=cache)

    print(f"\nEvaluating LLM Selector ({args.model}) on seed {args.seed} with +{args.distractors} distractors...")
    print(f"Candidates shortlisted to top-{args.shortlist_size} based on penalized cosine ranking.")

    # Save updated cache on exit
    if cache:
        with open(args.cache_file, "wb") as f:
            pickle.dump(cache, f)


if __name__ == "__main__":
    main()
