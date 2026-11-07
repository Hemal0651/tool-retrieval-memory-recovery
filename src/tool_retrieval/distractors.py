"""
Hard negative distractor mining and training dataset construction.
"""

from typing import Dict, List, Optional
import random
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset


def mine_distractors(
    tool_ids: List[str],
    tools_dict: Dict[str, str],
    encoder: SentenceTransformer,
    k: int = 30,
    batch_size: int = 128,
) -> Dict[str, List[str]]:
    """
    Identifies the top-k most semantically similar distractor tools in vector space
    using cosine similarity over normalized embeddings.

    Args:
        tool_ids: List of tool IDs in this split/pool.
        tools_dict: Map of tool_id -> tool description.
        encoder: SentenceTransformer model used to embed descriptions.
        k: Number of distractors to mine per tool (default 30).
        batch_size: Batch size for embedding computation.

    Returns:
        Dict mapping each tool_id to a list of its top-k nearest distractor tool_ids.
    """
    if len(tool_ids) <= 1:
        return {tid: [] for tid in tool_ids}

    texts = [tools_dict[tid] for tid in tool_ids]
    emb = encoder.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )

    sim = emb @ emb.T
    sim.fill_diagonal_(-1.0)  # Exclude self-similarity

    actual_k = min(k, len(tool_ids) - 1)
    top_indices = sim.topk(actual_k, dim=1).indices.cpu().numpy()

    neighbors: Dict[str, List[str]] = {}
    for i, tid in enumerate(tool_ids):
        neighbors[tid] = [tool_ids[j] for j in top_indices[i]]

    return neighbors


def build_triplets_dataset(
    train_queries: List[Dict[str, str]],
    tools_dict: Dict[str, str],
    train_neighbors: Dict[str, List[str]],
    top_n_candidates: int = 10,
    num_hard_negatives: int = 3,
    seed: Optional[int] = 42,
) -> Dataset:
    """
    Constructs a Dataset for contrastive fine-tuning using MultipleNegativesRankingLoss.
    For each query, randomly samples `num_hard_negatives` from the top `top_n_candidates`
    nearest neighbor distractors mined for the correct tool.
    """
    rnd = random.Random(seed)
    rows: Dict[str, List[str]] = {
        "anchor": [],
        "positive": [],
    }
    for j in range(1, num_hard_negatives + 1):
        rows[f"neg{j}"] = []

    for item in train_queries:
        tid = item["tool_id"]
        avail_negs = train_neighbors.get(tid, [])
        candidate_pool = avail_negs[:top_n_candidates]

        if not candidate_pool:
            continue

        sample_count = min(num_hard_negatives, len(candidate_pool))
        negs = rnd.sample(candidate_pool, sample_count)
        # Pad with wrapping if fewer than num_hard_negatives
        while len(negs) < num_hard_negatives:
            negs.append(candidate_pool[len(negs) % len(candidate_pool)])

        rows["anchor"].append(item["query"])
        rows["positive"].append(tools_dict[tid])
        for j, ng in enumerate(negs, 1):
            rows[f"neg{j}"].append(tools_dict[ng])

    return Dataset.from_dict(rows)
