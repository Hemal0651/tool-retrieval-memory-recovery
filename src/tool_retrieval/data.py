"""
Dataset loading, parsing, and API-level splitting for ToolAlpaca benchmark.
"""

from typing import Dict, List, Tuple, Any, Optional
import json
import random
import os


def load_toolalpaca(file_path: str) -> List[Dict[str, Any]]:
    """
    Load raw ToolAlpaca train_data.json file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ToolAlpaca data file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tools_and_queries(
    raw_data: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], List[Dict[str, str]], int]:
    """
    Extracts individual tools and single-tool queries from ToolAlpaca data.

    Returns:
        tools: Dict mapping tool_id ('APIName::function_name') to tool description.
        data: List of single-tool query instances [{'query', 'tool_id', 'api'}].
        skipped_count: Number of multi-tool or malformed instances skipped.
    """
    tools: Dict[str, str] = {}

    # 1. Tool pool: every function from every API = one candidate tool
    for api in raw_data:
        api_name = api.get("Name", "")
        function_desc = api.get("Function_Description", {})
        for fname, desc in function_desc.items():
            short = desc.split("\nParameters:")[0].strip()
            tool_id = f"{api_name}::{fname}"
            tools[tool_id] = f"{fname}: {short}"

    # 2. Queries with exactly ONE ground truth tool
    data: List[Dict[str, str]] = []
    skipped_count = 0

    for api in raw_data:
        api_name = api.get("Name", "")
        for inst in api.get("Instances", []):
            try:
                intermediate_steps = inst.get("intermediate_steps", [])
                used = {step[0][0] for step in intermediate_steps if step and step[0]}
            except Exception:
                used = set()

            # Paper Protocol: Skip multi-tool or empty queries
            if len(used) != 1:
                skipped_count += 1
                continue

            tool_name = used.pop()
            tool_id = f"{api_name}::{tool_name}"
            if tool_id in tools:
                data.append({
                    "query": inst.get("input", ""),
                    "tool_id": tool_id,
                    "api": api_name,
                })
            else:
                skipped_count += 1

    return tools, data, skipped_count


def split_by_api(
    data: List[Dict[str, str]],
    tools: Dict[str, str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[str]]]:
    """
    Randomly splits the dataset at the API level (zero API leakage between splits).

    Args:
        data: Extracted single-tool queries.
        tools: All available tools.
        seed: Random seed for reproducibility.
        train_ratio: Proportion of APIs in train set (default 0.70).
        val_ratio: Proportion of APIs in validation set (default 0.15).
        test_ratio: Proportion of APIs in test set (default 0.15).

    Returns:
        splits: Dict {'train': queries, 'val': queries, 'test': queries}
        pool: Dict {'train': tool_ids, 'val': tool_ids, 'test': tool_ids}
    """
    rnd = random.Random(seed)
    apis = sorted({x["api"] for x in data})
    rnd.shuffle(apis)

    n = len(apis)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    split_of: Dict[str, str] = {}
    for i, a in enumerate(apis):
        if i < train_end:
            split_of[a] = "train"
        elif i < val_end:
            split_of[a] = "val"
        else:
            split_of[a] = "test"

    splits = {
        s: [x for x in data if split_of.get(x["api"]) == s]
        for s in ("train", "val", "test")
    }
    pool = {
        s: [t for t in tools if split_of.get(t.split("::")[0]) == s]
        for s in ("train", "val", "test")
    }

    return splits, pool
