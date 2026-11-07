"""
End-to-end Retrieve-Select-Execute pipeline and multi-split evaluation protocol.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from tool_retrieval.memory import ExperienceMemory
from tool_retrieval.selector import CosineSelector, GroqLLMSelector


def attempt_query(
    Q: torch.Tensor,
    T: torch.Tensor,
    tool_idx_map: Dict[str, int],
    candidate_ids: List[str],
    query_idx: int,
    memory: Optional[ExperienceMemory],
    mode: str = "E12",
    R: int = 3,
) -> Tuple[Optional[int], List[int]]:
    """
    Executes a multi-attempt episode (up to R retries) for a single query.
    Hard-excludes tools attempted during previous attempts of this query episode.

    Args:
        Q: Query embeddings tensor (N_queries, embedding_dim).
        T: Tool embeddings tensor (N_tools, embedding_dim).
        tool_idx_map: Map of tool_id to its index in T.
        candidate_ids: List of candidate tool IDs (candidate_ids[0] = ground truth tool).
        query_idx: Index of this query in Q.
        memory: ExperienceMemory instance or None.
        mode: Penalty configuration ('none', 'E1', 'E12', 'E123').
        R: Maximum allowed attempts per query episode (default 3).

    Returns:
        attempt_succeeded: 1-based attempt index on which success occurred, or None if failed.
        picks: List of chosen candidate indices across attempts.
    """
    q_vec = Q[query_idx]
    cands_tensor_idx = [tool_idx_map[c] for c in candidate_ids]
    base_scores = q_vec @ T[cands_tensor_idx].T

    tried_indices = set()
    picks = []

    for a in range(1, R + 1):
        # 1. Apply memory penalties
        if memory is not None and mode != "none":
            s2, _ = memory.apply_penalties(base_scores, candidate_ids, q_vec, mode=mode)
        else:
            s2 = base_scores.clone()

        # 2. Hard-exclude tools already attempted in this query episode
        for idx in tried_indices:
            s2[idx] = -999.0

        # 3. Deterministic Top-1 selection
        chosen_idx = int(s2.argmax().item())
        picks.append(chosen_idx)

        # Candidate 0 is always the ground truth tool
        if chosen_idx == 0:
            return a, picks

        tried_indices.add(chosen_idx)

    return None, picks


class RetrieveSelectExecuteLoop:
    """
    High-level manager for the 2-phase benchmark protocol:
    - Phase 1: Warm-up on the first query per tool to build memory M.
    - Phase 2: Evaluation on remaining test queries with memory locked.
    """

    def __init__(
        self,
        lam: float = 0.3,
        tau: float = 0.6,
        R: int = 3,
        pollution_rate: float = 0.0,
    ):
        self.lam = lam
        self.tau = tau
        self.R = R
        self.pollution_rate = pollution_rate

    def evaluate_split(
        self,
        queries: List[Dict[str, str]],
        pool_ids: List[str],
        neighbors: Dict[str, List[str]],
        Q: torch.Tensor,
        T: torch.Tensor,
        k: int = 30,
        mode: str = "E12",
        seed: int = 42,
    ) -> List[Tuple[str, int, bool, bool, bool, int]]:
        """
        Runs the 2-phase evaluation on a specific data split (e.g. test set).

        Returns:
            List of result tuples for each test query:
            (tool_id, query_idx, is_familiar_error, first_attempt_ok, success_in_R, num_attempts)
        """
        tool_idx_map = {t: i for i, t in enumerate(pool_ids)}

        # Group queries by ground-truth tool
        queries_by_tool: Dict[str, List[int]] = {}
        for i, x in enumerate(queries):
            queries_by_tool.setdefault(x["tool_id"], []).append(i)

        # Only evaluate tools that have >= 2 queries in this split (1 for warmup, >=1 for test)
        multi_queries = {t: indices for t, indices in queries_by_tool.items() if len(indices) >= 2}

        # Initialize memory
        memory = ExperienceMemory(
            lam=self.lam,
            tau=self.tau,
            pollution_rate=self.pollution_rate,
            seed=seed,
        )
        failed_on_warmup: Dict[str, bool] = {}

        # ---------------- Phase 1: Build Memory from Warm-up Query ----------------
        for tool_id, indices in multi_queries.items():
            warmup_qi = indices[0]
            cand_ids = [tool_id] + neighbors.get(tool_id, [])[:k]

            # Warmup runs stateless (mode='none')
            a, picks = attempt_query(
                Q=Q,
                T=T,
                tool_idx_map=tool_idx_map,
                candidate_ids=cand_ids,
                query_idx=warmup_qi,
                memory=None,
                mode="none",
                R=self.R,
            )

            is_failed = (picks[0] != 0)
            failed_on_warmup[tool_id] = is_failed

            # Record all failed distractor picks into memory
            for p_idx in picks:
                if p_idx != 0:
                    memory.record_failure(
                        failed_tool_id=cand_ids[p_idx],
                        query_vec=Q[warmup_qi],
                        correct_tool_id=tool_id,
                    )

        # ---------------- Phase 2: Evaluate on Remaining Queries (Memory Locked) ----------------
        results = []
        for tool_id, indices in multi_queries.items():
            cand_ids = [tool_id] + neighbors.get(tool_id, [])[:k]
            for test_qi in indices[1:]:
                a, picks = attempt_query(
                    Q=Q,
                    T=T,
                    tool_idx_map=tool_idx_map,
                    candidate_ids=cand_ids,
                    query_idx=test_qi,
                    memory=memory,
                    mode=mode,
                    R=self.R,
                )
                first_ok = (picks[0] == 0)
                success_in_r = (a is not None)
                attempts_spent = a if a is not None else self.R
                is_fam = bool(failed_on_warmup[tool_id])

                results.append((
                    tool_id,
                    test_qi,
                    is_fam,
                    first_ok,
                    success_in_r,
                    attempts_spent,
                ))

        return results


def run_val_hyperparameter_sweep(
    val_queries: List[Dict[str, str]],
    pool_ids: List[str],
    neighbors: Dict[str, List[str]],
    Q: torch.Tensor,
    T: torch.Tensor,
    k: int = 30,
    lambdas: Tuple[float, ...] = (0.1, 0.3, 0.5),
    taus: Tuple[float, ...] = (0.6, 0.7, 0.8),
) -> Tuple[float, float]:
    """
    Grid search to determine optimal (lambda, tau) hyperparameters on validation split.
    """
    best_params = (0.3, 0.6)
    best_score = (-1.0, -1.0)

    for lam in lambdas:
        for tau in taus:
            loop = RetrieveSelectExecuteLoop(lam=lam, tau=tau, R=3)
            rows = loop.evaluate_split(
                queries=val_queries,
                pool_ids=pool_ids,
                neighbors=neighbors,
                Q=Q,
                T=T,
                k=k,
                mode="E12",
            )
            fam_rows = [r for r in rows if r[2]]
            f_top1 = float(np.mean([r[3] for r in fam_rows])) if fam_rows else 0.0
            o_top1 = float(np.mean([r[3] for r in rows])) if rows else 0.0

            if (f_top1, o_top1) > best_score:
                best_score = (f_top1, o_top1)
                best_params = (lam, tau)

    return best_params
