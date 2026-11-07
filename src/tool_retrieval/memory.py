"""
Experience Memory for failure tracking and similarity-weighted ranking penalties.
"""

from typing import Dict, List, Optional, Tuple, Set
import random
import torch


class ExperienceMemory:
    """
    Experience Memory component M that retains failed (query, tool) pairs
    and applies context-sensitive ranking penalties based on semantic similarity.

    Configurations:
        - 'none': Retry-only baseline without memory penalty.
        - 'E1': Flat ungated penalty: s'(q, t) = s(q, t) - lambda.
        - 'E12': Similarity-weighted penalty: s'(q, t) = s(q, t) - lambda * pi(q, t),
                 where pi(q, t) = max cos(f(q), f(q_i)) >= tau, else 0.
        - 'E123': Similarity-weighted penalty with hard shortlist filtering (E3)
                  when high-similarity collisions or multiple failures occur.
    """

    def __init__(
        self,
        lam: float = 0.3,
        tau: float = 0.6,
        filter_threshold: float = 0.85,
        pollution_rate: float = 0.0,
        seed: Optional[int] = 42,
    ):
        self.lam = lam
        self.tau = tau
        self.filter_threshold = filter_threshold
        self.pollution_rate = pollution_rate
        self.rnd = random.Random(seed)
        # tool_id -> list of query embedding tensors (1D normalized tensors)
        self.failures: Dict[str, List[torch.Tensor]] = {}

    def clear(self) -> None:
        """Reset the memory store."""
        self.failures.clear()

    def record_failure(
        self,
        failed_tool_id: str,
        query_vec: torch.Tensor,
        correct_tool_id: Optional[str] = None,
    ) -> None:
        """
        Record a failed query-tool interaction into memory M.

        Args:
            failed_tool_id: Identifier of the tool that incorrectly attempted/failed.
            query_vec: Normalized query embedding tensor.
            correct_tool_id: Optional correct tool ID. If pollution_rate > 0,
                             the correct tool may be erroneously marked as failed
                             to simulate infrastructure/permission errors.
        """
        vec = query_vec.detach().cpu().flatten()
        self.failures.setdefault(failed_tool_id, []).append(vec)

        # Simulate memory pollution: non-semantic infrastructure/permission failure
        if correct_tool_id and self.pollution_rate > 0.0:
            if self.rnd.random() < self.pollution_rate:
                self.failures.setdefault(correct_tool_id, []).append(vec)

    def apply_penalties(
        self,
        scores: torch.Tensor,
        candidate_ids: List[str],
        query_vec: torch.Tensor,
        mode: str = "E12",
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Applies ranking penalty to candidate retrieval scores based on memory M.

        Args:
            scores: Tensor of cosine similarity scores s(q, t) for candidates.
            candidate_ids: List of candidate tool IDs matching scores.
            query_vec: Normalized query embedding tensor.
            mode: Ranking configuration ('none', 'E1', 'E12', 'E123').

        Returns:
            penalized_scores: Re-ranked score tensor s'(q, t).
            penalties: List of applied penalty factors pi(q, t).
        """
        penalized = scores.clone()
        penalties: List[float] = []
        q_cpu = query_vec.detach().cpu().flatten()

        for j, tool_id in enumerate(candidate_ids):
            p = 0.0
            if mode != "none" and tool_id in self.failures:
                # Cosine similarities against past failed queries for this tool
                stack_mem = torch.stack(self.failures[tool_id])  # (num_failures, dim)
                sims = stack_mem @ q_cpu

                if mode == "E1":
                    # Flat penalty
                    p = 1.0
                    penalized[j] -= self.lam
                else:
                    max_sim = sims.max().item()
                    if max_sim >= self.tau:
                        p = max_sim
                        penalized[j] -= self.lam * p

                    # E3 filtering: hard exclusion on repeated or high-similarity failures
                    if mode == "E123":
                        if max_sim >= self.filter_threshold or (sims >= self.tau).sum().item() >= 2:
                            penalized[j] = -999.0

            penalties.append(p)

        return penalized, penalties

    def contains(self, tool_id: str) -> bool:
        """Check if a tool has any recorded failures."""
        return tool_id in self.failures

    def failure_count(self, tool_id: str) -> int:
        """Number of failure episodes recorded for this tool."""
        return len(self.failures.get(tool_id, []))
