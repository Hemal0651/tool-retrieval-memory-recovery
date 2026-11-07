"""
Statistical evaluation: Paired exact McNemar tests and tool-cluster bootstrap 95% CIs.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def cluster_boot(
    keys: np.ndarray,
    diff: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Computes a 95% confidence interval on the mean paired difference,
    resampling tool clusters (queries targeting the same ground truth tool are correlated).

    Args:
        keys: 1D array of tool cluster keys (e.g. tool_id strings).
        diff: 1D array of per-query differences (metric_b - metric_a).
        n_boot: Number of bootstrap iterations (default 10,000).
        seed: Random seed for RNG.

    Returns:
        (ci_lower, ci_upper): 2.5th and 97.5th percentiles of bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    _, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnt = np.bincount(inv)
    num_clusters = len(sums)

    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, num_clusters, num_clusters)
        out[b] = sums[pick].sum() / cnt[pick].sum()

    ci_lower, ci_upper = np.percentile(out, [2.5, 97.5])
    return float(ci_lower), float(ci_upper)


def compute_paired_mcnemar_and_ci(
    rows_baseline: List[Tuple],
    rows_memory: List[Tuple],
    col: int = 3,
    mask: Optional[np.ndarray] = None,
    n_boot: int = 10000,
) -> Dict[str, Any]:
    """
    Computes exact paired McNemar test and clustered bootstrap 95% CI.

    Row tuple layout:
      0: tool_id
      1: query_idx
      2: is_familiar_error (bool)
      3: first_attempt_ok (bool)
      4: success_within_R (bool)
      5: num_attempts (int)

    Args:
        rows_baseline: Result rows from baseline (e.g. retry-only).
        rows_memory: Result rows from memory configuration (e.g. E12).
        col: Index of metric to compare (default 3: first_attempt_ok).
        mask: Optional boolean filter array (e.g. familiar error subset).
        n_boot: Bootstrap resamples.

    Returns:
        Dict with metrics: baseline_mean, memory_mean, diff, ci_lower, ci_upper,
        helped_count, hurt_count, p_value.
    """
    assert len(rows_baseline) == len(rows_memory), "Rows must be paired 1:1"

    a = np.array([r[col] for r in rows_baseline], dtype=float)
    b = np.array([r[col] for r in rows_memory], dtype=float)
    keys = np.array([r[0] for r in rows_baseline])

    if mask is not None:
        a = a[mask]
        b = b[mask]
        keys = keys[mask]

    n = len(a)
    if n == 0:
        return {
            "n": 0,
            "baseline_mean": 0.0,
            "memory_mean": 0.0,
            "diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "helped": 0,
            "hurt": 0,
            "p_value": 1.0,
        }

    helped = int(((a == 0) & (b == 1)).sum())
    hurt = int(((a == 1) & (b == 0)).sum())
    both = int(((a == 1) & (b == 1)).sum())
    neither = int(((a == 0) & (b == 0)).sum())

    diff_arr = b - a
    ci_lower, ci_upper = cluster_boot(keys, diff_arr, n_boot=n_boot)

    # Compute exact McNemar p-value
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[both, hurt], [helped, neither]], exact=True)
        p_val = float(res.pvalue)
    except ImportError:
        # Fallback binomial test
        from scipy.stats import binomtest
        total_discordant = helped + hurt
        if total_discordant > 0:
            p_val = float(binomtest(min(helped, hurt), total_discordant, 0.5).pvalue)
        else:
            p_val = 1.0

    return {
        "n": n,
        "baseline_mean": float(a.mean()),
        "memory_mean": float(b.mean()),
        "diff": float(b.mean() - a.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "helped": helped,
        "hurt": hurt,
        "p_value": p_val,
    }
