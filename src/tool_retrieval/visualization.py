"""
Visualization module to generate paper plots (Figure 2: results.png).
"""

from typing import Optional
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_paper_figure(
    df_eval: pd.DataFrame,
    df_pollution: Optional[pd.DataFrame] = None,
    output_path: str = "figures/results.png",
) -> None:
    """
    Generates Figure 2 from Farouqe et al. (2026):
    - Left panel: Familiar-error Top-1 across +10, +20, and +30 distractors.
    - Middle panel: Overall Top-1 across distractors.
    - Right panel: Overall and familiar-error Top-1 under 0%–60% memory pollution.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    colors = {
        "none": "gray",
        "E1": "tab:red",
        "E12": "tab:green",
        "E123": "tab:blue",
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    d0 = df_eval[df_eval["pollution"] == 0] if "pollution" in df_eval.columns else df_eval

    # Panels 0 & 1: Performance across distractors
    for (e, m), g in d0.groupby(["encoder", "mode"]):
        if m == "E123":
            continue  # Identical to E12 in 1-query warmup setting
        ls = "-" if e == "finetuned" else "--"
        lab = "E1+E2 (=E1+E2+E3)" if m == "E12" else m
        color = colors.get(m, "black")

        ax[0].plot(
            g["distractors"],
            g["fam_top1"],
            ls,
            marker="o",
            c=color,
            label=f"{e}/{lab}",
        )
        ax[1].plot(
            g["distractors"],
            g["top1"],
            ls,
            marker="o",
            c=color,
        )

    ax[0].set_title("Familiar-error Top-1", fontsize=11, fontweight="bold")
    ax[1].set_title("Overall Top-1", fontsize=11, fontweight="bold")

    for a in ax[:2]:
        a.set_xlabel("# distractors", fontsize=10)
        a.set_xticks([10, 20, 30])
        a.grid(True, linestyle="--", alpha=0.5)

    ax[0].legend(fontsize=8, loc="upper left")

    # Panel 2: Memory Pollution curve
    dp = df_pollution if df_pollution is not None else df_eval[df_eval.get("distractors", 30) == 30]
    if "pollution" in dp.columns and len(dp["pollution"].unique()) > 1:
        ax[2].plot(
            dp["pollution"] * 100,
            dp["top1"],
            marker="o",
            color="tab:blue",
            label="Overall Top-1",
        )
        ax[2].plot(
            dp["pollution"] * 100,
            dp["fam_top1"],
            marker="s",
            color="tab:orange",
            label="Familiar Top-1",
        )
        # Horizontal reference line for no-memory baseline at +30
        ax[2].axhline(
            y=0.651,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="No-memory overall",
        )
        ax[2].axhline(
            y=0.410,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="No-memory familiar",
        )
        ax[2].set_title("Memory pollution (+30)", fontsize=11, fontweight="bold")
        ax[2].set_xlabel("% corrupted memory", fontsize=10)
        ax[2].set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax[2].grid(True, linestyle="--", alpha=0.5)
        ax[2].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")
