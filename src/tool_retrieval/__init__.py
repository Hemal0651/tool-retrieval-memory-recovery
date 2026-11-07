"""
Tool Retrieval under Distractor APIs with Memory-Based Recovery.

Official Python package implementing the retrieve-select-execute framework,
experience memory with similarity-weighted penalties, hard negative distractor mining,
and statistical evaluation protocols described in Farouqe et al. (2026).
"""

from tool_retrieval.data import (
    load_toolalpaca,
    extract_tools_and_queries,
    split_by_api,
)
from tool_retrieval.distractors import (
    mine_distractors,
    build_triplets_dataset,
)
from tool_retrieval.encoder import (
    train_contrastive_encoder,
)
from tool_retrieval.memory import (
    ExperienceMemory,
)
from tool_retrieval.selector import (
    CosineSelector,
    GroqLLMSelector,
)
from tool_retrieval.pipeline import (
    RetrieveSelectExecuteLoop,
)
from tool_retrieval.stats import (
    cluster_boot,
    compute_paired_mcnemar_and_ci,
)

__version__ = "1.0.0"
__authors__ = ["Hemal Farouqe", "Sheikh Mainul Islam", "Tanjida Sultana Maria"]

__all__ = [
    "load_toolalpaca",
    "extract_tools_and_queries",
    "split_by_api",
    "mine_distractors",
    "build_triplets_dataset",
    "train_contrastive_encoder",
    "ExperienceMemory",
    "CosineSelector",
    "GroqLLMSelector",
    "RetrieveSelectExecuteLoop",
    "cluster_boot",
    "compute_paired_mcnemar_and_ci",
]
