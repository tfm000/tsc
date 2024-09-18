"""
Contains code implementing the series-relation-aware (SRA) deciders described
in the Bi-Mamba4TS and Bi-Mamba+ papers, in addition to further experiments.
"""

import torch


def sra_decider_pearson(x: torch.Tensor) -> int:
    pass


def sra_decider_spearman(x: torch.Tensor) -> int:
    pass


def sra_decider_plus(x: torch.Tensor) -> tuple[tuple[int], tuple[int]]:
    pass
