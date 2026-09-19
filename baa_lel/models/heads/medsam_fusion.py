"""Shared projection layers for the MedSAM standard-fusion baselines."""
from __future__ import annotations

import torch.nn as nn


def build_medsam_fusion_projections(clinical_dim: int, hidden_dim: int = 256):
    """Return the image and clinical projections used by MedSAM fusion.

    Keeping this factory shared prevents the single-task and multi-task
    baselines from silently drifting in dimensions, normalization, or
    activation while preserving their existing state-dict key layout.
    """
    return (
        nn.Linear(256, hidden_dim),
        nn.Sequential(
            nn.Linear(int(clinical_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ),
    )
