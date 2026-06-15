from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

try:
    import torchinfo
except ImportError:
    torchinfo = None

logger = logging.getLogger(__name__)


class ModelSummary:
    """Keras-style model summary using torchinfo."""

    def __init__(self, model: nn.Module, input_data: Tuple[torch.Tensor, ...]):
        """Initialize summary wrapper with a model and representative input tensors."""
        self.model = model
        self.input_data = input_data

    def summary(self, branch_label: str = "") -> str:
        """Return a Keras-style summary string, or an empty string on failure."""
        if torchinfo is None:
            logger.warning(
                "torchinfo is not installed; skipping %s model summary",
                branch_label or "model",
            )
            return ""

        try:
            result = torchinfo.summary(
                self.model,
                input_data=self.input_data,
                col_names=["input_size", "output_size", "num_params"],
                row_settings=["var_names"],
                depth=8,
                verbose=0,
            )
        except Exception as exc:  # Intentionally broad: model summary must never block training.
            logger.warning("Failed to generate %s model summary: %s", branch_label or "model", exc)
            return ""

        text = str(result)
        if branch_label:
            text = f"=== {branch_label} model summary ===\n{text}"
        return text
