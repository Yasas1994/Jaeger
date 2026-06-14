"""PyTorch datasets backed by preprocessed NumPy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyFullDataset(Dataset):
    """Loads a fully-preprocessed ``.npz`` file.

    The archive is expected to contain at least an input array (e.g.
    ``translated``) and a label array (``label``). Each sample is returned
    together with a boolean mask indicating non-padding positions.
    """

    def __init__(
        self,
        path: str | Path,
        input_key: str = "translated",
        label_key: str = "label",
    ):
        data = np.load(path, allow_pickle=False)
        self.inputs = torch.from_numpy(data[input_key])
        self.labels = torch.from_numpy(data[label_key])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]
        if x.dim() == 2:
            # (frames, length)
            mask = x != 0
        elif x.dim() == 3:
            # (frames, length, channels) -> mask over frames and length
            mask = (x != 0).any(dim=-1)
        else:
            raise ValueError(
                f"NumpyFullDataset expects 2-D or 3-D inputs, got rank {x.dim()}"
            )
        return x, self.labels[idx], mask
