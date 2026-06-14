"""PyTorch datasets backed by preprocessed NumPy arrays."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from jaeger.data.pytorch.transforms import (
    apply_mutation,
    shuffle_frames as shuffle_frames_fn,
    translate_to_codons,
)


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


class NumpyRawDataset(Dataset):
    """Loads raw int8 sequences and applies runtime preprocessing."""

    def __init__(
        self,
        path: str | Path,
        seq_key: str = "sequences",
        label_key: str = "labels",
        crop_size: int = 500,
        num_classes: int = 3,
        codon_table: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        mutate: bool = False,
        mutation_rate: float = 0.1,
        shuffle_frames: bool = False,
    ):
        data = np.load(path, allow_pickle=False)
        self.seqs = data[seq_key]
        raw_labels = torch.from_numpy(data[label_key])
        self.labels = self._normalize_labels(raw_labels, num_classes)
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.codon_table = codon_table
        self.shuffle = shuffle
        self.mutate = mutate
        self.mutation_rate = mutation_rate
        self.shuffle_frames = shuffle_frames

    @staticmethod
    def _normalize_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert class-index labels to one-hot; leave one-hot labels unchanged."""
        if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
            indices = labels.view(-1).long()
            return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()
        return labels.float()

    def __len__(self) -> int:
        return len(self.labels)

    def _crop_pad_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Crop or pad a raw nucleotide sequence to ``crop_size``."""
        seq = np.asarray(seq)
        length = len(seq)
        if length > self.crop_size:
            start = random.randint(0, length - self.crop_size)
            return seq[start : start + self.crop_size]
        if length < self.crop_size:
            pad = self.crop_size - length
            return np.pad(seq, (0, pad), constant_values=4)
        return seq

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.seqs[idx]
        if self.shuffle:
            seq = seq[np.random.permutation(len(seq))]
        seq = self._crop_pad_sequence(seq)
        if self.mutate:
            seq = apply_mutation(seq, self.mutation_rate)
        x = translate_to_codons(seq, self.codon_table)
        mask = x != 0
        if self.shuffle_frames:
            x = shuffle_frames_fn(x)
        return x, self.labels[idx], mask
