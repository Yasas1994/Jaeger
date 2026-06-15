"""PyTorch dataset backed by raw CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from jaeger.dataops.pytorch.transforms import dna_to_indices, translate_to_codons


class CSVDataset(Dataset):
    """Reads label,sequence CSV files and applies runtime preprocessing."""

    def __init__(
        self,
        path: str | Path,
        crop_size: int = 500,
        num_classes: int = 3,
        codon_table: Optional[Dict[str, int]] = None,
        shuffle: bool = False,
        mutate: bool = False,
        mutation_rate: float = 0.1,
        shuffle_frames: bool = False,
        label_first: bool = True,
    ):
        self.path = Path(path)
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.codon_table = codon_table
        self.shuffle = shuffle
        self.mutate = mutate
        self.mutation_rate = mutation_rate
        self.shuffle_frames = shuffle_frames
        self.label_first = label_first
        self.rows = []
        with self.path.open("r", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 2:
                    continue
                if label_first:
                    label = int(row[0])
                    seq = row[1]
                else:
                    seq = row[0]
                    label = int(row[1])
                self.rows.append((label, seq))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        label, seq = self.rows[idx]
        seq_indices = dna_to_indices(seq)
        if self.mutate:
            from jaeger.dataops.pytorch.transforms import apply_mutation

            seq_indices = apply_mutation(seq_indices, self.mutation_rate)

        # Crop/pad nucleotide sequence
        if len(seq_indices) > self.crop_size:
            start = np.random.randint(0, len(seq_indices) - self.crop_size + 1)
            seq_indices = seq_indices[start : start + self.crop_size]
        elif len(seq_indices) < self.crop_size:
            pad = self.crop_size - len(seq_indices)
            seq_indices = np.pad(seq_indices, (0, pad), constant_values=4)

        if self.shuffle:
            seq_indices = np.random.permutation(seq_indices)

        x = translate_to_codons(seq_indices, self.codon_table)
        mask = x != 0

        if self.shuffle_frames:
            from jaeger.dataops.pytorch.transforms import shuffle_frames

            x = shuffle_frames(x)

        y = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=self.num_classes
        ).float()
        return x, y, mask
