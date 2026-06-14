"""Runtime preprocessing transforms for PyTorch datasets."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch


_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
_NUCLEOTIDES = ["A", "T", "G", "C", "N"]


def _reverse_complement(seq_str: str) -> str:
    return "".join(_COMPLEMENT.get(base, "N") for base in reversed(seq_str.upper()))


def translate_to_codons(seq: np.ndarray, codon_table: Dict[str, int]) -> torch.Tensor:
    """Translate an int8 nucleotide sequence into 6-frame codon indices.

    Valid codons are mapped using ``codon_table``; unknown or padded codons
    map to ``0`` so that masks can be computed as ``x != 0``. The six reading
    frames are trimmed to the same length before stacking.
    """
    seq_str = "".join(_NUCLEOTIDES[i] for i in seq)
    frames = []
    for offset in range(3):
        codons = [seq_str[i : i + 3] for i in range(offset, len(seq_str) - 2, 3)]
        indices = [codon_table.get(c, 0) for c in codons]
        rev = _reverse_complement(seq_str)
        rev_codons = [rev[i : i + 3] for i in range(offset, len(rev) - 2, 3)]
        rev_indices = [codon_table.get(c, 0) for c in rev_codons]
        frames.append(torch.tensor(indices, dtype=torch.long))
        frames.append(torch.tensor(rev_indices, dtype=torch.long))

    max_len = max(len(f) for f in frames)
    frames = [
        torch.nn.functional.pad(f, (0, max_len - len(f))) for f in frames
    ]
    return torch.stack(frames)


def dna_to_indices(seq: str) -> np.ndarray:
    mapping = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
    return np.array([mapping.get(base.upper(), 4) for base in seq], dtype=np.int8)


def apply_mutation(seq: np.ndarray, rate: float) -> np.ndarray:
    if rate == 0:
        return seq
    mask = np.random.rand(len(seq)) < rate
    mutated = seq.copy()
    mutated[mask] = np.random.randint(0, 4, size=mask.sum())
    return mutated


def shuffle_frames(x: torch.Tensor) -> torch.Tensor:
    """Shuffle the 6 reading frames of a translated tensor."""
    frames = list(range(x.shape[0]))
    random.shuffle(frames)
    return x[frames]
