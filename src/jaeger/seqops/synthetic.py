"""Synthetic DNA sequence generators.

Used for data augmentation, out-of-distribution testing, and generating
synthetic training examples (e.g., tandem repeats, low-entropy sequences).
"""

from __future__ import annotations

import random
from typing import List


from jaeger.seqops.stats import shannon_entropy


def generate_homopolymer(length: int, base: str = "A") -> str:
    """Generate a homopolymer (single-base repeat) of given length."""
    return base * length


def generate_tandem_repeat(motif: str, copies: int) -> str:
    """Repeat *motif* *copies* times to create a tandem repeat."""
    return motif * copies


def generate_random_tandem_repeats(
    num_sequences: int,
    motif_length_range: tuple = (3, 30),
    copy_number: int = 2000,
    alphabet: List[str] = ["A", "C", "G", "T"],
) -> List[str]:
    """Generate a list of random tandem repeat sequences.

    Parameters
    ----------
    num_sequences:
        Number of sequences to generate.
    motif_length_range:
        ``(min_len, max_len)`` for randomly sampled motifs.
    copy_number:
        Number of times to repeat each motif.
    alphabet:
        Nucleotide characters to sample motifs from.

    Returns
    -------
    List of generated sequences (each truncated to 2048 bp).
    """
    sequences = []
    for _ in range(num_sequences):
        motif_len = random.randint(*motif_length_range)
        motif = "".join(random.choices(alphabet, k=motif_len))
        seq = generate_tandem_repeat(motif, copy_number)
        sequences.append(seq[:2048])
    return sequences


def generate_biased_sequence(length: int, freqs: dict | None = None) -> str:
    """Generate a sequence with biased nucleotide frequencies.

    *freqs* should be a dict like ``{'A': 0.7, 'C': 0.1, 'G': 0.1, 'T': 0.1}``.
    """
    if freqs is None:
        freqs = {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1}
    bases = list(freqs.keys())
    weights = list(freqs.values())
    return "".join(random.choices(bases, weights=weights, k=length))


def generate_low_entropy_sequence(
    length: int, window_size: int, threshold: float, max_attempts: int = 10000
) -> str:
    """Generate a random sequence where all sliding windows have entropy < *threshold*."""
    for attempt in range(max_attempts):
        seq = generate_biased_sequence(length)
        valid = True
        for i in range(length - window_size + 1):
            if shannon_entropy(seq[i : i + window_size]) >= threshold:
                valid = False
                break
        if valid:
            return seq
    raise ValueError(
        f"Failed to generate low-entropy sequence after {max_attempts} attempts"
    )
