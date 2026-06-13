"""Sequence statistics utilities.

GC content, entropy, skew, and other DNA sequence metrics.
"""

from __future__ import annotations

from math import log2



def shannon_entropy(seq: str) -> float:
    """Calculate Shannon entropy (bits) for a DNA sequence."""
    counts = {}
    for base in seq:
        counts[base] = counts.get(base, 0) + 1
    entropy = 0.0
    length = len(seq)
    for count in counts.values():
        p = count / length
        entropy -= p * log2(p)
    return entropy


def gc_skew(seq: str, window: int = 2048) -> float:
    """Calculate GC skew for a sequence."""
    g = seq.count("G")
    c = seq.count("C")
    return (g - c) / (g + c) if (g + c) > 0 else 0.0


def calculate_gc_content(sequence: str) -> float:
    """Calculate the GC content of a DNA sequence."""
    gc = sequence.count("G") + sequence.count("C")
    return gc / len(sequence) if sequence else 0.0


def calculate_percentage_of_n(sequence: str) -> float:
    """Calculate the percentage of N characters in a sequence."""
    return sequence.count("N") / len(sequence) if sequence else 0.0
