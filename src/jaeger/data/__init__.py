"""Jaeger data package: re-exports from seqops and dataops.

This module provides a unified import point for external users.
Internal code should import directly from ``jaeger.seqops`` or
``jaeger.dataops``.
"""

from __future__ import annotations

# Maps and lookup tables
from jaeger.seqops.maps import (
    CODONS,
    AA,
    AA_ID,
    PC2,
    PC2_ID,
    MURPHY10,
    PC5,
    COD_ID,
    MURPHY10_ID,
    PC5_ID,
    CODON_ID,
    DICODONS,
    DICODON_ID,
)

# Shuffle utilities
from jaeger.seqops.transform import (
    dinuc_shuffle,
    kmer_shuffle,
    string_to_char_array,
    char_array_to_string,
    one_hot_to_tokens,
    tokens_to_one_hot,
)

# FASTA fragment generation
from jaeger.seqops.io import fragment_generator, fragment_generator_lib

# Public converter API
from jaeger.dataops.convert import convert_dataset

__all__ = [
    # Maps
    "CODONS",
    "AA",
    "AA_ID",
    "PC2",
    "PC2_ID",
    "MURPHY10",
    "PC5",
    "COD_ID",
    "MURPHY10_ID",
    "PC5_ID",
    "CODON_ID",
    "DICODONS",
    "DICODON_ID",
    # Shuffle
    "dinuc_shuffle",
    "kmer_shuffle",
    "string_to_char_array",
    "char_array_to_string",
    "one_hot_to_tokens",
    "tokens_to_one_hot",
    # FASTA
    "fragment_generator",
    "fragment_generator_lib",
    # Converters
    "convert_dataset",
]
