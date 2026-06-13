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

# Preprocessing functions
from jaeger.seqops.encode import process_string_train, process_string_inference

# Dataset loaders
from jaeger.data.loaders import (
    _load_numpy_full_dataset as load_numpy_full,
    _load_numpy_raw_dataset as load_numpy_raw,
    _load_numpy_raw_variable_dataset as load_numpy_raw_variable,
)

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
    # Preprocessing
    "process_string_train",
    "process_string_inference",
    # Loaders
    "load_numpy_full",
    "load_numpy_raw",
    "load_numpy_raw_variable",
    # Converters
    "convert_dataset",
]
