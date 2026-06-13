"""Sequence validation utilities.

Validators for DNA sequence quality, format checking, and constraint verification.
"""

from __future__ import annotations

import re


VALID_DNA_RE = re.compile(r"^[ATGCNatgcn]+$")


def is_valid_dna(sequence: str) -> bool:
    """Check if a sequence contains only valid DNA characters."""
    return bool(VALID_DNA_RE.match(sequence))


def validate_fasta_entry(header: str, sequence: str) -> tuple[bool, str | None]:
    """Validate a single FASTA entry.

    Returns ``(is_valid, error_message)``.
    """
    if not header:
        return False, "Empty header"
    if not sequence:
        return False, "Empty sequence"
    if not is_valid_dna(sequence):
        return (
            False,
            f"Invalid characters in sequence: {set(sequence) - set('ATGCNatgcn')}",
        )
    return True, None
