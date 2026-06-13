"""Tests for jaeger.seqops.validate."""

from __future__ import annotations

import pytest

from jaeger.seqops import validate


@pytest.mark.parametrize(
    "seq, valid",
    [
        ("ACGT", True),
        ("acgt", True),
        ("ACGTNX", False),
        ("", False),
        ("ACGT123", False),
    ],
)
def test_is_valid_dna(seq: str, valid: bool):
    assert validate.is_valid_dna(seq) is valid


@pytest.mark.parametrize(
    "header, sequence, ok",
    [
        ("seq1", "ACGT", True),
        ("seq2", "ACGTNX", False),
        ("seq3", "", False),
    ],
)
def test_validate_fasta_entry(header: str, sequence: str, ok: bool):
    is_valid, error = validate.validate_fasta_entry(header, sequence)
    assert is_valid is ok
    if ok:
        assert error is None
    else:
        assert error is not None
