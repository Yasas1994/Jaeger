"""Tests for jaeger.seqops.maps."""

from __future__ import annotations

from jaeger.seqops import maps


def test_codon_table_size():
    assert len(maps.CODONS) == 64
    assert len(maps.CODON_ID) == 64
    assert len(set(maps.CODON_ID)) == 64


def test_amino_acid_tables():
    # AA maps each codon to an amino acid (including stops); hence length 64.
    assert len(maps.AA) == 64
    assert len(maps.AA_ID) == 64


def test_reduced_alphabet_maps():
    # PC2 maps the 20 canonical amino acids to 2 groups.
    assert len(maps.PC2) == 20
    assert len(maps.PC2_ID) == 64
    assert len(maps.MURPHY10) == 20
    assert len(maps.MURPHY10_ID) == 64
    assert len(maps.PC5) == 20
    assert len(maps.PC5_ID) == 64


def test_dinucleotide_maps():
    assert len(maps.DICODONS) == 64 * 64
    assert len(maps.DICODON_ID) == 64 * 64
