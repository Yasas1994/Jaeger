"""Unit tests for prophage boundary refinement with pyrodigal-gv."""

import random
from pathlib import Path

import numpy as np
import pytest

from jaeger.postprocess.prophage_boundaries import (
    find_genes,
    refine_boundary,
    refine_prophage_boundaries,
    refine_region,
)


def test_refine_boundary_keeps_intergenic_left():
    genes = [(100, 200), (300, 400)]
    assert refine_boundary(50, genes, "left") == 50


def test_refine_boundary_keeps_intergenic_right():
    genes = [(100, 200), (300, 400)]
    assert refine_boundary(250, genes, "right") == 250


def test_refine_boundary_left_inside_gene_extends_to_gene_start():
    genes = [(100, 200)]
    assert refine_boundary(150, genes, "left") == 100


def test_refine_boundary_right_inside_gene_extends_to_gene_end():
    genes = [(100, 200)]
    assert refine_boundary(150, genes, "right") == 200


def test_refine_boundary_caps_left_extension():
    genes = [(0, 1000)]
    assert refine_boundary(900, genes, "left", max_extension=50) == 850


def test_refine_boundary_caps_right_extension():
    genes = [(0, 1000)]
    assert refine_boundary(100, genes, "right", max_extension=50) == 150


def test_refine_region_snaps_both_boundaries():
    genes = [(100, 200), (500, 600)]
    assert refine_region(150, 550, genes) == (100, 600)


def test_refine_region_keeps_intergenic_boundaries():
    genes = [(100, 200), (500, 600)]
    assert refine_region(250, 700, genes) == (250, 700)


def test_find_genes_returns_sorted_half_open_intervals_within_sequence():
    # A short random-ish sequence unlikely to contain real genes, but the
    # function should still return sane coordinates.
    seq = "ATG" + "ACGT" * 50 + "TAA"
    genes = find_genes(seq)
    assert isinstance(genes, list)
    for start, end in genes:
        assert 0 <= start < end <= len(seq)
    assert genes == sorted(genes)


def test_refine_boundary_rejects_invalid_side():
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        refine_boundary(50, [(0, 100)], "upstream")


def _make_orf_sequence(num_codons: int = 100, seed: int = 42) -> str:
    """Return a short sequence containing one clean ORF."""
    rng = random.Random(seed)
    stop_codons = {"TAA", "TAG", "TGA"}
    codons = [
        "".join(bases)
        for bases in __import__("itertools").product("ACGT", repeat=3)
        if "".join(bases) not in stop_codons
    ]
    internal = "".join(rng.choice(codons) for _ in range(num_codons))
    return "ATG" + internal + "TAA"


def test_refine_prophage_boundaries_snaps_to_predicted_orf(tmp_path: Path):
    seq = _make_orf_sequence(num_codons=100)
    genes = find_genes(seq)
    assert len(genes) >= 1, "pyrodigal-gv should predict the synthetic ORF"

    orf_start, orf_end = min(genes, key=lambda g: g[0])
    fasta = tmp_path / "test.fa"
    fasta.write_text(f">contig1\n{seq}\n")

    prophage_cordinates = {
        "contig1": [
            np.array([[orf_start + 5, orf_end - 5]]),
            np.array([1.0]),
        ]
    }

    refined = refine_prophage_boundaries(
        prophage_cordinates=prophage_cordinates,
        fasta_path=fasta,
        fsize=1,
        max_extension=1000,
    )

    assert refined == {"contig1": [(orf_start + 5, orf_end - 5, orf_start, orf_end)]}
