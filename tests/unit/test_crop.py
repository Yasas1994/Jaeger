from __future__ import annotations

import pytest


def test_codons_to_nucleotides():
    from jaeger.seqops.crop import codons_to_nucleotides

    assert codons_to_nucleotides(665) == 2000
    assert codons_to_nucleotides(500) == 1505
    assert codons_to_nucleotides(100) == 305


def test_nucleotides_to_codons():
    from jaeger.seqops.crop import nucleotides_to_codons

    assert nucleotides_to_codons(2000) == 665
    assert nucleotides_to_codons(1505) == 500


def test_roundtrip():
    from jaeger.seqops.crop import codons_to_nucleotides, nucleotides_to_codons

    for c in (50, 100, 500, 665):
        assert nucleotides_to_codons(codons_to_nucleotides(c)) == c


def test_codons_to_nucleotides_rejects_non_positive():
    from jaeger.seqops.crop import codons_to_nucleotides

    with pytest.raises(ValueError):
        codons_to_nucleotides(0)
    with pytest.raises(ValueError):
        codons_to_nucleotides(-10)


def test_tf_and_numba_conventions_agree_at_3c_plus_5():
    """The whole plan relies on this: at nt = 3*c + 5, the TF preprocessor
    (offset_lut) and the numba converter (//3-1) yield exactly c codon frames."""
    from jaeger.seqops.crop import (
        codons_to_nucleotides,
        numba_frame_length,
        tf_frame_length,
    )

    for c in (50, 100, 500, 665):
        nt = codons_to_nucleotides(c)
        assert tf_frame_length(nt) == c
        assert numba_frame_length(nt) == c


def test_real_tf_preprocessor_matches_helper():
    """Guard against drift between crop.py and the real encode.py preprocessor."""
    tf = pytest.importorskip("tensorflow")
    import random as _random

    from jaeger.seqops.crop import codons_to_nucleotides, tf_frame_length
    from jaeger.seqops.encode import process_string_train

    rng = _random.Random(0)
    seq = "".join(rng.choice("ACGT") for _ in range(2005))
    for c in (500, 665):
        nt = codons_to_nucleotides(c)
        p = process_string_train(
            crop_size=nt,
            input_type="translated",
            seq_onehot=False,
            class_label_onehot=True,
            num_classes=6,
            masking=False,
            ngram_width=3,
        )
        out, _ = p(tf.constant(f"0,{seq}"))
        got = int(out["translated"].shape[-1])
        assert got == c
        assert got == tf_frame_length(nt)


def test_resolve_crop_default_units_is_codon():
    from jaeger.seqops.crop import resolve_crop

    assert resolve_crop({"crop_size": 665}) == (665, 2000)


def test_resolve_crop_codon_units():
    from jaeger.seqops.crop import resolve_crop

    assert resolve_crop({"crop_units": "codon", "crop_size": 500}) == (500, 1505)


def test_resolve_crop_nucleotide_units():
    from jaeger.seqops.crop import resolve_crop

    assert resolve_crop({"crop_units": "nucleotide", "crop_size": 2000}) == (665, 2000)
    assert resolve_crop({"crop_units": "nucleotide", "crop_size": 1505}) == (500, 1505)


def test_resolve_crop_missing_crop_size_raises():
    from jaeger.seqops.crop import resolve_crop

    with pytest.raises(ValueError):
        resolve_crop({"crop_units": "codon"})


def test_resolve_crop_unknown_units_raises():
    from jaeger.seqops.crop import resolve_crop

    with pytest.raises(ValueError):
        resolve_crop({"crop_units": "weird", "crop_size": 100})
