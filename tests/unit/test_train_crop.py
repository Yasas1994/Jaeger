from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")


def test_tf_crop_size_nt_codon_units():
    from jaeger.commands.train import _sp_tf_crop_size_nt

    # codon crop_size converts to nucleotides for the TF preprocessor
    assert _sp_tf_crop_size_nt({"crop_size": 665, "crop_units": "codon"}) == 2000
    assert _sp_tf_crop_size_nt({"crop_size": 500, "crop_units": "codon"}) == 1505


def test_tf_crop_size_nt_legacy_passthrough():
    from jaeger.commands.train import _sp_tf_crop_size_nt

    # no crop_units -> legacy nucleotide behavior, unchanged
    assert _sp_tf_crop_size_nt({"crop_size": 2000}) == 2000
    assert _sp_tf_crop_size_nt({"crop_size": 2000, "crop_units": "nucleotide"}) == 2000


def test_tf_crop_size_nt_missing():
    from jaeger.commands.train import _sp_tf_crop_size_nt

    assert _sp_tf_crop_size_nt({}) is None


def test_tf_frame_len_codon_units():
    from jaeger.commands.train import _sp_tf_frame_len

    # codon crop_size IS the frame count
    assert _sp_tf_frame_len({"crop_size": 665, "crop_units": "codon"}) == 665


def test_tf_frame_len_legacy_nt():
    from jaeger.commands.train import _sp_tf_frame_len

    # legacy nucleotide behavior preserved (numba-style // 3 - 1)
    assert _sp_tf_frame_len({"crop_size": 2000}) == 665
