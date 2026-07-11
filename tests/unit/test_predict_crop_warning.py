from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")


def test_no_warning_when_lengths_match():
    from jaeger.commands.predict import _crop_length_warning

    assert _crop_length_warning(665, 2000, 2000) is None


def test_warning_when_fsize_maps_to_fewer_codons():
    from jaeger.commands.predict import _crop_length_warning

    msg = _crop_length_warning(665, 2000, 1995)
    assert msg is not None
    assert "663" in msg  # nucleotides_to_codons(1995)
    assert "665" in msg


def test_no_warning_when_trained_length_unknown():
    from jaeger.commands.predict import _crop_length_warning

    assert _crop_length_warning(None, None, 2000) is None
