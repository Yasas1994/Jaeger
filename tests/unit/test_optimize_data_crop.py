from __future__ import annotations

from unittest import mock


def _call(units, crop_size, stride=0, strides=None):
    from jaeger.commands import utils as u

    with mock.patch.object(u, "convert_dataset") as cd:
        u.optimize_data_core(
            input_path="in.csv",
            output_path="out.npz",
            format="translated",
            crop_size=tuple(crop_size),
            stride=stride,
            strides=strides,
            units=units,
        )
    return cd.call_args.kwargs


def test_optimize_data_codon_units_uses_canonical_nt():
    """665 codons must map to 2000 nt (3*665 + 5), not 1995 (3*665)."""
    kw = _call("codon", [665])
    assert kw["crop_size"] == (2000,)
    assert kw["stride"] == 0


def test_optimize_data_codon_units_converts_each_crop_and_stride():
    """Each codon crop -> 3c+5 nt; a codon stride is a shift -> 3*stride bp."""
    kw = _call("codon", [500, 665], stride=100)
    assert kw["crop_size"] == (1505, 2000)
    assert kw["stride"] == 300


def test_optimize_data_codon_units_with_strides_list():
    kw = _call("codon", [665], strides=[100])
    assert kw["crop_size"] == (2000,)
    assert kw["strides"] == [300]


def test_optimize_data_nuc_units_passthrough():
    kw = _call("nuc", [2000], stride=500)
    assert kw["crop_size"] == (2000,)
    assert kw["stride"] == 500
