"""Unit tests for jaeger.commands.utils."""

from __future__ import annotations

from pathlib import Path

from jaeger.commands import utils


def test_optimize_data_core_interprets_codon_units(monkeypatch, tmp_path: Path):
    """Crop sizes convert via 3*crop_size + 5 and strides scale by 3 (a shift)
    when units is codon: 10/20 codons -> 35/65 nt, stride 5 -> 15."""
    called = {}

    def _fake_convert_dataset(*, crop_size, stride, strides, **kwargs):
        called["crop_size"] = crop_size
        called["stride"] = stride
        called["strides"] = strides

    monkeypatch.setattr(utils, "convert_dataset", _fake_convert_dataset)

    utils.optimize_data_core(
        input_path=str(tmp_path / "in.csv"),
        output_path=str(tmp_path / "out.npz"),
        format="translated",
        crop_size=(10, 20),
        stride=5,
        units="codon",
        num_classes=3,
    )

    assert called["crop_size"] == (35, 65)
    assert called["stride"] == 15
    assert called["strides"] is None


def test_optimize_data_core_keeps_nucleotide_units(monkeypatch, tmp_path: Path):
    """crop_size and stride are unchanged when units is nuc."""
    called = {}

    def _fake_convert_dataset(*, crop_size, stride, **kwargs):
        called["crop_size"] = crop_size
        called["stride"] = stride

    monkeypatch.setattr(utils, "convert_dataset", _fake_convert_dataset)

    utils.optimize_data_core(
        input_path=str(tmp_path / "in.csv"),
        output_path=str(tmp_path / "out.npz"),
        format="translated",
        crop_size=(10, 20),
        stride=5,
        units="nuc",
        num_classes=3,
    )

    assert called["crop_size"] == (10, 20)
    assert called["stride"] == 5


def test_optimize_data_core_overlap_respects_codon_units(monkeypatch, tmp_path: Path):
    """Per-crop strides computed from overlap use codon-converted crop sizes."""
    called = {}

    def _fake_convert_dataset(*, strides, **kwargs):
        called["strides"] = strides

    monkeypatch.setattr(utils, "convert_dataset", _fake_convert_dataset)

    utils.optimize_data_core(
        input_path=str(tmp_path / "in.csv"),
        output_path=str(tmp_path / "out.npz"),
        format="translated",
        crop_size=(10,),
        overlap=0.5,
        units="codon",
        num_classes=3,
    )

    # 10 codons -> 3*10 + 5 = 35 nt; stride = int(35 * (1 - 0.5)) = 17
    assert called["strides"] == [17]
