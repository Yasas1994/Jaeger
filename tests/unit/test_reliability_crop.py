from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")


def _resolve(generator_cfg, sp_cfg):
    from jaeger.dataops.reliability_generator import _resolve_reliability_crop_size

    return _resolve_reliability_crop_size(generator_cfg, sp_cfg)


def test_generator_crop_sizes_plural_honored():
    """``crop_sizes: [665]`` (codons) -> 2000 nt; the plural key must win."""
    assert _resolve({"crop_sizes": [665], "units": "codon"}, {}) == 2000


def test_generator_crop_size_singular_codon_units():
    assert _resolve({"crop_size": 500, "crop_units": "codon"}, {}) == 1505


def test_falls_back_to_string_processor_crop_size():
    assert _resolve({"units": "codon"}, {"crop_size": 665}) == 2000


def test_nucleotide_units_passthrough():
    assert _resolve({"crop_size": 2000, "units": "nuc"}, {}) == 2000
    assert _resolve({"crop_size": 2000, "crop_units": "nucleotide"}, {}) == 2000


def test_default_is_500_codons():
    """No overrides -> 500 codons -> 1505 nt (canonical default)."""
    assert _resolve({}, {}) == 1505


def test_unknown_units_raises():
    with pytest.raises(ValueError):
        _resolve({"crop_size": 100, "units": "weird"}, {})


def test_translated_classifier_defaults_to_codon():
    assert _resolve({"crop_size": 665}, {"input_type": "translated"}) == 2000


def test_nucleotide_classifier_defaults_to_nt():
    """Nucleotide classifiers consume crop_size directly in nt; without an
    explicit unit the default must follow the model's input_type."""
    assert _resolve({"crop_size": 2000}, {"input_type": "nucleotide"}) == 2000
