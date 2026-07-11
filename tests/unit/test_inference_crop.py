from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")
tf = pytest.importorskip("tensorflow")


def _write_fasta(path, length: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(length))
    path.write_text(f">t\n{seq}\n")


def _sp_decoy() -> dict:
    """A NumPy-style string_processor config whose crop_size is a decoy.

    ``crop_size=100`` (codons) must NOT drive inference; prediction uses the
    runtime ``--fsize`` instead.
    """
    from jaeger.seqops.maps import CODON_ID, CODONS

    return {
        "data_format": "numpy",
        "input_type": "translated",
        "seq_onehot": False,
        "codon": list(CODONS),
        "codon_id": list(CODON_ID),
        "codon_depth": 1,
        "ngram_width": 3,
        "crop_size": 100,  # decoy: must be ignored by predict
        "masking": False,
        "mutate": False,
        "mutation_rate": 0.1,
        "shuffle": False,
    }


def test_build_prediction_dataset_uses_fsize_not_config_crop(tmp_path):
    """The inference window length must follow the runtime ``--fsize`` (2000 ->
    665 codon frames), not ``string_processor.crop_size`` (decoy 100 -> ~98)."""
    from jaeger.commands.predict import _build_prediction_dataset

    fasta = tmp_path / "t.fa"
    _write_fasta(fasta, 2000)

    ds = _build_prediction_dataset(
        input_file_path=fasta,
        num=4,
        string_processor_config=_sp_decoy(),
        fragsize=2000,
        stride=2000,
        batch=8,
        min_len=2000,
        max_len=None,
        dynamic_stride=False,
        dynamic_stride_threshold=0.0,
        use_padded_batch=True,
    )
    translated = next(iter(ds))[0]["translated"]
    assert int(translated.shape[-1]) == 665


def test_build_prediction_dataset_honors_smaller_fsize(tmp_path):
    """A smaller ``--fsize`` (1500 -> 498 codon frames) is honored end-to-end
    and is not overridden by the model config."""
    from jaeger.commands.predict import _build_prediction_dataset

    fasta = tmp_path / "t.fa"
    _write_fasta(fasta, 1500)

    ds = _build_prediction_dataset(
        input_file_path=fasta,
        num=4,
        string_processor_config=_sp_decoy(),
        fragsize=1500,
        stride=1500,
        batch=8,
        min_len=1500,
        max_len=None,
        dynamic_stride=False,
        dynamic_stride_threshold=0.0,
        use_padded_batch=True,
    )
    translated = next(iter(ds))[0]["translated"]
    assert int(translated.shape[-1]) == 498


def _write_project_yaml(
    path, *, crop_size: int, input_type: str = "translated", crop_units: str | None = None
) -> None:
    units_line = f"    crop_units: {crop_units}\n" if crop_units is not None else ""
    path.write_text(
        "model:\n"
        "  embedding:\n"
        f"    type: {input_type}\n"
        "    codon: null\n"
        "    codon_id: null\n"
        "  string_processor:\n"
        f"{units_line}"
        f"    crop_size: {crop_size}\n"
    )


def test_infer_loader_exposes_crop_size_codons_and_nt(tmp_path):
    """The InferModel string-processor loader annotates canonical codon/nt
    lengths, defaulting ``crop_units`` to ``codon`` (665 -> 2000 nt)."""
    from jaeger.nnlib.inference import InferModel

    cfg = tmp_path / "project.yaml"
    _write_project_yaml(cfg, crop_units="codon", crop_size=665)

    sp = InferModel.__new__(InferModel)._load_string_processor_config(cfg)
    assert sp["crop_units"] == "codon"
    assert sp["crop_size_codons"] == 665
    assert sp["crop_size_nt"] == 2000


def test_infer_loader_honors_nucleotide_crop_units(tmp_path):
    """With ``crop_units: nucleotide``, ``crop_size`` is nt and codons are
    derived (2000 nt -> 665 codons)."""
    from jaeger.nnlib.inference import InferModel

    cfg = tmp_path / "project.yaml"
    _write_project_yaml(cfg, crop_units="nucleotide", crop_size=2000)

    sp = InferModel.__new__(InferModel)._load_string_processor_config(cfg)
    assert sp["crop_size_codons"] == 665
    assert sp["crop_size_nt"] == 2000


def test_infer_loader_nucleotide_model_uses_nt(tmp_path):
    """For ``input_type: nucleotide`` there are no codon frames: ``crop_size``
    is the nucleotide window directly and must not be codon-converted."""
    from jaeger.nnlib.inference import InferModel

    cfg = tmp_path / "project.yaml"
    _write_project_yaml(cfg, crop_size=2000, input_type="nucleotide")

    sp = InferModel.__new__(InferModel)._load_string_processor_config(cfg)
    assert sp["crop_units"] == "nucleotide"
    assert sp["crop_size_nt"] == 2000
    assert sp.get("crop_size_codons") is None
