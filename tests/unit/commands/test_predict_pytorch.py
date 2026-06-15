"""Tests for the PyTorch-based ``jaeger predict`` command."""

from __future__ import annotations

import random

import torch
import yaml

from jaeger.commands.predict import run_core
from jaeger.nnlib.pytorch.builder import ModelBuilder


def _build_config(tmp_path, seq_length: int):
    """Create a minimal PyTorch model config."""
    return {
        "model": {
            "name": "test_predict_model",
            "classifier_out_dim": 3,
            "reliability_out_dim": 0,
            "class_label_map": [
                {"class": "chromosome", "label": 0},
                {"class": "virus", "label": 1},
                {"class": "plasmid", "label": 2},
            ],
            "embedding": {
                "input_type": "translated",
                "use_embedding_layer": True,
                "vocab_size": 65,
                "embedding_size": 4,
            },
            "string_processor": {
                "codon": "CODONS",
                "codon_id": "CODON_ID",
                "crop_size": seq_length,
                "seq_onehot": False,
                "shuffle": False,
                "mutate": False,
                "shuffle_frames": False,
            },
            "representation_learner": {
                "hidden_layers": [],
                "pooling": "average",
            },
            "classifier": {
                "input_shape": 4,
                "hidden_layers": [{"name": "dense", "config": {"units": 3}}],
            },
        },
        "training": {
            "batch_size": 2,
            "optimizer": "adam",
            "optimizer_params": {"lr": 1e-3},
        },
    }


def _write_fasta(path, records: dict[str, str]):
    """Write a small FASTA file from a header -> sequence mapping."""
    with open(path, "w") as fh:
        for header, seq in records.items():
            fh.write(f">{header}\n{seq}\n")


def test_predict_pytorch_run_core(tmp_path):
    """``run_core`` should produce TSV outputs for a tiny FASTA file."""
    seq_length = 100
    config = _build_config(tmp_path, seq_length)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    builder = ModelBuilder(config)
    models = builder.build_fragment_classifier()
    # Bias the classifier toward the virus class so the phage TSV is non-empty.
    with torch.no_grad():
        models["jaeger_model"].classification_head.net[-1].bias[:] = torch.tensor(
            [-10.0, 10.0, -10.0]
        )
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": models["jaeger_model"].state_dict()},
        checkpoint_path,
    )

    fasta_path = tmp_path / "input.fasta"
    bases = ["A", "C", "G", "T"]
    seq = "".join(random.choices(bases, k=200))  # 200 bp, longer than fsize
    _write_fasta(fasta_path, {"contig_1": seq, "contig_2": seq})

    output_dir = tmp_path / "output"
    run_core(
        input=str(fasta_path),
        output=str(output_dir),
        config=str(config_path),
        checkpoint=str(checkpoint_path),
        cpu=True,
        fsize=seq_length,
        stride=50,
        batch=2,
        overwrite=True,
        pc=-100,
    )

    out_model_dir = output_dir / config["model"]["name"]
    assert (out_model_dir / "input.tsv").exists()
    assert (out_model_dir / "input_phages.tsv").exists()
