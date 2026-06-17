"""Tests for jaeger.dataops.reliability_generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jaeger.dataops import reliability_generator as rg


def test_select_id_ood_records():
    records = [(0, "AAAA"), (1, "TTTT"), (0, "CCCC"), (1, "GGGG")]
    y_true = np.array([0, 1, 0, 1], dtype=np.int32)
    y_pred = np.array(
        [
            [0.9, 0.1],  # correct, high conf -> ID
            [0.2, 0.8],  # correct, high conf -> ID
            [0.1, 0.9],  # wrong, high conf -> OOD
            [0.6, 0.4],  # wrong, low conf -> dropped
        ],
        dtype=np.float32,
    )
    id_records, ood_records = rg._select_id_ood_records(
        records, y_true, y_pred, threshold=0.8
    )
    assert len(id_records) == 2
    assert all(label == 1 for label, _ in id_records)
    assert len(ood_records) == 1
    assert ood_records[0] == (0, "CCCC")


def test_generate_synthetic_sequences_cycles_perturbations(tmp_path: Path):
    records = [(0, "ATCG" * 10), (1, "TGCA" * 10)]
    seqs = rg._generate_synthetic_sequences(
        records,
        multiplier=0.5,
        perturbations_cfg={
            "shuffle": True,
            "subseq_repeat": True,
            "tandem_repeat": True,
        },
    )
    assert len(seqs) == 1  # 0.5 * 2 = 1
    assert len(seqs[0]) == len(records[0][1])


def test_write_and_read_csv(tmp_path: Path):
    path = str(tmp_path / "rel.csv")
    records = [(1, "AAAA"), (0, "TTTT")]
    rg._write_csv(records, path)
    read = rg._read_csv_records(path)
    assert read == records
