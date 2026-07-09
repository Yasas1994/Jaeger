"""Tests for jaeger.dataops.reliability_generator."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from jaeger.dataops import reliability_generator as rg
from jaeger.dataops import synthetic_perturbations as sp


def _make_streamed_inference_mock(y_pred: np.ndarray):
    """Return a mock for _run_classifier_inference_streamed."""

    def _mock(
        classifier,
        dataset,
        records: list[tuple[int, str]],
        threshold: float,
        id_records: list[tuple[int, str]],
        ood_records: list[tuple[int, str]],
        preds_csv_path: str | None = None,
        num_classes: int | None = None,
        seq_ids: list[str] | None = None,
    ) -> None:
        if preds_csv_path is not None:
            np.savetxt(preds_csv_path, y_pred, delimiter=",")
        y_true = np.array([label for label, _ in records], dtype=np.int32)
        conf = np.max(y_pred, axis=1)
        pred_class = np.argmax(y_pred, axis=1)
        for i, (label, seq) in enumerate(records):
            if conf[i] < threshold:
                continue
            if pred_class[i] == y_true[i]:
                id_records.append((1, seq))
            else:
                ood_records.append((0, seq))

    return _mock


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


def test_run_classifier_inference_streamed_uses_existing_csv(tmp_path: Path):
    """If a prediction CSV already exists, the function should use it directly."""
    records = [
        (0, "AAAA"),
        (0, "AAAC"),
        (1, "TTTT"),
        (1, "TTTG"),
    ]
    preds_csv = tmp_path / "preds.csv"
    probs = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    seq_ids = [f"seq_{i}" for i in range(len(records))]
    labels = np.array([label for label, _ in records], dtype=np.int32)
    rg._write_predictions_csv(str(preds_csv), seq_ids, labels, probs)

    # Passing None as classifier/dataset is safe because the existing CSV is used.
    id_records: list[tuple[int, str]] = []
    ood_records: list[tuple[int, str]] = []
    rg._run_classifier_inference_streamed(
        classifier=None,  # type: ignore[arg-type]
        dataset=None,  # type: ignore[arg-type]
        records=records,
        threshold=0.7,
        id_records=id_records,
        ood_records=ood_records,
        preds_csv_path=str(preds_csv),
        num_classes=2,
        seq_ids=seq_ids,
    )

    assert len(id_records) == 2
    assert len(ood_records) == 2


def test_generate_synthetic_sequences_cycles_perturbations(tmp_path: Path):
    records = [(0, "ATCG" * 10), (1, "TGCA" * 10)]
    seqs = list(
        sp.generate_synthetic_sequences(
            records,
            multiplier=0.5,
            perturbations_cfg={
                "shuffle": True,
                "subseq_repeat": True,
                "tandem_repeat": True,
            },
        )
    )
    assert len(seqs) == 1  # 0.5 * 2 = 1
    assert len(seqs[0]) == len(records[0][1])


def test_write_and_read_csv(tmp_path: Path):
    path = str(tmp_path / "rel.csv")
    records = [(1, "AAAA"), (0, "TTTT")]
    rg._write_csv(records, path)
    read = rg._read_csv_records(path)
    assert read == records


def test_normalize_perturbation_cfg_structured():
    cfg = {
        "shuffle": {"enabled": True, "mode": "dinuc"},
        "subseq_repeat": {"enabled": True, "window_fraction": 0.3},
        "tandem_repeat": {
            "enabled": True,
            "motif_length_range": [4, 8],
            "window_fraction": 0.3,
            "num_repeats": 10,
        },
    }
    specs = sp._normalize_perturbation_cfg(cfg)
    assert len(specs) == 3
    assert specs[0]["name"] == "shuffle"
    assert specs[0]["fn"] is sp.apply_dinuc_shuffle
    assert specs[1]["kwargs"]["window_fraction"] == 0.3
    assert specs[2]["kwargs"]["num_repeats"] == 10
    assert specs[2]["kwargs"]["motif_length_range"] == (4, 8)


def test_normalize_perturbation_cfg_multiple_shuffle_modes():
    cfg = {
        "shuffle": {"enabled": True, "mode": ["dinuc", "random"]},
        "subseq_repeat": False,
        "tandem_repeat": False,
    }
    specs = sp._normalize_perturbation_cfg(cfg)
    shuffle_specs = [s for s in specs if s["name"] == "shuffle"]
    assert len(shuffle_specs) == 2
    fns = {s["fn"] for s in shuffle_specs}
    assert sp.apply_dinuc_shuffle in fns
    assert sp.apply_shuffle in fns


def test_normalize_perturbation_cfg_legacy_booleans():
    cfg = {"shuffle": True, "subseq_repeat": False, "tandem_repeat": True}
    specs = sp._normalize_perturbation_cfg(cfg)
    names = [s["name"] for s in specs]
    assert "shuffle" in names
    assert "subseq_repeat" not in names
    assert "tandem_repeat" in names


def test_compute_perturbation_counts_with_global_multiplier():
    records = [(0, "A" * 100)] * 12
    specs = [{"name": "shuffle"}, {"name": "subseq_repeat"}, {"name": "tandem_repeat"}]
    cfg = {}
    counts = sp._compute_perturbation_counts(records, 1.0, specs, cfg)
    assert sum(counts) == 12
    assert all(c == 4 for c in counts)


def test_compute_perturbation_counts_with_explicit_counts():
    records = [(0, "A" * 100)] * 100
    specs = [{"name": "shuffle"}, {"name": "tandem_repeat"}]
    cfg = {"shuffle": {"count": 10}, "tandem_repeat": {"multiplier": 0.2}}
    counts = sp._compute_perturbation_counts(records, 1.0, specs, cfg)
    # When all specs are explicit, the global multiplier is ignored and the
    # explicit counts are honored exactly.
    assert counts[0] == 10
    assert counts[1] == 20


def test_compute_perturbation_counts_splits_across_multiple_shuffle_modes():
    records = [(0, "A" * 100)] * 12
    # Two shuffle specs plus two other perturbations -> four implicit specs.
    specs = [
        {"name": "shuffle"},
        {"name": "shuffle"},
        {"name": "subseq_repeat"},
        {"name": "tandem_repeat"},
    ]
    cfg = {}
    counts = sp._compute_perturbation_counts(records, 1.0, specs, cfg)
    assert sum(counts) == 12
    assert all(c > 0 for c in counts)


def test_generate_synthetic_sequences_uses_shuffle_mode():
    records = [(0, "ATCG" * 10)]
    seqs = list(
        sp.generate_synthetic_sequences(
            records,
            multiplier=2.0,
            perturbations_cfg={"shuffle": {"enabled": True, "mode": "dinuc"}},
        )
    )
    assert len(seqs) == 2
    assert all(len(s) == len(records[0][1]) for s in seqs)


def test_generate_reliability_data_smoke(monkeypatch, tmp_path: Path):
    """End-to-end smoke test with a mocked classifier and dataset pipeline."""
    import tensorflow as tf

    csv_path = str(tmp_path / "train.csv")
    rg._write_csv(
        [(0, "ATCG" * 50), (1, "TGCA" * 50), (0, "GCTA" * 50), (1, "CATG" * 50)],
        csv_path,
    )
    output_dir = str(tmp_path / "rel_out")

    records = rg._read_csv_records(csv_path)
    true_labels = np.array([label for label, _ in records], dtype=np.int32)

    # Mock dataset builder so we do not need a real TF preprocessing pipeline.
    def _mock_build_dataset(
        csv_path, string_processor_config, classifier_out_dim, batch_size, **kwargs
    ):
        n = len(rg._read_csv_records(csv_path))
        x = tf.zeros((n, 10), dtype=tf.float32)
        y = tf.one_hot(true_labels[:n], depth=classifier_out_dim)
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

    monkeypatch.setattr(rg, "_build_inference_dataset", _mock_build_dataset)

    # Mock streaming classifier inference on real data.
    real_preds = np.array(
        [
            [0.9, 0.1],  # true 0, pred 0 -> ID
            [0.1, 0.9],  # true 1, pred 1 -> ID
            [0.1, 0.9],  # true 0, pred 1 -> OOD
            [0.6, 0.4],  # true 1, low conf -> dropped
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        rg,
        "_run_classifier_inference_streamed",
        _make_streamed_inference_mock(real_preds),
    )

    # Mock classifier inference on synthetic OOD chunks.
    def _mock_run_inference(classifier, dataset):
        n = sum(int(batch[0].shape[0]) for batch in dataset)
        return np.tile(np.array([0.9, 0.1], dtype=np.float32), (n, 1))

    monkeypatch.setattr(rg, "_run_classifier_inference", _mock_run_inference)

    class FakeClassifier:
        pass

    result = rg.generate_reliability_data(
        classifier=FakeClassifier(),
        raw_csv_path=csv_path,
        output_dir=output_dir,
        string_processor_config={"crop_size": 100},
        model_cfg={"string_processor": {}},
        classifier_out_dim=2,
        reliability_out_dim=1,
        batch_size=2,
        id_threshold=0.8,
        synthetic_ood_threshold=0.8,
        synthetic_ood_multiplier=1.0,
        generator_cfg={
            "perturbations": {"shuffle": True, "tandem_repeat": False, "mix": True},
            "val_fraction": 0.5,
        },
    )

    assert (Path(output_dir) / "reliability_train.csv").exists()
    assert (Path(output_dir) / "reliability_val.csv").exists()
    assert (Path(output_dir) / "reliability_train.npz").exists()
    assert (Path(output_dir) / "reliability_val.npz").exists()
    assert result["train"]["paths"]
    assert result["validation"]["paths"]


def test_generate_reliability_data_accepts_raw_csv_paths(monkeypatch, tmp_path: Path):
    """Separate train/val CSVs are used directly instead of splitting one file."""
    import tensorflow as tf

    train_csv = str(tmp_path / "train.csv")
    val_csv = str(tmp_path / "val.csv")
    rg._write_csv(
        [(0, "ATCG" * 50), (1, "TGCA" * 50), (0, "GCTA" * 50), (1, "CATG" * 50)],
        train_csv,
    )
    rg._write_csv(
        [(0, "AAAA" * 50), (1, "TTTT" * 50)],
        val_csv,
    )
    output_dir = str(tmp_path / "rel_out")

    train_records = rg._read_csv_records(train_csv)

    def _mock_build_dataset(
        csv_path, string_processor_config, classifier_out_dim, batch_size, **kwargs
    ):
        records = rg._read_csv_records(csv_path)
        n = len(records)
        labels = np.array([label for label, _ in records], dtype=np.int32)
        x = tf.zeros((n, 10), dtype=tf.float32)
        y = tf.one_hot(labels, depth=classifier_out_dim)
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

    monkeypatch.setattr(rg, "_build_inference_dataset", _mock_build_dataset)

    train_real_preds = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=np.float32,
    )
    val_real_preds = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    _streamed_call_count = 0

    def _mock_run_inference_streamed(
        classifier,
        dataset,
        records,
        threshold,
        id_records,
        ood_records,
        preds_csv_path=None,
        num_classes=None,
        seq_ids=None,
    ):
        nonlocal _streamed_call_count
        _streamed_call_count += 1
        y_pred = train_real_preds if _streamed_call_count == 1 else val_real_preds
        if preds_csv_path is not None:
            np.savetxt(preds_csv_path, y_pred, delimiter=",")
        y_true = np.array([label for label, _ in records], dtype=np.int32)
        conf = np.max(y_pred, axis=1)
        pred_class = np.argmax(y_pred, axis=1)
        for i, (label, seq) in enumerate(records):
            if conf[i] < threshold:
                continue
            if pred_class[i] == y_true[i]:
                id_records.append((1, seq))
            else:
                ood_records.append((0, seq))

    monkeypatch.setattr(
        rg, "_run_classifier_inference_streamed", _mock_run_inference_streamed
    )

    class FakeClassifier:
        pass

    rg.generate_reliability_data(
        classifier=FakeClassifier(),
        raw_csv_path=train_csv,
        output_dir=output_dir,
        string_processor_config={"crop_size": 100},
        model_cfg={"string_processor": {}},
        classifier_out_dim=2,
        reliability_out_dim=1,
        batch_size=2,
        id_threshold=0.8,
        synthetic_ood_threshold=0.8,
        synthetic_ood_multiplier=0.0,
        generator_cfg={
            "raw_csv_paths": {"train": train_csv, "val": val_csv},
            "perturbations": {"shuffle": False},
            "val_fraction": 0.5,
        },
    )

    assert (Path(output_dir) / "reliability_train.csv").exists()
    assert (Path(output_dir) / "reliability_val.csv").exists()

    # Val file should contain exactly the two validation records (both ID).
    val_written = rg._read_csv_records(str(Path(output_dir) / "reliability_val.csv"))
    assert len(val_written) == 2
    assert all(label == 1 for label, _ in val_written)

    # Train file should contain ID + OOD from train CSV (no val records).
    train_written = rg._read_csv_records(
        str(Path(output_dir) / "reliability_train.csv")
    )
    assert all(seq in {r[1] for r in train_records} for _, seq in train_written)


def test_generate_reliability_data_raw_csv_paths_missing_val_falls_back(
    monkeypatch, tmp_path: Path
):
    """If raw_csv_paths.val is missing, fall back to val_fraction split of train."""
    import tensorflow as tf

    csv_path = str(tmp_path / "train.csv")
    rg._write_csv(
        [(0, "ATCG" * 50), (1, "TGCA" * 50), (0, "GCTA" * 50), (1, "CATG" * 50)],
        csv_path,
    )
    output_dir = str(tmp_path / "rel_out")

    true_labels = np.array([0, 1, 0, 1], dtype=np.int32)

    def _mock_build_dataset(
        csv_path, string_processor_config, classifier_out_dim, batch_size, **kwargs
    ):
        n = len(rg._read_csv_records(csv_path))
        x = tf.zeros((n, 10), dtype=tf.float32)
        y = tf.one_hot(true_labels[:n], depth=classifier_out_dim)
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

    monkeypatch.setattr(rg, "_build_inference_dataset", _mock_build_dataset)

    real_preds = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        rg,
        "_run_classifier_inference_streamed",
        _make_streamed_inference_mock(real_preds),
    )

    class FakeClassifier:
        pass

    result = rg.generate_reliability_data(
        classifier=FakeClassifier(),
        raw_csv_path=csv_path,
        output_dir=output_dir,
        string_processor_config={"crop_size": 100},
        model_cfg={"string_processor": {}},
        classifier_out_dim=2,
        reliability_out_dim=1,
        batch_size=2,
        id_threshold=0.8,
        synthetic_ood_threshold=0.8,
        synthetic_ood_multiplier=0.0,
        generator_cfg={
            "raw_csv_paths": {"train": csv_path},
            "perturbations": {"shuffle": False},
            "val_fraction": 0.5,
        },
    )

    assert (Path(output_dir) / "reliability_train.csv").exists()
    assert (Path(output_dir) / "reliability_val.csv").exists()
    assert result["train"]["paths"]
    assert result["validation"]["paths"]


def test_build_inference_dataset_produces_batched_dict_input(tmp_path: Path):
    """_build_inference_dataset pads dict inputs and infers default codon params."""
    csv_path = str(tmp_path / "seqs.csv")
    rg._write_csv(
        [(0, "ATCG" * 300), (1, "TGCA" * 300)],
        csv_path,
    )

    ds = rg._build_inference_dataset(
        csv_path,
        string_processor_config={"crop_size": 500, "seq_onehot": False},
        classifier_out_dim=3,
        batch_size=2,
    )

    assert ds.element_spec[0]["translated"].shape.as_list() == [None, 6, None]
    for x, y in ds:
        assert "translated" in x
        assert x["translated"].shape[0] == 2
        assert y.shape == (2, 3)
        break


def test_generate_reliability_data_skips_if_outputs_exist(monkeypatch, tmp_path: Path):
    """Existing reliability NPZs are reused instead of regenerating."""
    output_dir = str(tmp_path)
    train_npz = str(tmp_path / "reliability_train.npz")
    val_npz = str(tmp_path / "reliability_val.npz")
    Path(train_npz).touch()
    Path(val_npz).touch()

    def _raise(*args, **kwargs):
        raise AssertionError("Should not regenerate when outputs exist")

    monkeypatch.setattr(rg, "_build_inference_dataset", _raise)
    monkeypatch.setattr(rg, "_convert_to_npz", _raise)

    result = rg.generate_reliability_data(
        classifier=object(),
        raw_csv_path=str(tmp_path / "missing_train.csv"),
        output_dir=output_dir,
        string_processor_config={"crop_size": 100},
        model_cfg={"string_processor": {}},
        classifier_out_dim=2,
        reliability_out_dim=1,
        batch_size=2,
        generator_cfg={},
    )

    assert result["train"]["paths"] == [train_npz]
    assert result["validation"]["paths"] == [val_npz]


def test_convert_to_npz_derives_crop_size_from_crop_sizes(monkeypatch, tmp_path: Path):
    """_convert_to_npz falls back to max(crop_sizes) when crop_size is missing."""
    called = {}

    def _fake_convert_dataset(*, crop_size, **kwargs):
        called["crop_size"] = crop_size
        return None

    monkeypatch.setattr(rg, "convert_dataset", _fake_convert_dataset)

    rg._convert_to_npz(
        csv_path=str(tmp_path / "rel.csv"),
        npz_path=str(tmp_path / "rel.npz"),
        string_processor_config={
            "crop_sizes": [100, 200, 300, 400, 500, 600],
            "seq_onehot": False,
            "input_type": "translated",
        },
        reliability_out_dim=1,
        model_cfg={"string_processor": {}},
    )

    assert called["crop_size"] == 600


def test_convert_to_npz_uses_generator_cfg_crop_size(monkeypatch, tmp_path: Path):
    """_convert_to_npz prefers crop_size from generator_cfg over string_processor."""
    called = {}

    def _fake_convert_dataset(*, crop_size, **kwargs):
        called["crop_size"] = crop_size
        return None

    monkeypatch.setattr(rg, "convert_dataset", _fake_convert_dataset)

    rg._convert_to_npz(
        csv_path=str(tmp_path / "rel.csv"),
        npz_path=str(tmp_path / "rel.npz"),
        string_processor_config={
            "crop_sizes": [100, 200, 300, 400, 500, 600],
            "seq_onehot": False,
            "input_type": "translated",
        },
        reliability_out_dim=1,
        model_cfg={"string_processor": {}},
        generator_cfg={"crop_size": 250},
    )

    assert called["crop_size"] == 250


def test_convert_to_npz_converts_codon_units(monkeypatch, tmp_path: Path):
    """generator_cfg.units='codon' multiplies crop_size by 3."""
    called = {}

    def _fake_convert_dataset(*, crop_size, **kwargs):
        called["crop_size"] = crop_size
        return None

    monkeypatch.setattr(rg, "convert_dataset", _fake_convert_dataset)

    rg._convert_to_npz(
        csv_path=str(tmp_path / "rel.csv"),
        npz_path=str(tmp_path / "rel.npz"),
        string_processor_config={"crop_size": 100},
        reliability_out_dim=1,
        model_cfg={"string_processor": {}},
        generator_cfg={"crop_size": 250, "units": "codon"},
    )

    assert called["crop_size"] == 750


def test_build_inference_dataset_uses_passed_crop_size(tmp_path: Path):
    """_build_inference_dataset crops sequences to the provided crop_size."""
    csv_path = str(tmp_path / "seqs.csv")
    rg._write_csv(
        [(0, "ATCG" * 300), (1, "TGCA" * 300)],
        csv_path,
    )

    crop_size = 99
    ds = rg._build_inference_dataset(
        csv_path,
        string_processor_config={"seq_onehot": False, "input_type": "translated"},
        classifier_out_dim=3,
        batch_size=2,
        crop_size=crop_size,
    )

    for x, _ in ds:
        assert x["translated"].shape[2] <= crop_size // 3
        break


def test_generate_reliability_data_adds_synthetic_ood_to_validation(
    monkeypatch, tmp_path: Path
):
    """When a separate val CSV is provided, synthetic OOD is also added to validation."""
    import tensorflow as tf

    train_csv = str(tmp_path / "train.csv")
    val_csv = str(tmp_path / "val.csv")
    rg._write_csv([(0, "ATCG" * 50), (1, "TGCA" * 50)], train_csv)
    rg._write_csv([(0, "AAAA" * 50), (1, "TTTT" * 50)], val_csv)
    output_dir = str(tmp_path / "rel_out")

    def _mock_build_dataset(*args, **kwargs):
        records = rg._read_csv_records(args[0])
        n = len(records)
        labels = np.array([label for label, _ in records], dtype=np.int32)
        x = tf.zeros((n, 10), dtype=tf.float32)
        y = tf.one_hot(labels, depth=2)
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    monkeypatch.setattr(rg, "_build_inference_dataset", _mock_build_dataset)

    real_preds = np.tile(np.array([0.9, 0.1], dtype=np.float32), (2, 1))
    monkeypatch.setattr(
        rg,
        "_run_classifier_inference_streamed",
        _make_streamed_inference_mock(real_preds),
    )

    def _mock_generate_synthetic(
        records, multiplier, perturbations_cfg, crop_size=None, **kwargs
    ):
        return iter(["SYNTHETIC_SEQ"] * int(len(records) * multiplier))

    monkeypatch.setattr(rg, "generate_synthetic_sequences", _mock_generate_synthetic)

    def _mock_filter_synthetic_ood(*args, **kwargs):
        return [(0, seq) for seq in args[1]]

    monkeypatch.setattr(rg, "_filter_synthetic_ood", _mock_filter_synthetic_ood)

    class FakeClassifier:
        pass

    rg.generate_reliability_data(
        classifier=FakeClassifier(),
        raw_csv_path=train_csv,
        output_dir=output_dir,
        string_processor_config={"crop_size": 100},
        model_cfg={"string_processor": {}},
        classifier_out_dim=2,
        reliability_out_dim=1,
        batch_size=2,
        synthetic_ood_multiplier=1.0,
        generator_cfg={
            "raw_csv_paths": {"train": train_csv, "val": val_csv},
            "perturbations": {"shuffle": False},
        },
    )

    val_written = rg._read_csv_records(str(Path(output_dir) / "reliability_val.csv"))
    train_written = rg._read_csv_records(
        str(Path(output_dir) / "reliability_train.csv")
    )

    assert any(seq == "SYNTHETIC_SEQ" for _, seq in val_written)
    assert any(seq == "SYNTHETIC_SEQ" for _, seq in train_written)


def test_normalize_perturbation_cfg_mix_boolean():
    cfg = {
        "shuffle": False,
        "subseq_repeat": False,
        "tandem_repeat": False,
        "mix": True,
    }
    specs = sp._normalize_perturbation_cfg(cfg)
    assert len(specs) == 1
    assert specs[0]["name"] == "mix"
    assert specs[0]["n_segments"] == 2


def test_normalize_perturbation_cfg_mix_structured():
    cfg = {
        "shuffle": False,
        "subseq_repeat": False,
        "tandem_repeat": False,
        "mix": {"enabled": True, "n_segments": 3},
    }
    specs = sp._normalize_perturbation_cfg(cfg)
    assert len(specs) == 1
    assert specs[0]["name"] == "mix"
    assert specs[0]["n_segments"] == 3


def test_generate_synthetic_sequences_mix_requires_distinct_classes():
    records = [(0, "A" * 100)] * 5
    with pytest.raises(ValueError, match="mix perturbation requires at least 2"):
        list(
            sp.generate_synthetic_sequences(
                records,
                multiplier=1.0,
                perturbations_cfg={
                    "shuffle": False,
                    "subseq_repeat": False,
                    "tandem_repeat": False,
                    "mix": {"enabled": True, "n_segments": 2},
                },
                crop_size=50,
            )
        )


def test_generate_synthetic_sequences_mix_produces_chimera(monkeypatch):
    records = [(0, "A" * 100), (1, "C" * 100)]
    _original_sample = random.sample

    def _patched_sample(population, k):
        # Fix the interior cut in apply_mix; delegate label sampling normally.
        if isinstance(population, range):
            return [25]
        return _original_sample(population, k)

    monkeypatch.setattr("jaeger.seqops.synthetic.random.sample", _patched_sample)
    seqs = list(
        sp.generate_synthetic_sequences(
            records,
            multiplier=1.0,
            perturbations_cfg={
                "shuffle": False,
                "subseq_repeat": False,
                "tandem_repeat": False,
                "mix": {"enabled": True, "n_segments": 2},
            },
            crop_size=50,
        )
    )
    assert len(seqs) == 2  # 1.0 * 2 records
    for seq in seqs:
        assert len(seq) == 50
        assert "A" in seq
        assert "C" in seq


def test_generate_synthetic_sequences_bounded_chunks():
    """Sub-chunking must not change the total number or length of outputs."""
    records = [(0, "ATCG" * 10), (1, "TGCA" * 10)]
    seqs = list(
        sp.generate_synthetic_sequences(
            records,
            multiplier=2.0,
            perturbations_cfg={"shuffle": {"enabled": True, "mode": "random"}},
            generation_chunk_size=2,
        )
    )
    assert len(seqs) == 4  # 2 records * 2.0
    assert all(len(s) == len(records[0][1]) for s in seqs)
