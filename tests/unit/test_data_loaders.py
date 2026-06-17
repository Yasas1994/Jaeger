"""Tests for jaeger.data.loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from jaeger.data import loaders


NUM_SAMPLES = 4
SEQ_LEN = 16
CODON_DEPTH = 65  # len(CODON_ID) + 1
NUM_CLASSES = 3


@pytest.fixture
def translated_integer_npz(tmp_path: Path) -> str:
    path = tmp_path / "translated_int.npz"
    translated = np.random.randint(
        0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
    )
    labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
    np.savez(path, translated=translated, labels=labels, codon_map="codon_id")
    return str(path)


@pytest.fixture
def translated_onehot_npz(tmp_path: Path) -> str:
    path = tmp_path / "translated_oh.npz"
    translated = np.eye(CODON_DEPTH, dtype=np.float32)[
        np.random.randint(0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN))
    ]
    labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
    np.savez(path, translated=translated, labels=labels, codon_map="codon_id")
    return str(path)


@pytest.fixture
def nucleotide_integer_npz(tmp_path: Path) -> str:
    path = tmp_path / "nucleotide_int.npz"
    nucleotide = np.random.randint(0, 5, size=(NUM_SAMPLES, 2, SEQ_LEN), dtype=np.int32)
    labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
    np.savez(
        path,
        nucleotide=nucleotide,
        labels=labels,
        nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
    )
    return str(path)


@pytest.fixture
def nucleotide_onehot_npz(tmp_path: Path) -> str:
    path = tmp_path / "nucleotide_oh.npz"
    nucleotide = np.eye(4, dtype=np.float32)[
        np.random.randint(0, 4, size=(NUM_SAMPLES, 2, SEQ_LEN))
    ]
    labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
    np.savez(path, nucleotide=nucleotide, labels=labels)
    return str(path)


@pytest.fixture
def both_npz(tmp_path: Path) -> str:
    path = tmp_path / "both.npz"
    translated = np.random.randint(
        0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
    )
    nucleotide = np.random.randint(0, 5, size=(NUM_SAMPLES, 2, SEQ_LEN), dtype=np.int32)
    labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
    np.savez(
        path,
        translated=translated,
        nucleotide=nucleotide,
        labels=labels,
        codon_map="codon_id",
        nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
    )
    return str(path)


class TestLoadNumpyDataset:
    def test_translated_integer(self, translated_integer_npz: str):
        ds = loaders._load_numpy_dataset(
            translated_integer_npz,
            input_type="translated",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        features, label = next(iter(ds))
        assert "translated" in features
        assert features["translated"].shape == (6, SEQ_LEN)
        assert features["translated"].dtype == tf.int32
        assert label.shape == (NUM_CLASSES,)
        assert label.dtype == tf.float32

    def test_translated_integer_to_onehot(self, translated_integer_npz: str):
        ds = loaders._load_numpy_dataset(
            translated_integer_npz,
            input_type="translated",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert features["translated"].shape == (6, SEQ_LEN, CODON_DEPTH)
        assert features["translated"].dtype == tf.float32

    def test_translated_onehot_unchanged(self, translated_onehot_npz: str):
        ds = loaders._load_numpy_dataset(
            translated_onehot_npz,
            input_type="translated",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert features["translated"].shape == (6, SEQ_LEN, CODON_DEPTH)
        assert features["translated"].dtype == tf.float32

    def test_nucleotide_integer_default_onehot(self, nucleotide_integer_npz: str):
        ds = loaders._load_numpy_dataset(
            nucleotide_integer_npz,
            input_type="nucleotide",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert "nucleotide" in features
        assert features["nucleotide"].shape == (2, SEQ_LEN, 4)
        assert features["nucleotide"].dtype == tf.float32

    def test_nucleotide_integer_custom_onehot(self, tmp_path: Path):
        path = tmp_path / "nucleotide_custom.npz"
        # Controlled input: first position of the forward strand is token for A (1).
        nucleotide = np.zeros((1, 2, SEQ_LEN), dtype=np.int32)
        nucleotide[0, 0, 0] = 1  # A token
        labels = np.array([0], dtype=np.int32)
        np.savez(
            path,
            nucleotide=nucleotide,
            labels=labels,
            nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
        )

        custom_map = {
            "A": [0.0, 1.0, 0.0, 0.0],
            "G": [1.0, 0.0, 0.0, 0.0],
            "T": [0.0, 0.0, 0.0, 1.0],
            "C": [0.0, 0.0, 1.0, 0.0],
            "N": [0.0, 0.0, 0.0, 0.0],
        }
        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="nucleotide",
            seq_onehot=True,
            nucleotide_onehot_map=custom_map,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert features["nucleotide"].shape == (2, SEQ_LEN, 4)
        # Verify custom mapping: token for A (1) maps to [0,1,0,0].
        assert np.allclose(
            features["nucleotide"].numpy()[0, 0, :], [0.0, 1.0, 0.0, 0.0]
        )

    def test_nucleotide_onehot_unchanged(self, nucleotide_onehot_npz: str):
        ds = loaders._load_numpy_dataset(
            nucleotide_onehot_npz,
            input_type="nucleotide",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert features["nucleotide"].shape == (2, SEQ_LEN, 4)
        assert features["nucleotide"].dtype == tf.float32

    def test_both_input_type(self, both_npz: str):
        ds = loaders._load_numpy_dataset(
            both_npz,
            input_type="both",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert "translated" in features
        assert "nucleotide" in features
        assert features["translated"].shape == (6, SEQ_LEN, CODON_DEPTH)
        assert features["nucleotide"].shape == (2, SEQ_LEN, 4)

    def test_integer_labels_to_onehot(self, translated_integer_npz: str):
        ds = loaders._load_numpy_dataset(
            translated_integer_npz,
            input_type="translated",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
            one_hot_labels=True,
        )
        _, label = next(iter(ds))
        assert label.shape == (NUM_CLASSES,)
        assert label.dtype == tf.float32
        assert float(tf.reduce_sum(label)) == pytest.approx(1.0)

    def test_label_backward_compat(self, tmp_path: Path):
        path = tmp_path / "legacy.npz"
        translated = np.random.randint(
            0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
        np.savez(path, translated=translated, label=labels, codon_map="codon_id")
        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        _, label = next(iter(ds))
        assert label.shape == (NUM_CLASSES,)

    def test_binary_labels_num_classes_one(self, tmp_path: Path):
        """Binary reliability heads need scalar float labels, not one-hot."""
        path = tmp_path / "binary.npz"
        translated = np.random.randint(
            0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        np.savez(path, translated=translated, labels=labels, codon_map="codon_id")
        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=1,
            one_hot_labels=True,
        )
        _, label = next(iter(ds))
        assert label.shape == (1,)
        assert label.dtype == tf.float32
        assert set(label.numpy().tolist()).issubset({0.0, 1.0})

    def test_translated_integer_to_onehot_batchwise(self, translated_integer_npz: str):
        ds_batch = loaders._load_numpy_dataset(
            translated_integer_npz,
            input_type="translated",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
            buffer_size=2,
        )
        features, _ = next(iter(ds_batch))
        assert features["translated"].shape == (6, SEQ_LEN, CODON_DEPTH)
        assert features["translated"].dtype == tf.float32

    def test_nucleotide_integer_default_onehot_batchwise(
        self, nucleotide_integer_npz: str
    ):
        ds_full = loaders._load_numpy_dataset(
            nucleotide_integer_npz,
            input_type="nucleotide",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )
        ds_batch = loaders._load_numpy_dataset(
            nucleotide_integer_npz,
            input_type="nucleotide",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
            buffer_size=2,
        )
        full_features, _ = next(iter(ds_full))
        batch_features, _ = next(iter(ds_batch))
        assert batch_features["nucleotide"].shape == full_features["nucleotide"].shape
        assert np.allclose(
            batch_features["nucleotide"].numpy(),
            full_features["nucleotide"].numpy(),
        )

    def test_nucleotide_integer_custom_onehot_batchwise(self, tmp_path: Path):
        path = tmp_path / "nucleotide_custom_batch.npz"
        nucleotide = np.zeros((NUM_SAMPLES, 2, SEQ_LEN), dtype=np.int32)
        nucleotide[:, 0, 0] = 1  # A token
        labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
        np.savez(
            path,
            nucleotide=nucleotide,
            labels=labels,
            nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
        )

        custom_map = {
            "A": [0.0, 1.0, 0.0, 0.0],
            "G": [1.0, 0.0, 0.0, 0.0],
            "T": [0.0, 0.0, 0.0, 1.0],
            "C": [0.0, 0.0, 1.0, 0.0],
            "N": [0.0, 0.0, 0.0, 0.0],
        }
        ds_batch = loaders._load_numpy_dataset(
            str(path),
            input_type="nucleotide",
            seq_onehot=True,
            nucleotide_onehot_map=custom_map,
            num_classes=NUM_CLASSES,
            buffer_size=2,
        )
        features, _ = next(iter(ds_batch))
        assert features["nucleotide"].shape == (2, SEQ_LEN, 4)
        assert np.allclose(
            features["nucleotide"].numpy()[0, 0, :], [0.0, 1.0, 0.0, 0.0]
        )

    def test_both_input_type_batchwise(self, both_npz: str):
        ds_batch = loaders._load_numpy_dataset(
            both_npz,
            input_type="both",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
            buffer_size=2,
        )
        features, _ = next(iter(ds_batch))
        assert "translated" in features
        assert "nucleotide" in features
        assert features["translated"].shape == (6, SEQ_LEN, CODON_DEPTH)
        assert features["nucleotide"].shape == (2, SEQ_LEN, 4)


@pytest.fixture
def ragged_nucleotide_npz(tmp_path: Path) -> str:
    path = tmp_path / "ragged_nuc.npz"
    crops = [
        np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32),
        np.array([[1, 2], [2, 1]], dtype=np.int32),
    ]
    arr = np.empty(len(crops), dtype=object)
    arr[:] = crops
    labels = np.array([0, 1], dtype=np.int32)
    np.savez(
        path,
        nucleotide=arr,
        labels=labels,
        lengths=np.array([4, 2], dtype=np.int32),
        translated_lengths=np.array([0, 0], dtype=np.int32),
        nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
        crop_sizes=np.array([4], dtype=np.int32),
        strides=np.array([0], dtype=np.int32),
        pad_int=np.int32(0),
        padded=np.bool_(False),
    )
    return str(path)


class TestRaggedLoaders:
    def test_ragged_nucleotide_integer(self, ragged_nucleotide_npz: str):
        ds = loaders._load_numpy_dataset(
            ragged_nucleotide_npz,
            input_type="nucleotide",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        features, label = next(iter(ds))
        assert features["nucleotide"].shape == (2, 4)
        assert label.shape == (NUM_CLASSES,)

    def test_ragged_padded_batch(self, ragged_nucleotide_npz: str):
        ds = loaders._load_numpy_dataset(
            ragged_nucleotide_npz,
            input_type="nucleotide",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        batched = ds.padded_batch(
            2, padded_shapes=({"nucleotide": [2, None]}, [NUM_CLASSES])
        )
        features, labels = next(iter(batched))
        assert features["nucleotide"].shape == (2, 2, 4)

    def test_ragged_nucleotide_integer_to_onehot(self, ragged_nucleotide_npz: str):
        ds = loaders._load_numpy_dataset(
            ragged_nucleotide_npz,
            input_type="nucleotide",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert features["nucleotide"].shape == (2, 4, 4)
        assert features["nucleotide"].dtype == tf.float32

    def test_ragged_translated_integer_to_onehot(self, tmp_path: Path):
        path = tmp_path / "ragged_trans.npz"
        crops = [
            np.array(
                [
                    [1, 2, 3, 4],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            np.array([[1, 2], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        ]
        arr = np.empty(len(crops), dtype=object)
        arr[:] = crops
        labels = np.array([0, 1], dtype=np.int32)
        np.savez(
            path,
            translated=arr,
            labels=labels,
            lengths=np.array([0, 0], dtype=np.int32),
            translated_lengths=np.array([4, 2], dtype=np.int32),
            codon_map="codon_id",
            crop_sizes=np.array([4], dtype=np.int32),
            strides=np.array([0], dtype=np.int32),
            pad_int=np.int32(0),
            padded=np.bool_(False),
        )
        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )
        features, _ = next(iter(ds))
        assert features["translated"].shape == (6, 4, 65)
        assert features["translated"].dtype == tf.float32
