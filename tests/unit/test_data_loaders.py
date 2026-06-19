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


def test_convert_and_load_unpadded(simple_csv_path: str, tmp_path: Path):
    from jaeger.dataops import convert

    out = tmp_path / "unpadded.npz"
    convert.convert_dataset(
        input_path=simple_csv_path,
        output_path=str(out),
        format="nucleotide",
        crop_size=24,
        num_classes=2,
        num_workers=1,
        pad=False,
    )
    ds = loaders._load_numpy_dataset(
        str(out),
        input_type="nucleotide",
        seq_onehot=False,
        num_classes=2,
    )
    features, label = next(iter(ds))
    assert features["nucleotide"].shape[1] <= 24
    assert label.shape == (2,)


class TestResolveStrides:
    def test_explicit_strides(self):
        assert loaders._resolve_strides([300, 600], [270, 540], None) == [270, 540]

    def test_overlap(self):
        assert loaders._resolve_strides([300, 600], None, 0.1) == [270, 540]

    def test_default_strides(self):
        assert loaders._resolve_strides([300, 600], None, None) == [300, 600]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            loaders._resolve_strides([300, 600], [270], None)


class TestRuntimeCropLoader:
    def test_translated_integer_crops_match_manual_slices(self, tmp_path: Path):
        path = tmp_path / "translated_crops.npz"
        seq_len = 20
        translated = np.arange(6 * seq_len, dtype=np.int32).reshape(1, 6, seq_len) + 1
        labels = np.array([1], dtype=np.int32)
        translated_lengths = np.array([seq_len], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=2,
            crop_sizes=[10, 20],
            overlap=0.0,
        )
        crops = list(ds.as_numpy_iterator())
        assert len(crops) == 3  # two 10-length + one 20-length

        # Every crop is padded to the max crop size (20).
        feats, label = crops[0]
        assert feats["translated"].shape == (6, 20)
        np.testing.assert_array_equal(
            feats["translated"][:, :10], translated[0, :, :10]
        )
        np.testing.assert_array_equal(feats["translated"][:, 10:], 0)
        feats, label = crops[1]
        assert feats["translated"].shape == (6, 20)
        np.testing.assert_array_equal(
            feats["translated"][:, :10], translated[0, :, 10:20]
        )
        np.testing.assert_array_equal(feats["translated"][:, 10:], 0)
        feats, label = crops[2]
        assert feats["translated"].shape == (6, 20)
        np.testing.assert_array_equal(feats["translated"], translated[0, :, :20])
        np.testing.assert_array_equal(label, np.array([0.0, 1.0]))

    def test_tail_crop_trimmed_to_actual_length(self, tmp_path: Path):
        path = tmp_path / "translated_tail.npz"
        seq_len = 20
        translated = np.arange(6 * seq_len, dtype=np.int32).reshape(1, 6, seq_len) + 1
        labels = np.array([2], dtype=np.int32)
        # actual sequence is only 13 codons long
        translated_lengths = np.array([13], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=3,
            crop_sizes=[10],
            overlap=0.0,
        )
        crops = list(ds.as_numpy_iterator())
        assert len(crops) == 2
        assert crops[0][0]["translated"].shape == (6, 10)
        # Tail crop starts at seq_len - crop_size so it stays 10-long.
        assert crops[1][0]["translated"].shape == (6, 10)
        np.testing.assert_array_equal(crops[1][0]["translated"], translated[0, :, 3:13])

    def test_translated_onehot_conversion(self, tmp_path: Path):
        path = tmp_path / "translated_oh_crops.npz"
        seq_len = 12
        translated = np.random.randint(1, 65, size=(1, 6, seq_len), dtype=np.int32)
        labels = np.array([0], dtype=np.int32)
        translated_lengths = np.array([seq_len], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=True,
            num_classes=2,
            crop_sizes=[6],
            overlap=0.0,
        )
        features, label = next(iter(ds))
        assert features["translated"].shape == (6, 6, 65)
        assert features["translated"].dtype == tf.float32

    def test_nucleotide_onehot_crops(self, tmp_path: Path):
        path = tmp_path / "nucleotide_oh_crops.npz"
        seq_len = 16
        nucleotide = np.random.randint(0, 4, size=(1, 2, seq_len), dtype=np.int32)
        labels = np.array([1], dtype=np.int32)
        lengths = np.array([seq_len], dtype=np.int32)
        np.savez(
            path,
            nucleotide=nucleotide,
            labels=labels,
            lengths=lengths,
            nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="nucleotide",
            seq_onehot=True,
            num_classes=2,
            crop_sizes=[8, 16],
            overlap=0.0,
        )
        crops = list(ds.as_numpy_iterator())
        assert len(crops) == 3
        # All crops are padded to the max crop size.
        assert crops[0][0]["nucleotide"].shape == (2, 16, 4)
        assert crops[1][0]["nucleotide"].shape == (2, 16, 4)
        assert crops[2][0]["nucleotide"].shape == (2, 16, 4)
        np.testing.assert_array_equal(crops[0][0]["nucleotide"][:, 8:, :], 0)

    def test_overlapping_crops(self, tmp_path: Path):
        path = tmp_path / "translated_overlap.npz"
        seq_len = 30
        translated = np.random.randint(1, 65, size=(1, 6, seq_len), dtype=np.int32)
        labels = np.array([0], dtype=np.int32)
        translated_lengths = np.array([seq_len], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=2,
            crop_sizes=[10],
            overlap=0.5,
        )
        crops = list(ds.as_numpy_iterator())
        # stride = 5, starts [0, 5, 10, 15, 20]
        assert len(crops) == 5

    def test_runtime_crops_padded_to_max_size(self, tmp_path: Path):
        """All runtime crops share the same fixed length so cardinality/shape is known."""
        path = tmp_path / "translated_padded_crops.npz"
        seq_len = 20
        translated = np.arange(6 * seq_len, dtype=np.int32).reshape(1, 6, seq_len) + 1
        labels = np.array([1], dtype=np.int32)
        translated_lengths = np.array([seq_len], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=2,
            crop_sizes=[10, 20],
            overlap=0.0,
        )
        crops = list(ds.as_numpy_iterator())
        assert len(crops) == 3
        # Every crop is padded to the largest crop size.
        for feats, _label in crops:
            assert feats["translated"].shape == (6, 20)
        # Shorter crops have token-0 padding after the real sequence.
        np.testing.assert_array_equal(crops[0][0]["translated"][:, 10:], 0)
        np.testing.assert_array_equal(crops[1][0]["translated"][:, 10:], 0)
        # The largest crop is unpadded.
        np.testing.assert_array_equal(crops[2][0]["translated"], translated[0, :, :20])

    def test_runtime_crops_keep_natural_length(self, tmp_path: Path):
        path = tmp_path / "translated_natural.npz"
        seq_len = 20
        translated = np.arange(6 * seq_len, dtype=np.int32).reshape(1, 6, seq_len) + 1
        labels = np.array([1], dtype=np.int32)
        translated_lengths = np.array([seq_len], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=2,
            crop_sizes=[10, 20],
            overlap=0.0,
            pad_to_max=False,
        )
        crops = list(ds.as_numpy_iterator())
        assert len(crops) == 3
        assert crops[0][0]["translated"].shape == (6, 10)
        assert crops[1][0]["translated"].shape == (6, 10)
        assert crops[2][0]["translated"].shape == (6, 20)

    def test_int8_npz_cast_to_int32(self, tmp_path: Path):
        path = tmp_path / "translated_int8.npz"
        translated = np.random.randint(1, 65, size=(2, 6, 12), dtype=np.int8)
        labels = np.array([0, 1], dtype=np.int32)
        np.savez(path, translated=translated, labels=labels, codon_map="codon_id")

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=2,
        )
        features, _ = next(iter(ds))
        assert features["translated"].dtype == tf.int32

    def test_object_array_runtime_crops(self, tmp_path: Path):
        path = tmp_path / "translated_obj.npz"
        items = [
            np.arange(6 * 20, dtype=np.int32).reshape(6, 20) + 1,
            np.arange(6 * 15, dtype=np.int32).reshape(6, 15) + 1,
        ]
        translated = np.empty(2, dtype=object)
        translated[:] = items
        labels = np.array([0, 1], dtype=np.int32)
        translated_lengths = np.array([20, 15], dtype=np.int32)
        np.savez(
            path,
            translated=translated,
            labels=labels,
            translated_lengths=translated_lengths,
            codon_map="codon_id",
        )

        ds = loaders._load_numpy_dataset(
            str(path),
            input_type="translated",
            seq_onehot=False,
            num_classes=2,
            crop_sizes=[10],
            overlap=0.0,
        )
        crops = list(ds.as_numpy_iterator())
        assert len(crops) == 4  # 2 crops from seq 0, 2 from seq 1
        assert crops[0][0]["translated"].shape == (6, 10)
        np.testing.assert_array_equal(crops[0][0]["translated"], items[0][:, :10])
        # Tail crop for length 15 starts at 5 so it stays 10-long.
        np.testing.assert_array_equal(crops[-1][0]["translated"], items[1][:, 5:15])
