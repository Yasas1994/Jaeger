"""Smoke tests for the unified NumPy NPZ data format loading."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from jaeger.data.loaders import _load_numpy_dataset

# Test configuration matching typical Jaeger training setup
CROP_SIZE = 500
SEQ_LEN = CROP_SIZE // 3 - 1  # 165
NUM_CLASSES = 3
CODON_DEPTH = 65  # len(CODON_ID) + 1
BATCH_SIZE = 4
NUM_SAMPLES = 10


def create_test_csv(path, num_samples=10):
    """Create a small CSV file with random DNA sequences for testing."""
    bases = ["A", "T", "G", "C"]
    with open(path, "w") as f:
        for i in range(num_samples):
            label = i % NUM_CLASSES
            seq = "".join(np.random.choice(bases, size=CROP_SIZE))
            f.write(f"{label},{seq}\n")


def test_load_translated_integer_npz():
    """Test loading integer-encoded translated NPZ data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test.npz")
        sequences = np.random.randint(
            0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
        np.savez_compressed(
            npz_path, translated=sequences, labels=labels, codon_map="codon_id"
        )

        dataset = _load_numpy_dataset(
            npz_path,
            input_type="translated",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )

        count = 0
        for features, label in dataset:
            assert "translated" in features
            assert features["translated"].shape == (6, SEQ_LEN)
            assert features["translated"].dtype == tf.int32
            assert label.shape == (NUM_CLASSES,)
            assert label.dtype == tf.float32
            count += 1

        assert count == NUM_SAMPLES
        print(f"✓ Translated integer NPZ load: {count} samples, shapes correct")


def test_load_translated_integer_to_onehot():
    """Test converting integer translated NPZ to one-hot on load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_onehot.npz")
        sequences = np.random.randint(
            0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
        np.savez_compressed(
            npz_path, translated=sequences, labels=labels, codon_map="codon_id"
        )

        dataset = _load_numpy_dataset(
            npz_path,
            input_type="translated",
            seq_onehot=True,
            num_classes=NUM_CLASSES,
        )

        count = 0
        for features, label in dataset:
            assert features["translated"].shape == (6, SEQ_LEN, CODON_DEPTH)
            assert features["translated"].dtype == tf.float32
            assert label.shape == (NUM_CLASSES,)
            count += 1

        assert count == NUM_SAMPLES
        print(
            f"✓ Translated integer -> one-hot NPZ load: {count} samples, shapes correct"
        )


def test_load_nucleotide_integer_with_custom_onehot():
    """Test loading integer-encoded nucleotide NPZ with custom one-hot mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_nuc.npz")
        nucleotide = np.random.randint(
            0, 5, size=(NUM_SAMPLES, 2, CROP_SIZE), dtype=np.int32
        )
        labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
        custom_onehot_map = {
            "A": [0.0, 1.0, 0.0, 0.0],
            "G": [1.0, 0.0, 0.0, 0.0],
            "T": [0.0, 0.0, 0.0, 1.0],
            "C": [0.0, 0.0, 1.0, 0.0],
            "N": [0.0, 0.0, 0.0, 0.0],
        }
        np.savez_compressed(
            npz_path,
            nucleotide=nucleotide,
            labels=labels,
            nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
        )

        dataset = _load_numpy_dataset(
            npz_path,
            input_type="nucleotide",
            seq_onehot=True,
            nucleotide_onehot_map=custom_onehot_map,
            num_classes=NUM_CLASSES,
        )

        count = 0
        for features, label in dataset:
            assert "nucleotide" in features
            assert features["nucleotide"].shape == (2, CROP_SIZE, 4)
            assert features["nucleotide"].dtype == tf.float32
            assert label.shape == (NUM_CLASSES,)
            count += 1

        assert count == NUM_SAMPLES
        print(f"✓ Nucleotide integer with custom one-hot load: {count} samples")


def test_dataset_pipeline_with_batching():
    """Test that loaded datasets work correctly with batching and prefetching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test.npz")
        sequences = np.random.randint(
            0, CODON_DEPTH, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
        np.savez_compressed(
            npz_path, translated=sequences, labels=labels, codon_map="codon_id"
        )

        dataset = (
            _load_numpy_dataset(
                npz_path,
                input_type="translated",
                seq_onehot=False,
                num_classes=NUM_CLASSES,
            )
            .cache()
            .shuffle(buffer_size=100)
            .padded_batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        for features, label in dataset:
            assert features["translated"].shape == (BATCH_SIZE, 6, SEQ_LEN)
            assert label.shape == (BATCH_SIZE, NUM_CLASSES)
            break

        print("✓ Dataset pipeline with batching works correctly")


if __name__ == "__main__":
    print("Running data format smoke tests...\n")

    test_load_translated_integer_npz()
    test_load_translated_integer_to_onehot()
    test_load_nucleotide_integer_with_custom_onehot()
    test_dataset_pipeline_with_batching()

    print("\n✅ All data format tests passed!")
