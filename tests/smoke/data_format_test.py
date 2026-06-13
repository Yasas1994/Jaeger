"""Smoke tests for TFRecord and NumPy data format loading in training."""

import os
import tempfile
import numpy as np
import tensorflow as tf

from jaeger.data.tfrecord import (
    _make_parse_tfrecord_fn,
    _get_tfrecord_feature_description,
)
from jaeger.data.loaders import _load_numpy_full_dataset
from jaeger.seqops.encode import process_string_train
from jaeger.seqops.maps import CODONS, CODON_ID

# Test configuration matching typical Jaeger training setup
CROP_SIZE = 500
SEQ_LEN = CROP_SIZE // 3 - 1  # 165
NUM_CLASSES = 3
CODON_DEPTH = 21
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


def test_tfrecord_feature_description():
    """Test that feature descriptions are generated correctly."""
    # Embedding layer (int64)
    desc = _get_tfrecord_feature_description(
        input_type="translated",
        use_embedding_layer=True,
        codon_depth=CODON_DEPTH,
        crop_size=CROP_SIZE,
        num_classes=NUM_CLASSES,
    )
    assert "translated" in desc
    assert "label" in desc
    print("✓ TFRecord feature description (embedding) correct")

    # One-hot (float)
    desc = _get_tfrecord_feature_description(
        input_type="translated",
        use_embedding_layer=False,
        codon_depth=CODON_DEPTH,
        crop_size=CROP_SIZE,
        num_classes=NUM_CLASSES,
    )
    assert "translated" in desc
    assert "label" in desc
    print("✓ TFRecord feature description (one-hot) correct")


def test_parse_tfrecord_embedding():
    """Test parsing TFRecord examples with embedding layer (int indices)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        tfrecord_path = os.path.join(tmpdir, "test.tfrecord")
        create_test_csv(csv_path, NUM_SAMPLES)

        # Preprocess and write TFRecord
        preprocess_fn = process_string_train(
            crop_size=CROP_SIZE,
            seq_onehot=False,
            input_type="translated",
            class_label_onehot=True,
            num_classes=NUM_CLASSES,
            shuffle=False,
            ngram_width=3,
            codons=CODONS,
            codon_num=CODON_ID,
        )

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            with open(csv_path) as f:
                for line in f:
                    outputs, label = preprocess_fn(line.strip().encode())
                    translated = outputs["translated"]
                    translated_flat = tf.reshape(translated, [-1])
                    feature = {
                        "translated": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=translated_flat.numpy().tolist()
                            )
                        ),
                        "label": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=label.numpy().flatten().tolist()
                            )
                        ),
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example.SerializeToString())

        # Parse back
        parse_fn = _make_parse_tfrecord_fn(
            input_type="translated",
            use_embedding_layer=True,
            codon_depth=CODON_DEPTH,
            crop_size=CROP_SIZE,
            num_classes=NUM_CLASSES,
        )

        dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_fn)
        count = 0
        for features, label in dataset:
            assert "translated" in features
            assert features["translated"].shape == (6, SEQ_LEN)
            assert features["translated"].dtype == tf.int32
            assert label.shape == (NUM_CLASSES,)
            assert label.dtype == tf.float32
            count += 1

        assert count == NUM_SAMPLES
        print(f"✓ TFRecord embedding parse: {count} samples, shapes correct")


def test_load_numpy_full_dataset_with_legacy_params():
    """Test loading NumPy .npz files with legacy params (backward compatibility)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        npz_path = os.path.join(tmpdir, "test.npz")
        create_test_csv(csv_path, NUM_SAMPLES)

        # Preprocess and save as NumPy
        preprocess_fn = process_string_train(
            crop_size=CROP_SIZE,
            seq_onehot=False,
            input_type="translated",
            class_label_onehot=True,
            num_classes=NUM_CLASSES,
            shuffle=False,
            ngram_width=3,
            codons=CODONS,
            codon_num=CODON_ID,
        )

        sequences = np.zeros((NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32)
        labels = np.zeros((NUM_SAMPLES, NUM_CLASSES), dtype=np.float32)

        with open(csv_path) as f:
            for i, line in enumerate(f):
                outputs, label = preprocess_fn(line.strip().encode())
                sequences[i] = outputs["translated"].numpy()
                labels[i] = label.numpy()

        np.savez_compressed(npz_path, translated=sequences, label=labels)

        # Load back with legacy params
        dataset = _load_numpy_full_dataset(
            npz_path,
            input_type="translated",
            use_embedding_layer=True,
            codon_depth=CODON_DEPTH,
            crop_size=CROP_SIZE,
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
        print(f"✓ NumPy full load (legacy params): {count} samples, shapes correct")


def test_numpy_flattened_format():
    """Test loading NumPy .npz with flattened sequences (as saved by converter)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_flat.npz")

        # Create flattened data (as the conversion script produces)
        sequences = np.random.randint(
            0, 21, size=(NUM_SAMPLES, 6 * SEQ_LEN), dtype=np.int32
        )
        labels = np.random.rand(NUM_SAMPLES, NUM_CLASSES).astype(np.float32)

        np.savez_compressed(npz_path, translated=sequences, label=labels)

        # Load back
        dataset = _load_numpy_full_dataset(
            npz_path,
            input_type="translated",
            use_embedding_layer=True,
            codon_depth=CODON_DEPTH,
            crop_size=CROP_SIZE,
        )

        count = 0
        for features, label in dataset:
            assert features["translated"].shape == (6, SEQ_LEN)
            count += 1

        assert count == NUM_SAMPLES
        print(f"✓ NumPy full flattened load: {count} samples, reshaped correctly")


def test_dataset_pipeline_with_batching():
    """Test that loaded datasets work correctly with batching and prefetching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test.npz")

        sequences = np.random.randint(
            0, 21, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.random.rand(NUM_SAMPLES, NUM_CLASSES).astype(np.float32)
        np.savez_compressed(npz_path, translated=sequences, label=labels)

        dataset = (
            _load_numpy_full_dataset(
                npz_path,
                input_type="translated",
                use_embedding_layer=True,
                codon_depth=CODON_DEPTH,
                crop_size=CROP_SIZE,
            )
            .cache()
            .shuffle(buffer_size=100)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        for features, label in dataset:
            assert features["translated"].shape == (BATCH_SIZE, 6, SEQ_LEN)
            assert label.shape == (BATCH_SIZE, NUM_CLASSES)
            break

        print("✓ Dataset pipeline with batching works correctly")


def test_load_numpy_full_dataset():
    """Test loading fully-preprocessed NumPy .npz files (numpy_full format)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_full.npz")

        # Create preprocessed data (as produced by convert_to_numpy_full.py)
        sequences = np.random.randint(
            1, 65, size=(NUM_SAMPLES, 6, SEQ_LEN), dtype=np.int32
        )
        labels = np.eye(NUM_CLASSES, dtype=np.float32)[
            np.arange(NUM_SAMPLES) % NUM_CLASSES
        ]

        np.savez_compressed(npz_path, translated=sequences, label=labels)

        # Load back
        dataset = _load_numpy_full_dataset(npz_path)

        count = 0
        for features, label in dataset:
            assert "translated" in features
            assert features["translated"].shape == (6, SEQ_LEN)
            assert features["translated"].dtype == tf.int32
            assert label.shape == (NUM_CLASSES,)
            assert label.dtype == tf.float32
            count += 1

        assert count == NUM_SAMPLES
        print(f"✓ NumPy full load: {count} samples, shapes correct")

        # Test with batching pipeline
        batched = (
            dataset.cache()
            .shuffle(buffer_size=100)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        for features, label in batched:
            assert features["translated"].shape == (BATCH_SIZE, 6, SEQ_LEN)
            assert label.shape == (BATCH_SIZE, NUM_CLASSES)
            break

        print("✓ NumPy full dataset pipeline with batching works correctly")


if __name__ == "__main__":
    print("Running data format smoke tests...\n")

    test_tfrecord_feature_description()
    test_parse_tfrecord_embedding()
    test_load_numpy_full_dataset_with_legacy_params()
    test_numpy_flattened_format()
    test_dataset_pipeline_with_batching()
    test_load_numpy_full_dataset()

    print("\n✅ All data format tests passed!")
