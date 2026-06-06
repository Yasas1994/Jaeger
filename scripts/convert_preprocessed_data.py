#!/usr/bin/env python3
"""
Convert Jaeger CSV training data to TFRecord or NumPy format.

This pre-processes the data once, so training can skip live preprocessing
and load tensors directly — achieving 20-30x faster data loading.

Usage:
    python scripts/convert_preprocessed_data.py \\
        --csv train_shuffled.csv \\
        --output train_shuffled.npz \\
        --format numpy \\
        --crop-size 500

    python scripts/convert_preprocessed_data.py \\
        --csv train_shuffled.csv \\
        --output train_shuffled.tfrecord \\
        --format tfrecord \\
        --crop-size 500
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import numpy as np
import tensorflow as tf

from jaeger.preprocess.latest.convert import process_string_train
from jaeger.preprocess.latest.maps import CODONS, CODON_ID


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_tfrecord_embedding(translated, label):
    """Serialize a single example for embedding layer (int32 indices)."""
    translated_flat = tf.reshape(translated, [-1])
    feature = {
        "translated": _int64_feature(translated_flat.numpy().tolist()),
        "label": _float_feature(label.numpy().flatten().tolist()),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def serialize_tfrecord_onehot(translated, label):
    """Serialize a single example for one-hot encoded input (float32)."""
    translated_flat = tf.reshape(translated, [-1])
    feature = {
        "translated": _float_feature(translated_flat.numpy().flatten().tolist()),
        "label": _float_feature(label.numpy().flatten().tolist()),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def convert_csv_to_tfrecord(
    csv_path, output_path, crop_size=500, use_embedding_layer=True
):
    """Convert CSV to TFRecord with preprocessed tensors."""
    print(f"Converting {csv_path} -> {output_path}")

    total = sum(1 for _ in open(csv_path))
    print(f"Total samples: {total}")

    preprocess_fn = process_string_train(
        crop_size=crop_size,
        seq_onehot=False,
        input_type="translated",
        class_label_onehot=True,
        num_classes=3,
        shuffle=False,
        ngram_width=3,
        codons=CODONS,
        codon_num=CODON_ID,
    )

    serialize_fn = (
        serialize_tfrecord_embedding
        if use_embedding_layer
        else serialize_tfrecord_onehot
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tf.io.TFRecordWriter(output_path) as writer:
        start = time.time()

        with open(csv_path) as f:
            for i, line in enumerate(f):
                outputs, label = preprocess_fn(line.strip().encode())
                translated = outputs["translated"]
                example = serialize_fn(translated, label)
                writer.write(example)

                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    print(
                        f"  {i + 1}/{total} ({100 * (i + 1) / total:.1f}%) - {rate:.1f} samples/sec"
                    )

        elapsed = time.time() - start
        print(
            f"Done! {total} samples in {elapsed:.1f}s ({total / elapsed:.1f} samples/sec)"
        )


def convert_csv_to_numpy(
    csv_path, output_path, crop_size=500, use_embedding_layer=True
):
    """Convert CSV to NumPy .npz with preprocessed tensors."""
    print(f"Converting {csv_path} -> {output_path}")

    total = sum(1 for _ in open(csv_path))
    print(f"Total samples: {total}")

    preprocess_fn = process_string_train(
        crop_size=crop_size,
        seq_onehot=False,
        input_type="translated",
        class_label_onehot=True,
        num_classes=3,
        shuffle=False,
        ngram_width=3,
        codons=CODONS,
        codon_num=CODON_ID,
    )

    # For fixed crop_size, sequences have fixed length: crop_size // 3 - 1
    seq_len = crop_size // 3 - 1

    if use_embedding_layer:
        sequences = np.zeros((total, 6, seq_len), dtype=np.int32)
    else:
        # Need to determine codon_depth from first sample
        with open(csv_path) as f:
            first_line = next(f)
        outputs, _ = preprocess_fn(first_line.strip().encode())
        codon_depth = outputs["translated"].shape[-1]
        sequences = np.zeros((total, 6, seq_len, codon_depth), dtype=np.float32)
        print(f"Codon depth: {codon_depth}")

    labels = np.zeros((total, 3), dtype=np.float32)

    start = time.time()
    with open(csv_path) as f:
        for i, line in enumerate(f):
            outputs, label = preprocess_fn(line.strip().encode())
            seq = outputs["translated"].numpy()
            sequences[i] = seq
            labels[i] = label.numpy()

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(
                    f"  {i + 1}/{total} ({100 * (i + 1) / total:.1f}%) - {rate:.1f} samples/sec"
                )

    elapsed = time.time() - start
    print(
        f"Done! {total} samples in {elapsed:.1f}s ({total / elapsed:.1f} samples/sec)"
    )
    print(
        f"Memory: sequences={sequences.nbytes / (1024**3):.2f}GB, labels={labels.nbytes / (1024**3):.3f}GB"
    )

    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path, translated=sequences, label=labels)
    print("Saved!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Jaeger CSV training data to TFRecord or NumPy format"
    )
    parser.add_argument("--csv", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--format",
        required=True,
        choices=["tfrecord", "numpy"],
        help="Output format",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=500,
        help="Sequence crop size (default: 500)",
    )
    parser.add_argument(
        "--use-embedding-layer",
        action="store_true",
        default=True,
        help="Use embedding layer (int indices) vs one-hot (default: True)",
    )
    parser.add_argument(
        "--no-embedding-layer",
        action="store_false",
        dest="use_embedding_layer",
        help="Use one-hot encoding instead of embedding layer",
    )

    args = parser.parse_args()

    if args.format == "tfrecord":
        convert_csv_to_tfrecord(
            args.csv,
            args.output,
            crop_size=args.crop_size,
            use_embedding_layer=args.use_embedding_layer,
        )
    elif args.format == "numpy":
        convert_csv_to_numpy(
            args.csv,
            args.output,
            crop_size=args.crop_size,
            use_embedding_layer=args.use_embedding_layer,
        )


if __name__ == "__main__":
    main()
