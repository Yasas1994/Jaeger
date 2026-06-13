"""TFRecord parsing utilities for preprocessed training data.

Provides feature-description builders and parsing functions for
TensorFlow TFRecord datasets created by the conversion pipeline.
"""

from __future__ import annotations

import tensorflow as tf


def _get_tfrecord_feature_description(
    input_type: str,
    use_embedding_layer: bool,
    codon_depth: int,
    crop_size: int,
    num_classes: int,
) -> dict:
    """Returns TFRecord feature description for parsing preprocessed data."""
    if input_type == "translated":
        if use_embedding_layer:
            feature_description = {
                "translated": tf.io.FixedLenFeature(
                    [6 * (crop_size // 3 - 1)], tf.int64
                ),
                "label": tf.io.FixedLenFeature([num_classes], tf.float32),
            }
        else:
            feature_description = {
                "translated": tf.io.FixedLenFeature(
                    [6 * (crop_size // 3 - 1) * codon_depth], tf.float32
                ),
                "label": tf.io.FixedLenFeature([num_classes], tf.float32),
            }
    elif input_type == "nucleotide":
        feature_description = {
            "nucleotide": tf.io.FixedLenFeature([2 * crop_size * 4], tf.float32),
            "label": tf.io.FixedLenFeature([num_classes], tf.float32),
        }
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    return feature_description


def _make_parse_tfrecord_fn(
    input_type: str,
    use_embedding_layer: bool,
    codon_depth: int,
    crop_size: int,
    num_classes: int,
):
    """Creates a TFRecord parsing function for the given config."""
    feature_description = _get_tfrecord_feature_description(
        input_type, use_embedding_layer, codon_depth, crop_size, num_classes
    )

    if input_type == "translated":
        if use_embedding_layer:
            seq_shape = [6, crop_size // 3 - 1]
        else:
            seq_shape = [6, crop_size // 3 - 1, codon_depth]
    elif input_type == "nucleotide":
        seq_shape = [2, crop_size, 4]

    @tf.function
    def _parse_tfrecord(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        if input_type == "translated":
            if use_embedding_layer:
                seq = tf.cast(parsed["translated"], tf.int32)
            else:
                seq = tf.cast(parsed["translated"], tf.float32)
            seq = tf.reshape(seq, seq_shape)
            features = {"translated": seq}
        elif input_type == "nucleotide":
            seq = tf.cast(parsed["nucleotide"], tf.float32)
            seq = tf.reshape(seq, seq_shape)
            features = {"nucleotide": seq}
        label = parsed["label"]
        return features, label

    return _parse_tfrecord
