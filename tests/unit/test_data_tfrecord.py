"""Tests for jaeger.data.tfrecord."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.data import tfrecord


class TestFeatureDescription:
    def test_get_tfrecord_feature_description_translated_embedding(self):
        desc = tfrecord._get_tfrecord_feature_description(
            input_type="translated", use_embedding_layer=True, codon_depth=65, crop_size=12, num_classes=2
        )
        assert "translated" in desc
        assert "label" in desc

    def test_get_tfrecord_feature_description_translated_onehot(self):
        desc = tfrecord._get_tfrecord_feature_description(
            input_type="translated", use_embedding_layer=False, codon_depth=65, crop_size=12, num_classes=2
        )
        assert "translated" in desc


class TestParseFn:
    def test_make_parse_tfrecord_fn_translated_embedding(self):
        parse_fn = tfrecord._make_parse_tfrecord_fn(
            input_type="translated", use_embedding_layer=True, codon_depth=65, crop_size=12, num_classes=2
        )
        translated = np.random.randint(0, 65, size=(6, 3), dtype=np.int64)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "translated": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=translated.flatten().tolist())
                    ),
                    "label": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[1.0, 0.0])
                    ),
                }
            )
        ).SerializeToString()
        inputs, label = parse_fn(example)
        assert "translated" in inputs
        assert label.shape.as_list() == [2]
