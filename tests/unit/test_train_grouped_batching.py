"""Tests for jaeger.commands.train._apply_grouped_batching."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.commands.train import _apply_grouped_batching, _replica_round


def test_replica_round_single_replica():
    assert _replica_round(127, 1) == 127


def test_replica_round_multi_replica():
    assert _replica_round(127, 4) == 124
    assert _replica_round(128, 4) == 128


def _make_dataset(lengths):
    def generator():
        for length in lengths:
            yield (
                {"seq": tf.constant(np.zeros((3, length), dtype=np.int32))},
                tf.constant([1.0, 0.0], dtype=tf.float32),
            )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {"seq": tf.TensorSpec(shape=(3, None), dtype=tf.int32)},
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
        ),
    )


def test_grouped_batching_unique_length_per_batch():
    ds = _make_dataset([4, 4, 6, 6, 6, 4, 4])
    batching_cfg = {"length_batch_sizes": {4: 2, 6: 3}, "default_batch_size": 2}
    batched = _apply_grouped_batching(
        ds, batching_cfg, num_replicas=1, feature_key="seq"
    )

    lengths_seen = set()
    for features, _ in batched:
        actual_length = features["seq"].numpy().shape[2]
        lengths_seen.add(actual_length)
        assert actual_length in {4, 6}
    assert lengths_seen == {4, 6}


def test_grouped_batching_uses_default_batch_size():
    ds = _make_dataset([5, 5, 5, 5, 5])
    batching_cfg = {"length_batch_sizes": {4: 2}, "default_batch_size": 3}
    batched = _apply_grouped_batching(
        ds, batching_cfg, num_replicas=1, feature_key="seq"
    )

    for features, _ in batched:
        assert features["seq"].numpy().shape[2] == 5
        assert features["seq"].numpy().shape[0] == 3


def test_grouped_batching_rounds_for_replicas():
    ds = _make_dataset([4] * 10)
    batching_cfg = {"length_batch_sizes": {4: 7}, "default_batch_size": 4}
    batched = _apply_grouped_batching(
        ds, batching_cfg, num_replicas=4, feature_key="seq"
    )

    for features, _ in batched:
        batch_size = features["seq"].numpy().shape[0]
        assert batch_size % 4 == 0
        assert batch_size <= 7


def test_grouped_batching_zero_rounded_per_length_fallback():
    ds = _make_dataset([4] * 10)
    batching_cfg = {"length_batch_sizes": {4: 3}, "default_batch_size": 8}
    batched = _apply_grouped_batching(
        ds, batching_cfg, num_replicas=4, feature_key="seq"
    )

    for features, _ in batched:
        batch_size = features["seq"].numpy().shape[0]
        assert batch_size == 8


def test_grouped_batching_missing_default_raises():
    ds = _make_dataset([4])
    batching_cfg = {"length_batch_sizes": {4: 2}}
    with pytest.raises(ValueError):
        _apply_grouped_batching(ds, batching_cfg, num_replicas=1, feature_key="seq")


def test_grouped_batching_zero_rounded_default_raises():
    ds = _make_dataset([4])
    batching_cfg = {"default_batch_size": 3}
    with pytest.raises(ValueError):
        _apply_grouped_batching(ds, batching_cfg, num_replicas=4, feature_key="seq")
