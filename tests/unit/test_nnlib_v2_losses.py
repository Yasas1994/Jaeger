"""Tests for jaeger.nnlib.v2.losses."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.nnlib.v2 import losses


class TestSupervisedContrastiveLoss:
    def test_loss_shape(self):
        loss_fn = losses.SupervisedContrastiveLoss(temperature=0.5)
        labels = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        features = tf.random.normal((4, 8))
        loss = loss_fn(labels, features)
        assert loss.shape.rank == 0
        assert np.isfinite(loss.numpy())


class TestNpairsLoss:
    def test_loss_shape(self):
        y_true = tf.constant([0, 1, 0], dtype=tf.int32)
        y_pred = tf.random.normal((3, 3))
        loss = losses.npairs_loss(y_true, y_pred)
        assert loss.shape.rank == 0
        assert np.isfinite(loss.numpy())


class TestArcFaceLoss:
    def test_loss_and_gradients(self):
        layer = losses.ArcFaceLoss(
            num_classes=3, embedding_dim=8, margin=0.5, scale=30.0, onehot=True
        )
        labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
        embeddings = tf.random.normal((2, 8))
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            loss = layer(labels, embeddings)
        grads = tape.gradient(loss, embeddings)
        assert loss.shape.rank == 0
        assert grads is not None


class TestArcFaceLossMixedPrecision:
    def test_zero_and_small_embeddings_stay_finite(self):
        old_policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            layer = losses.ArcFaceLoss(
                num_classes=3, embedding_dim=8, margin=0.5, scale=64.0, onehot=True
            )
            labels = tf.constant(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32
            )
            # One zero vector, one near-zero vector, and one normal vector.
            embeddings = tf.constant(
                [[0.0] * 8, [1e-8] * 8, [1.0, 2.0, -1.0, 0.5, 0.1, -0.2, 0.3, 0.4]],
                dtype=tf.float16,
            )
            with tf.GradientTape() as tape:
                tape.watch(embeddings)
                loss = layer(labels, embeddings)
            grads = tape.gradient(loss, embeddings)
            assert loss.shape.rank == 0
            assert np.isfinite(loss.numpy())
            assert grads is not None
            assert np.all(np.isfinite(grads.numpy()))
        finally:
            tf.keras.mixed_precision.set_global_policy(old_policy)


class TestHierarchicalLoss:
    def test_loss_shape(self):
        parent_of = [0, 0, 1, 1, 2, 2]
        groups = [[0, 1], [2, 3], [4, 5]]
        loss_fn = losses.HierarchicalLoss(parent_of=parent_of, groups=groups)
        y_true = tf.constant([0, 3, 5], dtype=tf.int32)
        fine_logits = tf.random.normal((3, 6))
        loss = loss_fn(y_true, fine_logits)
        assert loss.shape.rank == 0
        assert np.isfinite(loss.numpy())

    def test_loss_onehot(self):
        parent_of = [0, 0, 1, 1, 2, 2]
        groups = [[0, 1], [2, 3], [4, 5]]
        loss_fn = losses.HierarchicalLoss(parent_of=parent_of, groups=groups)
        y_true = tf.one_hot([0, 3, 5], depth=6)
        fine_logits = tf.random.normal((3, 6))
        loss = loss_fn(y_true, fine_logits)
        assert loss.shape.rank == 0
