"""Tests for OODSignalLayer in jaeger.nnlib.v2.layers."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2 import layers


@pytest.fixture
def logits():
    return tf.constant([[1.0, 2.0, 3.0], [0.5, -1.0, 1.5]], dtype=tf.float32)


@pytest.fixture
def nmd():
    return tf.constant([[3.0, 4.0], [0.0, -5.0]], dtype=tf.float32)


class TestOODSignalLayerSignals:
    def test_max_prob(self, logits):
        layer = layers.OODSignalLayer(signals=["max_prob"])
        out = layer({"logits": logits})

        expected = tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_entropy(self, logits):
        layer = layers.OODSignalLayer(signals=["entropy"], epsilon=1e-10)
        out = layer({"logits": logits})

        probs = tf.nn.softmax(logits, axis=-1)
        safe_probs = tf.maximum(probs, 1e-10)
        expected = -tf.reduce_sum(
            safe_probs * tf.math.log(safe_probs), axis=-1, keepdims=True
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_energy(self, logits):
        layer = layers.OODSignalLayer(signals=["energy"])
        out = layer({"logits": logits})

        expected = tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_margin(self, logits):
        layer = layers.OODSignalLayer(signals=["margin"])
        out = layer({"logits": logits})

        probs = tf.nn.softmax(logits, axis=-1)
        top_2 = tf.nn.top_k(probs, k=2).values
        expected = top_2[..., 0:1] - top_2[..., 1:2]
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_nmd_norm(self, logits, nmd):
        layer = layers.OODSignalLayer(signals=["nmd_norm"])
        out = layer({"logits": logits, "nmd": nmd})

        expected = tf.norm(nmd, ord="euclidean", axis=-1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_all_signals_combined(self, logits, nmd):
        layer = layers.OODSignalLayer(
            signals=["max_prob", "entropy", "energy", "margin", "nmd_norm"]
        )
        out = layer({"logits": logits, "nmd": nmd})
        assert out.shape.as_list() == [2, 5]
        assert np.all(np.isfinite(out.numpy()))


class TestOODSignalLayerInputs:
    def test_dict_and_list_inputs_match(self, logits, nmd):
        layer = layers.OODSignalLayer(
            signals=["max_prob", "entropy", "energy", "margin", "nmd_norm"]
        )
        out_dict = layer({"logits": logits, "nmd": nmd})
        out_list = layer([logits, nmd])
        np.testing.assert_allclose(out_dict.numpy(), out_list.numpy(), rtol=1e-6)

    def test_dict_and_tuple_inputs_match(self, logits, nmd):
        layer = layers.OODSignalLayer(
            signals=["max_prob", "entropy", "energy", "margin", "nmd_norm"]
        )
        out_dict = layer({"logits": logits, "nmd": nmd})
        out_tuple = layer((logits, nmd))
        np.testing.assert_allclose(out_dict.numpy(), out_tuple.numpy(), rtol=1e-6)

    def test_plain_logits_tensor_uses_max_prob(self, logits):
        layer = layers.OODSignalLayer(signals=["max_prob"])
        out = layer(logits)

        expected = tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_plain_logits_tensor_with_entropy(self, logits):
        layer = layers.OODSignalLayer(signals=["entropy"])
        out = layer(logits)

        probs = tf.nn.softmax(logits, axis=-1)
        safe_probs = tf.maximum(probs, 1e-10)
        expected = -tf.reduce_sum(
            safe_probs * tf.math.log(safe_probs), axis=-1, keepdims=True
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_compute_output_shape_plain_tensor(self):
        layer = layers.OODSignalLayer(signals=["max_prob", "entropy"])
        shape = layer.compute_output_shape((None, 8))
        assert shape == (None, 2)

    def test_compute_output_shape_dict(self):
        layer = layers.OODSignalLayer(signals=["max_prob"])
        shape = layer.compute_output_shape({"logits": (None, 8), "nmd": (None, 4)})
        assert shape == (None, 1)

    def test_compute_output_shape_list(self):
        layer = layers.OODSignalLayer(signals=["max_prob", "nmd_norm"])
        shape = layer.compute_output_shape([(None, 8), (None, 4)])
        assert shape == (None, 2)


class TestOODSignalLayerErrors:
    def test_unsupported_signal_raises(self):
        with pytest.raises(ValueError, match="Unsupported signal"):
            layers.OODSignalLayer(signals=["unknown_signal"])

    def test_missing_nmd_raises(self, logits):
        layer = layers.OODSignalLayer(signals=["nmd_norm"])
        with pytest.raises(ValueError, match="nmd_norm"):
            layer({"logits": logits})

    def test_plain_logits_missing_nmd_raises(self, logits):
        layer = layers.OODSignalLayer(signals=["nmd_norm"])
        with pytest.raises(ValueError, match="nmd_norm"):
            layer(logits)


class TestOODSignalLayerConfig:
    def test_get_config_round_trip(self):
        layer = layers.OODSignalLayer(
            signals=["max_prob", "entropy", "nmd_norm"], epsilon=1e-8, name="ood"
        )
        config = layer.get_config()
        restored = layers.OODSignalLayer.from_config(config)

        assert restored.signals == layer.signals
        assert restored.epsilon == layer.epsilon
        assert restored.name == layer.name

    def test_default_signal_is_max_prob(self):
        layer = layers.OODSignalLayer()
        assert layer.signals == ["max_prob"]
