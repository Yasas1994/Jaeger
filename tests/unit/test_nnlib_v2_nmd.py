import tensorflow as tf

from jaeger.nnlib.v2.nmd import NMDLayer


class TestNMDLayer:
    def test_output_shape_without_mask(self):
        x = tf.random.normal((2, 6, 32, 8))
        layer = NMDLayer()
        out = layer(x)
        assert list(out.shape) == [2, 8]

    def test_output_shape_with_mask(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = NMDLayer()
        out = layer(x, mask=mask)
        assert list(out.shape) == [2, 8]

    def test_get_config_roundtrip(self):
        layer = NMDLayer(epsilon=1e-3, momentum=0.95, dtype="float32")
        config = layer.get_config()
        restored = NMDLayer.from_config(config)
        assert restored.epsilon == 1e-3
        assert restored.momentum == 0.95
