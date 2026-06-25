"""Tests for gradient accumulation in MetricModel."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2.layers import MetricModel


def _build_model(gradient_accumulation_steps: int = 1) -> MetricModel:
    inputs = tf.keras.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer="ones"
    )(inputs)
    model = MetricModel(inputs, outputs)
    model.gradient_accumulation_steps = gradient_accumulation_steps
    return model


def _mse_loss(args: list[tf.Tensor]) -> tf.Tensor:
    y_true, y_pred = args
    return tf.reduce_mean(tf.square(y_true - y_pred))


@pytest.mark.parametrize("accum_steps", [1, 2, 4])
def test_weights_update_every_accumulation_steps(accum_steps: int):
    model = _build_model(gradient_accumulation_steps=accum_steps)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss_fn=_mse_loss,
    )

    x = tf.constant([[1.0]], dtype=tf.float32)
    y = tf.constant([[0.0]], dtype=tf.float32)
    kernel_var = model.layers[1].kernel
    initial = kernel_var.numpy().copy()

    # Run (accum_steps - 1) steps; weights should not change yet.
    for _ in range(accum_steps - 1):
        model.train_step((x, y))
    np.testing.assert_allclose(kernel_var.numpy(), initial, atol=1e-6)

    # The next step applies the accumulated gradients.
    model.train_step((x, y))
    new_value = kernel_var.numpy()
    assert not np.allclose(new_value, initial, atol=1e-6)


def test_accumulated_gradients_are_averaged_over_effective_batch():
    """With accum=4, four steps should produce the same update as one full batch."""
    accum_steps = 4
    model = _build_model(gradient_accumulation_steps=accum_steps)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss_fn=_mse_loss,
    )

    x = tf.constant([[1.0]], dtype=tf.float32)
    y = tf.constant([[0.0]], dtype=tf.float32)
    kernel_var = model.layers[1].kernel

    for _ in range(accum_steps):
        model.train_step((x, y))

    # pred = 1*1 = 1, MSE = 1, dW/dstep = 0.5, accumulated over 4 steps = 2.0
    # SGD lr=1 => W = 1 - 2.0 = -1.0
    np.testing.assert_allclose(kernel_var.numpy(), np.array([[-1.0]]), atol=1e-5)


def test_flush_applies_remaining_gradients():
    model = _build_model(gradient_accumulation_steps=4)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss_fn=_mse_loss,
    )

    x = tf.constant([[1.0]], dtype=tf.float32)
    y = tf.constant([[0.0]], dtype=tf.float32)
    kernel_var = model.layers[1].kernel
    initial = kernel_var.numpy().copy()

    model.train_step((x, y))
    np.testing.assert_allclose(kernel_var.numpy(), initial, atol=1e-6)

    model.flush_accumulated_gradients()
    assert not np.allclose(kernel_var.numpy(), initial, atol=1e-6)


def test_loss_metric_uses_original_scale():
    model = _build_model(gradient_accumulation_steps=4)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss_fn=_mse_loss,
    )

    x = tf.constant([[2.0]], dtype=tf.float32)
    y = tf.constant([[0.0]], dtype=tf.float32)
    out = model.train_step((x, y))

    # pred = 2, MSE = 4. The reported loss must be the original MSE, not MSE/4.
    np.testing.assert_allclose(out["loss"].numpy(), 4.0, atol=1e-4)


def test_callback_flushes_on_epoch_end():
    from jaeger.nnlib.builder import GradientAccumulationCallback

    model = _build_model(gradient_accumulation_steps=4)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss_fn=_mse_loss,
    )
    cb = GradientAccumulationCallback()
    cb.set_model(model)

    x = tf.constant([[1.0]], dtype=tf.float32)
    y = tf.constant([[0.0]], dtype=tf.float32)
    kernel_var = model.layers[1].kernel
    initial = kernel_var.numpy().copy()

    model.train_step((x, y))
    np.testing.assert_allclose(kernel_var.numpy(), initial, atol=1e-6)

    cb.on_epoch_end(0)
    assert not np.allclose(kernel_var.numpy(), initial, atol=1e-6)


@pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"),
    reason="XLA jit_compile is only tested on GPU",
)
def test_callback_flushes_on_epoch_end_with_xla():
    """Regression test for Brain job 7504069.

    The flush callback must be able to apply gradients after an epoch when the
    model is compiled with XLA/jit_compile, where the optimizer can fail in
    eager context with ``'NoneType' object has no attribute 'merge_call'``.
    """
    from jaeger.nnlib.builder import GradientAccumulationCallback

    model = _build_model(gradient_accumulation_steps=4)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss_fn=_mse_loss,
        jit_compile=True,
    )
    cb = GradientAccumulationCallback()
    cb.set_model(model)

    x = tf.constant([[1.0]], dtype=tf.float32)
    y = tf.constant([[0.0]], dtype=tf.float32)
    kernel_var = model.layers[1].kernel
    initial = kernel_var.numpy().copy()

    model.train_step((x, y))
    np.testing.assert_allclose(kernel_var.numpy(), initial, atol=1e-6)

    cb.on_epoch_end(0)
    assert not np.allclose(kernel_var.numpy(), initial, atol=1e-6)
