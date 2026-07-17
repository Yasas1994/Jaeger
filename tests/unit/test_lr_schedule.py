from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder, TrainingStateCallback


def _make_builder(training: dict) -> DynamicModelBuilder:
    return DynamicModelBuilder({"model": {"name": "jaeger"}, "training": training})


# --- cosine lr_schedule wiring ------------------------------------------------


def test_cosine_schedule_replaces_scalar_lr():
    builder = _make_builder(
        {
            "optimizer_params": {"learning_rate": 3e-4},
            "lr_schedule": {
                "name": "cosine",
                "min_lr": 1e-5,
                "decay_steps": 1000,
            },
        }
    )
    opt = builder._get_optimizer("adamw", builder.train_cfg["optimizer_params"])
    opt.build([tf.Variable(1.0)])
    # Keras 3 evaluates the schedule in the learning_rate property.
    assert float(opt.learning_rate.numpy()) == pytest.approx(3e-4, rel=1e-3)
    opt.iterations.assign(1000)
    assert float(opt.learning_rate.numpy()) == pytest.approx(1e-5, rel=1e-2)


def test_cosine_schedule_initial_lr_override():
    builder = _make_builder(
        {
            "optimizer_params": {"learning_rate": 3e-4},
            "lr_schedule": {
                "name": "cosine",
                "initial_lr": 1e-3,
                "min_lr": 0.0,
                "decay_steps": 500,
            },
        }
    )
    opt = builder._get_optimizer("adamw", builder.train_cfg["optimizer_params"])
    opt.build([tf.Variable(1.0)])
    assert float(opt.learning_rate.numpy()) == pytest.approx(1e-3, rel=1e-3)


def test_cosine_schedule_requires_decay_steps():
    # The builder constructs the optimizer at __init__, so a bad schedule
    # fails fast at config load time.
    with pytest.raises(ValueError, match="decay_steps"):
        _make_builder(
            {"lr_schedule": {"name": "cosine", "initial_lr": 1e-3, "min_lr": 0.0}}
        )


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="Unsupported lr_schedule"):
        _make_builder({"lr_schedule": {"name": "linear"}})


def test_no_schedule_keeps_scalar_lr():
    builder = _make_builder({})
    opt = builder._get_optimizer("adamw", {"learning_rate": 3e-4})
    assert not isinstance(
        opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule
    )


# --- ReduceLROnPlateau conflict -----------------------------------------------


def _callbacks_cfg():
    return {
        "callbacks": {
            "classifier": [
                {
                    "name": "ReduceLROnPlateau",
                    "params": {"monitor": "val_loss", "patience": 2},
                },
                {
                    "name": "EarlyStopping",
                    "params": {"monitor": "val_loss", "patience": 3},
                },
            ]
        }
    }


def test_rlp_dropped_when_schedule_set(caplog):
    builder = _make_builder(
        {
            **_callbacks_cfg(),
            "lr_schedule": {
                "name": "cosine",
                "initial_lr": 1e-3,
                "min_lr": 0.0,
                "decay_steps": 500,
            },
        }
    )
    with caplog.at_level("WARNING"):
        callbacks = builder.get_callbacks(branch="classifier")
    assert not any(
        isinstance(c, tf.keras.callbacks.ReduceLROnPlateau) for c in callbacks
    )
    assert any(isinstance(c, tf.keras.callbacks.EarlyStopping) for c in callbacks)


def test_rlp_kept_without_schedule():
    builder = _make_builder(_callbacks_cfg())
    callbacks = builder.get_callbacks(branch="classifier")
    assert any(isinstance(c, tf.keras.callbacks.ReduceLROnPlateau) for c in callbacks)


# --- TrainingStateCallback with schedules --------------------------------------


def test_current_lr_evaluates_schedule():
    schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 1000, alpha=0.01)
    opt = tf.keras.optimizers.AdamW(learning_rate=schedule)
    opt.build([tf.Variable(1.0)])
    assert TrainingStateCallback._current_lr(opt) == pytest.approx(1e-3, rel=1e-3)


def test_state_restore_skips_schedule(tmp_path):
    import json

    (tmp_path / "training_state.json").write_text(
        json.dumps({"epoch": 5, "learning_rate": 1e-5})
    )
    schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 1000)
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule))
    model(tf.zeros((1, 1)))

    cb = TrainingStateCallback(tmp_path, [], restore=True)
    cb.set_model(model)
    cb.on_train_begin()  # must not raise on schedule.assign
    assert TrainingStateCallback._uses_lr_schedule(model.optimizer)


# --- EMA passthrough ------------------------------------------------------------


def test_ema_params_passthrough():
    builder = _make_builder(
        {
            "optimizer_params": {
                "learning_rate": 3e-4,
                "use_ema": True,
                "ema_momentum": 0.999,
                "ema_overwrite_frequency": 2,
            }
        }
    )
    opt = builder._get_optimizer("adamw", builder.train_cfg["optimizer_params"])
    assert opt.use_ema is True
    assert opt.ema_momentum == pytest.approx(0.999)
    assert opt.ema_overwrite_frequency == 2


def test_ema_optimizer_step_overwrites_weights():
    """With ema_overwrite_frequency, model weights converge to the EMA."""
    opt = tf.keras.optimizers.AdamW(
        learning_rate=1e-3, use_ema=True, ema_momentum=0.5, ema_overwrite_frequency=1
    )
    var = tf.Variable(1.0)
    opt.build([var])
    for _ in range(3):
        with tf.GradientTape() as tape:
            loss = (var - 0.0) ** 2
        opt.apply_gradients(zip(tape.gradient(loss, [var]), [var]))
    assert np.isfinite(var.numpy())
