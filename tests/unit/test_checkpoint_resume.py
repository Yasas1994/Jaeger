from __future__ import annotations

import json
import os

import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder, TrainingStateCallback


def _make_builder() -> DynamicModelBuilder:
    return DynamicModelBuilder({"model": {"name": "jaeger"}, "training": {}})


def _write_ckpt(directory, epoch: int, loss: float, mtime: float) -> None:
    path = directory / f"epoch:{epoch:02d}-loss:{loss:.2f}.weights.h5"
    path.write_bytes(b"weights")
    os.utime(path, (mtime, mtime))


def test_latest_checkpoint_selected_by_epoch_not_mtime(tmp_path):
    # Newest mtime belongs to the *oldest* epoch (e.g. after an rsync).
    _write_ckpt(tmp_path, epoch=5, loss=1.20, mtime=3000)
    _write_ckpt(tmp_path, epoch=30, loss=0.40, mtime=1000)
    _write_ckpt(tmp_path, epoch=12, loss=0.80, mtime=2000)

    result = _make_builder().get_latest_h5_with_metadata(tmp_path)

    assert result["epoch"] == 30
    assert result["loss"] == 0.40
    assert result["path"].name.startswith("epoch:30")


def test_latest_checkpoint_empty_dir(tmp_path):
    result = _make_builder().get_latest_h5_with_metadata(tmp_path)
    assert result == {"path": None, "epoch": 0, "loss": None, "is_converged": False}


def _tiny_model(lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr), loss="mse")
    return model


def test_training_state_callback_round_trip(tmp_path):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    # --- original run: write state at epoch end ---
    model = _tiny_model(lr=3e-4)
    writer = TrainingStateCallback(tmp_path, [reduce_lr, early_stop])
    writer.set_model(model)
    model.optimizer.learning_rate.assign(5e-5)  # as if RLP had decayed it
    reduce_lr.best = 0.42
    reduce_lr.wait = 2
    reduce_lr.cooldown_counter = 1
    early_stop.best = 0.42
    early_stop.wait = 2
    writer.on_epoch_end(7)

    state = json.loads((tmp_path / TrainingStateCallback.STATE_FILENAME).read_text())
    assert state["epoch"] == 7
    assert state["learning_rate"] == pytest.approx(5e-5)
    assert state["reduce_lr_on_plateau"]["best"] == pytest.approx(0.42)
    assert state["early_stopping"]["wait"] == 2

    # --- resumed run: restore state at train begin ---
    fresh_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5)
    fresh_early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    resumed = _tiny_model(lr=3e-4)
    restorer = TrainingStateCallback(
        tmp_path, [fresh_reduce_lr, fresh_early_stop], restore=True
    )
    restorer.set_model(resumed)
    restorer.on_train_begin()

    assert float(resumed.optimizer.learning_rate.numpy()) == pytest.approx(5e-5)
    assert fresh_reduce_lr.best == pytest.approx(0.42)
    assert fresh_reduce_lr.wait == 2
    assert fresh_reduce_lr.cooldown_counter == 1
    assert fresh_early_stop.best == pytest.approx(0.42)
    assert fresh_early_stop.wait == 2


def test_training_state_callback_no_restore_without_flag(tmp_path):
    (tmp_path / TrainingStateCallback.STATE_FILENAME).write_text(
        json.dumps({"epoch": 3, "learning_rate": 1e-6})
    )
    model = _tiny_model(lr=3e-4)
    cb = TrainingStateCallback(tmp_path, [], restore=False)
    cb.set_model(model)
    cb.on_train_begin()
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(3e-4)


def test_arcface_sidecar_never_selected_as_checkpoint(tmp_path):
    # The MetricModel arcface sidecar shares the epoch/loss prefix and is
    # written *after* the main file (newer mtime); it must never be picked.
    _write_ckpt(tmp_path, epoch=2, loss=1.00, mtime=1000)
    sidecar = tmp_path / "epoch:02-loss:1.00.weights.arcface.weights.h5"
    sidecar.write_bytes(b"arcface")
    os.utime(sidecar, (3000, 3000))

    result = _make_builder().get_latest_h5_with_metadata(tmp_path)

    assert result["epoch"] == 2
    assert result["path"].name == "epoch:02-loss:1.00.weights.h5"


def _tiny_projection_model(num_classes: int = 3, embedding_dim: int = 4):
    from jaeger.nnlib.v2.layers import MetricModel
    from jaeger.nnlib.v2.losses import ArcFaceLoss

    inputs = tf.keras.Input(shape=(embedding_dim,), name="rep")
    x = tf.keras.layers.Dense(embedding_dim, name="proj_dense")(inputs)
    model = MetricModel(inputs=inputs, outputs=x, name="proj")

    labels = tf.keras.Input(shape=(num_classes,), name="labels")
    embeddings = tf.keras.Input(shape=(embedding_dim,), name="embeddings")
    arcface = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=0.5,
        scale=16.0,
        onehot=True,
        name="Arcface",
    )
    loss = arcface(labels, embeddings)
    arcface_model = tf.keras.Model(
        inputs=[labels, embeddings], outputs=loss, name="Arcface_model"
    )
    model.set_arcface_loss(arcface_model)
    return model, arcface_model


def test_metric_model_arcface_round_trip_strict(tmp_path):
    import h5py

    ckpt = tmp_path / "epoch:01-loss:1.00.weights.h5"

    model, arcface_model = _tiny_projection_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss_fn=arcface_model)
    # Build the ArcFace variables, then set them to a known value.
    arcface_model([tf.one_hot([0, 1], 3), tf.zeros((2, 4))], training=False)
    model._get_arcface_loss().trainable_variables[0].assign(tf.ones((3, 4)) * 0.7)
    model.save_weights(str(ckpt))

    sidecar = tmp_path / "epoch:01-loss:1.00.weights.arcface.weights.h5"
    assert sidecar.exists()
    with h5py.File(ckpt, "r") as store:
        assert "_arcface_loss" not in store

    fresh, _ = _tiny_projection_model()
    fresh.load_weights(str(ckpt), skip_mismatch=False)

    for a, b in zip(model.weights, fresh.weights):
        tf.debugging.assert_near(a, b)
    centroids = fresh._get_arcface_loss().trainable_variables[0]
    tf.debugging.assert_near(centroids, tf.ones((3, 4)) * 0.7)
