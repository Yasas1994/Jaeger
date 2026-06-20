"""Custom Keras callbacks used by Jaeger.

The standard Keras ``TerminateOnNaN`` callback overrides ``on_batch_end``.
With the TensorFlow backend Keras may dispatch callbacks asynchronously, so
a NaN detected in ``on_batch_end`` can set ``model.stop_training`` too late
to stop the current epoch. The callbacks below override the training-specific
hooks so they run synchronously.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class SyncTerminateOnNaN(tf.keras.callbacks.Callback):
    """Synchronously terminate training when the loss becomes non-finite.

    Parameters
    ----------
    raise_error : bool
        If ``True``, raise ``RuntimeError`` immediately on NaN/Inf loss.
        If ``False`` (default), set ``model.stop_training = True`` so Keras
        exits gracefully at the end of the current batch.
    """

    def __init__(self, raise_error: bool = False):
        super().__init__()
        self.raise_error = raise_error

    def _check_loss(self, batch, logs):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None:
            return
        if np.isnan(loss) or np.isinf(loss):
            msg = f"Batch {batch}: Invalid loss ({loss}), terminating training"
            if self.raise_error:
                raise RuntimeError(msg)
            tf.print(msg)
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self._check_loss(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        # Also catch NaNs during validation/evaluation.
        self._check_loss(batch, logs)
