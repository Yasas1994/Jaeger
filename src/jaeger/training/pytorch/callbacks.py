import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from jaeger.utils.logging import get_logger


logger = get_logger(log_file=None, log_path=None, level=3)


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = False,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.wait = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self.stopped_epoch = 0

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (
            self.mode == "min" and current < self.best_value - self.min_delta
        ) or (self.mode == "max" and current > self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
                self.stopped_epoch = epoch

    def on_train_end(self, trainer):
        if self.restore_best_weights and self.best_state is not None:
            trainer.model.load_state_dict(self.best_state)


class ModelCheckpoint:
    """Save the model whenever a monitored metric improves."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: int = 0,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (self.mode == "min" and current < self.best_value) or (
            self.mode == "max" and current > self.best_value
        )

        if improved:
            self.best_value = current
            if self.save_best_only:
                self._save(trainer, epoch, logs)
                if self.verbose:
                    print(
                        f"Epoch {epoch}: {self.monitor} improved to {current:.4f}; saving model"
                    )
        elif not self.save_best_only:
            self._save(trainer, epoch, logs)

    def _save(self, trainer, epoch: int, logs: Optional[Dict[str, float]] = None):
        logs = dict(logs or {})
        # ``epoch`` is passed explicitly so templates without it still work.
        logs.setdefault("epoch", epoch)
        formatted = str(self.filepath).format(**logs)
        path = Path(formatted)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            },
            path,
        )
        if self.verbose:
            print(f"Epoch {epoch}: saved checkpoint to {path}")

    def on_train_end(self, trainer):
        pass


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 2,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        verbose: int = 0,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (
            self.mode == "min" and current < self.best_value
        ) or (self.mode == "max" and current > self.best_value)

        if improved:
            self.best_value = current
            self.wait = 0
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.wait = 0
            self._reduce_lr(trainer.optimizer)

    def _reduce_lr(self, optimizer: torch.optim.Optimizer):
        for param_group in optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr > new_lr:
                param_group["lr"] = new_lr
                if self.verbose:
                    logger.info(
                        "Reducing learning rate: %.6f -> %.6f", old_lr, new_lr
                    )

    def on_train_end(self, trainer):
        pass


class TerminateOnNaN:
    """Stop training if a monitored metric becomes NaN or Inf."""

    def __init__(self, monitor: str = "train_loss"):
        self.monitor = monitor

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        if math.isnan(current) or math.isinf(current):
            logger.warning(
                "Stopping training: %s is %s at epoch %d",
                self.monitor,
                current,
                epoch,
            )
            trainer.should_stop = True

    def on_train_end(self, trainer):
        pass
