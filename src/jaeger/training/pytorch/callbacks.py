from pathlib import Path
from typing import Any, Dict, Optional

import torch


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

        improved = (self.mode == "min" and current < self.best_value - self.min_delta) or (
            self.mode == "max" and current > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
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
                self._save(trainer, epoch)
                if self.verbose:
                    print(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}; saving model")
        elif not self.save_best_only:
            self._save(trainer, epoch)

    def _save(self, trainer, epoch: int):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            },
            self.filepath,
        )

    def on_train_end(self, trainer):
        pass
