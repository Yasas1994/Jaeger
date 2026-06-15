import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jaeger.training.pytorch.engine import evaluate, train_one_epoch


class Trainer:
    """High-level training loop for Jaeger PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        metrics: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        checkpoint_dir: Optional[str] = None,
        history_path: Optional[str] = None,
        branch: str = "classifier",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.history_path = Path(history_path) if history_path else None
        self.branch = branch
        self.history: List[Dict[str, float]] = []

    def fit(self) -> List[Dict[str, float]]:
        self.model.to(self.device)
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(self)

        for epoch in range(1, self.epochs + 1):
            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_begin"):
                    callback.on_epoch_begin(self, epoch)

            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.loss_fn,
                self.optimizer,
                self.device,
                metrics=self.metrics,
                branch=self.branch,
            )
            val_metrics = evaluate(
                self.model,
                self.val_loader,
                self.loss_fn,
                self.device,
                metrics=self.metrics,
                branch=self.branch,
            )

            epoch_log = {"epoch": epoch}
            for k, v in train_metrics.items():
                epoch_log[f"train_{k}"] = v
            for k, v in val_metrics.items():
                epoch_log[f"val_{k}"] = v
            self.history.append(epoch_log)

            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(self, epoch, epoch_log)

            if self.checkpoint_dir:
                self._save_checkpoint(epoch)
            if self.history_path:
                self._save_history()

        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self)

        return self.history

    def _save_checkpoint(self, epoch: int):
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def _save_history(self):
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w") as fh:
            json.dump(self.history, fh, indent=2)
