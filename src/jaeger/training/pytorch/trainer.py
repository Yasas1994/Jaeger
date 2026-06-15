from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from torch.utils.data import DataLoader

from jaeger.training.pytorch.engine import evaluate, train_one_epoch
from jaeger.utils.logging import get_logger


logger = get_logger(log_file=None, log_path=None, level=3)


class _SpeedMsColumn(ProgressColumn):
    """Rich column showing average milliseconds per batch."""

    def render(self, task):
        if task.completed == 0:
            return Text("? ms/batch")
        ms = task.elapsed / task.completed * 1000
        return Text(f"{ms:.1f} ms/batch")


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
        branch: str = "classifier",
        progress_bar: bool = False,
        profile: bool = False,
        train_steps: Optional[int] = None,
        validation_steps: Optional[int] = None,
        start_epoch: int = 0,
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
        self.branch = branch
        self.progress_bar = progress_bar
        self.profile = profile
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.start_epoch = start_epoch
        self.history: List[Dict[str, float]] = []
        self.should_stop = False

    def fit(self) -> List[Dict[str, float]]:
        self.model.to(self.device)
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(self)

        if self.start_epoch >= self.epochs:
            logger.info(
                "Checkpoint epoch %d is >= requested epochs %d; nothing to train",
                self.start_epoch,
                self.epochs,
            )
            for callback in self.callbacks:
                if hasattr(callback, "on_train_end"):
                    callback.on_train_end(self)
            return self.history

        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_begin"):
                    callback.on_epoch_begin(self, epoch)

            train_total = (
                self.train_steps
                if self.train_steps is not None
                else self._loader_length(self.train_loader)
            )
            val_total = (
                self.validation_steps
                if self.validation_steps is not None
                else self._loader_length(self.val_loader)
            )

            if self.progress_bar:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    _SpeedMsColumn(),
                ) as progress:
                    train_task = progress.add_task(
                        f"train epoch {epoch}/{self.epochs}",
                        total=train_total,
                    )
                    val_task = progress.add_task(
                        f"val epoch {epoch}/{self.epochs}",
                        total=val_total,
                    )
                    train_metrics = train_one_epoch(
                        self.model,
                        self.train_loader,
                        self.loss_fn,
                        self.optimizer,
                        self.device,
                        metrics=self.metrics,
                        branch=self.branch,
                        progress=progress,
                        task_id=train_task,
                        profile=self.profile,
                        train_steps=self.train_steps,
                    )
                    val_metrics = evaluate(
                        self.model,
                        self.val_loader,
                        self.loss_fn,
                        self.device,
                        metrics=self.metrics,
                        branch=self.branch,
                        progress=progress,
                        task_id=val_task,
                        profile=self.profile,
                        validation_steps=self.validation_steps,
                    )
            else:
                train_metrics = train_one_epoch(
                    self.model,
                    self.train_loader,
                    self.loss_fn,
                    self.optimizer,
                    self.device,
                    metrics=self.metrics,
                    branch=self.branch,
                    profile=self.profile,
                    train_steps=self.train_steps,
                )
                val_metrics = evaluate(
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    self.device,
                    metrics=self.metrics,
                    branch=self.branch,
                    profile=self.profile,
                    validation_steps=self.validation_steps,
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

            if getattr(self, "should_stop", False):
                break

        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self)

        return self.history

    @staticmethod
    def _loader_length(loader: DataLoader) -> Optional[int]:
        try:
            return len(loader)
        except TypeError:
            return None

