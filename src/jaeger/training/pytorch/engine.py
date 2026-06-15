from collections import deque
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: Optional[Dict[str, Any]] = None,
    forward_key: str = "prediction",
    label_key: str = "label",
    branch: str = "classifier",
    progress: Optional[Any] = None,
    task_id: Optional[Any] = None,
    loss_window_size: int = 50,
) -> Dict[str, float]:
    """Train for one epoch and return averaged loss and metrics."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    recent_losses: deque[float] = deque(maxlen=loss_window_size)
    if metrics:
        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

    for batch in dataloader:
        if branch == "classifier":
            x, y, mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(x, mask=mask)
            if isinstance(outputs, dict):
                preds = outputs[forward_key]
            else:
                preds = outputs
            loss = loss_fn(preds, y)
        elif branch == "reliability":
            x, y, mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(x, mask=mask)
            preds = outputs[forward_key]
            loss = loss_fn(preds, y)
        else:
            raise ValueError(f"Unknown branch: {branch}")

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        loss_value = loss.item()
        total_loss += loss_value * batch_size
        total_samples += batch_size
        recent_losses.append(loss_value)

        if progress is not None and task_id is not None:
            moving_avg = sum(recent_losses) / len(recent_losses)
            progress.advance(task_id, 1)
            progress.update(task_id, description=f"loss={moving_avg:.4f}")

        if metrics:
            for metric in metrics.values():
                if hasattr(metric, "update"):
                    metric.update(preds.detach().cpu(), y.detach().cpu())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    result = {"loss": avg_loss}
    if metrics:
        result.update(
            {
                name: metric.compute() if hasattr(metric, "compute") else float(metric)
                for name, metric in metrics.items()
            }
        )
    return result


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    metrics: Optional[Dict[str, Any]] = None,
    forward_key: str = "prediction",
    branch: str = "classifier",
    progress: Optional[Any] = None,
    task_id: Optional[Any] = None,
    loss_window_size: int = 50,
) -> Dict[str, float]:
    """Evaluate and return averaged loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    recent_losses: deque[float] = deque(maxlen=loss_window_size)
    if metrics:
        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

    with torch.no_grad():
        for batch in dataloader:
            x, y, mask = [b.to(device) for b in batch]
            outputs = model(x, mask=mask)
            if isinstance(outputs, dict):
                preds = outputs[forward_key]
            else:
                preds = outputs
            loss = loss_fn(preds, y)

            batch_size = y.size(0)
            loss_value = loss.item()
            total_loss += loss_value * batch_size
            total_samples += batch_size
            recent_losses.append(loss_value)

            if progress is not None and task_id is not None:
                moving_avg = sum(recent_losses) / len(recent_losses)
                progress.advance(task_id, 1)
                progress.update(task_id, description=f"loss={moving_avg:.4f}")

            if metrics:
                for metric in metrics.values():
                    if hasattr(metric, "update"):
                        metric.update(preds.cpu(), y.cpu())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    result = {"loss": avg_loss}
    if metrics:
        result.update(
            {
                name: metric.compute() if hasattr(metric, "compute") else float(metric)
                for name, metric in metrics.items()
            }
        )
    return result
