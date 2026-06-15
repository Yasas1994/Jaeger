import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _format_timing(data_ms: float, forward_ms: float, backward_ms: float = 0.0, optim_ms: float = 0.0, metrics_ms: float = 0.0) -> str:
    """Return a compact timing string for the progress bar."""
    if backward_ms > 0 or optim_ms > 0:
        return f" d={data_ms:.1f}ms f={forward_ms:.1f}ms b={backward_ms:.1f}ms o={optim_ms:.1f}ms m={metrics_ms:.1f}ms"
    return f" d={data_ms:.1f}ms f={forward_ms:.1f}ms m={metrics_ms:.1f}ms"


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
    profile: bool = False,
    train_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Train for one epoch and return averaged loss and metrics.

    When ``profile=True``, per-section timings are included in the progress
    bar description and returned under keys prefixed with ``time_``.

    If ``train_steps`` is a non-negative integer, training stops after that
    many batches even if the dataloader has more data. Negative values or
    ``None`` mean iterate until the dataloader is exhausted.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress_loss = 0.0
    progress_batches = 0
    timing = {
        "data": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optim": 0.0,
        "metrics": 0.0,
    }
    if metrics:
        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

    data_start = time.perf_counter()
    for batch_idx, batch in enumerate(dataloader):
        if train_steps is not None and train_steps >= 0 and batch_idx >= train_steps:
            break
        data_time = time.perf_counter() - data_start
        timing["data"] += data_time

        if branch == "classifier":
            x, y, mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            forward_start = time.perf_counter()
            outputs = model(x, mask=mask)
            if isinstance(outputs, dict):
                preds = outputs[forward_key]
            else:
                preds = outputs
            loss = loss_fn(preds, y)
            forward_time = time.perf_counter() - forward_start
            timing["forward"] += forward_time
        elif branch == "reliability":
            x, y, mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            forward_start = time.perf_counter()
            outputs = model(x, mask=mask)
            preds = outputs[forward_key]
            loss = loss_fn(preds, y)
            forward_time = time.perf_counter() - forward_start
            timing["forward"] += forward_time
        else:
            raise ValueError(f"Unknown branch: {branch}")

        backward_start = time.perf_counter()
        loss.backward()
        backward_time = time.perf_counter() - backward_start
        timing["backward"] += backward_time

        optim_start = time.perf_counter()
        optimizer.step()
        optim_time = time.perf_counter() - optim_start
        timing["optim"] += optim_time

        batch_size = y.size(0)
        loss_value = loss.item()
        total_loss += loss_value * batch_size
        total_samples += batch_size
        progress_loss += loss_value
        progress_batches += 1

        metrics_start = time.perf_counter()
        if metrics:
            for metric in metrics.values():
                if hasattr(metric, "update"):
                    metric.update(preds.detach().cpu(), y.detach().cpu())
        metrics_time = time.perf_counter() - metrics_start
        timing["metrics"] += metrics_time

        if progress is not None and task_id is not None:
            mean_loss = progress_loss / progress_batches
            description = f"loss={mean_loss:.4f}"
            if profile:
                description += _format_timing(
                    data_time * 1000,
                    forward_time * 1000,
                    backward_time * 1000,
                    optim_time * 1000,
                    metrics_time * 1000,
                )
            progress.advance(task_id, 1)
            progress.update(task_id, description=description)

        data_start = time.perf_counter()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    result: Dict[str, float] = {"loss": avg_loss}
    if metrics:
        result.update(
            {
                name: metric.compute() if hasattr(metric, "compute") else float(metric)
                for name, metric in metrics.items()
            }
        )
    if profile:
        n = progress_batches if progress_batches > 0 else 1
        result.update(
            {
                "time_data_ms": timing["data"] / n * 1000,
                "time_forward_ms": timing["forward"] / n * 1000,
                "time_backward_ms": timing["backward"] / n * 1000,
                "time_optim_ms": timing["optim"] / n * 1000,
                "time_metrics_ms": timing["metrics"] / n * 1000,
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
    profile: bool = False,
    validation_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate and return averaged loss and metrics.

    When ``profile=True``, per-section timings are included in the progress
    bar description and returned under keys prefixed with ``time_``.

    If ``validation_steps`` is a non-negative integer, evaluation stops after
    that many batches even if the dataloader has more data. Negative values or
    ``None`` mean iterate until the dataloader is exhausted.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    progress_loss = 0.0
    progress_batches = 0
    timing = {"data": 0.0, "forward": 0.0, "metrics": 0.0}
    if metrics:
        for m in metrics.values():
            if hasattr(m, "reset"):
                m.reset()

    with torch.no_grad():
        data_start = time.perf_counter()
        for batch_idx, batch in enumerate(dataloader):
            if (
                validation_steps is not None
                and validation_steps >= 0
                and batch_idx >= validation_steps
            ):
                break
            data_time = time.perf_counter() - data_start
            timing["data"] += data_time

            x, y, mask = [b.to(device) for b in batch]
            forward_start = time.perf_counter()
            outputs = model(x, mask=mask)
            if isinstance(outputs, dict):
                preds = outputs[forward_key]
            else:
                preds = outputs
            loss = loss_fn(preds, y)
            forward_time = time.perf_counter() - forward_start
            timing["forward"] += forward_time

            batch_size = y.size(0)
            loss_value = loss.item()
            total_loss += loss_value * batch_size
            total_samples += batch_size
            progress_loss += loss_value
            progress_batches += 1

            metrics_start = time.perf_counter()
            if metrics:
                for metric in metrics.values():
                    if hasattr(metric, "update"):
                        metric.update(preds.cpu(), y.cpu())
            metrics_time = time.perf_counter() - metrics_start
            timing["metrics"] += metrics_time

            if progress is not None and task_id is not None:
                mean_loss = progress_loss / progress_batches
                description = f"loss={mean_loss:.4f}"
                if profile:
                    description += _format_timing(
                        data_time * 1000,
                        forward_time * 1000,
                        metrics_ms=metrics_time * 1000,
                    )
                progress.advance(task_id, 1)
                progress.update(task_id, description=description)

            data_start = time.perf_counter()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    result: Dict[str, float] = {"loss": avg_loss}
    if metrics:
        result.update(
            {
                name: metric.compute() if hasattr(metric, "compute") else float(metric)
                for name, metric in metrics.items()
            }
        )
    if profile:
        n = progress_batches if progress_batches > 0 else 1
        result.update(
            {
                "time_data_ms": timing["data"] / n * 1000,
                "time_forward_ms": timing["forward"] / n * 1000,
                "time_metrics_ms": timing["metrics"] / n * 1000,
            }
        )
    return result
