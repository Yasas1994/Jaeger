"""Receptive-field utilities for Jaeger fragment models."""

from __future__ import annotations

import math
from typing import Any


def _conv_rf_delta(kernel_size: int, dilation_rate: int) -> int:
    """Return the receptive-field growth of a single 1-D convolution."""
    return (kernel_size - 1) * dilation_rate


def _format_rf(rf: int | float) -> str:
    if math.isinf(rf):
        return "full sequence"
    return str(int(rf))


def _attends_over_length(attention_axes: int | tuple | list) -> bool:
    """Return True if the attention axis set includes the length axis (2)."""
    if attention_axes is None:
        return False
    if isinstance(attention_axes, int):
        return attention_axes == 2
    return 2 in tuple(attention_axes)


def _compute_rf(
    hidden_layers: list[dict[str, Any]],
    current_rf: int | float = 1,
) -> tuple[int | float, list[tuple[str, int | float]]]:
    """Compute RF from ``hidden_layers`` starting at ``current_rf``."""
    rf: int | float = current_rf
    trace: list[tuple[str, int | float]] = []

    for layer in hidden_layers:
        name = layer.get("name", "unknown")
        cfg = layer.get("config", {}) or {}

        if name == "masked_conv1d":
            rf += _conv_rf_delta(
                int(cfg.get("kernel_size", 1)),
                int(cfg.get("dilation_rate", 1)),
            )
        elif name == "residual_block":
            block_size = int(cfg.get("block_size", 2))
            kernel_size = int(cfg.get("kernel_size", 3))
            dilation_rate = int(cfg.get("dilation_rate", 1))
            delta = _conv_rf_delta(kernel_size, dilation_rate)
            rf += block_size * delta
        elif name == "masked_bilstm":
            rf = math.inf
        elif name == "axial_attention":
            rf = math.inf
        elif name == "transformer_encoder" and _attends_over_length(
            cfg.get("attention_axes", 2)
        ):
            rf = math.inf
        elif name == "parallel_branches":
            branches = cfg.get("branches", [])
            branch_rfs: list[int | float] = []
            for idx, branch in enumerate(branches):
                branch_layers = branch.get("hidden_layers", [])
                branch_rf, _ = _compute_rf(branch_layers, rf)
                trace.append((f"parallel_branches.branch_{idx}", branch_rf))
                branch_rfs.append(branch_rf)
            if branch_rfs:
                rf = max(branch_rfs)
            continue
        # All other layers do not change the RF size.

        trace.append((name, rf))

    return rf, trace


def compute_receptive_field(
    hidden_layers: list[dict[str, Any]],
) -> tuple[int | float, list[tuple[str, int | float]]]:
    """Compute the 1-D sequence receptive field of a representation learner.

    The model receives a tensor of shape ``(batch, frames, length)``; the
    receptive field reported here is the number of sequence positions on the
    length axis that influence one output position of the convolutional stack.

    Layers that do not change the receptive field (normalization, activation,
    dropout, NMD, pooling) are tracked in the trace but do not increase the
    value. Bidirectional LSTM and axial/length attention make every output
    position depend on the entire sequence, so the receptive field becomes the
    full sequence from that point onward.

    ``parallel_branches`` layers are recursed into so that each branch's RF is
    reported separately; the overall RF is taken as the maximum branch RF.

    Parameters
    ----------
    hidden_layers:
        The ``model.representation_learner.hidden_layers`` list from a Jaeger
        config.

    Returns
    -------
    rf:
        The final receptive-field size in sequence positions, or ``inf`` when
        a bidirectional LSTM is present.
    trace:
        A per-layer breakdown of ``(layer_name, receptive_field)``.
    """
    rf, trace = _compute_rf(hidden_layers, 1)
    return rf, [("input", 1)] + trace


def receptive_field_summary(
    hidden_layers: list[dict[str, Any]],
    crop_size: int | None = None,
) -> str:
    """Return a human-readable receptive-field summary."""
    rf, trace = compute_receptive_field(hidden_layers)
    lines = [f"Receptive field: {_format_rf(rf)}"]
    for name, layer_rf in trace:
        lines.append(f"  {name}: {_format_rf(layer_rf)}")
    if crop_size is not None and not math.isinf(rf):
        coverage = min(100, int(rf / crop_size * 100)) if crop_size else 0
        lines.append(f"  crop size: {crop_size} ({coverage}% coverage)")
    elif crop_size is not None:
        lines.append(f"  crop size: {crop_size}")
    return "\n".join(lines)
