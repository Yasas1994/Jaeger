"""Collators for variable-length PyTorch batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple:
    """Collate a list of ``(x, y, mask)`` tuples by padding to the max length.

    Padding is applied to the sequence (length) dimension:

    - 2-D inputs ``(frames, length)`` are padded on the right of ``length``.
    - 3-D inputs ``(frames, length, channels)`` are padded on the right of
      ``length``; channels are left untouched.

    Masks are always ``(frames, length)`` and padded consistently.
    """
    xs, ys, masks = zip(*batch)

    def _seq_len(x: torch.Tensor) -> int:
        if x.dim() == 2:
            return x.shape[-1]
        if x.dim() == 3:
            return x.shape[-2]
        raise ValueError(f"pad_collate supports 2-D and 3-D inputs, got rank {x.dim()}")

    max_len = max(_seq_len(x) for x in xs)
    padded_xs: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for x, mask in zip(xs, masks):
        pad = max_len - _seq_len(x)
        if pad > 0:
            if x.dim() == 2:
                padded_xs.append(F.pad(x, (0, pad)))
            else:
                # Pad the length dimension (second-to-last) on the right.
                padded_xs.append(F.pad(x, (0, 0, 0, pad)))
            padded_masks.append(F.pad(mask, (0, pad)))
        else:
            padded_xs.append(x)
            padded_masks.append(mask)
    return torch.stack(padded_xs), torch.stack(ys), torch.stack(padded_masks)
