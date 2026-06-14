"""Collators for variable-length PyTorch batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple:
    """Collate a list of ``(x, y, mask)`` tuples by padding to the max length.

    All tensors in the batch are stacked after padding the last dimension of
    ``x`` and ``mask`` to ``max(x.shape[-1])``.
    """
    xs, ys, masks = zip(*batch)
    max_len = max(x.shape[-1] for x in xs)
    padded_xs: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for x, mask in zip(xs, masks):
        pad = max_len - x.shape[-1]
        if pad > 0:
            padded_xs.append(F.pad(x, (0, pad)))
            padded_masks.append(F.pad(mask, (0, pad)))
        else:
            padded_xs.append(x)
            padded_masks.append(mask)
    return torch.stack(padded_xs), torch.stack(ys), torch.stack(padded_masks)
