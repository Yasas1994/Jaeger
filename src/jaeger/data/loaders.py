"""Dataset loaders for the unified NumPy NPZ training format.

Provides a single ``tf.data.Dataset`` builder for the NPZ schema produced by
``jaeger.dataops.convert._convert_to_npz``. The schema supports ``nucleotide``,
``translated``, or ``both`` input representations, integer or one-hot encodings,
and configurable nucleotide one-hot mappings.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from jaeger.dataops.convert import _get_codon_map


def _one_hot_integer_np(indices: np.ndarray, depth: int) -> np.ndarray:
    """Convert integer indices to float32 one-hot along a new last axis.

    Index 0 and any invalid index produce an all-zero vector. Valid indices
    are in [1, depth-1]. Output shape is indices.shape + (depth,).
    """
    out = np.zeros(indices.shape + (depth,), dtype=np.float32)
    flat = indices.ravel()
    valid = flat > 0
    if np.any(valid):
        out.reshape(-1, depth)[np.arange(flat.size)[valid], flat[valid]] = 1.0
    return out


def _build_nucleotide_onehot_lookup(
    nucleotide_map: dict[str, int],
    nucleotide_onehot_map: dict[str, list[float]] | None,
) -> np.ndarray:
    """Build a token -> one-hot vector lookup table.

    The NPZ ``nucleotide_map`` maps base letters to integer tokens. The config
    ``nucleotide_onehot_map`` maps base letters to one-hot vectors. This helper
    combines them so that token ``nucleotide_map[base]`` maps to
    ``nucleotide_onehot_map[base]``.

    If ``nucleotide_onehot_map`` is None, the default 4-D one-hot order
    A=[1,0,0,0], G=[0,1,0,0], T=[0,0,1,0], C=[0,0,0,1], N=[0,0,0,0] is used.
    Unknown tokens map to the zero vector.
    """
    if nucleotide_onehot_map is None:
        nucleotide_onehot_map = {
            "A": [1.0, 0.0, 0.0, 0.0],
            "G": [0.0, 1.0, 0.0, 0.0],
            "T": [0.0, 0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 0.0, 0.0],
        }
    depth = len(next(iter(nucleotide_onehot_map.values())))
    max_token = max(nucleotide_map.values())
    lookup = np.zeros((max_token + 1, depth), dtype=np.float32)
    for base, token in nucleotide_map.items():
        vec = nucleotide_onehot_map.get(base)
        if vec is not None:
            lookup[token] = np.asarray(vec, dtype=np.float32)
    return lookup


def _load_numpy_dataset(
    path: str,
    input_type: str = "translated",
    seq_onehot: bool = False,
    codon_depth: int | None = None,
    nucleotide_onehot_map: dict[str, list[float]] | None = None,
    num_classes: int | None = None,
    one_hot_labels: bool = True,
) -> tf.data.Dataset:
    """Load a unified NPZ file and return a ``tf.data.Dataset``.

    Parameters
    ----------
    path:
        Path to the ``.npz`` archive.
    input_type:
        One of ``"translated"``, ``"nucleotide"``, or ``"both"``.
    seq_onehot:
        If True and the stored sequence array is integer-encoded, convert it to
        one-hot floats on the fly.
    codon_depth:
        Depth to use for translated one-hot conversion. If None, the depth is
        inferred from the ``codon_map`` entry stored in the NPZ.
    nucleotide_onehot_map:
        Optional base -> one-hot mapping for nucleotide inputs. Defaults to
        A/G/T/C/N order when omitted.
    num_classes:
        Number of classes; used to one-hot integer labels when
        ``one_hot_labels`` is True.
    one_hot_labels:
        Convert integer labels to one-hot float vectors.

    Returns
    -------
    ``tf.data.Dataset`` yielding ``(features, labels)`` tuples.
    """
    data = np.load(path, allow_pickle=False)

    valid_input_types = {"translated", "nucleotide", "both"}
    if input_type not in valid_input_types:
        raise ValueError(
            f"Unsupported input_type: {input_type}. "
            f"Use one of {sorted(valid_input_types)}."
        )

    features: dict[str, np.ndarray] = {}

    if input_type in ("translated", "both"):
        translated = data["translated"]
        if translated.ndim == 3:
            if seq_onehot:
                if codon_depth is None:
                    codon_map_name = str(data["codon_map"])
                    codon_map = _get_codon_map(codon_map_name)
                    codon_depth = len(codon_map) + 1
                translated = _one_hot_integer_np(translated, codon_depth)
        elif translated.ndim != 4:
            raise ValueError(
                f"Unexpected translated array ndim: {translated.ndim} "
                "(expected 3 for integer or 4 for one-hot)"
            )
        features["translated"] = translated

    if input_type in ("nucleotide", "both"):
        nucleotide = data["nucleotide"]
        if nucleotide.ndim == 3:
            if seq_onehot:
                nucleotide_map = json.loads(str(data["nucleotide_map"]))
                lookup = _build_nucleotide_onehot_lookup(
                    nucleotide_map, nucleotide_onehot_map
                )
                nucleotide = lookup[nucleotide]
        elif nucleotide.ndim != 4:
            raise ValueError(
                f"Unexpected nucleotide array ndim: {nucleotide.ndim} "
                "(expected 3 for integer or 4 for one-hot)"
            )
        features["nucleotide"] = nucleotide

    if "labels" in data:
        labels = data["labels"]
    elif "label" in data:
        labels = data["label"]
    else:
        raise KeyError("NPZ must contain 'labels' or 'label' array")

    if (
        one_hot_labels
        and num_classes is not None
        and num_classes > 1
        and (labels.ndim == 1 or np.issubdtype(labels.dtype, np.integer))
    ):
        labels = np.eye(num_classes, dtype=np.float32)[labels]
    else:
        labels = labels.astype(np.float32)

    # Binary heads expect shape (N, 1), multi-class heads expect (N, num_classes).
    if labels.ndim == 1 and num_classes == 1:
        labels = labels[:, np.newaxis]

    return tf.data.Dataset.from_tensor_slices((features, labels))
