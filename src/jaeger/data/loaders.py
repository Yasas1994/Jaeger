"""Dataset loaders for NumPy-based training formats.

Provides `tf.data.Dataset` builders for:
- ``numpy_raw`` — int8 DNA sequences processed on-the-fly to 6-frame codon IDs.
- ``numpy_raw_variable`` — variable-length int8 sequences.
- ``numpy_full`` — fully preprocessed arrays (fastest, no runtime augmentations).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.seqops.encode import (
    _make_process_raw_sequence_fn,
    _make_process_variable_sequence_fn,
)


def _load_numpy_raw_dataset(
    path: str,
    crop_size: int = 500,
    ngram_width: int = 3,
    num_classes: int = 3,
    shuffle: bool = False,
    mutate: bool = False,
    mutation_rate: float = 0.1,
    shuffle_frames: bool = False,
):
    """Loads a raw NumPy ``.npz`` file (int8 sequences) and returns a ``tf.data.Dataset``.

    The ``.npz`` must contain:
        - ``sequences``: int8 array of shape ``(N, crop_size)``
        - ``labels``: float32 array of shape ``(N, num_classes)``

    Each sequence is converted to 6-frame codon IDs via
    :func:`jaeger.data.raw_processors._make_process_raw_sequence_fn`.
    """
    data = np.load(path, allow_pickle=False)
    sequences = data["sequences"]
    labels = data["labels"]

    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))

    process_fn = _make_process_raw_sequence_fn(
        crop_size=crop_size,
        ngram_width=ngram_width,
        num_classes=num_classes,
        shuffle=shuffle,
        mutate=mutate,
        mutation_rate=mutation_rate,
        shuffle_frames=shuffle_frames,
    )

    dataset = dataset.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def _load_numpy_raw_variable_dataset(
    path: str,
    ngram_width: int = 3,
    num_classes: int = 3,
    shuffle: bool = False,
    mutate: bool = False,
    mutation_rate: float = 0.1,
    shuffle_frames: bool = False,
):
    """Loads a variable-length raw NumPy ``.npz`` file and returns a ``tf.data.Dataset``.

    The ``.npz`` must contain:
        - ``sequences``: int8 array of shape ``(N, max_length)`` (padded)
        - ``lengths``: int array of shape ``(N,)`` with actual lengths
        - ``labels``: float32 array of shape ``(N, num_classes)``
    """
    data = np.load(path, allow_pickle=False)
    sequences = data["sequences"]
    lengths = data["lengths"]
    labels = data["labels"]

    dataset = tf.data.Dataset.from_tensor_slices((sequences, lengths, labels))

    process_fn = _make_process_variable_sequence_fn(
        ngram_width=ngram_width,
        num_classes=num_classes,
        shuffle=shuffle,
        mutate=mutate,
        mutation_rate=mutation_rate,
        shuffle_frames=shuffle_frames,
    )

    dataset = dataset.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def _load_numpy_full_dataset(
    path: str,
    input_type: str = "translated",
    use_embedding_layer: bool = True,
    codon_depth: int = 21,
    crop_size: int = 500,
):
    """Loads a fully-preprocessed NumPy ``.npz`` file and returns a ``tf.data.Dataset``.

    Supports both structured arrays ``(N, 6, seq_len)`` and flattened arrays
    ``(N, flat)`` for backward compatibility. Also supports nucleotide input.

    The ``.npz`` file must contain:
        - ``translated`` or ``nucleotide``: preprocessed sequences
        - ``label``: float32 array of shape ``(N, num_classes)``

    No further preprocessing is applied — the data is ready for direct training.
    This gives the fastest possible data loading at the cost of no runtime
    augmentations (shuffle, mutate, frame_shuffle).
    """
    data = np.load(path, allow_pickle=False)

    if input_type == "translated":
        if use_embedding_layer:
            seq_shape = [6, crop_size // 3 - 1]
        else:
            seq_shape = [6, crop_size // 3 - 1, codon_depth]
        seq_key = "translated"
    elif input_type == "nucleotide":
        seq_shape = [2, crop_size, 4]
        seq_key = "nucleotide"
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    seqs = data[seq_key]
    labels = data["label"]

    # Ensure correct shapes if needed (flattened -> structured)
    expected_flat = np.prod(seq_shape)
    if seqs.ndim == 2 and seqs.shape[1] == expected_flat:
        seqs = seqs.reshape([-1] + list(seq_shape))

    dataset = tf.data.Dataset.from_tensor_slices(({seq_key: seqs}, labels))
    return dataset
