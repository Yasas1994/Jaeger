"""Dataset format converters.

Converts CSV training data to a compressed NumPy ``.npz`` archive.
Supported output representations:
- ``nucleotide`` — integer or one-hot encoded DNA crops.
- ``translated`` — 6-frame codon / dicodon embedding crops.
- ``both`` — both nucleotide and translated representations.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import zipfile
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import psutil

from jaeger.utils.logging import get_logger

logger = get_logger(log_file=None, log_path=None, level=3)


# ---------------------------------------------------------------------------
# Memory guard helpers (borrowed from the refactored converter)
# ---------------------------------------------------------------------------
def _count_lines(path: str) -> int:
    """Count newline-terminated rows without loading the whole file."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def _estimate_onehot_memory(
    total_rows: int,
    crop_size: int,
    fmt: str,
    one_hot: bool,
    codon_map_len: int | None = None,
    stride: int = 0,
) -> int:
    """Return the estimated raw float32 bytes needed for one-hot output."""
    if not one_hot:
        return 0

    total_rows = max(0, total_rows)
    crop_size = max(0, crop_size)
    estimated = 0

    # Overlapping crops increase the number of rows generated from the input.
    multiplier = 1
    if stride > 0 and crop_size > 0:
        multiplier = math.ceil(crop_size / stride)
    total_rows = total_rows * multiplier

    if fmt in ("nucleotide", "both"):
        estimated += total_rows * 2 * crop_size * 4 * np.dtype(np.float32).itemsize

    if fmt in ("translated", "both") and codon_map_len is not None:
        vocab_size = codon_map_len + 1
        max_codon_len = max(0, crop_size // 3 - 1)
        estimated += (
            total_rows * 6 * max_codon_len * vocab_size * np.dtype(np.float32).itemsize
        )

    return int(estimated)


def _estimate_output_bytes_per_row(
    crop_size: int,
    fmt: str,
    one_hot: bool,
    codon_map_len: int | None,
) -> int:
    """Safe upper-bound bytes for one output crop, assuming worst-case length."""
    crop_size = max(0, crop_size)
    per_row = 0
    if fmt in ("nucleotide", "both"):
        if one_hot:
            per_row += 2 * crop_size * 4 * np.dtype(np.float32).itemsize
        else:
            per_row += 2 * crop_size * np.dtype(np.int32).itemsize

    if fmt in ("translated", "both") and codon_map_len is not None:
        # Use codon length (crop_size // 3 - 1) because it is >= dicodon length.
        seq_len = max(0, crop_size // 3 - 1)
        if one_hot:
            per_row += 6 * seq_len * (codon_map_len + 1) * np.dtype(np.float32).itemsize
        else:
            per_row += 6 * seq_len * np.dtype(np.int32).itemsize

    return max(1, per_row)


def _estimate_total_bytes_per_input_row(
    crop_sizes: list[int],
    strides: list[int],
    fmt: str,
    one_hot: bool,
    codon_map_len: int | None,
) -> int:
    """Upper-bound bytes produced from one input sequence across all crop sizes.

    The estimate assumes every input sequence is at least as long as the largest
    crop size, so each crop size can contribute its maximum number of overlapping
    crops. This is intentionally conservative so the streaming batch size does
    not blow through the memory budget.
    """
    if not crop_sizes:
        return 1
    max_crop = max(crop_sizes)
    total = 0
    for crop_size, stride in zip(crop_sizes, strides):
        per_crop = _estimate_output_bytes_per_row(
            crop_size, fmt, one_hot, codon_map_len
        )
        max_crops = 1
        if stride > 0 and max_crop > 0:
            max_crops = math.ceil(max_crop / stride)
        total += per_crop * max_crops
    return max(1, total)


def _check_onehot_memory(
    estimated_bytes: int,
    available_bytes: int,
    fraction: float = 0.75,
) -> None:
    """Raise MemoryError if the estimated one-hot buffer is too large."""
    if estimated_bytes <= 0:
        return

    if estimated_bytes > available_bytes * fraction:
        raise MemoryError(
            f"Estimated one-hot output requires {estimated_bytes / (1024**3):.1f} GiB, "
            f"but only {available_bytes / (1024**3):.1f} GiB RAM is available "
            f"(safety fraction: {fraction:.0%}). "
            "Avoid one-hot encoding for large files; use integer encoding instead "
            "and let the dataloader convert to one-hot at training time."
        )


# ---------------------------------------------------------------------------
# Fast compression helper
# ---------------------------------------------------------------------------
def _save_npz(output_path: str, data: dict[str, np.ndarray], compress: str) -> None:
    """Save arrays to NPZ with the requested compression level."""
    compress = compress.lower()
    if compress == "default":
        np.savez_compressed(output_path, **data)
    elif compress == "none":
        np.savez(output_path, **data)
    elif compress == "fast":
        with zipfile.ZipFile(
            output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for key, arr in data.items():
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{key}.npy", buf.getvalue())
    else:
        raise ValueError(
            f"Invalid compress: {compress}. Choose from: default, none, fast"
        )


# ---------------------------------------------------------------------------
# Codon / nucleotide lookup helpers
# ---------------------------------------------------------------------------
_CODON_MAP_NAMES = {
    "codon_id": "CODON_ID",
    "aa_id": "AA_ID",
    "pc5_id": "PC5_ID",
    "murphy10_id": "MURPHY10_ID",
    "cod_id": "DICODON_ID",
    "pc2_id": "PC2_ID",
}

_DEFAULT_NUCLEOTIDE_MAP = {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}


def _get_codon_map(name: str) -> list[int]:
    """Return the codon map list from ``jaeger.seqops.maps`` by CLI name."""
    from jaeger.seqops import maps

    attr = _CODON_MAP_NAMES.get(name.lower())
    if attr is None or not hasattr(maps, attr):
        raise ValueError(f"Unknown codon map: {name}")
    return list(getattr(maps, attr))


def _parse_nucleotide_map(json_str: str | None) -> dict[str, int]:
    """Parse and validate a JSON nucleotide-to-int mapping."""
    if json_str is None:
        return dict(_DEFAULT_NUCLEOTIDE_MAP)
    try:
        mapping = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --nucleotide-map JSON: {exc}") from exc
    for base in "ACGTN":
        if base not in mapping:
            raise ValueError(f"--nucleotide-map must contain a mapping for {base}")
    return {k: int(v) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Numba availability
# ---------------------------------------------------------------------------
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator fallback when numba is not installed."""

        def wrapper(func):
            return func

        return wrapper


# ---------------------------------------------------------------------------
# Nucleotide batch encoder
# ---------------------------------------------------------------------------
def _build_nucleotide_lookups(user_map: dict[str, int]):
    """Build ASCII lookup tables for integer and one-hot nucleotide encoding.

    Returns
    -------
    ascii_to_user : np.ndarray, shape (256,)
        Maps ASCII code to the user-defined integer token. Unknown -> N value.
    comp_user : np.ndarray, shape (max_token + 1,)
        Maps a user integer token to its reverse-complement user token.
    ascii_to_oh : np.ndarray, shape (256,)
        Maps ASCII code to 0-3 one-hot index; N/unknown -> -1.
    comp_oh : np.ndarray, shape (4,)
        Maps one-hot base index to its reverse-complement one-hot index.
    """
    ascii_to_user = np.full(256, user_map["N"], dtype=np.int32)
    ascii_to_oh = np.full(256, -1, dtype=np.int32)

    oh_order = ["A", "G", "T", "C"]
    oh_index = {b: i for i, b in enumerate(oh_order)}
    comp_letters = {"A": "T", "T": "A", "G": "C", "C": "G"}

    for base in "ACGTN":
        ascii_to_user[ord(base)] = user_map[base]
        ascii_to_user[ord(base.lower())] = user_map[base]
        if base in oh_index:
            ascii_to_oh[ord(base)] = oh_index[base]
            ascii_to_oh[ord(base.lower())] = oh_index[base]

    max_token = max(user_map.values())
    comp_user = np.full(max_token + 1, user_map["N"], dtype=np.int32)
    for base, token in user_map.items():
        if base == "N":
            continue
        comp_letter = comp_letters.get(base, "N")
        comp_user[token] = user_map.get(comp_letter, user_map["N"])

    comp_oh = np.array([oh_index[comp_letters[b]] for b in oh_order], dtype=np.int32)
    return ascii_to_user, comp_user, ascii_to_oh, comp_oh


@njit(cache=False)
def _encode_nucleotide_batch_int(
    sequences,
    lengths,
    crop_size,
    ascii_to_user,
    comp_user,
    pad_int,
):
    """Encode a batch of DNA crops to integer nucleotide arrays."""
    n = len(lengths)
    out = np.full((n, 2, crop_size), pad_int, dtype=np.int32)

    for s in range(n):
        length = min(lengths[s], crop_size)
        for i in range(length):
            au = ascii_to_user[sequences[s, i]]
            out[s, 0, i] = au
            out[s, 1, i] = comp_user[au]
    return out


@njit(cache=False)
def _encode_nucleotide_batch_oh(
    sequences,
    lengths,
    crop_size,
    ascii_to_oh,
    comp_oh,
):
    """Encode a batch of DNA crops to one-hot nucleotide arrays."""
    n = len(lengths)
    out = np.zeros((n, 2, crop_size, 4), dtype=np.float32)

    for s in range(n):
        length = min(lengths[s], crop_size)
        for i in range(length):
            aoh = ascii_to_oh[sequences[s, i]]
            if 0 <= aoh < 4:
                out[s, 0, i, aoh] = 1.0
            coh = comp_oh[aoh] if 0 <= aoh < 4 else -1
            if 0 <= coh < 4:
                out[s, 1, i, coh] = 1.0
    return out


def _encode_nucleotide_batch(
    sequences,
    lengths,
    crop_size,
    ascii_to_user,
    comp_user,
    ascii_to_oh,
    comp_oh,
    one_hot,
    pad_int,
):
    """Encode a batch of DNA crops to nucleotide integer or one-hot arrays."""
    if one_hot:
        return _encode_nucleotide_batch_oh(
            sequences, lengths, crop_size, ascii_to_oh, comp_oh
        )
    return _encode_nucleotide_batch_int(
        sequences, lengths, crop_size, ascii_to_user, comp_user, pad_int
    )


# ---------------------------------------------------------------------------
# Crop generation and padding helpers
# ---------------------------------------------------------------------------
def _crop_starts(
    seq_len: int, crop_size: int, stride: int, pad_to_max: bool = True
) -> list[int]:
    """Return start indices for sliding-window crops.

    If ``stride`` is 0 or the sequence fits in one crop, a single start at 0 is
    returned. Otherwise the sequence is tiled with overlapping crops of
    ``crop_size``; a final tail window is appended when needed so the last base
    is covered.

    When ``pad_to_max`` is ``False``, starts are aligned to ``stride`` without
    the overlap-extension tail, so the final crop may be shorter than
    ``crop_size``.
    """
    if stride == 0 or seq_len <= crop_size:
        return [0]
    if pad_to_max:
        starts = list(range(0, seq_len - crop_size + 1, stride))
        if starts[-1] + crop_size < seq_len:
            starts.append(seq_len - crop_size)
    else:
        starts = list(range(0, seq_len, stride))
    return starts


def _generate_crops(seq_len: int, crop_sizes: list[int], strides: list[int]):
    """Generate (crop_size, start, length) tuples for one sequence.

    Yields one record per crop, ordered by ``crop_sizes`` then by start index.
    ``strides`` must have the same length as ``crop_sizes`` and provides the
    stride for each corresponding crop size.
    """
    for crop_size, stride in zip(crop_sizes, strides):
        starts = _crop_starts(seq_len, crop_size, stride)
        for start in starts:
            length = min(crop_size, seq_len - start)
            yield crop_size, start, length


def _pad_axis(
    arr: np.ndarray, target_len: int, axis: int = -1, pad_value: int | float = 0
) -> np.ndarray:
    """Pad or truncate ``arr`` along any axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    target_len : int
        Desired length along ``axis``.
    axis : int, optional
        Axis to pad or truncate (default: last axis).
    pad_value : int | float, optional
        Value used for padding (default: 0).

    Returns
    -------
    np.ndarray
        Array with shape ``(..., target_len, ...)`` along ``axis``.
    """
    axis = axis % arr.ndim
    if arr.shape[axis] >= target_len:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(None, target_len)
        return arr[tuple(slices)]
    pad_shape = list(arr.shape)
    pad_shape[axis] = target_len - arr.shape[axis]
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=axis)


def _pad_array(arr: np.ndarray, target_len: int, pad_value: int | float) -> np.ndarray:
    """Pad a variable-length encoded array along its last axis.

    Parameters
    ----------
    arr : np.ndarray
        Encoded crop with shape ``(..., length)``.
    target_len : int
        Desired length of the last axis.
    pad_value : int | float
        Value used for padding; must match ``arr.dtype``.

    Returns
    -------
    np.ndarray
        Array with shape ``(..., target_len)``.
    """
    return _pad_axis(arr, target_len, axis=-1, pad_value=pad_value)


# ---------------------------------------------------------------------------
# Codon / dicodon lookup helpers
# ---------------------------------------------------------------------------
_STANDARD_CODON_LUT3 = None


def _build_codon_lut(codon_map: list[int]) -> np.ndarray:
    """Build a flat 5-base lookup table mapping triplets to ``codon_map`` values.

    Unknown/invalid triplets map to ``-1``.
    """
    from jaeger.seqops.maps import CODONS

    lut = np.full(125, -1, dtype=np.int32)
    bases = ["A", "T", "G", "C", "N"]
    codon_to_id = {c: i for i, c in enumerate(CODONS)}
    for i in range(5):
        for j in range(5):
            for k in range(5):
                codon = bases[i] + bases[j] + bases[k]
                cid = codon_to_id.get(codon, -1)
                if cid >= 0:
                    lut[i * 25 + j * 5 + k] = codon_map[cid]
    return lut


def _build_standard_codon_lut3() -> np.ndarray:
    """Build a flat 5-base lookup table mapping valid triplets to CODONS indices.

    N-containing triplets map to ``-1``. The result is cached globally.
    """
    global _STANDARD_CODON_LUT3
    if _STANDARD_CODON_LUT3 is not None:
        return _STANDARD_CODON_LUT3

    from jaeger.seqops.maps import CODONS

    lut = np.full(125, -1, dtype=np.int32)
    bases = ["A", "T", "G", "C", "N"]
    codon_to_id = {c: i for i, c in enumerate(CODONS)}
    for i in range(5):
        for j in range(5):
            for k in range(5):
                codon = bases[i] + bases[j] + bases[k]
                cid = codon_to_id.get(codon, -1)
                if cid >= 0:
                    lut[i * 25 + j * 5 + k] = cid
    _STANDARD_CODON_LUT3 = lut
    return lut


def _build_dicodon_lut(codon_map: list[int]) -> np.ndarray:
    """Build a lookup table of dicodon map values.

    For ``DICODON_ID`` the returned table is the identity mapping ``0..4095``.
    """
    return np.array(codon_map, dtype=np.int32)


@njit(cache=False)
def _single_codon_actual_lengths(lengths, crop_size):
    """Return the number of valid codons in the shortest frame per crop."""
    n = len(lengths)
    out = np.zeros(n, dtype=np.int32)
    max_len = crop_size // 3 - 1
    for s in range(n):
        length = min(lengths[s], crop_size)
        if length < 3:
            continue
        offset = [-2, -1, 0][length % 3]
        ngrams_len = length - 3 + 1
        end_f1 = ngrams_len - 3 + offset
        end_f2 = ngrams_len - 2 + offset
        end_f3 = ngrams_len - 1 + offset
        nf1 = max(0, (end_f1 + 2) // 3)
        nf2 = max(0, (end_f2 + 1) // 3)
        nf3 = max(0, end_f3 // 3)
        min_len = min(nf1, nf2, nf3)
        if min_len <= 0:
            continue
        out[s] = min(min_len, max_len)
    return out


@njit(cache=False)
def _dicodon_actual_lengths(lengths, crop_size):
    """Return the number of valid overlapping dicodons per crop."""
    single = _single_codon_actual_lengths(lengths, crop_size)
    max_len = max(0, crop_size // 3 - 2)
    n = len(single)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        val = single[i] - 1
        if val < 0:
            out[i] = 0
        elif val > max_len:
            out[i] = max_len
        else:
            out[i] = val
    return out


def _one_hot_integer(indices: np.ndarray, depth: int) -> np.ndarray:
    """Convert an integer array to a float32 one-hot array along a new last axis.

    Only indices in ``[1, depth - 1]`` produce a one-hot vector; index ``0`` and
    negative values stay all-zero.
    """
    out = np.zeros(indices.shape + (depth,), dtype=np.float32)
    valid_mask = (indices > 0) & (indices < depth)
    if not np.any(valid_mask):
        return out
    coords = np.where(valid_mask)
    channel_indices = indices[valid_mask] - 1
    out[coords + (channel_indices,)] = 1.0
    return out


@njit(cache=False)
def _process_batch_numba_dicodon(
    sequences, lengths, crop_size, seq_len, codon_lut3, dicodon_lut, comp_lut, ascii_lut
):
    """Process a batch of DNA sequences to 6-frame dicodon embeddings."""
    n = len(lengths)
    out = np.zeros((n, 6, seq_len), dtype=np.int32)

    for s in range(n):
        length = min(lengths[s], crop_size)
        if length < 6:
            continue

        bases = np.empty(length, dtype=np.int8)
        for i in range(length):
            bases[i] = ascii_lut[sequences[s, i]]

        rev_bases = np.empty(length, dtype=np.int8)
        for i in range(length):
            rev_bases[i] = comp_lut[bases[length - 1 - i]]

        offset = [-2, -1, 0][length % 3]
        ngrams_len = length - 3 + 1
        end_f1 = ngrams_len - 3 + offset
        end_f2 = ngrams_len - 2 + offset
        end_f3 = ngrams_len - 1 + offset
        nf1 = max(0, (end_f1 + 2) // 3)
        nf2 = max(0, (end_f2 + 1) // 3)
        nf3 = max(0, end_f3 // 3)
        min_len = min(nf1, nf2, nf3)
        if min_len <= 1:
            continue
        dicodon_len = min(min_len - 1, seq_len)

        for f in range(3):
            for j in range(dicodon_len):
                start1 = f + j * 3
                start2 = start1 + 3
                idx0 = bases[start1]
                idx1 = bases[start1 + 1]
                idx2 = bases[start1 + 2]
                c1 = codon_lut3[idx0 * 25 + idx1 * 5 + idx2]
                idx0 = bases[start2]
                idx1 = bases[start2 + 1]
                idx2 = bases[start2 + 2]
                c2 = codon_lut3[idx0 * 25 + idx1 * 5 + idx2]
                if c1 < 0 or c2 < 0:
                    continue
                dicodon_idx = c1 * 64 + c2
                out[s, f, j] = dicodon_lut[dicodon_idx] + 1

        for f in range(3):
            for j in range(dicodon_len):
                start1 = f + j * 3
                start2 = start1 + 3
                idx0 = rev_bases[start1]
                idx1 = rev_bases[start1 + 1]
                idx2 = rev_bases[start1 + 2]
                c1 = codon_lut3[idx0 * 25 + idx1 * 5 + idx2]
                idx0 = rev_bases[start2]
                idx1 = rev_bases[start2 + 1]
                idx2 = rev_bases[start2 + 2]
                c2 = codon_lut3[idx0 * 25 + idx1 * 5 + idx2]
                if c1 < 0 or c2 < 0:
                    continue
                dicodon_idx = c1 * 64 + c2
                out[s, f + 3, j] = dicodon_lut[dicodon_idx] + 1

    return out


# ---------------------------------------------------------------------------
# Numba-optimized codon processing
# ---------------------------------------------------------------------------
# Precomputed lookup tables for numba
_CODON_LUT = None
_ASCII_TO_IDX = None
_COMP_LUT = None


def _build_numba_lookups():
    """Build flat lookup tables for numba-optimized processing."""
    global _CODON_LUT, _ASCII_TO_IDX, _COMP_LUT
    if _CODON_LUT is not None:
        return _CODON_LUT, _ASCII_TO_IDX, _COMP_LUT

    from jaeger.seqops.maps import CODONS

    base_to_idx = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}

    # Complement lookup: A<->T, G<->C, N<->N
    _COMP_LUT = np.array([1, 0, 3, 2, 4], dtype=np.int8)

    # Flat codon lookup: 125 entries (5x5x5)
    codon_to_id = {c: i for i, c in enumerate(CODONS)}
    _CODON_LUT = np.full(125, -1, dtype=np.int32)
    bases = ["A", "T", "G", "C", "N"]
    for i in range(5):
        for j in range(5):
            for k in range(5):
                codon = bases[i] + bases[j] + bases[k]
                _CODON_LUT[i * 25 + j * 5 + k] = codon_to_id.get(codon, -1)

    # ASCII to base index lookup
    _ASCII_TO_IDX = np.full(256, 4, dtype=np.int8)
    for c, i in base_to_idx.items():
        _ASCII_TO_IDX[ord(c)] = i

    return _CODON_LUT, _ASCII_TO_IDX, _COMP_LUT


@njit(cache=False)
def _process_batch_numba(
    sequences, lengths, crop_size, seq_len, codon_lut, comp_lut, ascii_lut
):
    """Process a batch of DNA sequences to 6-frame codon embeddings."""
    n = len(lengths)
    out = np.zeros((n, 6, seq_len), dtype=np.int32)

    for s in range(n):
        length = min(lengths[s], crop_size)

        # Convert ASCII to base indices
        bases = np.empty(length, dtype=np.int8)
        for i in range(length):
            bases[i] = ascii_lut[sequences[s, i]]

        # Compute frame lengths
        offset = [-2, -1, 0][length % 3]
        ngrams_len = length - 3 + 1

        end_f1 = ngrams_len - 3 + offset
        end_f2 = ngrams_len - 2 + offset
        end_f3 = ngrams_len - 1 + offset

        nf1 = max(0, (end_f1 + 2) // 3)
        nf2 = max(0, (end_f2 + 1) // 3)
        nf3 = max(0, end_f3 // 3)

        # Reverse complement
        rev_bases = np.empty(length, dtype=np.int8)
        for i in range(length):
            rev_bases[i] = comp_lut[bases[length - 1 - i]]

        nr1 = max(0, (end_f1 + 2) // 3)
        nr2 = max(0, (end_f2 + 1) // 3)
        nr3 = max(0, end_f3 // 3)

        min_len = min(nf1, nf2, nf3, nr1, nr2, nr3)
        if min_len <= 0:
            continue
        min_len = min(min_len, seq_len)

        for i in range(min_len):
            idx0 = bases[i * 3]
            idx1 = bases[i * 3 + 1]
            idx2 = bases[i * 3 + 2]
            out[s, 0, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2] + 1

            idx0 = bases[i * 3 + 1]
            idx1 = bases[i * 3 + 2]
            idx2 = bases[i * 3 + 3]
            out[s, 1, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2] + 1

            idx0 = bases[i * 3 + 2]
            idx1 = bases[i * 3 + 3]
            idx2 = bases[i * 3 + 4]
            out[s, 2, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2] + 1

        for i in range(min_len):
            idx0 = rev_bases[i * 3]
            idx1 = rev_bases[i * 3 + 1]
            idx2 = rev_bases[i * 3 + 2]
            out[s, 3, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2] + 1

            idx0 = rev_bases[i * 3 + 1]
            idx1 = rev_bases[i * 3 + 2]
            idx2 = rev_bases[i * 3 + 3]
            out[s, 4, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2] + 1

            idx0 = rev_bases[i * 3 + 2]
            idx1 = rev_bases[i * 3 + 3]
            idx2 = rev_bases[i * 3 + 4]
            out[s, 5, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2] + 1

    return out


# ---------------------------------------------------------------------------
# NPZ converter (nucleotide / translated / dicodon)
# ---------------------------------------------------------------------------
def _process_chunk_npz(
    lines,
    fmt,
    crop_sizes,
    strides,
    one_hot,
    pad_int,
    nucleotide_lookups,
    codon_lut,
    codon_map_len,
    standard_codon_lut3,
    dicodon_lut,
    ascii_lut,
    comp_lut,
):
    """Process a chunk of CSV lines into encoded crops for ``.npz`` output.

    Returns a dict with per-crop ``nucleotide``, ``translated``, ``labels``,
    ``lengths``, and ``translated_lengths``.
    """
    ascii_to_user, comp_user, ascii_to_oh, comp_oh = nucleotide_lookups
    parsed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        comma_idx = line.find(",")
        if comma_idx == -1:
            continue
        label = int(line[:comma_idx])
        seq = line[comma_idx + 1 :]
        comma2 = seq.find(",")
        if comma2 != -1:
            seq = seq[:comma2]
        seq = seq.strip()
        parsed.append((label, seq))

    nucleotide_arrays = []
    translated_arrays = []
    all_labels = []
    all_lengths = []
    all_translated_lengths = []

    for crop_size, stride in zip(crop_sizes, strides):
        batch_seqs = []
        batch_lengths = []
        batch_labels = []

        for label, seq in parsed:
            seq_bytes = seq.encode()
            seq_len = len(seq_bytes)
            starts = _crop_starts(seq_len, crop_size, stride)
            for start in starts:
                length = min(crop_size, seq_len - start)
                crop = np.full(crop_size, ord("N"), dtype=np.uint8)
                crop[:length] = np.frombuffer(
                    seq_bytes[start : start + length], dtype=np.uint8
                )
                batch_seqs.append(crop)
                batch_lengths.append(length)
                batch_labels.append(label)

        if batch_seqs:
            batch = np.stack(batch_seqs, axis=0)
            lengths_arr = np.array(batch_lengths, dtype=np.int32)

            if fmt in ("nucleotide", "both"):
                encoded = _encode_nucleotide_batch(
                    batch,
                    lengths_arr,
                    crop_size,
                    ascii_to_user,
                    comp_user,
                    ascii_to_oh,
                    comp_oh,
                    one_hot,
                    pad_int,
                )
                nucleotide_arrays.append(encoded)

            if fmt in ("translated", "both"):
                if codon_map_len != 4096:
                    seq_len = crop_size // 3 - 1
                    encoded = _process_batch_numba(
                        batch,
                        lengths_arr,
                        crop_size,
                        seq_len,
                        codon_lut,
                        comp_lut,
                        ascii_lut,
                    )
                    actual_lengths = _single_codon_actual_lengths(
                        lengths_arr, crop_size
                    )
                else:
                    seq_len = max(0, crop_size // 3 - 2)
                    encoded = _process_batch_numba_dicodon(
                        batch,
                        lengths_arr,
                        crop_size,
                        seq_len,
                        standard_codon_lut3,
                        dicodon_lut,
                        comp_lut,
                        ascii_lut,
                    )
                    actual_lengths = _dicodon_actual_lengths(lengths_arr, crop_size)
                translated_arrays.append(encoded)
                all_translated_lengths.extend(actual_lengths.tolist())
            else:
                all_translated_lengths.extend([0] * len(batch_labels))

            all_labels.extend(batch_labels)
            all_lengths.extend(batch_lengths)
        else:
            # Keep per-crop-size list length consistent across workers.
            if fmt in ("nucleotide", "both"):
                empty_shape = (0, 2, crop_size, 4) if one_hot else (0, 2, crop_size)
                empty_dtype = np.float32 if one_hot else np.int32
                nucleotide_arrays.append(np.empty(empty_shape, dtype=empty_dtype))
            if fmt in ("translated", "both"):
                if codon_map_len != 4096:
                    seq_len = max(0, crop_size // 3 - 1)
                else:
                    seq_len = max(0, crop_size // 3 - 2)
                translated_arrays.append(np.empty((0, 6, seq_len), dtype=np.int32))

    return {
        "nucleotide": nucleotide_arrays,
        "translated": translated_arrays,
        "labels": np.array(all_labels, dtype=np.int32),
        "lengths": np.array(all_lengths, dtype=np.int32),
        "translated_lengths": np.array(all_translated_lengths, dtype=np.int32),
    }


def _finalize_batch_arrays(
    result: dict,
    fmt: str,
    crop_sizes: list[int],
    one_hot: bool,
    codon_map_len: int | None,
    pad: bool,
    pad_int: int,
    nucleotide_map: dict[str, int],
    codon_map_name: str,
    feature_dtype: np.dtype | None = None,
) -> dict[str, np.ndarray | str]:
    """Turn a processed chunk into save-ready arrays.

    ``result`` is the dict returned by ``_process_chunk_npz``. If ``pad`` is
    True the arrays are padded to the global maximum length (legacy behavior).
    If ``pad`` is False each crop is trimmed to its actual length and stored
    in a 1-D object array.
    """
    if feature_dtype is None:
        feature_dtype = np.dtype(np.float32) if one_hot else np.dtype(np.int32)

    save_dict: dict[str, np.ndarray | str] = {
        "labels": result["labels"],
        "lengths": result["lengths"],
        "translated_lengths": result["translated_lengths"],
    }

    def _to_object_array(items: list[np.ndarray]) -> np.ndarray:
        arr = np.empty(len(items), dtype=object)
        arr[:] = items
        return arr

    if fmt in ("nucleotide", "both"):
        arrays = result["nucleotide"]
        if arrays:
            if pad:
                if one_hot:
                    max_len = max(a.shape[-2] for a in arrays)
                    padded = [
                        _pad_axis(a, max_len, axis=-2, pad_value=0.0) for a in arrays
                    ]
                    save_dict["nucleotide"] = np.concatenate(padded, axis=0).astype(
                        feature_dtype, copy=False
                    )
                else:
                    max_len = max(a.shape[-1] for a in arrays)
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=pad_int)
                        for a in arrays
                    ]
                    save_dict["nucleotide"] = np.concatenate(padded, axis=0).astype(
                        feature_dtype, copy=False
                    )
            else:
                items: list[np.ndarray] = []
                offset = 0
                for arr, _ in zip(arrays, crop_sizes):
                    n = arr.shape[0]
                    lens = result["lengths"][offset : offset + n]
                    for i in range(n):
                        L = int(lens[i])
                        if one_hot:
                            items.append(
                                arr[i, :, :L, :].astype(feature_dtype, copy=False)
                            )
                        else:
                            items.append(
                                arr[i, :, :L].astype(feature_dtype, copy=False)
                            )
                    offset += n
                save_dict["nucleotide"] = _to_object_array(items)
        else:
            save_dict["nucleotide"] = np.empty((0,), dtype=feature_dtype)
        save_dict["nucleotide_map"] = json.dumps(nucleotide_map)

    if fmt in ("translated", "both"):
        arrays = result["translated"]
        if arrays:
            if pad:
                max_len = max(a.shape[-1] for a in arrays)
                if one_hot and codon_map_len is not None:
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=0) for a in arrays
                    ]
                    stacked = np.concatenate(padded, axis=0)
                    save_dict["translated"] = _one_hot_integer(
                        stacked, codon_map_len + 1
                    ).astype(feature_dtype, copy=False)
                else:
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=0) for a in arrays
                    ]
                    save_dict["translated"] = np.concatenate(padded, axis=0).astype(
                        feature_dtype, copy=False
                    )
            else:
                items: list[np.ndarray] = []
                offset = 0
                for arr, _ in zip(arrays, crop_sizes):
                    n = arr.shape[0]
                    if one_hot and codon_map_len is not None:
                        arr = _one_hot_integer(arr, codon_map_len + 1)
                    lens = result["translated_lengths"][offset : offset + n]
                    for i in range(n):
                        L = int(lens[i])
                        if one_hot:
                            items.append(
                                arr[i, :, :L, :].astype(feature_dtype, copy=False)
                            )
                        else:
                            items.append(
                                arr[i, :, :L].astype(feature_dtype, copy=False)
                            )
                    offset += n
                save_dict["translated"] = _to_object_array(items)
        else:
            save_dict["translated"] = np.empty((0,), dtype=feature_dtype)
        save_dict["codon_map"] = codon_map_name

    return save_dict


def _convert_to_npz(
    input_path: str,
    output_path: str,
    fmt: str,
    crop_sizes: list[int],
    strides: list[int],
    num_classes: int,
    num_workers: int | None,
    one_hot: bool,
    pad_int: int,
    codon_map_name: str,
    nucleotide_map: dict[str, int],
    compress: str,
    pad: bool = True,
    dtype: str = "auto",
) -> None:
    """Convert a CSV dataset to an ``.npz`` file.

    Supports ``nucleotide``, ``translated``, and ``both`` output formats, with
    optional one-hot encoding and per-crop-size strides.
    """
    import warnings

    valid_fmts = ["nucleotide", "translated", "both"]
    if fmt not in valid_fmts:
        raise ValueError(f"Invalid format: {fmt}. Choose from: {', '.join(valid_fmts)}")

    if len(strides) != len(crop_sizes):
        raise ValueError(
            f"strides ({len(strides)}) must match crop_sizes ({len(crop_sizes)})"
        )

    if not output_path.endswith(".npz"):
        warnings.warn(
            f"Output path {output_path!r} does not end with '.npz'", stacklevel=2
        )

    with open(input_path) as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    if num_workers is None:
        num_workers = cpu_count()
    if len(lines) < 1000:
        num_workers = 1
    num_workers = max(1, min(num_workers, len(lines)))
    logger.info(f"Using {num_workers} workers")

    nucleotide_lookups = _build_nucleotide_lookups(nucleotide_map)
    _, ascii_lut, comp_lut = _build_numba_lookups()

    codon_map = _get_codon_map(codon_map_name)
    codon_map_len = len(codon_map)
    feature_dtype = _resolve_feature_dtype(
        fmt, one_hot, dtype, codon_map_len, nucleotide_map
    )
    if codon_map_len == 4096:
        standard_codon_lut3 = _build_standard_codon_lut3()
        dicodon_lut = _build_dicodon_lut(codon_map)
        codon_lut = np.empty(0, dtype=np.int32)
    else:
        codon_lut = _build_codon_lut(codon_map)
        standard_codon_lut3 = np.empty(0, dtype=np.int32)
        dicodon_lut = np.empty(0, dtype=np.int32)

    # Warm up Numba functions in the parent process so worker threads use
    # already-compiled code and don't each pay the compilation cost.
    if HAS_NUMBA and num_workers > 1:
        _crop_size = max(crop_sizes) if crop_sizes else 500
        _warmup = np.full((1, _crop_size), ord("A"), dtype=np.uint8)
        _warmup_len = np.array([_crop_size], dtype=np.int32)
        if fmt in ("nucleotide", "both"):
            _encode_nucleotide_batch(
                _warmup,
                _warmup_len,
                _crop_size,
                *nucleotide_lookups,
                one_hot,
                pad_int,
            )
        if fmt in ("translated", "both"):
            if codon_map_len == 4096:
                _process_batch_numba_dicodon(
                    _warmup,
                    _warmup_len,
                    _crop_size,
                    max(0, _crop_size // 3 - 2),
                    standard_codon_lut3,
                    dicodon_lut,
                    comp_lut,
                    ascii_lut,
                )
            else:
                _process_batch_numba(
                    _warmup,
                    _warmup_len,
                    _crop_size,
                    _crop_size // 3 - 1,
                    codon_lut,
                    comp_lut,
                    ascii_lut,
                )

    worker_kwargs = {
        "fmt": fmt,
        "crop_sizes": crop_sizes,
        "strides": strides,
        "one_hot": one_hot,
        "pad_int": pad_int,
        "nucleotide_lookups": nucleotide_lookups,
        "codon_lut": codon_lut,
        "codon_map_len": codon_map_len,
        "standard_codon_lut3": standard_codon_lut3,
        "dicodon_lut": dicodon_lut,
        "ascii_lut": ascii_lut,
        "comp_lut": comp_lut,
    }

    if num_workers <= 1:
        result = _process_chunk_npz(lines, **worker_kwargs)
    else:
        chunk_size = max(1, len(lines) // num_workers)
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        process_fn = partial(_process_chunk_npz, **worker_kwargs)
        with ThreadPool(num_workers) as pool:
            results = pool.map(process_fn, chunks)
        result = {
            "nucleotide": [
                np.concatenate(
                    [r["nucleotide"][i] for r in results],
                    axis=0,
                )
                for i in range(len(results[0]["nucleotide"]))
            ],
            "translated": [
                np.concatenate(
                    [r["translated"][i] for r in results],
                    axis=0,
                )
                for i in range(len(results[0]["translated"]))
            ],
            "labels": np.concatenate([r["labels"] for r in results], axis=0),
            "lengths": np.concatenate([r["lengths"] for r in results], axis=0),
            "translated_lengths": np.concatenate(
                [r["translated_lengths"] for r in results], axis=0
            ),
        }

    save_dict = _finalize_batch_arrays(
        result,
        fmt=fmt,
        crop_sizes=crop_sizes,
        one_hot=one_hot,
        codon_map_len=codon_map_len,
        pad=pad,
        pad_int=pad_int,
        nucleotide_map=nucleotide_map,
        codon_map_name=codon_map_name,
        feature_dtype=feature_dtype,
    )
    save_dict.update(
        {
            "pad_int": np.int32(pad_int),
            "crop_sizes": np.array(crop_sizes, dtype=np.int32),
            "strides": np.array(strides, dtype=np.int32),
            "padded": np.bool_(pad),
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _save_npz(output_path, save_dict, compress)


def _zip_compression(compress: str) -> tuple[int, int | None]:
    """Return ``(zipfile compression type, compresslevel)`` for NPZ output."""
    compress = compress.lower()
    if compress == "default":
        return zipfile.ZIP_DEFLATED, None
    if compress == "fast":
        return zipfile.ZIP_DEFLATED, 1
    if compress == "none":
        return zipfile.ZIP_STORED, None
    raise ValueError(f"Invalid compress: {compress}. Choose from: default, none, fast")


def _resolve_feature_dtype(
    fmt: str,
    one_hot: bool,
    dtype_arg: str,
    codon_map_len: int | None,
    nucleotide_map: dict[str, int],
) -> np.dtype:
    """Return the smallest NumPy dtype that fits the integer vocabulary.

    One-hot float outputs always use ``float32``. For integer outputs the
    default ``auto`` mode picks ``int8`` / ``uint8`` / ``int16`` / ``int32``
    based on the maximum token value.
    """
    if one_hot:
        return np.dtype(np.float32)

    dtype_arg = dtype_arg.lower()
    if dtype_arg != "auto":
        return np.dtype(dtype_arg)

    max_token = 0
    if fmt in ("translated", "both") and codon_map_len is not None:
        max_token = max(max_token, codon_map_len)
    if fmt in ("nucleotide", "both"):
        max_token = max(max_token, max(nucleotide_map.values()))

    if max_token < 128:
        return np.dtype(np.int8)
    if max_token < 256:
        return np.dtype(np.uint8)
    if max_token < 32768:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def _convert_to_npz_streaming(
    input_path: str,
    output_path: str,
    fmt: str,
    crop_sizes: list[int],
    strides: list[int],
    num_classes: int,
    one_hot: bool,
    pad_int: int,
    codon_map_name: str,
    nucleotide_map: dict[str, int],
    compress: str,
    max_memory_bytes: int,
    pad: bool,
    dtype: str = "auto",
) -> None:
    """Memory-bounded CSV -> NPZ converter that writes sharded batches.

    Each encoded batch is written directly into the output ``.npz`` archive as
    a separate ``.npy`` entry (e.g. ``translated_00000.npy``). The loader
    reassembles samples on-the-fly, so the converter never needs to hold the
    final arrays in memory. The only memory spike is the single batch being
    encoded, which is bounded by ``max_memory_bytes``.
    """
    codon_map_len: int | None = None
    if fmt in ("translated", "both"):
        codon_map_len = len(_get_codon_map(codon_map_name))

    feature_dtype = _resolve_feature_dtype(
        fmt, one_hot, dtype, codon_map_len, nucleotide_map
    )

    total_per_row = _estimate_total_bytes_per_input_row(
        crop_sizes, strides, fmt, one_hot, codon_map_len
    )
    batch_rows = max(1, int(max_memory_bytes * 0.5 / total_per_row))

    nucleotide_lookups = _build_nucleotide_lookups(nucleotide_map)
    _, ascii_lut, comp_lut = _build_numba_lookups()

    codon_map = _get_codon_map(codon_map_name)
    if len(codon_map) == 4096:
        standard_codon_lut3 = _build_standard_codon_lut3()
        dicodon_lut = _build_dicodon_lut(codon_map)
        codon_lut = np.empty(0, dtype=np.int32)
    else:
        codon_lut = _build_codon_lut(codon_map)
        standard_codon_lut3 = np.empty(0, dtype=np.int32)
        dicodon_lut = np.empty(0, dtype=np.int32)

    worker_kwargs = {
        "fmt": fmt,
        "crop_sizes": crop_sizes,
        "strides": strides,
        "one_hot": one_hot,
        "pad_int": pad_int,
        "nucleotide_lookups": nucleotide_lookups,
        "codon_lut": codon_lut,
        "codon_map_len": codon_map_len if codon_map_len is not None else 0,
        "standard_codon_lut3": standard_codon_lut3,
        "dicodon_lut": dicodon_lut,
        "ascii_lut": ascii_lut,
        "comp_lut": comp_lut,
    }

    compress_type, compress_level = _zip_compression(compress)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        keys: list[str] = []
        batch_idx = 0
        with zipfile.ZipFile(
            output_path,
            "w",
            compression=compress_type,
            compresslevel=compress_level or 6,
        ) as zf:
            with open(input_path) as f:
                while True:
                    batch_lines = list(itertools.islice(f, batch_rows))
                    if not batch_lines:
                        break

                    result = _process_chunk_npz(batch_lines, **worker_kwargs)
                    batch_dict = _finalize_batch_arrays(
                        result,
                        fmt=fmt,
                        crop_sizes=crop_sizes,
                        one_hot=one_hot,
                        codon_map_len=codon_map_len,
                        pad=pad,
                        pad_int=pad_int,
                        nucleotide_map=nucleotide_map,
                        codon_map_name=codon_map_name,
                        feature_dtype=feature_dtype,
                    )

                    if not keys:
                        keys = [
                            k
                            for k in batch_dict.keys()
                            if k not in ("nucleotide_map", "codon_map")
                        ]

                    for key in keys:
                        buf = io.BytesIO()
                        np.save(buf, batch_dict[key])
                        zf.writestr(
                            f"{key}_{batch_idx:05d}.npy",
                            buf.getvalue(),
                            compress_type=compress_type,
                            compresslevel=compress_level,
                        )
                    batch_idx += 1

            if batch_idx == 0:
                raise ValueError(f"Input file is empty: {input_path}")

            manifest = {
                "version": 1,
                "keys": keys,
                "num_shards": batch_idx,
                "format": fmt,
                "crop_sizes": crop_sizes,
                "strides": strides,
                "padded": pad,
                "pad_int": int(pad_int),
                "one_hot": one_hot,
                "num_classes": int(num_classes),
                "codon_map": codon_map_name if fmt in ("translated", "both") else None,
                "nucleotide_map": (
                    json.dumps(nucleotide_map)
                    if fmt in ("nucleotide", "both")
                    else None
                ),
            }
            buf = io.BytesIO()
            np.save(buf, np.array(json.dumps(manifest), dtype=object))
            zf.writestr(
                "_jaeger_manifest.npy",
                buf.getvalue(),
                compress_type=compress_type,
                compresslevel=compress_level,
            )
    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def convert_dataset(
    input_path: str,
    output_path: str,
    format: str,
    crop_size: int | tuple[int, ...] | list[int] = 500,
    stride: int = 0,
    strides: list[int] | None = None,
    num_classes: int = 3,
    num_workers: int | None = None,
    one_hot: bool = False,
    pad_int: int = 0,
    codon_map: str = "codon_id",
    nucleotide_map: str | None = None,
    compress: str = "default",
    dtype: str = "auto",
    max_length: int = 5000,  # deprecated, ignored
    use_embedding_layer: bool = True,  # deprecated, ignored
    max_memory_mb: int | None = None,
    pad: bool = False,
) -> None:
    """Convert a CSV dataset to a compressed NumPy ``.npz`` file.

    This function dispatches to either :func:`_convert_to_npz` or
    :func:`_convert_to_npz_streaming` depending on the available memory budget.
    It supports three output representations: ``nucleotide``, ``translated``,
    and ``both``. The resulting ``.npz`` archive contains encoded crops, labels,
    crop metadata, and (depending on the format) the nucleotide/codon mapping
    used during encoding.

    When the streaming converter is used, the output is a *sharded* NPZ: each
    encoded batch is stored as a separate ``.npy`` entry (e.g.
    ``translated_00000.npy``) and a ``_jaeger_manifest.npy`` entry records the
    shard layout. This keeps peak memory bounded by ``max_memory_mb`` instead
    of materialising the full dataset in RAM.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (label,sequence format).
    output_path : str
        Path to output ``.npz`` file.
    format : str
        One of ``nucleotide``, ``translated``, or ``both`` (case-insensitive).
    crop_size : int | tuple[int, ...] | list[int], optional
        Sequence crop size(s). An int is wrapped as ``[crop_size]``; a tuple or
        list is converted to a list (default: 500).
    stride : int, optional
        Sliding-window stride applied to every crop size. ``0`` means a single
        crop per sequence (default: 0). Ignored when ``strides`` is provided.
    strides : list[int] | None, optional
        Per-crop-size strides. If given, it overrides ``stride`` and must have
        the same length as the resolved ``crop_sizes``.
    num_classes : int, optional
        Number of output classes (default: 3).
    num_workers : int | None, optional
        Number of parallel workers for CPU-bound encoding. ``None`` uses all
        CPUs (default: None). For very small datasets (< 1000 samples) the
        converter falls back to a single worker to avoid multiprocessing
        overhead.
    one_hot : bool, optional
        Encode nucleotide crops as one-hot float tensors instead of integer
        tokens (default: False).
    pad_int : int, optional
        Integer value used for padding nucleotide crops (default: 0).
    codon_map : str, optional
        Codon map name, e.g. ``codon_id``, ``aa_id``, ``cod_id``
        (default: ``codon_id``).
    nucleotide_map : str | None, optional
        JSON string with mappings for ``A``, ``C``, ``G``, ``T``, ``N``. Uses
        the default mapping when ``None``.
    compress : str, optional
        Compression mode for ``np.savez``: ``default`` uses
        ``np.savez_compressed``; ``none`` uses ``np.savez``.
    max_length : int, optional
        Deprecated and ignored. Kept for backward compatibility with old CLI
        calls.
    use_embedding_layer : bool, optional
        Deprecated and ignored. Kept for backward compatibility with old CLI
        calls.
    max_memory_mb : int | None, optional
        Memory budget in megabytes used to decide whether to use the streaming
        converter. ``None`` uses 75% of available RAM. ``0`` or negative values
        disable the budget check and always use the fast path.
    pad : bool, optional
        If True, pad all crops to the maximum crop length (legacy behavior).
        If False, trim each crop to its actual length and store as object
        arrays.
    """
    if isinstance(crop_size, int):
        crop_sizes = [crop_size]
    else:
        crop_sizes = list(crop_size)

    if strides is None:
        strides = [stride] * len(crop_sizes)
    elif len(strides) != len(crop_sizes):
        raise ValueError(
            f"strides ({len(strides)}) must match crop_sizes ({len(crop_sizes)})"
        )

    format = format.lower()
    valid_formats = ["nucleotide", "translated", "both"]
    if format not in valid_formats:
        raise ValueError(
            f"Invalid format: {format}. Choose from: {', '.join(valid_formats)}"
        )

    nucleotide_map_dict = _parse_nucleotide_map(nucleotide_map)

    logger.info(f"Converting {input_path} -> {output_path}")
    logger.info(
        f"Format: {format}, Crop sizes: {crop_sizes}, Strides: {strides}, "
        f"Num classes: {num_classes}, Pad: {pad}"
    )

    total_lines = _count_lines(input_path)
    if crop_sizes:
        max_crop_idx = int(np.argmax(crop_sizes))
        max_crop = crop_sizes[max_crop_idx]
        max_stride = strides[max_crop_idx]
    else:
        max_crop = 500
        max_stride = 0

    codon_map_len: int | None = None
    if format in ("translated", "both"):
        codon_map_arr = _get_codon_map(codon_map)
        codon_map_len = len(codon_map_arr)

    if max_memory_mb is not None and max_memory_mb > 0:
        budget = max_memory_mb * 1024 * 1024
    elif max_memory_mb is None:
        budget = int(psutil.virtual_memory().available * 0.75)
    else:
        budget = None

    stream = False
    if budget is not None:
        per_row = _estimate_total_bytes_per_input_row(
            crop_sizes, strides, format, one_hot, codon_map_len
        )
        stream = total_lines * per_row > budget

    if one_hot and not stream:
        estimated = _estimate_onehot_memory(
            total_rows=total_lines,
            crop_size=max_crop,
            fmt=format,
            one_hot=one_hot,
            codon_map_len=codon_map_len,
            stride=max_stride,
        )
        _check_onehot_memory(estimated, psutil.virtual_memory().available)

    if stream:
        _convert_to_npz_streaming(
            input_path=input_path,
            output_path=output_path,
            fmt=format,
            crop_sizes=crop_sizes,
            strides=strides,
            num_classes=num_classes,
            one_hot=one_hot,
            pad_int=pad_int,
            codon_map_name=codon_map,
            nucleotide_map=nucleotide_map_dict,
            compress=compress,
            dtype=dtype,
            max_memory_bytes=budget,
            pad=pad,
        )
    else:
        _convert_to_npz(
            input_path=input_path,
            output_path=output_path,
            fmt=format,
            crop_sizes=crop_sizes,
            strides=strides,
            num_classes=num_classes,
            num_workers=num_workers,
            one_hot=one_hot,
            pad_int=pad_int,
            codon_map_name=codon_map,
            nucleotide_map=nucleotide_map_dict,
            compress=compress,
            dtype=dtype,
            pad=pad,
        )
