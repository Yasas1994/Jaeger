"""Dataset format converters.

Converts CSV training data to optimized, variable-length padded NPZ formats:
- ``nucleotide``   - 2-strand nucleotide indices or one-hot tensors.
- ``translated``   - 6-frame codon indices or one-hot tensors.
- ``both``         - store both nucleotide and translated arrays.

Integer outputs reserve ``pad_int`` (default 0) for padding. One-hot outputs use
all-zero vectors for padding and for ambiguous bases (N).
"""

from __future__ import annotations

import multiprocessing
import time
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
import psutil


def _save_npz(output_path: str, data: Dict[str, np.ndarray], compress: str):
    """Save arrays to NPZ with the requested compression level."""
    if compress == "default":
        np.savez_compressed(output_path, **data)
    elif compress == "none":
        np.savez(output_path, **data)
    else:  # fast
        import io
        import zipfile

        with zipfile.ZipFile(
            output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as zf:
            for key, arr in data.items():
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{key}.npy", buf.getvalue())


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
# Lookup tables for Numba kernels
# ---------------------------------------------------------------------------
def _build_ascii_to_base_idx() -> np.ndarray:
    """Return a (256,) int8 array mapping ASCII bytes to A/T/G/C/N indices."""
    table = np.full(256, 4, dtype=np.int8)
    table[ord("A")] = 0
    table[ord("T")] = 1
    table[ord("G")] = 2
    table[ord("C")] = 3
    table[ord("a")] = 0
    table[ord("t")] = 1
    table[ord("g")] = 2
    table[ord("c")] = 3
    return table


def _build_complement_lut() -> np.ndarray:
    """Return a (5,) int8 array mapping A<->T, G<->C, N->N."""
    return np.array([1, 0, 3, 2, 4], dtype=np.int8)


def _build_codon_lut(codon_map: np.ndarray) -> np.ndarray:
    """Return a flat (125,) int32 lookup table for 3-base combinations.

    Index = base0 * 25 + base1 * 5 + base2 where each base is 0..4.
    Unknown codons map to -1.
    """
    from jaeger.seqops.maps import CODONS

    codon_to_id = {c: i for i, c in enumerate(CODONS)}
    lut = np.full(125, -1, dtype=np.int32)
    bases = ["A", "T", "G", "C", "N"]
    for i in range(5):
        for j in range(5):
            for k in range(5):
                codon = bases[i] + bases[j] + bases[k]
                cid = codon_to_id.get(codon, -1)
                if 0 <= cid < len(codon_map):
                    lut[i * 25 + j * 5 + k] = int(codon_map[cid])
    return lut


_ASCII_TO_BASE_IDX = _build_ascii_to_base_idx()
_COMPLEMENT_LUT = _build_complement_lut()


# Use forkserver on Linux to avoid fork-deadlocks when the parent process has
# background threads (e.g. from PyTorch/NumPy/OpenMP). Fall back to the default
# context on platforms where forkserver is unavailable.
try:
    _MP_CONTEXT = multiprocessing.get_context("forkserver")
except ValueError:
    _MP_CONTEXT = multiprocessing.get_context()


# ---------------------------------------------------------------------------
# Memory guard helpers
# ---------------------------------------------------------------------------
def _estimate_onehot_memory(
    total_rows: int,
    crop_size: int,
    format: str,
    one_hot: bool,
    codon_map_arr: np.ndarray | None = None,
) -> int:
    """Return the estimated raw float32 bytes needed for one-hot output.

    Returns 0 when ``one_hot`` is False or ``format`` does not require it.
    """
    if not one_hot:
        return 0

    total_rows = max(0, total_rows)
    crop_size = max(0, crop_size)
    estimated = 0

    if format in ("nucleotide", "both"):
        # (2 strands, crop_size, 4 bases) float32
        estimated += total_rows * 2 * crop_size * 4 * np.dtype(np.float32).itemsize

    if format in ("translated", "both") and codon_map_arr is not None:
        vocab_size = int(codon_map_arr.max()) + 1
        max_codon_len = max(0, crop_size // 3 - 1)
        # (6 frames, max_codon_len, vocab_size) float32
        estimated += (
            total_rows * 6 * max_codon_len * vocab_size * np.dtype(np.float32).itemsize
        )

    return int(estimated)


def _check_onehot_memory(
    estimated_bytes: int,
    available_bytes: int,
    fraction: float = 0.75,
) -> None:
    """Raise MemoryError if the estimated one-hot buffer is too large.

    The ``fraction`` sets a safety margin below 100 % of available RAM to
    leave room for the OS, worker overhead, and intermediate copies.
    """
    if estimated_bytes <= 0:
        return

    if estimated_bytes > available_bytes * fraction:
        raise MemoryError(
            f"Estimated one-hot output requires {estimated_bytes / (1024 ** 3):.1f} GiB, "
            f"but only {available_bytes / (1024 ** 3):.1f} GiB RAM is available "
            f"(safety fraction: {fraction:.0%}). "
            "Avoid one-hot encoding for large files; use integer encoding instead "
            "and let the dataloader/model convert to one-hot at training time."
        )


def _count_lines(path: str) -> int:
    """Count newline-terminated rows without loading the whole file."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_MAP = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
_BASE_MAP_LOWER = {"a": 0, "t": 1, "g": 2, "c": 3, "n": 4}
_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
_NUCLEOTIDE_ONEHOT_ORDER = "ATGC"

_CODON_TO_ID: Dict[str, int] | None = None


def _get_codon_to_id() -> Dict[str, int]:
    """Lazy-load codon mapping to avoid E402 module-level import."""
    global _CODON_TO_ID
    if _CODON_TO_ID is None:
        from jaeger.seqops.maps import CODONS

        _CODON_TO_ID = {c: i for i, c in enumerate(CODONS)}
    return _CODON_TO_ID


def _get_codon_map(codon_map_name: str) -> np.ndarray:
    """Return the requested length-64 codon map from jaeger.seqops.maps."""
    import jaeger.seqops.maps as maps

    mapping = getattr(maps, codon_map_name, None)
    if mapping is None:
        raise ValueError(f"Unknown codon map: {codon_map_name}")
    mapping = np.asarray(mapping)
    if mapping.shape != (64,):
        raise ValueError(
            f"Codon map {codon_map_name} must have shape (64,), got {mapping.shape}"
        )
    return mapping


def _base_index(char: str) -> int:
    """Return 0..4 for ACGTN (case-insensitive)."""
    if char in _BASE_MAP:
        return _BASE_MAP[char]
    return _BASE_MAP_LOWER.get(char, 4)


def _reverse_complement(seq: str) -> str:
    """Return the reverse-complement DNA string."""
    return "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


# ---------------------------------------------------------------------------
# Nucleotide encoding
# ---------------------------------------------------------------------------
@njit(cache=False)
def _encode_nucleotide_integer_numba(
    seq_bytes: np.ndarray, crop_size: int, pad_int: int
) -> Tuple[np.ndarray, int]:
    """Numba kernel for 2-strand integer nucleotide encoding."""
    length = min(seq_bytes.shape[0], crop_size)
    arr = np.full((2, crop_size), pad_int, dtype=np.int32)
    ascii_lut = _ASCII_TO_BASE_IDX
    comp_lut = _COMPLEMENT_LUT

    for i in range(length):
        idx = ascii_lut[seq_bytes[i]]
        arr[0, i] = idx + 1

    for i in range(length):
        idx = comp_lut[ascii_lut[seq_bytes[length - 1 - i]]]
        arr[1, i] = idx + 1

    return arr, length


def _encode_nucleotide_integer(
    seq: str, crop_size: int, pad_int: int
) -> Tuple[np.ndarray, int]:
    """Encode a DNA sequence as integer nucleotide indices (2 strands)."""
    if HAS_NUMBA:
        seq_bytes = np.frombuffer(seq[:crop_size].encode("ascii"), dtype=np.uint8)
        return _encode_nucleotide_integer_numba(seq_bytes, crop_size, pad_int)

    seq = seq[:crop_size]
    length = len(seq)
    arr = np.full((2, crop_size), pad_int, dtype=np.int32)

    for i, ch in enumerate(seq):
        arr[0, i] = _base_index(ch) + 1

    rev = _reverse_complement(seq)
    for i, ch in enumerate(rev):
        arr[1, i] = _base_index(ch) + 1

    return arr, length


@njit(cache=False)
def _encode_nucleotide_onehot_numba(
    seq_bytes: np.ndarray, crop_size: int
) -> Tuple[np.ndarray, int]:
    """Numba kernel for 2-strand one-hot nucleotide encoding."""
    length = min(seq_bytes.shape[0], crop_size)
    arr = np.zeros((2, crop_size, 4), dtype=np.float32)
    ascii_lut = _ASCII_TO_BASE_IDX
    comp_lut = _COMPLEMENT_LUT

    for i in range(length):
        idx = ascii_lut[seq_bytes[i]]
        if idx < 4:
            arr[0, i, idx] = 1.0

    for i in range(length):
        idx = comp_lut[ascii_lut[seq_bytes[length - 1 - i]]]
        if idx < 4:
            arr[1, i, idx] = 1.0

    return arr, length


def _encode_nucleotide_onehot(seq: str, crop_size: int) -> Tuple[np.ndarray, int]:
    """Encode a DNA sequence as a one-hot nucleotide tensor (2 strands)."""
    if HAS_NUMBA:
        seq_bytes = np.frombuffer(seq[:crop_size].encode("ascii"), dtype=np.uint8)
        return _encode_nucleotide_onehot_numba(seq_bytes, crop_size)

    seq = seq[:crop_size]
    length = len(seq)
    arr = np.zeros((2, crop_size, 4), dtype=np.float32)

    for i, ch in enumerate(seq):
        idx = _base_index(ch)
        if idx < 4:
            arr[0, i, idx] = 1.0

    rev = _reverse_complement(seq)
    for i, ch in enumerate(rev):
        idx = _base_index(ch)
        if idx < 4:
            arr[1, i, idx] = 1.0

    return arr, length


# ---------------------------------------------------------------------------
# Translated (codon) encoding
# ---------------------------------------------------------------------------
@njit(cache=False)
def _translate_sequence_integer_numba(
    seq_bytes: np.ndarray, crop_size: int, codon_lut: np.ndarray, pad_int: int
) -> Tuple[np.ndarray, int]:
    """Numba kernel for 6-frame codon indices using *codon_lut*."""
    length = min(seq_bytes.shape[0], crop_size)
    ngram_width = 3
    ngrams_len = max(0, length - ngram_width + 1)
    offset = [-2, -1, 0][length % ngram_width]
    ascii_lut = _ASCII_TO_BASE_IDX
    comp_lut = _COMPLEMENT_LUT

    end_f1 = ngrams_len - 3 + offset
    end_f2 = ngrams_len - 2 + offset
    end_f3 = ngrams_len - 1 + offset

    nf1 = max(0, (end_f1 + 2) // ngram_width)
    nf2 = max(0, (end_f2 + 1) // ngram_width)
    nf3 = max(0, end_f3 // ngram_width)

    # Reverse complement bytes
    rev_len = length
    rev_ngrams_len = max(0, rev_len - ngram_width + 1)
    end_r1 = rev_ngrams_len - 3 + offset
    end_r2 = rev_ngrams_len - 2 + offset
    end_r3 = rev_ngrams_len - 1 + offset

    nr1 = max(0, (end_r1 + 2) // ngram_width)
    nr2 = max(0, (end_r2 + 1) // ngram_width)
    nr3 = max(0, end_r3 // ngram_width)

    min_len = min(nf1, nf2, nf3, nr1, nr2, nr3)
    max_codon_len = max(0, crop_size // 3 - 1)
    frames = np.full((6, max_codon_len), pad_int, dtype=np.int32)

    if min_len <= 0:
        return frames, 0

    for i in range(min_len):
        # Forward frame 1
        b0 = ascii_lut[seq_bytes[i * 3]]
        b1 = ascii_lut[seq_bytes[i * 3 + 1]]
        b2 = ascii_lut[seq_bytes[i * 3 + 2]]
        cid = codon_lut[b0 * 25 + b1 * 5 + b2]
        frames[0, i] = pad_int if cid < 0 else cid + 1

        # Forward frame 2
        b0 = ascii_lut[seq_bytes[i * 3 + 1]]
        b1 = ascii_lut[seq_bytes[i * 3 + 2]]
        b2 = ascii_lut[seq_bytes[i * 3 + 3]]
        cid = codon_lut[b0 * 25 + b1 * 5 + b2]
        frames[1, i] = pad_int if cid < 0 else cid + 1

        # Forward frame 3
        b0 = ascii_lut[seq_bytes[i * 3 + 2]]
        b1 = ascii_lut[seq_bytes[i * 3 + 3]]
        b2 = ascii_lut[seq_bytes[i * 3 + 4]]
        cid = codon_lut[b0 * 25 + b1 * 5 + b2]
        frames[2, i] = pad_int if cid < 0 else cid + 1

        # Reverse frames use reversed sequence
        ri = rev_len - 1 - i * 3
        b0 = comp_lut[ascii_lut[seq_bytes[ri]]]
        b1 = comp_lut[ascii_lut[seq_bytes[ri - 1]]]
        b2 = comp_lut[ascii_lut[seq_bytes[ri - 2]]]
        cid = codon_lut[b0 * 25 + b1 * 5 + b2]
        frames[3, i] = pad_int if cid < 0 else cid + 1

        b0 = comp_lut[ascii_lut[seq_bytes[ri - 1]]]
        b1 = comp_lut[ascii_lut[seq_bytes[ri - 2]]]
        b2 = comp_lut[ascii_lut[seq_bytes[ri - 3]]]
        cid = codon_lut[b0 * 25 + b1 * 5 + b2]
        frames[4, i] = pad_int if cid < 0 else cid + 1

        b0 = comp_lut[ascii_lut[seq_bytes[ri - 2]]]
        b1 = comp_lut[ascii_lut[seq_bytes[ri - 3]]]
        b2 = comp_lut[ascii_lut[seq_bytes[ri - 4]]]
        cid = codon_lut[b0 * 25 + b1 * 5 + b2]
        frames[5, i] = pad_int if cid < 0 else cid + 1

    return frames, min_len


def _translate_sequence_integer(
    seq: str, crop_size: int, codon_map: np.ndarray, pad_int: int, codon_lut: np.ndarray | None = None
) -> Tuple[np.ndarray, int]:
    """Encode a DNA sequence as 6-frame codon indices using *codon_map*."""
    if HAS_NUMBA:
        seq_bytes = np.frombuffer(
            seq[:crop_size].upper().encode("ascii"), dtype=np.uint8
        )
        if codon_lut is None:
            codon_lut = _build_codon_lut(codon_map)
        return _translate_sequence_integer_numba(
            seq_bytes, crop_size, codon_lut, pad_int
        )

    seq = seq[:crop_size].upper()
    length = len(seq)
    codon_to_id = _get_codon_to_id()

    ngram_width = 3
    ngrams_len = max(0, length - ngram_width + 1)
    ngrams = [seq[i : i + ngram_width] for i in range(ngrams_len)]
    offset = [-2, -1, 0][length % ngram_width]

    end_f1 = ngrams_len - 3 + offset
    end_f2 = ngrams_len - 2 + offset
    end_f3 = ngrams_len - 1 + offset

    f1 = [ngrams[i] for i in range(0, end_f1, ngram_width)]
    f2 = [ngrams[i] for i in range(1, end_f2, ngram_width)]
    f3 = [ngrams[i] for i in range(2, end_f3, ngram_width)]

    rev_comp = _reverse_complement(seq)
    rev_ngrams_len = max(0, len(rev_comp) - ngram_width + 1)
    rev_ngrams = [rev_comp[i : i + ngram_width] for i in range(rev_ngrams_len)]

    end_r1 = rev_ngrams_len - 3 + offset
    end_r2 = rev_ngrams_len - 2 + offset
    end_r3 = rev_ngrams_len - 1 + offset

    r1 = [rev_ngrams[i] for i in range(0, end_r1, ngram_width)]
    r2 = [rev_ngrams[i] for i in range(1, end_r2, ngram_width)]
    r3 = [rev_ngrams[i] for i in range(2, end_r3, ngram_width)]

    min_len = min(len(f1), len(f2), len(f3), len(r1), len(r2), len(r3))
    max_codon_len = max(0, crop_size // 3 - 1)
    frames = np.full((6, max_codon_len), pad_int, dtype=np.int32)

    for frame_idx, codons in enumerate([f1, f2, f3, r1, r2, r3]):
        for i, codon in enumerate(codons[:min_len]):
            cid = codon_to_id.get(codon, -1)
            if 0 <= cid < len(codon_map):
                frames[frame_idx, i] = int(codon_map[cid]) + 1
            else:
                frames[frame_idx, i] = pad_int

    return frames, min_len


def _to_one_hot(arr: np.ndarray, vocab_size: int, pad_int: int) -> np.ndarray:
    """Convert an integer array to a one-hot float tensor.

    Padding positions (``arr == pad_int``) remain all-zero.
    """
    one_hot = np.zeros(arr.shape + (vocab_size,), dtype=np.float32)
    mask = arr != pad_int
    if mask.any():
        one_hot[mask] = np.eye(vocab_size)[arr[mask] - 1]
    return one_hot


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------
def _parse_csv_line(line: str) -> Tuple[int, str]:
    """Parse a 'label,sequence[,id]' CSV line."""
    parts = line.strip().split(",", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid CSV line (no comma): {line!r}")
    return int(parts[0]), parts[1]


def _sliding_windows(seq: str, crop_size: int, stride: int):
    """Yield windows from *seq* using a sliding window of *crop_size*.

    If *stride* is 0 (or negative), a single truncated window is returned.
    Sequences shorter than *crop_size* yield one padded window.
    """
    if stride <= 0:
        yield seq[:crop_size]
        return

    length = len(seq)
    if length <= crop_size:
        yield seq
        return

    for start in range(0, length, stride):
        window = seq[start : start + crop_size]
        if not window:
            break
        yield window


def _process_chunk(
    lines: List[str],
    crop_size: int,
    stride: int,
    num_classes: int,
    codon_map: np.ndarray,
    format: str,
    one_hot: bool,
    pad_int: int,
) -> Dict[str, List[np.ndarray]]:
    """Process a chunk of CSV lines and return per-sample arrays."""
    out: Dict[str, List[np.ndarray]] = {
        "nucleotide": [],
        "translated": [],
        "labels": [],
        "lengths": [],
    }

    codon_lut = None
    if format in ("translated", "both"):
        codon_lut = _build_codon_lut(codon_map)

    for line in lines:
        if not line.strip():
            continue
        label, seq = _parse_csv_line(line)
        label_vec = np.zeros(num_classes, dtype=np.float32)
        label_vec[label] = 1.0

        for window in _sliding_windows(seq, crop_size, stride):
            if format in ("nucleotide", "both"):
                if one_hot:
                    nuc, nuc_len = _encode_nucleotide_onehot(window, crop_size)
                else:
                    nuc, nuc_len = _encode_nucleotide_integer(
                        window, crop_size, pad_int
                    )
                out["nucleotide"].append(nuc)

            if format in ("translated", "both"):
                trans, trans_len = _translate_sequence_integer(
                    window, crop_size, codon_map, pad_int, codon_lut=codon_lut
                )
                if one_hot:
                    trans = _to_one_hot(trans, int(codon_map.max()) + 1, pad_int)
                out["translated"].append(trans)

            out["labels"].append(label_vec)
            # Store nucleotide length when available; otherwise codon length.
            out["lengths"].append(
                nuc_len if format in ("nucleotide", "both") else trans_len
            )

    return out


def _trim_to_max(arrays: List[np.ndarray], lengths: List[int]) -> np.ndarray:
    """Stack a list of padded arrays and trim to the actual maximum length."""
    stacked = np.stack(arrays, axis=0)
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        return stacked
    # Time/length is the second-to-last axis for 1-D feature arrays and the
    # second axis for nucleotide/translated arrays. Trim all axes after batch
    # and frame/strand dims to max_len.
    if stacked.ndim == 3:
        return stacked[:, :, :max_len]
    if stacked.ndim == 4:
        return stacked[:, :, :max_len, :]
    return stacked


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
def convert_dataset(
    input_path: str,
    output_path: str,
    format: str,
    crop_size: int = 500,
    stride: int = 0,
    num_classes: int = 3,
    num_workers: int | None = None,
    one_hot: bool = False,
    pad_int: int = 0,
    codon_map: str = "CODON_ID",
    max_length: int = 5000,
    compress: str = "fast",
    _available_memory_bytes: int | None = None,
):
    """Convert a CSV dataset to an optimized training format.

    Args:
        input_path: Path to input CSV file (label,sequence format).
        output_path: Path to output file.
        format: Target representation — ``nucleotide``, ``translated``, or ``both``.
        crop_size: Maximum sequence length to process per window.
        stride: Sliding-window step in nucleotides. 0 = one window per sequence.
        num_classes: Number of output classes.
        num_workers: Number of parallel workers (``None`` = auto).
        one_hot: Output float one-hot tensors instead of integer indices.
        pad_int: Integer value used for padding and stored as ``pad_int``.
        codon_map: Name of a length-64 codon map in ``jaeger.seqops.maps``.
        max_length: Deprecated; kept for compatibility.
        compress: NPZ compression level — ``fast`` (zlib 1), ``default`` (zlib 6),
            or ``none``.
        _available_memory_bytes: Internal override for memory guard tests; do not
            use from production code.

    Raises:
        MemoryError: If ``one_hot=True`` and the estimated raw float32 tensor
            exceeds the available RAM safety margin.
    """
    del max_length  # no longer used

    format = format.lower()
    total_lines = _count_lines(input_path)
    valid_formats = ["nucleotide", "translated", "both"]
    if format not in valid_formats:
        raise ValueError(
            f"Invalid format: {format}. Choose from: {', '.join(valid_formats)}"
        )

    compress = compress.lower()
    valid_compress = ["fast", "default", "none"]
    if compress not in valid_compress:
        raise ValueError(
            f"Invalid compress: {compress}. Choose from: {', '.join(valid_compress)}"
        )

    codon_map_arr = _get_codon_map(codon_map) if format in ("translated", "both") else None

    # Guard against one-hot outputs that cannot fit in RAM.
    if one_hot:
        estimated = _estimate_onehot_memory(
            total_rows=total_lines,
            crop_size=crop_size,
            format=format,
            one_hot=one_hot,
            codon_map_arr=codon_map_arr,
        )
        available = (
            _available_memory_bytes
            if _available_memory_bytes is not None
            else psutil.virtual_memory().available
        )
        _check_onehot_memory(estimated, available)

    # Validate pad_int does not collide with actual token indices.
    if not one_hot and format in ("nucleotide", "both"):
        actual_nuc = {1, 2, 3, 4, 5}  # A,T,G,C,N after +1 offset
        if pad_int in actual_nuc:
            raise ValueError(
                f"pad_int ({pad_int}) collides with a nucleotide token index "
                f"({sorted(actual_nuc)})"
            )
    if not one_hot and format in ("translated", "both") and codon_map_arr is not None:
        actual_trans = set(int(v) + 1 for v in codon_map_arr)
        if pad_int in actual_trans:
            raise ValueError(
                f"pad_int ({pad_int}) collides with a translated token index "
                f"for codon map {codon_map}"
            )

    print(f"Converting {input_path} -> {output_path}")
    print(
        f"Format: {format}, one_hot: {one_hot}, pad_int: {pad_int}, "
        f"codon_map: {codon_map}, crop_size: {crop_size}, stride: {stride}, "
        f"num_classes: {num_classes}, compress: {compress}"
    )

    with open(input_path) as f:
        lines = f.readlines()

    if num_workers is None:
        num_workers = cpu_count()
    print(f"Total CSV rows: {total_lines}, using {num_workers} workers")

    chunk_size = max(1, len(lines) // num_workers)
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    start = time.time()
    process_fn = partial(
        _process_chunk,
        crop_size=crop_size,
        stride=stride,
        num_classes=num_classes,
        codon_map=codon_map_arr if codon_map_arr is not None else np.arange(64),
        format=format,
        one_hot=one_hot,
        pad_int=pad_int,
    )

    if num_workers == 1:
        results = [process_fn(chunk) for chunk in chunks]
    elif HAS_NUMBA:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_fn, chunks))
    else:
        with _MP_CONTEXT.Pool(num_workers) as pool:
            results = pool.map(process_fn, chunks)

    # Merge chunks
    nuc_arrays = [a for r in results for a in r["nucleotide"]]
    trans_arrays = [a for r in results for a in r["translated"]]
    labels = np.stack([a for r in results for a in r["labels"]], axis=0)
    lengths = np.array([a for r in results for a in r["lengths"]], dtype=np.int32)

    to_save: Dict[str, np.ndarray] = {"label": labels, "lengths": lengths, "pad_int": np.array(pad_int)}

    if format in ("nucleotide", "both"):
        to_save["nucleotide"] = _trim_to_max(nuc_arrays, lengths.tolist())
    if format in ("translated", "both") and codon_map_arr is not None:
        to_save["translated"] = _trim_to_max(trans_arrays, [t.shape[1] for t in trans_arrays])
        to_save["vocab_size"] = np.array(int(codon_map_arr.max()) + 1)

    _save_npz(output_path, to_save, compress)

    total_windows = len(labels)
    elapsed = time.time() - start
    print(
        f"Processed {total_lines} CSV rows into {total_windows} windows in "
        f"{elapsed:.2f}s ({total_windows / elapsed:.0f} windows/sec)"
    )
    for key, arr in to_save.items():
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
