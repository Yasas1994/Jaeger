"""Dataset format converters.

Converts CSV training data to optimized formats:
- ``tfrecord`` — TensorFlow TFRecord with preprocessed tensors.
- ``numpy_raw`` — int8 DNA sequences (fast loading, runtime preprocessing).
- ``numpy_full`` — fully preprocessed tensors (fastest loading, no augmentations).
- ``numpy_raw_variable`` — variable-length int8 sequences.

The ``numpy_full`` converter uses Numba JIT when available for ~5× speedup.
"""

from __future__ import annotations

import json
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

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
):
    """Encode a batch of DNA crops to integer nucleotide arrays."""
    n = len(lengths)
    out = np.zeros((n, 2, crop_size), dtype=np.int32)

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
):
    """Encode a batch of DNA crops to nucleotide integer or one-hot arrays."""
    if one_hot:
        return _encode_nucleotide_batch_oh(
            sequences, lengths, crop_size, ascii_to_oh, comp_oh
        )
    return _encode_nucleotide_batch_int(
        sequences, lengths, crop_size, ascii_to_user, comp_user
    )


# ---------------------------------------------------------------------------
# Crop generation and padding helpers
# ---------------------------------------------------------------------------
def _crop_starts(seq_len: int, crop_size: int, stride: int) -> list[int]:
    """Return start indices for sliding-window crops.

    If ``stride`` is 0 or the sequence fits in one crop, a single start at 0 is
    returned. Otherwise the sequence is tiled with overlapping crops of
    ``crop_size``; a final tail window is appended when needed so the last base
    is covered.
    """
    if stride == 0 or seq_len <= crop_size:
        return [0]
    starts = list(range(0, seq_len - crop_size + 1, stride))
    if starts[-1] + crop_size < seq_len:
        starts.append(seq_len - crop_size)
    return starts


def _generate_crops(seq_len: int, crop_sizes: list[int], stride: int):
    """Generate (crop_size, start, length) tuples for one sequence.

    Yields one record per crop, ordered by ``crop_sizes`` then by start index.
    """
    for crop_size in crop_sizes:
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
# TFRecord helpers
# ---------------------------------------------------------------------------
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    import tensorflow as tf

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _serialize_tfrecord_embedding(translated, label):
    """Serialize a single example for embedding layer (int32 indices)."""
    import tensorflow as tf

    translated_flat = tf.reshape(translated, [-1])
    feature = {
        "translated": _int64_feature(translated_flat.numpy().tolist()),
        "label": _float_feature(label.numpy().flatten().tolist()),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def _serialize_tfrecord_onehot(translated, label):
    """Serialize a single example for one-hot encoded input (float32)."""
    import tensorflow as tf

    translated_flat = tf.reshape(translated, [-1])
    feature = {
        "translated": _float_feature(translated_flat.numpy().flatten().tolist()),
        "label": _float_feature(label.numpy().flatten().tolist()),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def _convert_to_tfrecord(
    csv_path, output_path, total, preprocess_fn, use_embedding_layer
):
    """Convert CSV to TFRecord with preprocessed tensors."""
    import tensorflow as tf

    serialize_fn = (
        _serialize_tfrecord_embedding
        if use_embedding_layer
        else _serialize_tfrecord_onehot
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tf.io.TFRecordWriter(output_path) as writer:
        start = time.time()
        with open(csv_path) as f:
            for i, line in enumerate(f):
                outputs, label = preprocess_fn(line.strip().encode())
                translated = outputs["translated"]
                example = serialize_fn(translated, label)
                writer.write(example)
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    print(
                        f"  {i + 1}/{total} ({100 * (i + 1) / total:.1f}%) - {rate:.1f} samples/sec"
                    )
        elapsed = time.time() - start
        print(
            f"Done! {total} samples in {elapsed:.1f}s ({total / elapsed:.1f} samples/sec)"
        )


def _convert_with_tf(
    csv_path: str,
    output_path: str,
    format: str,
    crop_size: int,
    num_classes: int,
    use_embedding_layer: bool,
):
    """Convert CSV using TensorFlow preprocessing (tfrecord only)."""
    from jaeger.seqops.encode import process_string_train
    from jaeger.seqops.maps import CODONS, CODON_ID

    total = sum(1 for _ in open(csv_path))
    print(f"Total samples: {total}")

    preprocess_fn = process_string_train(
        crop_size=crop_size,
        seq_onehot=False,
        input_type="translated",
        class_label_onehot=True,
        num_classes=num_classes,
        shuffle=False,
        ngram_width=3,
        codons=CODONS,
        codon_num=CODON_ID,
    )

    _convert_to_tfrecord(
        csv_path, output_path, total, preprocess_fn, use_embedding_layer
    )


# ---------------------------------------------------------------------------
# numpy_raw (int8 sequences)
# ---------------------------------------------------------------------------
_BASE_MAP = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
_BASE_MAP_LOWER = {"a": 0, "t": 1, "g": 2, "c": 3, "n": 4}


def _encode_sequence_int8(seq: str, crop_size: int) -> np.ndarray:
    """Encode DNA sequence to int8 array."""
    arr = np.zeros(crop_size, dtype=np.int8)
    length = min(len(seq), crop_size)
    for i in range(length):
        c = seq[i]
        if c in _BASE_MAP:
            arr[i] = _BASE_MAP[c]
        elif c in _BASE_MAP_LOWER:
            arr[i] = _BASE_MAP_LOWER[c]
        else:
            arr[i] = 4
    return arr


def _process_chunk_numpy_raw(
    lines: list[str], crop_size: int, num_classes: int
) -> tuple:
    """Process a chunk of CSV lines to int8 sequences."""
    n = len(lines)
    sequences = np.zeros((n, crop_size), dtype=np.int8)
    labels = np.zeros((n, num_classes), dtype=np.float32)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        comma_idx = line.find(",")
        if comma_idx == -1:
            continue
        label = int(line[:comma_idx])
        seq = line[comma_idx + 1 :]
        sequences[i] = _encode_sequence_int8(seq, crop_size)
        labels[i, label] = 1.0

    return sequences, labels


def _convert_to_numpy_raw(
    csv_path: str,
    output_path: str,
    crop_size: int,
    num_classes: int,
    num_workers: int | None,
):
    """Convert CSV to NumPy raw (int8 sequences)."""
    total = sum(1 for _ in open(csv_path))
    print(f"Total samples: {total}")

    if num_workers is None:
        num_workers = cpu_count()
    print(f"Using {num_workers} workers")

    with open(csv_path) as f:
        lines = f.readlines()

    chunk_size = max(1, len(lines) // num_workers)
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(f"Split into {len(chunks)} chunks")

    start = time.time()
    process_fn = partial(
        _process_chunk_numpy_raw, crop_size=crop_size, num_classes=num_classes
    )

    with Pool(num_workers) as pool:
        results = pool.map(process_fn, chunks)

    all_sequences = np.concatenate([r[0] for r in results], axis=0)
    all_labels = np.concatenate([r[1] for r in results], axis=0)

    elapsed = time.time() - start
    print(
        f"Processed {total} samples in {elapsed:.2f}s ({total / elapsed:.0f} samples/sec)"
    )

    np.savez_compressed(output_path, sequences=all_sequences, labels=all_labels)
    seq_mb = all_sequences.nbytes / (1024 * 1024)
    label_mb = all_labels.nbytes / (1024 * 1024)
    print(f"Saved: sequences={seq_mb:.1f}MB, labels={label_mb:.1f}MB")


# ---------------------------------------------------------------------------
# numpy_full (fully preprocessed)
# ---------------------------------------------------------------------------
_CODON_TO_ID = None
_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

# Precomputed lookup tables for numba
_CODON_LUT = None
_ASCII_TO_IDX = None
_COMP_LUT = None


def _get_codon_to_id():
    """Lazy-load codon mapping to avoid E402 module-level import."""
    global _CODON_TO_ID
    if _CODON_TO_ID is None:
        from jaeger.seqops.maps import CODONS

        _CODON_TO_ID = {c: i for i, c in enumerate(CODONS)}
    return _CODON_TO_ID


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
def _process_sequence_numba(
    seq_bytes, crop_size, seq_len, codon_lut, comp_lut, ascii_lut
):
    """Process DNA sequence to 6-frame codon embeddings (numba-optimized)."""
    length = min(len(seq_bytes), crop_size)

    # Convert ASCII bytes to base indices
    bases = np.empty(length, dtype=np.int8)
    for i in range(length):
        bases[i] = ascii_lut[seq_bytes[i]]

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
        return np.empty((6, 0), dtype=np.int32)
    min_len = min(min_len, seq_len)

    frames = np.empty((6, min_len), dtype=np.int32)

    # Forward frames
    for i in range(min_len):
        idx0 = bases[i * 3]
        idx1 = bases[i * 3 + 1]
        idx2 = bases[i * 3 + 2]
        frames[0, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2]

        idx0 = bases[i * 3 + 1]
        idx1 = bases[i * 3 + 2]
        idx2 = bases[i * 3 + 3]
        frames[1, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2]

        idx0 = bases[i * 3 + 2]
        idx1 = bases[i * 3 + 3]
        idx2 = bases[i * 3 + 4]
        frames[2, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2]

    # Reverse frames
    for i in range(min_len):
        idx0 = rev_bases[i * 3]
        idx1 = rev_bases[i * 3 + 1]
        idx2 = rev_bases[i * 3 + 2]
        frames[3, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2]

        idx0 = rev_bases[i * 3 + 1]
        idx1 = rev_bases[i * 3 + 2]
        idx2 = rev_bases[i * 3 + 3]
        frames[4, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2]

        idx0 = rev_bases[i * 3 + 2]
        idx1 = rev_bases[i * 3 + 3]
        idx2 = rev_bases[i * 3 + 4]
        frames[5, i] = codon_lut[idx0 * 25 + idx1 * 5 + idx2]

    # +1 for embedding layer
    for i in range(6):
        for j in range(min_len):
            frames[i, j] = frames[i, j] + 1

    return frames


def _process_sequence_full(
    seq: str, crop_size: int, ngram_width: int = 3
) -> np.ndarray:
    """Process a single DNA sequence to 6-frame codon embeddings (pure Python)."""
    codon_to_id = _get_codon_to_id()
    seq = seq[:crop_size].upper()
    length = len(seq)
    offset = [-2, -1, 0][length % ngram_width]

    ngrams_len = length - ngram_width + 1
    ngrams = [seq[i : i + ngram_width] for i in range(ngrams_len)]

    end_f1 = ngrams_len - 3 + offset
    end_f2 = ngrams_len - 2 + offset
    end_f3 = ngrams_len - 1 + offset

    f1 = [codon_to_id.get(ngrams[i], -1) for i in range(0, end_f1, ngram_width)]
    f2 = [codon_to_id.get(ngrams[i], -1) for i in range(1, end_f2, ngram_width)]
    f3 = [codon_to_id.get(ngrams[i], -1) for i in range(2, end_f3, ngram_width)]

    rev_comp = "".join(_COMP.get(b, "N") for b in reversed(seq))
    rev_ngrams_len = len(rev_comp) - ngram_width + 1
    rev_ngrams = [rev_comp[i : i + ngram_width] for i in range(rev_ngrams_len)]

    end_r1 = rev_ngrams_len - 3 + offset
    end_r2 = rev_ngrams_len - 2 + offset
    end_r3 = rev_ngrams_len - 1 + offset

    r1 = [codon_to_id.get(rev_ngrams[i], -1) for i in range(0, end_r1, ngram_width)]
    r2 = [codon_to_id.get(rev_ngrams[i], -1) for i in range(1, end_r2, ngram_width)]
    r3 = [codon_to_id.get(rev_ngrams[i], -1) for i in range(2, end_r3, ngram_width)]

    min_len = min(len(f1), len(f2), len(f3), len(r1), len(r2), len(r3))
    frames = np.array(
        [
            f1[:min_len],
            f2[:min_len],
            f3[:min_len],
            r1[:min_len],
            r2[:min_len],
            r3[:min_len],
        ],
        dtype=np.int32,
    )
    frames = frames + 1
    return frames


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


def _process_chunk_numpy_full(
    lines: list[str], crop_size: int, num_classes: int
) -> tuple:
    """Process a chunk of CSV lines to fully preprocessed tensors."""
    n = len(lines)
    seq_len = crop_size // 3 - 1
    sequences = np.zeros((n, 6, seq_len), dtype=np.int32)
    labels = np.zeros((n, num_classes), dtype=np.float32)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        comma_idx = line.find(",")
        if comma_idx == -1:
            continue
        label = int(line[:comma_idx])
        seq = line[comma_idx + 1 :]
        frames = _process_sequence_full(seq, crop_size)
        actual_len = frames.shape[1]
        sequences[i, :, :actual_len] = frames
        labels[i, label] = 1.0

    return sequences, labels


def _convert_to_numpy_full(
    csv_path: str,
    output_path: str,
    crop_size: int,
    num_classes: int,
    num_workers: int | None,
):
    """Convert CSV to fully preprocessed NumPy format."""
    total = sum(1 for _ in open(csv_path))
    print(f"Total samples: {total}")

    if num_workers is None:
        num_workers = cpu_count()
    print(f"Using {num_workers} workers")

    # Read and parse CSV
    with open(csv_path) as f:
        lines = f.readlines()

    start = time.time()

    if HAS_NUMBA and total >= 1000:
        # Fast path: batch numba processing (no multiprocessing overhead)
        print("Using numba-optimized batch processing")
        codon_lut, ascii_lut, comp_lut = _build_numba_lookups()
        seq_len = crop_size // 3 - 1

        # Parse labels and encode sequences as uint8 array
        labels = np.zeros((total, num_classes), dtype=np.float32)
        sequences = np.full((total, crop_size), ord("N"), dtype=np.uint8)
        lengths = np.zeros(total, dtype=np.int32)

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            comma_idx = line.find(",")
            if comma_idx == -1:
                continue
            label = int(line[:comma_idx])
            seq = line[comma_idx + 1 :]
            seq_bytes = seq.encode()
            length = min(len(seq_bytes), crop_size)
            sequences[i, :length] = np.frombuffer(seq_bytes[:length], dtype=np.uint8)
            lengths[i] = length
            labels[i, label] = 1.0

        # Process entire batch in numba
        all_sequences = _process_batch_numba(
            sequences, lengths, crop_size, seq_len, codon_lut, comp_lut, ascii_lut
        )
        all_labels = labels
    else:
        # Fallback: multiprocessing with pure Python
        if HAS_NUMBA:
            print("Numba available but dataset too small for batch mode")
        else:
            print("Numba not available, using pure Python")

        chunk_size = max(1, len(lines) // num_workers)
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        process_fn = partial(
            _process_chunk_numpy_full, crop_size=crop_size, num_classes=num_classes
        )
        with Pool(num_workers) as pool:
            results = pool.map(process_fn, chunks)
        all_sequences = np.concatenate([r[0] for r in results], axis=0)
        all_labels = np.concatenate([r[1] for r in results], axis=0)

    elapsed = time.time() - start
    print(
        f"Processed {total} samples in {elapsed:.2f}s ({total / elapsed:.0f} samples/sec)"
    )

    np.savez_compressed(output_path, translated=all_sequences, label=all_labels)
    seq_mb = all_sequences.nbytes / (1024 * 1024)
    label_mb = all_labels.nbytes / (1024 * 1024)
    print(f"Saved: translated={seq_mb:.1f}MB, label={label_mb:.1f}MB")


# ---------------------------------------------------------------------------
# numpy_raw_variable (variable-length int8)
# ---------------------------------------------------------------------------


def _encode_variable_length(line: str, max_length: int) -> tuple:
    """Encode a single CSV line to variable-length int8 sequence."""
    line = line.strip()
    if not line:
        return None, None, None
    comma_idx = line.find(",")
    if comma_idx == -1:
        return None, None, None
    label = int(line[:comma_idx])
    seq = line[comma_idx + 1 :]
    length = min(len(seq), max_length)

    arr = np.full(max_length, 4, dtype=np.int8)
    for i in range(length):
        c = seq[i]
        if c in _BASE_MAP:
            arr[i] = _BASE_MAP[c]
        elif c in _BASE_MAP_LOWER:
            arr[i] = _BASE_MAP_LOWER[c]
        else:
            arr[i] = 4
    return arr, length, label


def _process_chunk_numpy_raw_variable(
    lines: list[str], max_length: int, num_classes: int
) -> tuple:
    """Process a chunk of CSV lines with variable lengths."""
    n = len(lines)
    sequences = np.full((n, max_length), 4, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int32)
    labels = np.zeros((n, num_classes), dtype=np.float32)

    for i, line in enumerate(lines):
        seq_arr, length, label = _encode_variable_length(line, max_length)
        if seq_arr is None:
            continue
        sequences[i] = seq_arr
        lengths[i] = length
        labels[i, label] = 1.0

    return sequences, lengths, labels


def _convert_to_numpy_raw_variable(
    csv_path: str,
    output_path: str,
    max_length: int,
    num_classes: int,
    num_workers: int | None,
):
    """Convert CSV to NumPy with variable-length sequences."""
    total = sum(1 for _ in open(csv_path))
    print(f"Total samples: {total}")

    if num_workers is None:
        num_workers = cpu_count()
    print(f"Using {num_workers} workers")

    with open(csv_path) as f:
        lines = f.readlines()

    chunk_size = max(1, len(lines) // num_workers)
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(f"Split into {len(chunks)} chunks")

    start = time.time()
    process_fn = partial(
        _process_chunk_numpy_raw_variable,
        max_length=max_length,
        num_classes=num_classes,
    )

    with Pool(num_workers) as pool:
        results = pool.map(process_fn, chunks)

    all_sequences = np.concatenate([r[0] for r in results], axis=0)
    all_lengths = np.concatenate([r[1] for r in results], axis=0)
    all_labels = np.concatenate([r[2] for r in results], axis=0)

    elapsed = time.time() - start
    print(
        f"Processed {total} samples in {elapsed:.2f}s ({total / elapsed:.0f} samples/sec)"
    )

    np.savez_compressed(
        output_path,
        sequences=all_sequences,
        lengths=all_lengths,
        labels=all_labels,
    )
    seq_mb = all_sequences.nbytes / (1024 * 1024)
    length_mb = all_lengths.nbytes / (1024 * 1024)
    label_mb = all_labels.nbytes / (1024 * 1024)
    print(
        f"Saved: sequences={seq_mb:.1f}MB, lengths={length_mb:.1f}MB, labels={label_mb:.1f}MB"
    )


# ---------------------------------------------------------------------------
# npz converter (nucleotide / translated / dicodon)
# ---------------------------------------------------------------------------


def _process_chunk_npz(
    lines,
    fmt,
    crop_sizes,
    stride,
    one_hot,
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

    for crop_size in crop_sizes:
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

        if not batch_seqs:
            continue

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
            )
            for i in range(encoded.shape[0]):
                nucleotide_arrays.append(encoded[i, :, : lengths_arr[i]])

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
                actual_lengths = _single_codon_actual_lengths(lengths_arr, crop_size)
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
            for i in range(encoded.shape[0]):
                translated_arrays.append(encoded[i, :, : actual_lengths[i]])
            all_translated_lengths.extend(actual_lengths.tolist())
        else:
            all_translated_lengths.extend([0] * len(batch_labels))

        all_labels.extend(batch_labels)
        all_lengths.extend(batch_lengths)

    return {
        "nucleotide": nucleotide_arrays,
        "translated": translated_arrays,
        "labels": np.array(all_labels, dtype=np.int32),
        "lengths": np.array(all_lengths, dtype=np.int32),
        "translated_lengths": np.array(all_translated_lengths, dtype=np.int32),
    }


def _convert_to_npz(
    input_path: str,
    output_path: str,
    fmt: str,
    crop_sizes: list[int],
    stride: int,
    num_classes: int,
    num_workers: int | None,
    one_hot: bool,
    pad_int: int,
    codon_map_name: str,
    nucleotide_map: dict[str, int],
    compress: str,
) -> None:
    """Convert a CSV dataset to an ``.npz`` file.

    Supports ``nucleotide``, ``translated``, and ``both`` output formats, with
    optional one-hot encoding and multiple crop sizes sharing one stride.
    """
    import warnings

    valid_fmts = ["nucleotide", "translated", "both"]
    if fmt not in valid_fmts:
        raise ValueError(f"Invalid format: {fmt}. Choose from: {', '.join(valid_fmts)}")

    if not output_path.endswith(".npz"):
        warnings.warn(
            f"Output path {output_path!r} does not end with '.npz'", stacklevel=2
        )

    with open(input_path) as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    nucleotide_lookups = _build_nucleotide_lookups(nucleotide_map)
    _, ascii_lut, comp_lut = _build_numba_lookups()

    codon_map = _get_codon_map(codon_map_name)
    codon_map_len = len(codon_map)
    if codon_map_len == 4096:
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
        "stride": stride,
        "one_hot": one_hot,
        "nucleotide_lookups": nucleotide_lookups,
        "codon_lut": codon_lut,
        "codon_map_len": codon_map_len,
        "standard_codon_lut3": standard_codon_lut3,
        "dicodon_lut": dicodon_lut,
        "ascii_lut": ascii_lut,
        "comp_lut": comp_lut,
    }

    if num_workers is None or num_workers <= 1:
        result = _process_chunk_npz(lines, **worker_kwargs)
    else:
        chunk_size = max(1, len(lines) // num_workers)
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        process_fn = partial(_process_chunk_npz, **worker_kwargs)
        with Pool(num_workers) as pool:
            results = pool.map(process_fn, chunks)
        result = {
            "nucleotide": sum((r["nucleotide"] for r in results), []),
            "translated": sum((r["translated"] for r in results), []),
            "labels": np.concatenate([r["labels"] for r in results], axis=0),
            "lengths": np.concatenate([r["lengths"] for r in results], axis=0),
            "translated_lengths": np.concatenate(
                [r["translated_lengths"] for r in results], axis=0
            ),
        }

    save_dict: dict[str, np.ndarray | str] = {
        "lengths": result["lengths"],
        "labels": result["labels"],
        "pad_int": np.int32(pad_int),
        "crop_sizes": np.array(crop_sizes, dtype=np.int32),
        "stride": np.int32(stride),
    }

    if fmt in ("nucleotide", "both"):
        arrays = result["nucleotide"]
        if arrays:
            max_len = max(a.shape[-1] for a in arrays)
            if one_hot:
                padded = [_pad_axis(a, max_len, axis=-1, pad_value=0.0) for a in arrays]
                nucleotide_arr = np.stack(padded, axis=0)
            else:
                padded = [
                    _pad_axis(a, max_len, axis=-1, pad_value=pad_int) for a in arrays
                ]
                nucleotide_arr = np.stack(padded, axis=0)
            save_dict["nucleotide"] = nucleotide_arr
            save_dict["nucleotide_map"] = json.dumps(nucleotide_map)
        else:
            save_dict["nucleotide"] = np.empty(
                (0,), dtype=np.float32 if one_hot else np.int32
            )
            save_dict["nucleotide_map"] = json.dumps(nucleotide_map)

    if fmt in ("translated", "both"):
        arrays = result["translated"]
        if arrays:
            max_len = max(a.shape[-1] for a in arrays)
            if one_hot:
                padded = [_pad_axis(a, max_len, axis=-1, pad_value=0) for a in arrays]
                stacked = np.stack(padded, axis=0)
                translated_arr = _one_hot_integer(stacked, codon_map_len + 1)
            else:
                padded = [_pad_axis(a, max_len, axis=-1, pad_value=0) for a in arrays]
                translated_arr = np.stack(padded, axis=0)
            save_dict["translated"] = translated_arr
            save_dict["codon_map"] = codon_map_name
        else:
            save_dict["translated"] = np.empty(
                (0,), dtype=np.float32 if one_hot else np.int32
            )
            save_dict["codon_map"] = codon_map_name

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if compress == "none":
        np.savez(output_path, **save_dict)
    else:
        np.savez_compressed(output_path, **save_dict)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def convert_dataset(
    input_path: str,
    output_path: str,
    format: str,
    crop_size: int = 500,
    num_classes: int = 3,
    use_embedding_layer: bool = True,
    max_length: int = 1000,
    num_workers: int | None = None,
):
    """Convert a CSV dataset to an optimized training format.

    Args:
        input_path: Path to input CSV file (label,sequence format).
        output_path: Path to output file.
        format: Target format — ``tfrecord``, ``numpy_raw``, ``numpy_full``,
            or ``numpy_raw_variable``.
        crop_size: Sequence crop size (for fixed-length formats).
        num_classes: Number of output classes.
        use_embedding_layer: Whether model uses embedding layer (TFRecord only).
        max_length: Maximum sequence length (variable-length only).
        num_workers: Number of parallel workers (``None`` = auto).
    """
    valid_formats = ["tfrecord", "numpy_raw", "numpy_full", "numpy_raw_variable"]
    if format not in valid_formats:
        raise ValueError(
            f"Invalid format: {format}. Choose from: {', '.join(valid_formats)}"
        )

    print(f"Converting {input_path} -> {output_path}")
    print(f"Format: {format}, Crop size: {crop_size}, Num classes: {num_classes}")

    if format == "tfrecord":
        _convert_with_tf(
            input_path, output_path, format, crop_size, num_classes, use_embedding_layer
        )
    elif format == "numpy_raw":
        _convert_to_numpy_raw(
            input_path, output_path, crop_size, num_classes, num_workers
        )
    elif format == "numpy_full":
        _convert_to_numpy_full(
            input_path, output_path, crop_size, num_classes, num_workers
        )
    elif format == "numpy_raw_variable":
        _convert_to_numpy_raw_variable(
            input_path, output_path, max_length, num_classes, num_workers
        )
