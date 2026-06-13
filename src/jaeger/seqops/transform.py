"""Sequence transformation utilities.

DNA sequence shuffling, reverse complement, and tandem repeat masking.
"""

from __future__ import annotations

import numpy as np


def reverse_complement(dna_sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement_dict = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "-": "-",
        "N": "N",
        "W": "W",
        "S": "S",
        "Y": "R",
        "R": "Y",
        "M": "K",
        "K": "M",
        "B": "V",
        "V": "B",
        "H": "D",
        "D": "H",
        "a": "T",
        "t": "A",
        "g": "C",
        "c": "G",
    }
    return "".join(complement_dict.get(base, "N") for base in reversed(dna_sequence))


# ------------------------------------------------------------------
# String / one-hot conversions (used by dinuc_shuffle)
# ------------------------------------------------------------------


def string_to_char_array(seq: str) -> np.ndarray:
    """Convert an ASCII string to a NumPy array of byte-long ASCII codes.

    e.g. ``"ACGT"`` becomes ``[65, 67, 71, 84]``.
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr: np.ndarray) -> str:
    """Convert a NumPy array of byte-long ASCII codes into an ASCII string."""
    return arr.tobytes().decode("ascii")


def one_hot_to_tokens(one_hot: np.ndarray) -> np.ndarray:
    """Convert an L x D one-hot encoding into an L-vector of integers.

    The token ``D`` is used when the one-hot encoding is all 0.
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens: np.ndarray, one_hot_dim: int) -> np.ndarray:
    """Convert an L-vector of integers in ``[0, D]`` to an L x D one-hot encoding.

    A token of ``D`` means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]
    return identity[tokens]


# ------------------------------------------------------------------
# Shuffling
# ------------------------------------------------------------------


def shuffle_dna(seq: str) -> str:
    """Randomly shuffle a DNA sequence (preserves mononucleotide frequencies)."""
    seq_list = list(seq)
    np.random.shuffle(seq_list)
    return "".join(seq_list)


def kmer_shuffle(seq: str, k: int = 1) -> str:
    """Shuffle a sequence by breaking into k-mers, shuffling, and concatenating.

    Args:
        seq: DNA sequence string.
        k: k-mer size. ``k=1`` is random shuffle, ``k=2`` preserves dinucleotides
           (but uses a simpler algorithm than :func:`dinuc_shuffle`).

    Returns:
        Shuffled sequence string.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    kmers = [seq[i : i + k] for i in range(0, len(seq), k)]
    last = kmers[-1] if len(kmers[-1]) < k else None
    if last is not None:
        kmers = kmers[:-1]
    rng = np.random.default_rng()
    rng.shuffle(kmers)
    if last is not None:
        kmers.append(last)
    return "".join(kmers)


def dinuc_shuffle(seq: str | np.ndarray, num_shufs: int | None = None, rng=None):
    """Create shuffles preserving dinucleotide frequencies.

    Args:
        seq: Either a string of length L, or an L x D NumPy array of one-hot
            encodings.
        num_shufs: Number of shuffles to create. If ``None``, only one shuffle.
        rng: NumPy RandomState object for performing shuffles.

    Returns:
        If *seq* is a string: a list of N strings (or a single string if
        *num_shufs* is ``None``).
        If *seq* is a 2D array: an N x L x D NumPy array (or L x D if
        *num_shufs* is ``None``).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    chars, tokens = np.unique(arr, return_inverse=True)

    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)

    return all_results if num_shufs else all_results[0]


def kmer_mix_shuffle(seq1: str, seq2: str, k: int) -> str:
    """Shuffle k-mers from two sequences of different classes.

    .. note::
       Currently a no-op stub.
    """
    pass
