"""Tests for jaeger.seqops.transform."""

from __future__ import annotations

import numpy as np
import pytest

from jaeger.seqops import transform


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("ATGC", "GCAT"),
        ("A", "T"),
        ("N", "N"),
        ("ATGCATGC", "GCATGCAT"),
    ],
)
def test_reverse_complement(seq: str, expected: str):
    assert transform.reverse_complement(seq) == expected


def test_string_to_char_array_roundtrip():
    seq = "ACGT"
    arr = transform.string_to_char_array(seq)
    assert arr.dtype == np.int8
    assert transform.char_array_to_string(arr) == seq


def test_one_hot_to_tokens_and_back():
    tokens = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    one_hot = np.eye(4, dtype=np.float32)[tokens]
    recovered = transform.one_hot_to_tokens(one_hot)
    assert np.array_equal(recovered, tokens)

    one_hot_again = transform.tokens_to_one_hot(recovered, one_hot_dim=4)
    assert np.array_equal(one_hot_again, one_hot)


def test_one_hot_to_tokens_all_zero_token():
    one_hot = np.zeros((3, 4), dtype=np.float32)
    tokens = transform.one_hot_to_tokens(one_hot)
    assert np.all(tokens == 4)


def test_shuffle_dna_preserves_base_counts():
    seq = "AATTGGCCNN"
    shuffled = transform.shuffle_dna(seq)
    assert sorted(shuffled) == sorted(seq)
    assert len(shuffled) == len(seq)


def test_kmer_shuffle_preserves_kmers():
    seq = "AATTGGCC"
    shuffled = transform.kmer_shuffle(seq, k=2)
    kmers_orig = sorted(seq[i : i + 2] for i in range(0, len(seq), 2))
    kmers_shuf = sorted(shuffled[i : i + 2] for i in range(0, len(shuffled), 2))
    assert kmers_shuf == kmers_orig


def test_kmer_shuffle_invalid_k():
    with pytest.raises(ValueError):
        transform.kmer_shuffle("ACGT", k=0)


def test_dinuc_shuffle_string():
    seq = "AATTAATTAATT"
    shuffled = transform.dinuc_shuffle(seq)
    assert isinstance(shuffled, str)
    assert len(shuffled) == len(seq)
    assert sorted(shuffled) == sorted(seq)


def test_dinuc_shuffle_one_hot():
    tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    one_hot = np.eye(4, dtype=np.float32)[tokens]
    shuffled = transform.dinuc_shuffle(one_hot)
    assert shuffled.shape == one_hot.shape
    assert np.allclose(np.sum(shuffled, axis=-1), 1.0)


def test_dinuc_shuffle_multiple():
    seq = "AATTAATTAATT"
    shuffles = transform.dinuc_shuffle(seq, num_shufs=3)
    assert isinstance(shuffles, list)
    assert len(shuffles) == 3


def test_kmer_mix_shuffle_stub():
    assert transform.kmer_mix_shuffle("AAAA", "TTTT", k=2) is None
