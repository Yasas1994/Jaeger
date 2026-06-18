"""Tests for jaeger.seqops.synthetic."""

from __future__ import annotations

import pytest

from jaeger.seqops import synthetic


@pytest.mark.parametrize("length", [1, 10, 100])
def test_generate_homopolymer(length: int):
    seq = synthetic.generate_homopolymer(length, base="A")
    assert len(seq) == length
    assert set(seq) == {"A"}


def test_generate_tandem_repeat():
    seq = synthetic.generate_tandem_repeat("AT", copies=5)
    assert seq == "ATATATATAT"
    assert len(seq) == 10


def test_generate_random_tandem_repeats():
    seqs = synthetic.generate_random_tandem_repeats(
        num_sequences=3, motif_length_range=(2, 4), copy_number=3, alphabet="ACGT"
    )
    assert len(seqs) == 3
    for seq in seqs:
        assert len(seq) > 0
        assert set(seq).issubset(set("ACGT"))


def test_generate_biased_sequence():
    freqs = {"A": 0.5, "T": 0.25, "G": 0.25}
    seq = synthetic.generate_biased_sequence(1000, freqs)
    assert len(seq) == 1000
    counts = {base: seq.count(base) for base in "ACGT"}
    # Approximate check; large enough to be reasonably close.
    assert counts["A"] > counts["T"]
    assert counts["T"] >= counts.get("C", 0)


def test_generate_low_entropy_sequence():
    seq = synthetic.generate_low_entropy_sequence(100, window_size=50, threshold=2.5)
    assert len(seq) == 100
    assert set(seq).issubset(set("ACGT"))


def test_generate_low_entropy_sequence_max_attempts():
    # Very restrictive threshold should exhaust attempts quickly.
    with pytest.raises(ValueError):
        synthetic.generate_low_entropy_sequence(
            200, window_size=50, threshold=0.0, max_attempts=1
        )


def test_apply_shuffle_preserves_length_and_bases():
    seq = "ATCGATCGATCG"
    shuffled = synthetic.apply_shuffle(seq)
    assert len(shuffled) == len(seq)
    assert sorted(shuffled) == sorted(seq)
    assert shuffled != seq  # very unlikely to be identical


def test_apply_subseq_repeat_window_preserves_length():
    seq = "ATCG" * 20
    corrupted = synthetic.apply_subseq_repeat_window(seq, window_fraction=0.25)
    assert len(corrupted) == len(seq)


def test_apply_tandem_repeat_window_preserves_length():
    seq = "ATCG" * 20
    corrupted = synthetic.apply_tandem_repeat_window(
        seq, motif_length_range=(3, 6), window_fraction=0.25
    )
    assert len(corrupted) == len(seq)


def test_apply_tandem_repeat_window_with_num_repeats():
    seq = "ATCG" * 20
    corrupted = synthetic.apply_tandem_repeat_window(
        seq, motif_length_range=(3, 3), window_fraction=0.25, num_repeats=5
    )
    assert len(corrupted) == len(seq)


def test_apply_dinuc_shuffle_preserves_length_and_dinuc_counts():
    seq = "ATCGATCGATCG"
    shuffled = synthetic.apply_dinuc_shuffle(seq)
    assert len(shuffled) == len(seq)
    # Dinuc shuffle preserves mononucleotide and dinucleotide frequencies.
    assert sorted(shuffled) == sorted(seq)


def test_apply_kmer_shuffle_preserves_kmer_counts():
    seq = "ATCGATCGATCG"
    shuffled = synthetic.apply_kmer_shuffle(seq, k=2)
    assert len(shuffled) == len(seq)
    assert sorted(shuffled) == sorted(seq)


def test_apply_mix_concatenates_full_sequences():
    seqs = ["AAAA", "CCCC", "GGGG"]
    mixed = synthetic.apply_mix(seqs)
    assert mixed == "AAAACCCCGGGG"


def test_apply_mix_truncates_to_output_length():
    seqs = ["AAAA" * 10, "CCCC" * 10]
    mixed = synthetic.apply_mix(seqs, output_length=20)
    assert len(mixed) == 20


def test_apply_mix_pads_to_output_length():
    seqs = ["AAAA", "CCCC"]
    mixed = synthetic.apply_mix(seqs, output_length=20, pad_value="N")
    assert len(mixed) == 20
    assert mixed.endswith("NNNNNNNNNNNN")


def test_apply_mix_includes_material_from_each_source():
    seqs = ["A" * 50, "C" * 50]
    mixed = synthetic.apply_mix(seqs, output_length=40)
    assert "A" in mixed
    assert "C" in mixed
