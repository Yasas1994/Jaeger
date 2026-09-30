"""Synthetic DNA sequence generators.

Used for data augmentation, out-of-distribution testing, and generating
synthetic training examples (e.g., tandem repeats, low-entropy sequences).
"""

from __future__ import annotations

import random
from typing import List


from jaeger.seqops.stats import shannon_entropy
from jaeger.seqops.transform import dinuc_shuffle, kmer_shuffle


def generate_homopolymer(length: int, base: str = "A") -> str:
    """Generate a homopolymer (single-base repeat) of given length."""
    return base * length


def generate_tandem_repeat(motif: str, copies: int) -> str:
    """Repeat *motif* *copies* times to create a tandem repeat."""
    return motif * copies


def generate_random_tandem_repeats(
    num_sequences: int,
    motif_length_range: tuple = (3, 30),
    copy_number: int = 2000,
    alphabet: List[str] = ["A", "C", "G", "T"],
) -> List[str]:
    """Generate a list of random tandem repeat sequences.

    Parameters
    ----------
    num_sequences:
        Number of sequences to generate.
    motif_length_range:
        ``(min_len, max_len)`` for randomly sampled motifs.
    copy_number:
        Number of times to repeat each motif.
    alphabet:
        Nucleotide characters to sample motifs from.

    Returns
    -------
    List of generated sequences (each truncated to 2048 bp).
    """
    sequences = []
    for _ in range(num_sequences):
        motif_len = random.randint(*motif_length_range)
        motif = "".join(random.choices(alphabet, k=motif_len))
        seq = generate_tandem_repeat(motif, copy_number)
        sequences.append(seq[:2048])
    return sequences


def generate_biased_sequence(length: int, freqs: dict | None = None) -> str:
    """Generate a sequence with biased nucleotide frequencies.

    *freqs* should be a dict like ``{'A': 0.7, 'C': 0.1, 'G': 0.1, 'T': 0.1}``.
    """
    if freqs is None:
        freqs = {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1}
    bases = list(freqs.keys())
    weights = list(freqs.values())
    return "".join(random.choices(bases, weights=weights, k=length))


def generate_low_entropy_sequence(
    length: int, window_size: int, threshold: float, max_attempts: int = 10000
) -> str:
    """Generate a random sequence where all sliding windows have entropy < *threshold*."""
    for attempt in range(max_attempts):
        seq = generate_biased_sequence(length)
        valid = True
        for i in range(length - window_size + 1):
            if shannon_entropy(seq[i : i + window_size]) >= threshold:
                valid = False
                break
        if valid:
            return seq
    raise ValueError(
        f"Failed to generate low-entropy sequence after {max_attempts} attempts"
    )


def _random_window(seq_len: int, window_fraction: float) -> tuple[int, int]:
    """Return ``(start, end)`` of a random window covering ``window_fraction`` of the sequence."""
    window_len = max(1, int(seq_len * window_fraction))
    start = random.randint(0, max(0, seq_len - window_len))
    return start, start + window_len


def apply_shuffle(seq: str) -> str:
    """Return a randomly permuted version of *seq*."""
    chars = list(seq)
    random.shuffle(chars)
    return "".join(chars)


def apply_dinuc_shuffle(seq: str) -> str:
    """Return a dinucleotide-frequency-preserving shuffle of *seq*."""
    return dinuc_shuffle(seq)


def apply_kmer_shuffle(seq: str, k: int = 2) -> str:
    """Return a k-mer-preserving shuffle of *seq*."""
    return kmer_shuffle(seq, k=k)


def apply_subseq_repeat_window(seq: str, window_fraction: float = 0.25) -> str:
    """Replace a random window with a repeated subsequence from *seq*."""
    if not seq:
        return seq
    seq_len = len(seq)
    start, end = _random_window(seq_len, window_fraction)
    window_len = end - start
    # choose a subsequence from the original sequence
    sub_len = random.randint(1, min(window_len, seq_len))
    sub_start = random.randint(0, seq_len - sub_len)
    sub = seq[sub_start : sub_start + sub_len]
    repeats = (window_len // sub_len) + 1
    fill = (sub * repeats)[:window_len]
    return seq[:start] + fill + seq[end:]


def apply_tandem_repeat_window(
    seq: str,
    motif_length_range: tuple[int, int] = (3, 10),
    window_fraction: float = 0.25,
    num_repeats: int | None = None,
) -> str:
    """Replace a random window with a tandem repeat of a short random motif.

    If *num_repeats* is provided, the motif is repeated exactly that many times
    (truncated or padded to the window length). Otherwise the motif is repeated
    enough times to fill the window.
    """
    if not seq:
        return seq
    seq_len = len(seq)
    start, end = _random_window(seq_len, window_fraction)
    window_len = end - start
    motif_len = random.randint(*motif_length_range)
    motif = "".join(random.choices("ACGT", k=motif_len))
    if num_repeats is not None and num_repeats > 0:
        repeat_block = motif * num_repeats
        fill = (repeat_block * ((window_len // len(repeat_block)) + 1))[:window_len]
    else:
        fill = (motif * ((window_len // motif_len) + 1))[:window_len]
    return seq[:start] + fill + seq[end:]


def apply_random_seq(
    seq: str,
    gc_range: tuple[float, float] = (0.25, 0.75),
) -> str:
    """Replace *seq* with an iid random sequence of the same length.

    The G+C content of the random sequence is sampled uniformly from
    *gc_range*. Unlike shuffles, this destroys all k-mer/compositional
    structure of the input — the strongest form of out-of-distribution
    sequence. The input is only used for its length.
    """
    if not seq:
        return seq
    gc = random.uniform(*gc_range)
    return "".join(random.choices("ACGT", weights=[1 - gc, gc, gc, 1 - gc], k=len(seq)))


def apply_gc_shift(
    seq: str,
    rate_range: tuple[float, float] = (0.05, 0.20),
    direction: str | None = None,
) -> str:
    """Shift the G+C content of *seq* by directed point mutations.

    A rate r is sampled uniformly from *rate_range*. With *direction* ``"up"``,
    r·len(seq) random A/T positions are mutated to G or C; with ``"down"``,
    r·len(seq) random G/C positions are mutated to A or T. When *direction* is
    ``None`` (default) one of the two directions is chosen at random. The
    sequence length and all non-ACGT characters are preserved.
    """
    if not seq:
        return seq
    if direction is None:
        direction = random.choice(("up", "down"))
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up', 'down' or None, got {direction!r}")

    chars = list(seq)
    if direction == "up":
        pool = [i for i, c in enumerate(chars) if c in "ATat"]
        targets = "GC"
    else:
        pool = [i for i, c in enumerate(chars) if c in "GCgc"]
        targets = "AT"
    n = min(len(pool), int(round(len(seq) * random.uniform(*rate_range))))
    for i in random.sample(pool, k=n):
        chars[i] = random.choice(targets)
    return "".join(chars)


def apply_pad_truncate(
    seq: str,
    length_range: tuple[int, int] = (500, 1900),
    output_length: int | None = None,
    pad_char: str = "M",
) -> str:
    """Truncate *seq* to a random short subsequence and right-pad it.

    A length L is sampled uniformly from *length_range* and a random-start
    subsequence of length L is kept; the result is right-padded with
    *pad_char* to *output_length* (defaults to the input length). This mirrors
    the inference-time padding of short contigs in
    ``jaeger.seqops.io.fragment_generator``, which pads with ``'M'`` so that
    padding positions are masked rather than counted as ambiguous (N) content.
    """
    if not seq:
        return seq
    seq_len = len(seq)
    if output_length is None:
        output_length = seq_len
    lo, hi = length_range
    hi = min(hi, output_length)
    if lo > hi:
        raise ValueError(
            f"length_range {length_range} incompatible with output_length {output_length}"
        )
    target_len = random.randint(lo, hi)
    keep = min(target_len, seq_len)
    start = random.randint(0, seq_len - keep)
    truncated = seq[start : start + keep]
    if len(truncated) < output_length:
        truncated += pad_char * (output_length - len(truncated))
    return truncated[:output_length]


def apply_n_stretch(
    seq: str,
    n_fraction_range: tuple[float, float] = (0.3, 1.0),
    max_stretches: int = 3,
    point_n_share: float = 0.2,
) -> str:
    """Replace a random fraction of *seq* with ambiguous bases (N).

    The total N fraction is sampled uniformly from *n_fraction_range*. A fixed
    share of it (*point_n_share*) is scattered as single Ns at random
    positions; the rest is distributed over 1 to *max_stretches* contiguous,
    non-overlapping stretches at random positions. Sequence length is preserved
    and the realised N fraction matches the sampled target (up to integer
    rounding).
    """
    if not seq:
        return seq
    seq_len = len(seq)
    lo, hi = n_fraction_range
    fraction = random.uniform(lo, hi)
    total_n = min(seq_len, max(1, int(round(seq_len * fraction))))

    # Split the N budget between scattered single-base Ns and stretches.
    n_points = min(total_n, int(round(total_n * point_n_share)))
    stretch_n = total_n - n_points

    chars = list(seq)
    if stretch_n > 0:
        n_stretches = random.randint(1, max(1, min(max_stretches, stretch_n)))
        # Partition stretch_n into n_stretches positive stretch lengths.
        remaining = stretch_n
        lengths: list[int] = []
        for i in range(n_stretches - 1):
            take = random.randint(1, remaining - (n_stretches - 1 - i))
            lengths.append(take)
            remaining -= take
        lengths.append(remaining)

        # Random composition of the remaining bases into (n_stretches + 1) gaps
        # (zeros allowed) so stretches never overlap.
        n_keep = seq_len - stretch_n
        cuts = sorted(random.choices(range(n_keep + 1), k=n_stretches))
        gaps = (
            [cuts[0]]
            + [cuts[i + 1] - cuts[i] for i in range(n_stretches - 1)]
            + [n_keep - cuts[-1]]
        )
        pos = 0
        for i in range(n_stretches):
            pos += gaps[i]
            chars[pos : pos + lengths[i]] = "N" * lengths[i]
            pos += lengths[i]

    if n_points > 0:
        free = [i for i, c in enumerate(chars) if c != "N"]
        for i in random.sample(free, k=min(n_points, len(free))):
            chars[i] = "N"

    return "".join(chars)


def apply_mix(
    sequences: list[str],
    output_length: int | None = None,
    pad_value: str = "N",
) -> str:
    """Build a chimeric sequence from random subsegments of *sequences*.

    A random subsegment is drawn from each source sequence and concatenated.
    If *output_length* is provided, the result is truncated or padded with
    *pad_value* to that length. Otherwise the full concatenation is returned.
    """
    if not sequences:
        raise ValueError("apply_mix requires at least one sequence")

    if output_length is not None and output_length < 0:
        raise ValueError("output_length must be non-negative")

    if output_length is None:
        return "".join(sequences)

    if output_length == 0:
        return ""

    n = len(sequences)
    if output_length < n:
        # Fallback: give each of the first output_length sources one base;
        # remaining sources receive a zero-length segment.
        cuts = list(range(output_length)) + [output_length]
    else:
        cuts = sorted(random.sample(range(output_length), k=n - 1))
    cuts = [0] + cuts + [output_length]
    segment_lengths = [cuts[i + 1] - cuts[i] for i in range(n)]

    segments: list[str] = []
    for seq, seg_len in zip(sequences, segment_lengths):
        seq_len = len(seq)
        if seq_len == 0 or seg_len <= 0:
            segments.append("")
            continue
        actual_len = min(seg_len, seq_len)
        start = random.randint(0, seq_len - actual_len)
        segments.append(seq[start : start + actual_len])

    chimera = "".join(segments)
    if len(chimera) < output_length:
        chimera += pad_value * (output_length - len(chimera))
    return chimera
