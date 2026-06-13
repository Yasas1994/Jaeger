"""Metagenomic fragment sampling (genome splitting).

Sample random fragments from genomes to mimic metagenomic assemblies
for a given size distribution.

Two modes:
1. Sequentially sample random fragments of varying size (given a size distribution):
   - Used when *coverage* is NOT provided.
   - Walks along each genome with random fragment lengths and fixed overlap.
   - When minlen == maxlen -> sliding window with constant window size.

2. Coverage-based random sampling:
   - Used when *coverage* is provided.
   - For each genome, sample random fragments until
     target_bases = coverage * genome_length is reached (approx).
"""

from __future__ import annotations

import random

import pyfastx

from jaeger.seqops.transform import dinuc_shuffle
from jaeger.utils.logging import get_logger

logger = get_logger(log_file=None, log_path=None, level=3)


def split_core(**kwargs):
    """Sample random fragments from genomes (to mimic metagenome assemblies)
    for a given size distribution.

    Parameters
    ----------
    minlen : int
        Minimum fragment length (min of discrete uniform distribution).
    maxlen : int
        Maximum fragment length (max of discrete uniform distribution).
    overlap : int
        Number of overlapping nucleotides between two consecutive fragments.
    coverage : int | None
        Switch from sequential sampling to coverage-based sampling if not None.
    circular : bool
        Treat sequences as circular (connects the ends).
    max_n_prop : float
        Maximum proportion of Ns allowed in a fragment.
    seed : int | None
        Seed for the random number generator.
    input : str
        Path to input FASTA file.
    output : str
        Path to output FASTA file.
    shuffle : bool
        Whether to dinuc-shuffle sequences before sampling.
    """
    input_path = kwargs.get("input")
    output_path = kwargs.get("output")

    min_len = kwargs.get("minlen", 2000)
    max_len = kwargs.get("maxlen", 50000)
    overlap = kwargs.get("overlap", 0)
    shuffle = kwargs.get("shuffle", False)
    coverage = kwargs.get("coverage", None)
    circular = kwargs.get("circular", False)
    max_n_prop = kwargs.get("max_n_prop", 0.3)
    seed = kwargs.get("seed", None)

    if seed is not None:
        random.seed(seed)
        logger.info(f"using seed: {seed}")

    if min_len <= 0 or max_len < min_len:
        raise ValueError("Invalid minlen/maxlen: ensure 0 < minlen <= maxlen")

    f = pyfastx.Fasta(input_path, build_index=False)

    def sample_fragment(seq, frag_len, circular=False):
        """Sample a fragment of length *frag_len* from *seq*.

        If *circular* is True, allow wrapping around the end.
        Returns ``(start, fragment_seq)``.
        """
        G = len(seq)
        if frag_len > G:
            frag_len = G
        if circular:
            start = random.randint(0, G - 1)
            end = start + frag_len
            if end <= G:
                fragment = seq[start:end]
            else:
                # wrap around
                end_part = seq[start:]
                wrap_part = seq[: (end - G)]
                fragment = end_part + wrap_part
        else:
            # ensure we don't go out of bounds
            start = random.randint(0, G - frag_len)
            fragment = seq[start : start + frag_len]
        return start, fragment

    with open(output_path, "w") as fh:
        for name, seq in f:
            seq = str(seq)
            if shuffle:
                seq = dinuc_shuffle(seq)

            genome_len = len(seq)
            frag_id = 0

            # If genome is shorter than min_len, skip it
            if genome_len < min_len:
                continue

            # --- MODE 1: coverage-based random sampling ---
            if coverage is not None:
                target_bases = coverage * genome_len
                bases_so_far = 0

                while bases_so_far < target_bases:
                    frag_len = random.randint(min_len, max_len)
                    if frag_len > genome_len:
                        frag_len = genome_len

                    start, fragment = sample_fragment(seq, frag_len, circular=circular)

                    Ns = fragment.count("N")
                    n_prop = Ns / len(fragment)

                    if n_prop <= max_n_prop and len(fragment) >= min_len:
                        header = (
                            f">{name}_frag{frag_id}_start{start}_"
                            f"len{len(fragment)}_cov{coverage}\n"
                        )
                        fh.write(header)
                        for i in range(0, len(fragment), 60):
                            fh.write(fragment[i : i + 60] + "\n")

                        bases_so_far += len(fragment)
                        frag_id += 1

            # --- MODE 2: original sequential tiling with overlap ---
            else:
                start = 0
                while start < genome_len:
                    frag_len = random.randint(min_len, max_len)
                    end = min(start + frag_len, genome_len)
                    fragment = seq[start:end]

                    Ns = fragment.count("N")
                    if len(fragment) > 0:
                        n_prop = Ns / len(fragment)
                    else:
                        n_prop = 1.0  # force skip

                    # Write FASTA record if passes filters
                    if n_prop <= max_n_prop and len(fragment) >= min_len:
                        fh.write(
                            f">{name}_frag{frag_id}_start{start}_len{len(fragment)}\n"
                        )
                        for i in range(0, len(fragment), 60):
                            fh.write(fragment[i : i + 60] + "\n")

                    # Move to next fragment start (with overlap)
                    if end == genome_len:
                        break

                    start = end - overlap
                    frag_id += 1
