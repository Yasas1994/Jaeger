"""Single source of truth for the codon <-> nucleotide crop mapping.

Jaeger encodes DNA into 6 reading frames with a codon (ngram) width of 3. Two
implementations do the frame extraction:

* the TensorFlow preprocessors in :mod:`jaeger.seqops.encode`
  (``process_string_train`` / ``process_string_inference``), used by CSV
  training and by ``jaeger predict``;
* the numba converter in :mod:`jaeger.dataops.convert` and the raw-numpy
  loaders, used by ``jaeger utils optimize-data`` and NumPy training.

They use different arithmetic (an ``offset_lut`` vs ``crop_size // 3 - 1``) and
only agree on the exact codon count when the nucleotide crop is
``3 * codons + 5``. Centralising that mapping here keeps train, predict,
reliability data generation and optimize-data consistent, and gives the test
suite one place to pin the contract.
"""

from __future__ import annotations

#: Codon (ngram) width used by the translated encoding.
NGRAM_WIDTH = 3


def codons_to_nucleotides(codons: int) -> int:
    """Return the nucleotide crop that yields exactly ``codons`` frames.

    ``3 * codons + 5`` lands in the mod-2 branch of the TF preprocessor and
    simultaneously satisfies ``nt // 3 - 1 == codons`` for the numba converter,
    so both code paths produce the same number of frames.
    """
    if not isinstance(codons, int) or codons <= 0:
        raise ValueError(f"codons must be a positive integer, got {codons!r}")
    return NGRAM_WIDTH * codons + 5


def nucleotides_to_codons(nucleotides: int) -> int:
    """Inverse of :func:`codons_to_nucleotides` (for validation / display)."""
    if not isinstance(nucleotides, int) or nucleotides <= 0:
        raise ValueError(f"nucleotides must be a positive integer, got {nucleotides!r}")
    return (nucleotides - 5) // NGRAM_WIDTH


def tf_frame_length(nucleotides: int) -> int:
    """Codon frames produced by the TF preprocessor for a nucleotide crop.

    Mirrors ``process_string_*`` in ``encode.py`` (``offset_lut = [-2, -1, 0]``,
    ``ngram_width = 3``). Diagnostics / tests only.
    """
    nt = int(nucleotides)
    if nt < NGRAM_WIDTH:
        return 0
    offset_lut = (-2, -1, 0)
    offset = offset_lut[nt % NGRAM_WIDTH]
    ngrams = nt - NGRAM_WIDTH + 1  # nt - 2
    # slice tri[0 : -3 + offset : 3]  ->  end is negative: k = 3 - offset
    k = NGRAM_WIDTH - offset
    usable = ngrams - k
    if usable <= 0:
        return 0
    return -(-usable // NGRAM_WIDTH)  # ceil(usable / 3)


def numba_frame_length(nucleotides: int) -> int:
    """Codon frames produced by the numba converter (``crop_size // 3 - 1``)."""
    nt = int(nucleotides)
    return max(0, nt // NGRAM_WIDTH - 1)


def resolve_crop(string_processor: dict) -> tuple[int, int]:
    """Resolve ``(codons, nucleotides)`` from a ``string_processor`` config.

    ``crop_units`` defaults to ``"codon"``. With codon units, ``crop_size`` is
    the codon count and the nucleotide length is ``3 * crop_size + 5``. With
    ``"nucleotide"`` units, ``crop_size`` is the nucleotide length and the
    codon count is derived via :func:`nucleotides_to_codons`.
    """
    if "crop_size" not in string_processor:
        raise ValueError("string_processor config must define 'crop_size'")
    size = string_processor["crop_size"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"crop_size must be a positive integer, got {size!r}")
    units = string_processor.get("crop_units", "codon")
    if units == "codon":
        return size, codons_to_nucleotides(size)
    if units == "nucleotide":
        return nucleotides_to_codons(size), size
    raise ValueError(f"crop_units must be 'codon' or 'nucleotide', got {units!r}")
