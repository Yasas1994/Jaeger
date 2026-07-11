"""Refine prophage boundaries using pyrodigal-gv gene predictions.

The segmentation step in `jaeger.postprocess.prophages` produces prophage
coordinates as multiples of the sliding-window size. This module snaps those
raw coordinates to the nearest intergenic region by running `pyrodigal-gv` on
each contig, so that predicted prophage ends do not fall inside coding genes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyfastx
import pyrodigal_gv

logger = logging.getLogger("jaeger")


# Lazily instantiated metagenomic gene finder.
_FINDER: pyrodigal_gv.ViralGeneFinder | None = None


def _get_gene_finder() -> pyrodigal_gv.ViralGeneFinder:
    """Return a shared `pyrodigal-gv` gene finder instance."""
    global _FINDER
    if _FINDER is None:
        _FINDER = pyrodigal_gv.ViralGeneFinder(meta=True)
    return _FINDER


def find_genes(sequence: str) -> list[tuple[int, int]]:
    """Run pyrodigal-gv on *sequence* and return 0-based half-open gene intervals.

    Args:
        sequence: Nucleotide sequence (IUPAC alphabet is accepted; non-ATGC
            characters are ignored by pyrodigal).

    Returns:
        Sorted list of ``(start, end)`` tuples in 0-based, half-open coordinates.
    """
    finder = _get_gene_finder()
    genes = finder.find_genes(sequence)
    # pyrodigal returns 1-based closed intervals; convert to 0-based half-open.
    intervals = [(int(g.begin) - 1, int(g.end)) for g in genes]
    intervals.sort()
    return intervals


def _is_intergenic(position: int, genes: list[tuple[int, int]]) -> bool:
    """Return True if *position* is not inside any gene interval."""
    for start, end in genes:
        if start <= position < end:
            return False
        if start > position:
            break
    return True


def refine_boundary(
    position: int,
    genes: list[tuple[int, int]],
    side: str,
    max_extension: int | None = None,
) -> int:
    """Snap a single boundary to the nearest intergenic region.

    Args:
        position: Raw boundary coordinate (0-based).
        genes: Sorted list of 0-based half-open gene intervals.
        side: ``"left"`` or ``"right"``. A left boundary is extended leftward,
            a right boundary rightward.
        max_extension: Maximum number of bases the boundary may be moved.
            If ``None``, the extension is unbounded.

    Returns:
        Refined boundary coordinate.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    if _is_intergenic(position, genes):
        return position

    containing_gene = next(
        ((start, end) for start, end in genes if start <= position < end),
        None,
    )
    if containing_gene is None:
        return position

    gene_start, gene_end = containing_gene
    refined = gene_start if side == "left" else gene_end

    if max_extension is not None:
        extension = abs(refined - position)
        if extension > max_extension:
            logger.warning(
                "Boundary refinement exceeded max_extension (%d bp); "
                "capping %s boundary.",
                max_extension,
                side,
            )
            refined = (
                position + max_extension
                if side == "right"
                else position - max_extension
            )

    return refined


def refine_region(
    raw_start: int,
    raw_end: int,
    genes: list[tuple[int, int]],
    max_extension: int | None = None,
) -> tuple[int, int]:
    """Refine both boundaries of a single prophage region.

    Args:
        raw_start: Raw left boundary (0-based).
        raw_end: Raw right boundary (0-based, exclusive).
        genes: Sorted list of 0-based half-open gene intervals.
        max_extension: Maximum extension allowed for each boundary.

    Returns:
        ``(refined_start, refined_end)``.
    """
    refined_start = refine_boundary(
        raw_start, genes, "left", max_extension=max_extension
    )
    refined_end = refine_boundary(raw_end, genes, "right", max_extension=max_extension)
    return refined_start, refined_end


def refine_prophage_boundaries(
    prophage_cordinates: dict[str, list[Any]],
    fasta_path: str | Path,
    fsize: int,
    max_extension: int | None = None,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Refine window-based prophage boundaries against gene coordinates.

    Args:
        prophage_cordinates: Output of ``segment()``:
            ``{contig_id: [window_index_ranges, scores]}`` where ranges is an
            ``(N, 2)`` array of window indices.
        fasta_path: Path to the input FASTA file.
        fsize: Sliding-window size used by Jaeger.
        max_extension: Maximum boundary extension allowed (default: ``2 * fsize``).

    Returns:
        ``{contig_id: [(raw_start, raw_end, refined_start, refined_end), ...]}``.
    """
    if max_extension is None:
        max_extension = 2 * fsize

    fasta_path = Path(fasta_path)
    refined: dict[str, list[tuple[int, int, int, int]]] = {}

    fa = pyfastx.Fasta(str(fasta_path), build_index=False)
    for record in fa:
        header = record[0].strip().replace(",", "___")
        if header not in prophage_cordinates:
            continue

        cords, _ = prophage_cordinates[header]
        if len(cords) == 0:
            refined[header] = []
            continue

        sequence = str(record[1])
        genes = find_genes(sequence)

        contig_refined: list[tuple[int, int, int, int]] = []
        for start_idx, end_idx in cords:
            raw_start = int(start_idx * fsize)
            raw_end = int(end_idx * fsize)
            refined_start, refined_end = refine_region(
                raw_start, raw_end, genes, max_extension=max_extension
            )
            refined_start = max(refined_start, 0)
            refined_end = min(refined_end, len(sequence))
            contig_refined.append((raw_start, raw_end, refined_start, refined_end))
        refined[header] = contig_refined

    return refined
