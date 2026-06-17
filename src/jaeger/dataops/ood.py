"""Out-of-distribution (OOD) detection utilities for sequence shuffling and splitting.

Provides functions for generating shuffled (negative) sequence datasets and
sampling genome fragments to mimic metagenomic assemblies.
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl
import pyfastx
from typing import Callable, Set, Optional, List
from dataclasses import dataclass

from jaeger.seqops.transform import dinuc_shuffle, kmer_shuffle
from jaeger.seqops.synthetic import generate_random_tandem_repeats
from jaeger.seqops.io import write_fasta_entry
from jaeger.utils.logging import get_logger

logger = get_logger(log_file=None, log_path=None, level=3)


# ───────────────────────────────
# Constants & Config
# ───────────────────────────────

N_THRESHOLD = 0.3
FASTA_LINE_WIDTH = 70

PREDICTION_MAP = {
    0: "bacteria",
    1: "phage",
    2: "eukarya",
    3: "archaea",
    4: "plasmid",
    5: "virus",
}


# ───────────────────────────────
# Data Model
# ───────────────────────────────


@dataclass
class SequenceRecord:
    """Unified container for a sequence and its metadata."""

    seq_id: str
    sequence: str
    label: int


# ───────────────────────────────
# Core Builder
# ───────────────────────────────


class OODDatasetBuilder:
    """
    Generates OOD (out-of-distribution) datasets by:
      1. loading original sequences,
      2. optionally filtering by Jaeger prediction correctness,
      3. masking low-complexity (>N%) sequences,
      4. appending shuffled negatives,
      5. appending random tandem repeats,
      6. shuffling the combined pool and writing to CSV or FASTA.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        input_type: str,
        output_type: str,
        input_predictions: Optional[str] = None,
        dinuc: bool = False,
        k: int = 1,
        num_tandem_repeats: int = 0,
        n_threshold: float = N_THRESHOLD,
        fasta_line_width: int = FASTA_LINE_WIDTH,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.input_type = input_type.upper()
        self.output_type = output_type.upper()
        self.input_predictions = input_predictions
        self.dinuc = dinuc
        self.k = k
        self.num_tandem_repeats = num_tandem_repeats
        self.n_threshold = n_threshold
        self.fasta_line_width = fasta_line_width

        self.shuffle_fn = self._resolve_shuffle_fn()
        self.correct_ids: Set[str] = self._load_correct_predictions()

    # ── internal helpers ──────────────────────────

    def _resolve_shuffle_fn(self) -> Callable[[str], str]:
        if self.dinuc:
            return dinuc_shuffle
        return lambda seq: kmer_shuffle(seq=seq, k=self.k)

    def _load_correct_predictions(self) -> Set[str]:
        if not self.input_predictions:
            return set()

        logger.info("Using Jaeger predictions to generate the OOD dataset")

        df = pl.read_csv(
            self.input_predictions,
            truncate_ragged_lines=True,
            separator="\t",
            columns=[0, 2],
        )

        df = (
            df.with_columns(
                true_class=pl.col("contig_id").str.split("__class=").list.last()
            )
            .with_columns(true_class=pl.col("true_class").replace(PREDICTION_MAP))
            .with_columns(is_correct=(pl.col("true_class") == pl.col("prediction")) * 1)
            .with_columns(
                contig_id=pl.col("contig_id").str.split("__class=").list.first()
            )
        )

        return set(df.filter(pl.col("is_correct") == 1)["contig_id"].to_list())

    def _compute_label(self, sequence: str, seq_id: str) -> int:
        """Return 0 if low-complexity or not in the 'correct' set; else 1."""
        if not sequence:
            return 0

        n_ratio = sequence.count("N") / len(sequence)
        low_complexity = n_ratio > self.n_threshold
        is_correct = seq_id in self.correct_ids

        return 0 if (low_complexity or not is_correct) else 1

    # ── loaders ───────────────────────────────────

    def _load_csv(self) -> List[SequenceRecord]:
        df = pl.read_csv(
            self.input_path,
            truncate_ragged_lines=True,
            has_header=False,
        )

        # Only expect/need a 3rd column when we have predictions to reconcile
        if self.correct_ids:
            df = df.select(["column_1", "column_2", "column_3"])
            df = df.with_columns(
                pl.when(pl.col("column_3").is_in(self.correct_ids))
                .then(pl.lit(1, dtype=pl.Int64))
                .otherwise(pl.lit(0, dtype=pl.Int64))
                .alias("column_1")
            )
        else:
            df = df.select(["column_1", "column_2"])

        # Mask high-N sequences regardless of prediction mode
        df = df.with_columns(
            pl.when(
                (
                    pl.col("column_2").str.count_matches("N")
                    / pl.col("column_2").str.len_chars()
                )
                > self.n_threshold
            )
            .then(pl.lit(0, dtype=pl.Int64))
            .otherwise(pl.col("column_1"))
            .alias("column_1")
        )

        records: List[SequenceRecord] = []
        for row in df.iter_rows(named=True):
            # Use column_3 as ID if present, otherwise synthesize one
            raw_id = row.get("column_3", f"seq_{len(records)}")
            clean_id = (
                raw_id.split("__class=")[0] if isinstance(raw_id, str) else str(raw_id)
            )

            records.append(
                SequenceRecord(
                    seq_id=clean_id,
                    sequence=row["column_2"],
                    label=row["column_1"],
                )
            )
        return records

    def _load_fasta(self) -> List[SequenceRecord]:
        records: List[SequenceRecord] = []
        for name, seq in pyfastx.Fasta(self.input_path, build_index=False):
            clean_id = name.split("__class=")[0]
            label = self._compute_label(seq, clean_id)
            records.append(SequenceRecord(seq_id=clean_id, sequence=seq, label=label))
        return records

    # ── OOD generators ────────────────────────────

    def _generate_shuffled(
        self, originals: List[SequenceRecord]
    ) -> List[SequenceRecord]:
        return [
            SequenceRecord(
                seq_id=r.seq_id,
                sequence=self.shuffle_fn(r.sequence),
                label=0,
            )
            for r in originals
        ]

    def _generate_tandem_repeats(self) -> List[SequenceRecord]:
        if self.num_tandem_repeats <= 0:
            return []

        sequences = generate_random_tandem_repeats(
            num_sequences=self.num_tandem_repeats
        )
        return [
            SequenceRecord(
                seq_id=f"tandem_repeat_{i}",
                sequence=seq,
                label=0,
            )
            for i, seq in enumerate(sequences)
        ]

    # ── writers ───────────────────────────────────

    def _write_csv(self, records: List[SequenceRecord]) -> None:
        df = pl.DataFrame(
            [
                {
                    "column_1": r.label,
                    "column_2": r.sequence,
                    "column_3": r.seq_id,
                }
                for r in records
            ]
        )
        df.write_csv(self.output_path, include_header=False)

    def _write_fasta(self, records: List[SequenceRecord]) -> None:
        with open(self.output_path, "w") as fh:
            for r in records:
                fh.write(f">{r.seq_id}__class={r.label}\n")
                for i in range(0, len(r.sequence), self.fasta_line_width):
                    fh.write(r.sequence[i : i + self.fasta_line_width] + "\n")

    # ── public API ────────────────────────────────

    def build(self) -> None:
        # 1. Load originals
        if self.input_type == "CSV":
            originals = self._load_csv()
        elif self.input_type == "FASTA":
            originals = self._load_fasta()
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")

        # 2. Generate negatives
        shuffled = self._generate_shuffled(originals)
        tandem = self._generate_tandem_repeats()

        # 3. Merge & shuffle
        pool = originals + shuffled + tandem
        random.shuffle(pool)

        logger.info(
            f"id: {len(originals)} | ood_shuffled: {len(shuffled)} | "
            f"ood_tandem: {len(tandem)} | total: {len(pool)}"
        )

        # 4. Write
        if self.output_type == "CSV":
            self._write_csv(pool)
        elif self.output_type == "FASTA":
            self._write_fasta(pool)
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")


# ───────────────────────────────
# Backward-compatible wrapper
# ───────────────────────────────


def shuffle_core(**kwargs):
    """
    Drop-in replacement for the original shuffle_core() signature.
    All kwargs are mapped to the new OODDatasetBuilder class.
    """
    builder = OODDatasetBuilder(
        input_path=kwargs.get("input"),
        output_path=kwargs.get("output"),
        input_type=kwargs.get("itype", "FASTA"),
        output_type=kwargs.get("otype", "FASTA"),
        input_predictions=kwargs.get("input_predictions"),
        dinuc=kwargs.get("dinuc", False),
        k=kwargs.get("k", 1),
        num_tandem_repeats=kwargs.get("num_tandem_repeats", 0),
        n_threshold=kwargs.get("n_threshold", N_THRESHOLD),
    )
    builder.build()
