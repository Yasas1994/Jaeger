"""Out-of-distribution (OOD) detection utilities for sequence shuffling and splitting.

Provides functions for generating shuffled (negative) sequence datasets and
sampling genome fragments to mimic metagenomic assemblies.
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl
import pyfastx

from jaeger.seqops.transform import dinuc_shuffle, kmer_shuffle
from jaeger.seqops.synthetic import generate_random_tandem_repeats
from jaeger.seqops.io import write_fasta_entry
from jaeger.utils.logging import get_logger

logger = get_logger(log_file=None, log_path=None, level=3)


def shuffle_core(**kwargs):
    """
    Shuffle sequences while maintaining the
    1. dinuc composition or
    2. break a sequence into k-mers -> shuffle -> concat
    3. random shuffling
    """
    if kwargs.get("dinuc"):
        shuffle_fn = dinuc_shuffle
    else:

        def shuffle_fn(x):
            return kmer_shuffle(seq=x, k=kwargs.get("k", 1))

    n_tandem_repeats = kwargs.get("num_tandem_repeats")
    correct = set()
    if kwargs.get("input_predictions"):
        logger.info("using jaeger predictions to generate the ood dataset")
        map_ = {
            0: "bacteria",
            1: "phage",
            2: "eukarya",
            3: "archaea",
            4: "plasmid",
            5: "virus",
        }
        ip = pl.read_csv(
            kwargs.get("input_predictions"),
            truncate_ragged_lines=True,
            separator="\t",
            columns=[0, 2],
        )
        ip = ip.with_columns(
            true_class=pl.col("contig_id").str.split("__class=").list.last()
        )
        ip = ip.with_columns(true_class=pl.col("true_class").replace(map_))
        ip = ip.with_columns(
            is_correct=(pl.col("true_class") == pl.col("prediction")) * 1
        )
        ip = ip.with_columns(
            contig_id=pl.col("contig_id").str.split("__class=").list.first()
        )
        correct = set(ip.filter(pl.col("is_correct") == 1)["contig_id"].to_list())
    match kwargs.get("itype"):
        case "CSV":
            f = pl.read_csv(
                kwargs.get("input"), truncate_ragged_lines=True, has_header=False
            )
            f = f.select(["column_1", "column_2", "column_3"])
            f = f.with_columns(
                pl.when(pl.col("column_3").is_in(correct))
                .then(pl.lit(1, dtype=pl.Int64))
                .otherwise(pl.lit(0, dtype=pl.Int64))
                .alias("column_1")
            )
            f = f.with_columns(
                pl.when(
                    (
                        pl.col("column_2").str.count_matches("N")
                        / pl.col("column_2").str.len_chars()
                    )
                    > 0.3
                )
                .then(pl.lit(0, dtype=pl.Int64))
                .otherwise(pl.col("column_1"))
                .alias("column_1")
            )
            print(f.head())
            fs = f.with_columns(
                pl.col("column_2").map_elements(
                    lambda x: shuffle_fn(x), return_dtype=pl.String
                ),
                pl.lit(0, dtype=pl.Int64).alias("column_1"),
            )
            print(fs.head())
            ft = pl.DataFrame()
            if n_tandem_repeats > 0:
                ft = pl.from_dict(
                    dict(
                        column_1=np.array(
                            [0 for _ in range(n_tandem_repeats)], dtype=np.int64
                        ),
                        column_2=generate_random_tandem_repeats(
                            num_sequences=n_tandem_repeats
                        ),
                        column_3=[
                            f"tandem_repeat_{i}" for i in range(n_tandem_repeats)
                        ],
                    )
                )
            print(ft.head())
            logger.info(
                f"id : {len(f)} ood: {len(fs)} ood_tandem: {0 if ft.is_empty() else len(ft)}"
            )

            f = pl.concat(
                [i for i in [f, fs, ft] if not i.is_empty()], how="vertical"
            ).sample(fraction=1.0, shuffle=True, with_replacement=False)

            match kwargs.get("otype"):
                case "CSV":
                    f.select(["column_1", "column_2", "column_3"]).write_csv(
                        kwargs.get("output"), include_header=False
                    )
                case "FASTA":
                    with open(kwargs.get("output"), "w") as fh:
                        for row in f.iter_rows(named=True):
                            label = row["column_1"]
                            seq = row["column_2"]
                            seq_id = row["column_3"].split("__class=")[0]
                            # use the 3rd column as header
                            fh.write(f">{seq_id}__class={label}\n")
                            for i in range(0, len(seq), 70):
                                fh.write(seq[i : i + 70] + "\n")

        case "FASTA":
            input_path = kwargs.get("input")
            output_path = kwargs.get("output")
            otype = kwargs.get("otype")  # "FASTA" or "CSV"
            fasta_iter = pyfastx.Fasta(input_path, build_index=False)

            # 1) pre‑generate your tandem dict
            tandem = dict(
                column_1=[0 for _ in range(n_tandem_repeats)],
                column_2=generate_random_tandem_repeats(num_sequences=n_tandem_repeats),
                column_3=[f"tandem_repeat_{i}" for i in range(n_tandem_repeats)],
            )

            # 2) collect ALL entries into a list
            entries = []

            # 2a) from your input FASTA
            for name, seq in fasta_iter:
                ncount = seq.count("N")
                lowcomplex = len(seq) > 0 and (ncount / len(seq)) > 0.3
                not_in_correct = name not in correct
                shuffled = shuffle_fn(seq)
                id_ = name.split("__class=")[0]
                orig_label = 0 if (lowcomplex or not_in_correct) else 1

                # store tuples of (id, sequence, label)
                entries.append((id_, seq, orig_label))
                entries.append((id_, shuffled, 0))

            # 2b) from your tandem dict
            for label, tandem_seq, tandem_id in zip(
                tandem["column_1"], tandem["column_2"], tandem["column_3"]
            ):
                entries.append((tandem_id, tandem_seq, label))

            # 3) shuffle the entire pool
            random.shuffle(entries)

            # 4) write them out in the new order
            with open(output_path, "w") as fh:
                for seq_id, sequence, label in entries:
                    if otype == "FASTA":
                        write_fasta_entry(fh, seq_id, sequence, label)
                    else:  # CSV
                        fh.write(f"{label},{sequence},{seq_id}\n")
