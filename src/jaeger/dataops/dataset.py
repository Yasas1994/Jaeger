"""Dataset creation and management utilities.

Functions for reading sequences, generating fragments, splitting datasets,
writing outputs, and creating non-redundant training databases via MMseqs2.
"""

from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pyfastx


def read_sequences(
    input_path: Path, intype: str, seq_col=None, class_col=None, class_id=None
):
    """Read sequences from a FASTA or CSV file."""
    records = []
    if intype == "FASTA":
        for name, seq in pyfastx.Fasta(str(input_path), build_index=False):
            records.append((name, str(seq), class_id))
    elif intype == "CSV":
        with open(input_path) as fh:
            reader = csv.reader(fh)
            for row in reader:
                seq = row[seq_col]
                cls = row[class_col]
                name = f"seq_{len(records)}"
                records.append((name, seq, cls))
    else:
        raise ValueError(f"Unsupported input type: {intype}")
    return records


def generate_fragments(records, frag_len=2048, overlap=1024):
    """Generate fragments from sequences."""
    fragments = []
    for name, seq, cls in records:
        seq = str(seq)
        start = 0
        frag_id = 0
        L = len(seq)
        if L >= frag_len:
            while start < L:
                end = min(start + frag_len, L)
                offset = frag_len - (end - start)
                start = start if offset == 0 else start - offset
                frag = seq[start:end]
                frag_name = (
                    f"{name}_frag{frag_id}_start{start}_len{len(frag)}_cls={cls}"
                )
                fragments.append((frag_name, frag, cls))
                frag_id += 1
                if end == L:
                    break
                start = end - overlap
    return fragments


def write_fasta(records, output_path):
    """Write sequences to a FASTA file."""
    with open(output_path, "w") as fh:
        for name, seq, _ in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")


def run_mmseqs_cluster(frag_fasta, out_prefix, tmpdir, min_id, min_cov):
    """Run MMseqs2 easy-cluster."""
    if shutil.which("mmseqs") is None:
        sys.exit("Error: MMseqs2 not found in PATH.")
    subprocess.run(
        [
            "mmseqs",
            "easy-cluster",
            frag_fasta,
            out_prefix,
            tmpdir,
            "--min-seq-id",
            str(min_id),
            "-c",
            str(min_cov),
        ],
        check=True,
    )


def split_dataset(records, trainperc, valperc, testperc):
    """Split records into train, val, and test sets."""

    random.shuffle(records)
    N = len(records)
    n_train = int(trainperc * N)
    n_val = int(valperc * N)
    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]
    return train, val, test


def write_output(train, val, test, out_prefix, outtype="CSV"):
    """Write output subsets in FASTA or CSV format."""
    subsets = {"train": train, "val": val, "test": test}
    for name, subset in subsets.items():
        if len(subset) > 0:
            if outtype == "FASTA":
                out_file = out_prefix / out_prefix.with_name(
                    f"{out_prefix.name}_{name}.fasta"
                )
                write_fasta(subset, out_file)
            elif outtype == "CSV":
                out_file = out_prefix / out_prefix.with_name(
                    f"{out_prefix.name}_{name}.csv"
                )
                with open(out_file, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    for seq_id, seq, cls in subset:
                        writer.writerow([cls, seq, seq_id])
            else:
                raise ValueError(f"Unsupported output type: {outtype}")


def build_dataset(**kwargs):
    """Generate a non-redundant fragment database from a FASTA/CSV file using MMseqs2.

    Required kwargs:
      input      : path to input FASTA/CSV of contigs
      output     : prefix for output FASTA/CSV files
      valperc    : 0.1    # fraction for validation set
      trainperc  : 0.8    # fraction for training set
      testperc   : 0.1    # fraction for test set
      maxiden    : 0.6    # minimum sequence identity for clustering
      maxcov     : 0.6    # minimum coverage fraction for clustering
      method     : "ANI"  # or "AAI" (to do: AAI)
      outtype    : "CSV"  # or "FASTA"
      intype     : "CSV"  # or "FASTA"
      class      : int       # class label as an int
      class_col  : int     # col index of CSV with class id
      seq_col    : int     # col index of CSV with sequence
    """
    inp = Path(kwargs["input"])
    out_pref = Path(kwargs["output"])
    valperc = kwargs.get("valperc", 0.1)
    trainperc = kwargs.get("trainperc", 0.8)
    testperc = kwargs.get("testperc", 0.1)
    maxiden = kwargs.get("maxiden", 0.6)
    maxcov = kwargs.get("maxcov", 0.6)
    fraglen = kwargs.get("fraglen", 2048)
    overlap = kwargs.get("overlap", 1024)
    method = kwargs.get("method", "ANI").upper()
    outtype = kwargs.get("outtype", "CSV").upper()
    intype = kwargs.get("intype", "CSV").upper()
    class_col = kwargs.get("class_col")
    seq_col = kwargs.get("seq_col")
    class_id = kwargs.get("class")

    assert abs(trainperc + valperc + testperc - 1.0) < 1e-6, (
        "train+val+test must sum to 1"
    )

    def get_class(x):
        return x.split("=")[-1]

    out_pref.mkdir(exist_ok=True, parents=True)

    # 1. Read input sequences
    records = read_sequences(inp, intype, seq_col, class_col, class_id)

    # 2. Generate fragments
    fragments = generate_fragments(records, frag_len=fraglen, overlap=overlap)
    frag_fasta = out_pref / f"{inp.name}.fragments.fasta"
    write_fasta(fragments, frag_fasta)

    # 3. Run MMseqs2
    clusters = out_pref / f"{inp.name}.fragments.clusters"
    tmpdir = out_pref / "tmp"
    match method:
        case "ANI":
            run_mmseqs_cluster(frag_fasta, clusters, tmpdir, maxiden, maxcov)
        case "AAI":
            NotImplementedError("AAI method is not yet implemented")

    # 4. Load representatives
    rep_seq = out_pref / f"{inp.name}.fragments.clusters_rep_seq.fasta"
    if not rep_seq.exists():
        raise FileNotFoundError(f"Expected MMseqs2 rep file: {rep_seq}")
    if class_id:
        reps = [
            (h, str(s), class_id)
            for h, s in pyfastx.Fasta(str(rep_seq), build_index=False)
        ]
    else:
        reps = [
            (h, str(s), get_class(h))
            for h, s in pyfastx.Fasta(str(rep_seq), build_index=False)
        ]

    # 5. Split datasets
    train, val, test = split_dataset(reps, trainperc, valperc, testperc)

    # 6. Write outputs
    write_output(train, val, test, out_pref, outtype)

    print(
        f"{len(fragments)} fragments → {len(reps)} reps → "
        f"{len(train)} train, {len(val)} val, {len(test)} test"
    )
