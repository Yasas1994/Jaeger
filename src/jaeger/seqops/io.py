"""Sequence I/O operations for FASTA files.

Fragment generation, reading, writing, and validation of DNA sequences.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Generator

import pyfastx
import pydustmasker
from rich.progress import Progress

from jaeger.utils.misc import safe_divide, signal_l

logger = logging.getLogger("Jaeger")

# TRF parameters for tandem repeat detection
TRF_EXECUTABLE = "trf"
TRF_MATCH, TRF_MISMATCH, TRF_DELTA, TRF_PM, TRF_PI, TRF_MINSCORE, TRF_MAXPERIOD = (
    2,
    7,
    7,
    80,
    10,
    50,
    500,
)


def _window_indices(
    seqlen: int,
    fragsize: int,
    stride: int | None,
    dynamic_stride: bool,
    dynamic_stride_threshold: float,
) -> list[int]:
    """Return window start indices for a contig.

    If *dynamic_stride* is enabled and the contig is shorter than
    *dynamic_stride_threshold* * *fragsize*, windows are distributed evenly so
    that the final window reaches the contig end. Otherwise the legacy fixed
    stride is used.
    """
    if not dynamic_stride or seqlen >= dynamic_stride_threshold * fragsize:
        step = fragsize if stride is None else stride
        return list(range(0, seqlen - (fragsize - 1), step))

    n_windows = max(1, math.ceil(seqlen / fragsize))
    if n_windows == 1:
        return [0]

    raw_stride = (seqlen - fragsize) / (n_windows - 1)
    indices = [int(round(i * raw_stride)) for i in range(n_windows)]
    indices[-1] = seqlen - fragsize

    # Deduplicate while preserving order.
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def fragment_generator(
    file_path: str,
    fragsize: int | None = None,
    stride: int | None = None,
    num: int | None = None,
    no_progress: bool = True,
    dustmask: bool = True,
    dynamic_stride: bool = False,
    dynamic_stride_threshold: float = 10.0,
    min_len: int | None = None,
    max_len: int | None = None,
) -> Generator[str, None, None]:
    """Generate fragments of DNA sequences from a FASTA file.

    Optionally masks low-complexity regions (recommended for eukaryotic
    host-associated metagenomes).

    Yields strings of the form:
    ``sequence,header,index,contig_end,i,seqlen,g,c,a,t,gc_skew``
    """
    if min_len is None:
        min_len = fragsize

    def _gen():
        fa = pyfastx.Fasta(file_path, build_index=False)
        with Progress(transient=True, disable=no_progress) as progress:
            task = progress.add_task("[cyan]hunting...", total=num)
            for j, record in enumerate(fa):
                progress.update(task, advance=1)
                seqlen = len(record[1])
                sequence = record[1].strip().upper()
                if dustmask:
                    sequence = pydustmasker.DustMasker(
                        sequence, window_size=64, score_threshold=20
                    ).mask()
                header = record[0].strip().replace(",", "___")
                if max_len is not None and seqlen > max_len:
                    continue
                if seqlen >= fragsize:
                    if fragsize is None:
                        yield f"{sequence},{header}"
                    else:
                        indices = _window_indices(
                            seqlen,
                            fragsize,
                            stride,
                            dynamic_stride,
                            dynamic_stride_threshold,
                        )
                        for i, (b, index) in enumerate(signal_l(indices)):
                            g = sequence[index : index + fragsize].count("G")
                            c = sequence[index : index + fragsize].count("C")
                            a = sequence[index : index + fragsize].count("A")
                            t = sequence[index : index + fragsize].count("T")
                            gc_skew = safe_divide((g - c), (g + c))
                            yield (
                                f"{sequence[index : index + fragsize]},"
                                f"{header},{index},{b},{i},{seqlen},{g},{c},{a},{t},"
                                f"{gc_skew: .3f}"
                            )
                elif seqlen >= min_len:
                    # Whole-contig window for short sequences.
                    g = sequence.count("G")
                    c = sequence.count("C")
                    a = sequence.count("A")
                    t = sequence.count("T")
                    gc_skew = safe_divide((g - c), (g + c))
                    yield (
                        f"{sequence},"
                        f"{header},0,1,0,{seqlen},{g},{c},{a},{t},"
                        f"{gc_skew: .3f}"
                    )

    return _gen()


def fragment_generator_lib(
    filename: str,
    fragsize: int | None = None,
    stride: int | None = None,
    num: int | None = None,
    dynamic_stride: bool = False,
    dynamic_stride_threshold: float = 10.0,
    min_len: int | None = None,
    max_len: int | None = None,
) -> Generator[str, None, None]:
    """Simpler fragment generator for library use."""
    if min_len is None:
        min_len = fragsize
    head = False
    if isinstance(filename, str):
        tmpfn = pyfastx.Fasta(filename, build_index=False)
        head = True
    else:
        raise ValueError("Not a supported input type")

    def _gen():
        for n, record in enumerate(tmpfn):
            if head:
                seqlen = len(record[1])
                seq = record[1]
                headder = record[0].replace(",", "___")
            else:
                seqlen = len(record)
                seq = record
                headder = f"seq_{n}"
            if max_len is not None and seqlen > max_len:
                continue
            if seqlen >= fragsize:
                if fragsize is None:
                    yield f"{str(seq)},{str(headder)}"
                else:
                    indices = _window_indices(
                        seqlen,
                        fragsize,
                        stride,
                        dynamic_stride,
                        dynamic_stride_threshold,
                    )
                    for i, (b, index) in enumerate(signal_l(indices)):
                        yield (
                            f"{str(seq)[index : index + fragsize]},"
                            f"{str(headder)},{str(index)},{str(b)},{str(i)},"
                            f"{str(seqlen)}"
                        )
            elif seqlen >= min_len:
                yield (f"{str(seq)},{str(headder)},0,1,0,{str(seqlen)}")

    return _gen()


def read_sequences(
    input_path: str,
    input_type: str = "FASTA",
    frag_len: int = 2048,
    overlap: int = 1024,
) -> list[dict]:
    """Read sequences from FASTA or CSV file.

    Returns a list of dicts with keys ``id``, ``seq``, ``label``.
    """
    records = []
    if input_type == "FASTA":
        fa = pyfastx.Fasta(input_path, build_index=False)
        for seq in fa:
            records.append({"id": seq[0], "seq": seq[1], "label": None})
    elif input_type == "CSV":
        import csv

        with open(input_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    records.append(
                        {
                            "id": row[0],
                            "seq": row[1],
                            "label": row[2] if len(row) > 2 else None,
                        }
                    )
    else:
        raise ValueError("input_type must be 'CSV' or 'FASTA'")
    return records


def write_fasta(records: list[dict], output_path: str | Path) -> None:
    """Write a list of sequence records to a FASTA file.

    Each record should be a dict with keys ``id`` and ``seq``.
    """
    with open(output_path, "w") as f:
        for record in records:
            f.write(f">{record['id']}\n{record['seq']}\n")


def write_fasta_entry(fh, header, seq, label):
    """Write a single FASTA entry to an open file handle.

    Format: >{header}__class={label} followed by sequence wrapped at 70 chars.
    """
    fh.write(f">{header}__class={label}\n")
    for i in range(0, len(seq), 70):
        fh.write(seq[i : i + 70] + "\n")


def write_fasta_record(
    fh,
    header: str,
    seq: str,
    label: str | None = None,
) -> None:
    """Write a single FASTA/CSV record to an open file handle.

    If *label* is provided, it is appended as a CSV field after the sequence.
    """
    if label is not None:
        fh.write(f">{header}\n{seq},{label}\n")
    else:
        fh.write(f">{header}\n{seq}\n")


def write_fasta_from_results(
    data: dict,
    output_path: str | Path,
    reliability_cutoff: float = 0.5,
    phage_score: int = 1,
) -> None:
    """Write FASTA entries from inference results.

    *data* is a dict mapping contig_id -> (DataFrame, host_label, length).
    Only entries with phage-like predictions above the reliability cutoff
    are written.
    """
    with open(output_path, "w") as fh:
        for contig_id, (df, host_label, length) in data.items():
            for _, row in df.iterrows():
                if row.get("reliability_score", 0) >= reliability_cutoff:
                    if row.get("prediction") == phage_score:
                        fh.write(f">{contig_id}_{row.get('index', 0)}\n")
                        fh.write(f"{row.get('sequence', '')}\n")


def validate_fasta_entries(
    input_file_path: str, min_len: int = 2048
) -> int | Exception:
    """Validate a FASTA file and count entries above *min_len*."""
    num = 0
    gt_min_len = 0
    logger.debug("validating fasta file")
    fa = pyfastx.Fasta(input_file_path, build_index=False)
    for seq in fa:
        num += 1
        gt_min_len += 1 if len(seq[1]) >= min_len else 0
    logger.info(f"{gt_min_len}/{num} entries in {input_file_path}")

    if gt_min_len == 0:
        raise Exception(f"all records in {input_file_path} are < {min_len}bp")

    return num


# ------------------------------------------------------------------
# Tandem repeat masking (TRF + pyfastx)
# ------------------------------------------------------------------


def split_fasta_with_pyfastx(
    input_fasta: str,
    output_dir: str,
    chunks: int | None = None,
    counts: int | None = None,
) -> list[str]:
    """Split a FASTA file into chunks using pyfastx.

    Returns paths to non-empty chunk files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if chunks:
        cmd = [
            "pyfastx",
            "split",
            "-n",
            str(chunks),
            "-o",
            str(output_dir),
            input_fasta,
        ]
    elif counts:
        cmd = [
            "pyfastx",
            "split",
            "-c",
            str(counts),
            "-o",
            str(output_dir),
            input_fasta,
        ]
    else:
        raise ValueError("You must specify either chunks (-n) or counts (-c)")

    logger.info("Splitting FASTA with pyfastx...")
    subprocess.run(cmd, check=True)

    all_chunks = list(output_dir.glob(f"*{Path(input_fasta).suffix}"))
    non_empty_chunks = []
    for f in sorted(all_chunks):
        if f.stat().st_size == 0:
            f.unlink()
        else:
            non_empty_chunks.append(str(f))

    logger.info(f"Split into {len(non_empty_chunks)} non-empty chunk files.")
    return non_empty_chunks


def run_trf_and_collect_mask(fasta_path: str, out_dir: str) -> Path:
    """Run TRF on a single FASTA chunk and return the masked file path."""
    fasta_path = Path(fasta_path).resolve()
    fasta_name = fasta_path.name
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        TRF_EXECUTABLE,
        str(fasta_path),
        str(TRF_MATCH),
        str(TRF_MISMATCH),
        str(TRF_DELTA),
        str(TRF_PM),
        str(TRF_PI),
        str(TRF_MINSCORE),
        str(TRF_MAXPERIOD),
        "-h",
        "-m",
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=out_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    param_suffix = f".{TRF_MATCH}.{TRF_MISMATCH}.{TRF_DELTA}.{TRF_PM}.{TRF_PI}.{TRF_MINSCORE}.{TRF_MAXPERIOD}.mask"
    mask_file = out_dir / f"{fasta_name}{param_suffix}"

    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    # Clean up TRF auxiliary files
    for ext in [".dat", ".repeat"]:
        extra = (
            fasta_name
            + f".{TRF_MATCH}.{TRF_MISMATCH}.{TRF_DELTA}.{TRF_PM}.{TRF_PI}.{TRF_MINSCORE}.{TRF_MAXPERIOD}{ext}"
        )
        if os.path.exists(out_dir / extra):
            os.remove(out_dir / extra)

    return mask_file


def run_trf_batch(
    fasta_files: list[str], out_dir: str = "masked_chunks", n_threads: int | None = None
) -> list[Path]:
    """Run TRF on a list of FASTA files in parallel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_threads is None:
        n_threads = max(1, cpu_count() - 1)

    args = [(f, out_dir) for f in fasta_files]
    with Pool(processes=n_threads) as pool:
        results = pool.starmap(run_trf_and_collect_mask, args)

    logger.debug(f"All masked chunk files saved to: {out_dir}")
    return results


def merge_masked_files(masked_files: list[str | Path], output_file: str | Path) -> None:
    """Merge all masked FASTA files into a single FASTA file."""
    with open(output_file, "w") as outfile:
        for mf in masked_files:
            with open(mf) as infile:
                shutil.copyfileobj(infile, outfile)
    logger.debug(f"Merged masked FASTAs into: {output_file}")
