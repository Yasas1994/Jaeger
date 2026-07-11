import csv
import random
import subprocess
import shutil
import sys
import pyfastx
import numpy as np
from pathlib import Path
from rich.progress import track
from jaeger.utils.logging import get_logger

from jaeger.dataops.dataset import build_dataset
from jaeger.dataops.convert import convert_dataset


logger = get_logger(log_file=None, log_path=None, level=3)


def mask_core(**kwargs):
    import numpy as np

    _rng = np.random.default_rng()

    # Pre‐define your alt‐nuc_map once
    _ALT = {
        ord("A"): ("T", "G", "C"),
        ord("T"): ("A", "G", "C"),
        ord("G"): ("A", "T", "C"),
        ord("C"): ("A", "T", "G"),
    }
    _DEFAULT_ALTS = ("N", "N", "N")

    input_path = kwargs.get("input")
    output_path = kwargs.get("output")
    min_perc = kwargs.get("minperc", 0.0)
    max_perc = kwargs.get("maxperc", 1.0)
    step = kwargs.get("step", 0.01)  # increment in mutation percentage
    mutate = kwargs.get("mutate", False)  # replace with random nucleotides

    f = pyfastx.Fasta(input_path, build_index=False)

    # def soft_mutation(seq: str, indices):
    #     """
    #     Turn seq[i]→lowercase for each i in indices, but leave other letters untouched.
    #     Works in‐place on a bytearray.
    #     """
    #     ba = bytearray(seq, "ascii")        # O(N) once
    #     mask = 0x20                         # bit to flip uppercase→lowercase
    #     for i in indices:
    #         # only flip if currently uppercase A–Z
    #         c = ba[i]
    #         if 0x41 <= c <= 0x5A:           # 'A'..'Z'
    #             ba[i] = c | mask
    #             # print(chr(c |  mask))
    #     return ba.decode("ascii")

    def hard_mask(seq: str, indices):
        """
        Turn seq[i]→N for each i in indices, but leave other letters untouched.
        Works in‐place on a bytearray.
        """
        ba = bytearray(seq, "ascii")  # O(N) once
        for i in indices:
            ba[i] = 0x4E
        return ba.decode("ascii")

    def replacement_mutation(seq: str, indices):
        """
        For each i in indices, replace seq[i] with one of its 3 alternatives
        uniformly at random. Other positions remain unchanged.
        """
        ba = bytearray(seq, "ascii")  # O(N)
        choices = _rng.integers(0, 3, size=len(indices))  # vectorized integer sampling

        for i, choice in zip(indices, choices):
            alts = _ALT.get(ba[i], _DEFAULT_ALTS)
            # ba[i] = ord(alts[choice])      # if you want to mutate the bytearray
            # but since alts[...] is a str of length 1:
            ba[i] = ord(alts[choice])

        return ba.decode("ascii")

    with open(output_path, "w") as fh:
        for name, seq in track(f, description="Processing..."):
            seq = str(seq)
            seqlen = len(seq)
            current_perc = min_perc
            used_indices = set()

            while current_perc <= max_perc:
                # Write mutated FASTA entry
                fh.write(f">{name}_mutperc_{current_perc * 100:.2f}\n")
                for i in range(0, len(seq), 70):
                    fh.write(seq[i : i + 70] + "\n")
                # Determine number of new positions to mutate
                num_mutate = int(seqlen * step)
                # Choose from unused indices to avoid re-mutating
                available = list(set(np.arange(seqlen)) - used_indices)
                if not available:
                    break
                new_indices = np.random.choice(
                    available, min(num_mutate, len(available)), replace=False
                )
                used_indices.update(new_indices)

                # Apply mutation
                if mutate is not True:
                    seq = hard_mask(seq, new_indices)
                else:
                    seq = replacement_mutation(seq, new_indices)

                current_perc += step


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
                    # writer.writerow(["class", "sequence", "id"])
                    for seq_id, seq, cls in subset:
                        writer.writerow([cls, seq, seq_id])
            else:
                raise ValueError(f"Unsupported output type: {outtype}")


def dataset_core(**kwargs):
    """
    Generate a non-redundant fragment database from a FASTA/CSV file using MMseqs2.

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
    build_dataset(**kwargs)


def convert_core(**kwargs):
    import pandas as pd

    """
    Convert between CSV and FASTA using pandas and pyfastx.

    Parameters
    ----------
    input_path : str
        Path to the input file (CSV or FASTA).
    output_path : str
        Path to the output file (FASTA or CSV).
    input_type : str
        Type of the input file: 'csv' or 'fasta'.
    """
    input_path = Path(kwargs.get("input"))
    output_path = Path(kwargs.get("output"))
    input_type = kwargs.get("itype")
    if input_type == "CSV":
        # CSV -> FASTA
        df = pd.read_csv(
            input_path, usecols=[0, 1, 2], names=["class", "sequence", "id"], dtype=str
        )
        with open(output_path, "w") as fasta_out:
            for idx, row in df.iterrows():
                seq_id = row["id"].strip()
                cls_id = row["class"].strip()
                seq = row["sequence"].strip()
                fasta_out.write(f">{seq_id}__class={cls_id}\n{seq}\n")
        print(f"[✓] Converted CSV to FASTA: {output_path}")

    elif input_type == "FASTA":
        # FASTA -> CSV
        fasta = pyfastx.Fasta(str(input_path), build_index=False)
        records = []
        for name, seq in fasta:
            seq_id, cls_id = name.split("__class=")
            records.append((cls_id, seq, seq_id))
        df = pd.DataFrame(records, columns=["class", "sequence", "id"])
        df.to_csv(output_path, index=False, header=False)
        print(f"[✓] Converted FASTA to CSV: {output_path}")

    else:
        raise ValueError("input_type must be 'CSV' or 'FASTA'")


# ------------------------------------------------------------------
# Stats (late import to avoid heavy deps at module load time)
# ------------------------------------------------------------------
from jaeger.utils.stats import welch_t_one_tailed  # noqa: E402


def stats_core(**kwargs):
    import matplotlib.pyplot as plt

    import seaborn as sns
    import pandas as pd

    """
    Calculate stats and create plots from jaeger output/s
    
    1. percentage of each class
    2. reliability score distribution
    3. class score distributions

    """
    input_path = Path(kwargs.get("input"))
    output_path = Path(kwargs.get("output"))
    output_path.mkdir(exist_ok=True, parents=True)
    pct_class = output_path / "class_percentages.png"
    pct_class_pval = output_path / "class_percentages_pval.png"
    relscore = output_path / "reliability_scores.png"
    relscore_len = output_path / "reliability_scores_by_length.png"
    ent = output_path / "entropy.png"
    eng = output_path / "energy.png"
    clscores = output_path / "class_scores.png"
    tsv_with_pvals = output_path / "jaeger_output_with_pvals.tsv"

    df = pd.read_table(input_path)
    sns.set_context("paper", font_scale=1.2)
    reliability_available = pd.api.types.is_numeric_dtype(df["reliability_score"])
    if not reliability_available:
        logger.warning(
            "Reliability score is unavailable in the input; skipping reliability-related plots."
        )
    if len(df) > 1:
        if reliability_available:
            # Create the count plot
            df["above_threshold"] = df["reliability_score"].apply(
                lambda x: "passed" if x >= 0.8 else "failed"
            )
            ax = sns.countplot(
                data=df,
                x="prediction",
                hue="above_threshold",
                palette="pastel",
                stat="percent",
            )
            # Annotate bars with percentage values (already in percent)
            for p in ax.patches:
                percentage = p.get_height()
                if percentage > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2,
                        p.get_height(),
                        f"{percentage:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
            # Style tweaks
            ax.set_ylabel("Percentage")
            ax.set_xlabel("Prediction")
            ax.set_title("Class Distribution (%)")
            sns.despine()
            plt.tight_layout()
            plt.savefig(pct_class, dpi=150, bbox_inches="tight")
            plt.close()

            # Calculate per-class distribution of reliability scores
            ax = sns.violinplot(df, x="prediction", y="reliability_score")
            sns.stripplot(
                df,
                x="prediction",
                y="reliability_score",
                s=1,
                alpha=0.1,
                color="gray",
                ax=ax,
            )
            ax.set_ylabel("Reliability score")
            ax.set_xlabel("Class")
            ax.set_title("Per-class distribution of reliability scores")
            sns.despine()
            plt.tight_layout()
            plt.savefig(relscore, dpi=150, bbox_inches="tight")
            plt.close()

        # Calculate per-class distribution of entropy
        ax = sns.violinplot(df, x="prediction", y="entropy")
        sns.stripplot(
            df, x="prediction", y="entropy", s=1, alpha=0.1, color="gray", ax=ax
        )
        ax.set_ylabel("Entropy")
        ax.set_xlabel("Class")
        ax.set_title("Per-class distribution of entropy")
        sns.despine()
        plt.tight_layout()
        plt.savefig(ent, dpi=150, bbox_inches="tight")
        plt.close()

        # Calculate per-class distribution of energy
        if "energy" in df.columns:
            ax = sns.violinplot(df, x="prediction", y="energy")
            sns.stripplot(
                df, x="prediction", y="energy", s=1, alpha=0.1, color="gray", ax=ax
            )
            ax.set_ylabel("Energy")
            ax.set_xlabel("Class")
            ax.set_title("Per-class distribution of Energy")
            sns.despine()
            plt.tight_layout()
            plt.savefig(eng, dpi=150, bbox_inches="tight")
            plt.close()

        # Calculate perclass score distributions
        # Create the grid
        df_long = pd.melt(
            df[
                ["contig_id", "length", "prediction"]
                + [
                    i
                    for i in df.columns
                    if i.endswith("_score") and i != "reliability_score"
                ]
            ],
            id_vars=["contig_id", "length", "prediction"],
            var_name="score_class",
            value_name="scores",
        )
        g = sns.FacetGrid(
            df_long,
            row="prediction",
            hue="score_class",
            margin_titles=False,
            height=2,
            aspect=3.5,
        )
        g.map(
            sns.kdeplot,
            "scores",
            fill=True,
            common_norm=False,
            alpha=0.2,
            linewidth=0.5,
        )
        g.add_legend()
        # Add titles and adjust layout
        g.set_axis_labels("Score", "Density")
        # g.set_titles("Per-class score distributions")
        g.savefig(clscores, dpi=150, bbox_inches="tight")
        plt.close()
        try:
            # quantile bins
            bins = pd.qcut(df["length"], q=5)

            # Extract bin edges
            bin_edges = bins.cat.categories

            # Create labels with numeric min–max
            labels = [
                f"{int(interval.left):,}–{int(interval.right):,}"
                for interval in bin_edges
            ]

            # Recreate qcut with readable labels
            df["length_bin"] = pd.qcut(df["length"], q=5, labels=labels)
            # Calculate per-class distribution of reliability scores
            ax = sns.violinplot(df, x="length_bin", y="reliability_score")
            sns.stripplot(
                df,
                x="length_bin",
                y="reliability_score",
                s=1,
                alpha=0.1,
                color="red",
                ax=ax,
            )
            ax.set_ylabel("Reliability score")
            ax.set_xlabel("Length range")
            ax.set_title("Legth-wise (quantile) distribution of reliability scores")
            plt.xticks(rotation=45)
            sns.despine()
            plt.tight_layout()
            plt.savefig(relscore_len, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(e)
            logger.warning("Legth-wise (quantile) plot was not created")

    # perform welch t-tests to check if there is a statistically significant difference
    # between the top-k classes
    mean_scores = df[
        [i for i in df.columns if i.endswith("_score") and "reliability" not in i]
    ].to_numpy()
    var_scores = df[[i for i in df.columns if i.endswith("_var")]].to_numpy()
    windows = (
        df[[i for i in df.columns if i.endswith("_windows") and "reliability" not in i]]
        .to_numpy()
        .sum(axis=-1)
    )
    rows = np.arange(mean_scores.shape[0])[:, None]
    sorted_indices = np.flip(np.argsort(mean_scores, axis=-1), axis=-1)
    sorted_means = mean_scores[rows, sorted_indices[:, :2]]
    sorted_vars = var_scores[rows, sorted_indices[:, :2]]
    pvals = []
    for means, vars, n in zip(sorted_means, sorted_vars, windows):
        _, _, p = welch_t_one_tailed(
            mean1=means[0], var1=vars[0], mean2=means[1], var2=vars[1], n1=n, n2=n
        )
        pvals.append(p)
    df["pval"] = pvals

    df.to_csv(tsv_with_pvals, index=None, sep="\t", float_format="%.3f")
    # Create the count plot

    if len(df) > 1:
        df["above_pval_threshold"] = df["pval"].apply(
            lambda x: "passed" if x <= 0.05 else "failed"
        )
        ax = sns.countplot(
            data=df,
            x="prediction",
            hue="above_pval_threshold",
            palette="pastel",
            stat="percent",
        )
        # Annotate bars with percentage values (already in percent)
        for p in ax.patches:
            percentage = p.get_height()
            if percentage > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_height(),
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        # Style tweaks
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Prediction")
        ax.set_title("Class Distribution (%)")
        sns.despine()
        plt.tight_layout()
        plt.savefig(pct_class_pval, dpi=150, bbox_inches="tight")
        plt.close()


# =============================================================================
# Data optimization / format conversion
# =============================================================================


def optimize_data_core(
    input_path: str,
    output_path: str,
    format: str,
    crop_size: tuple[int, ...] = (500,),
    stride: int = 0,
    strides: list[int] | None = None,
    num_classes: int = 3,
    num_workers: int | None = None,
    one_hot: bool = False,
    pad_int: int = 0,
    codon_map: str = "codon_id",
    nucleotide_map: str | None = None,
    compress: str = "default",
    dtype: str = "auto",
    max_length: int = 5000,  # deprecated, ignored
    max_memory_mb: int | None = None,
    pad: bool = False,
    units: str = "nuc",
    overlap: float | None = None,
):
    """Convert Jaeger CSV training data to an optimized ``.npz`` format.

    This is a thin wrapper around :func:`jaeger.dataops.convert.convert_dataset`.
    See that function for details on supported formats and output contents.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (label,sequence format).
    output_path : str
        Path to output ``.npz`` file.
    format : str
        One of ``nucleotide``, ``translated``, or ``both``.
    crop_size : tuple[int, ...]
        Sequence crop size(s) (default: ``(500,)``).
    stride : int, optional
        Sliding-window stride applied to every crop size (default: 0).
    strides : list[int] | None, optional
        Per-crop-size strides. If given, overrides ``stride``.
    num_classes : int, optional
        Number of classes (default: 3).
    num_workers : int | None, optional
        Number of parallel workers. ``None`` processes in a single worker.
    one_hot : bool, optional
        Encode nucleotide crops as one-hot float tensors (default: False).
    pad_int : int, optional
        Integer padding value for nucleotide crops (default: 0).
    codon_map : str, optional
        Codon map name (default: ``codon_id``).
    nucleotide_map : str | None, optional
        JSON string with mappings for ``A``, ``C``, ``G``, ``T``, ``N``.
    compress : str, optional
        Compression mode for the output archive (default: ``default``).
    dtype : str, optional
        Integer dtype for encoded features: ``int8``, ``uint8``, ``int16``,
        ``int32``, or ``auto``. ``auto`` selects the smallest dtype that fits
        the vocabulary (default: ``auto``).
    max_memory_mb : int | None, optional
        Memory budget in MB for encoded output buffers. ``None`` uses ~75% of
        available RAM. ``0`` disables streaming.
    pad : bool, optional
        If True, pad all crops to the global maximum length. Default is False.
    max_length : int, optional
        Deprecated and ignored. Kept for backward compatibility.
    units : str, optional
        Units for ``crop_size`` and ``stride``: ``nuc`` (nucleotides) or
        ``codon`` (codons; crop sizes convert to nucleotides via
        ``3*codons + 5``; strides scale by 3).
    overlap : float | None, optional
        Overlap between crops as a fraction of each crop size (0.0-1.0).
        If provided, per-crop strides are computed from the (unit-converted)
        crop sizes and ``stride`` is ignored.
    """
    if units not in {"nuc", "codon"}:
        raise ValueError("units must be 'nuc' or 'codon'")

    if units == "codon":
        from jaeger.seqops.crop import codons_to_nucleotides

        # Codon crops must land on the mod-2 branch so both frame extractors
        # agree: nucleotide length = 3*codons + 5. Stride is a shift, so it
        # scales by 3 without the +5 window offset.
        crop_size = tuple(codons_to_nucleotides(cs) for cs in crop_size)
        stride = stride * 3
        if strides is not None:
            strides = [s * 3 for s in strides]

    if strides is None and overlap is not None:
        strides = [int(cs * (1 - overlap)) for cs in crop_size]

    convert_dataset(
        input_path=input_path,
        output_path=output_path,
        format=format,
        crop_size=crop_size,
        stride=stride,
        strides=strides,
        num_classes=num_classes,
        num_workers=num_workers,
        one_hot=one_hot,
        pad_int=pad_int,
        codon_map=codon_map,
        nucleotide_map=nucleotide_map,
        compress=compress,
        dtype=dtype,
        max_length=max_length,
        max_memory_mb=max_memory_mb,
        pad=pad,
    )
