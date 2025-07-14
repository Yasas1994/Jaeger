from jaeger.preprocess.shuffle_dna import dinuc_shuffle
import polars as pl
import pyfastx
import random
import subprocess
import shutil
import sys
import numpy as np
from pathlib import Path
from rich.progress import track


def shuffle_dna(seq: str) -> str:
    """Randomly shuffles a DNA sequence."""
    seq_list = list(seq)
    random.shuffle(seq_list)
    return "".join(seq_list)

def kmer_shuffle(seq: str, k: int) -> str:
    """Randomly shuffles a DNA sequence at the non-overlapping k-mer level"""
    # Trim sequence to nearest multiple of k
    trimmed_len = len(seq) - (len(seq) % k)
    trimmed_seq = seq[:trimmed_len]
    
    # Split into non-overlapping k-mers
    kmers = [trimmed_seq[i:i+k] for i in range(0, trimmed_len, k)]
    
    # Shuffle
    np.random.shuffle(kmers)
    
    # Join and return
    return ''.join(kmers)


def shuffle_core(**kwargs):
    if kwargs.get("dinuc"):
        shuffle_fn = dinuc_shuffle
    else:
        def shuffle_fn(x):
            return kmer_shuffle(seq=x, k=kwargs.get('k', 1))

    match kwargs.get("itype"):
        case "CSV":
            f = pl.read_csv(
                kwargs.get("input"), truncate_ragged_lines=True, has_header=False
            )
            f = f.with_columns(pl.lit(1).alias("column_1"))
            fs = f.with_columns(
                pl.col("column_2").map_elements(
                    lambda x: shuffle_fn(x), return_dtype=pl.String
                ),
                pl.lit(0).alias("column_1"),
            )
            f = pl.concat([f, fs]).sample(
                fraction=1.0, shuffle=True, with_replacement=False
            )
            f.write_csv(kwargs.get("output"), include_header=False)
        case "FASTA":
            f = pyfastx.Fasta(kwargs.get("input"), build_index=False)

            with open(kwargs.get("output"), "w") as fh:
                for name, seq in f:
                    fh.write(f">{name}\n")
                    shuffled = shuffle_fn(seq)
                    for i in range(0, len(shuffled), 70):
                        fh.write(shuffled[i : i + 70] + "\n")


def split_core(**kwargs):
    # split records in f into fragments of varying sizes

    input_path = kwargs.get("input")
    output_path = kwargs.get("output")
    min_len = kwargs.get("minlen", 2000)
    max_len = kwargs.get("maxlen", 50000)
    overlap = kwargs.get("overlap", 0)  # how many bases to overlap
    shuffle = kwargs.get("shuffle", False)

    f = pyfastx.Fasta(input_path, build_index=False)

    with open(output_path, "w") as fh:
        for name, seq in track(f, description="Processing..."):
            seq = str(seq)
            if shuffle:
                seq = dinuc_shuffle(seq)

            start = 0
            frag_id = 0

            while start < len(seq):
                frag_len = random.randint(min_len, max_len)
                end = min(start + frag_len, len(seq))
                fragment = seq[start:end]

                # Write FASTA record
                Ns = fragment.count("N")
                if Ns / len(fragment) < 0.3 and len(fragment) >= min_len:
                    fh.write(f">{name}_frag{frag_id}_start{start}_len{len(fragment)}\n")
                    for i in range(0, len(fragment), 60):
                        fh.write(fragment[i : i + 60] + "\n")

                # Move to next fragment start (with overlap)
                if end == len(seq):
                    break

                start = end - overlap
                frag_id += 1


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


def dataset_core(**kwargs):
    """
    Generate a non‐redundant fragment database from a FASTA file for
    model training/validation via MMseqs2 clustering.

    Required kwargs:
      input      : path to input FASTA of contigs
      output     : prefix for output FASTA files
    Optional kwargs (with defaults):
      valperc    : 0.1    # fraction for validation set
      trainperc  : 0.8    # fraction for training set
      testperc   : 0.1    # fraction for test set
      maxiden    : 0.6    # minimum sequence identity for clustering
      maxcov     : 0.6    # minimum coverage fraction for clustering
      method     : "ANI"  # or "AAI" (to do: AAI)
      outtype    : "CSV"  # or "FASTA
    """
    # unpack arguments
    inp = Path(kwargs["input"])
    out_pref = Path(kwargs["output"])  # output directory
    valperc = kwargs.get("valperc", 0.1)
    trainperc = kwargs.get("trainperc", 0.8)
    testperc = kwargs.get("testperc", 0.1)
    maxiden = kwargs.get("maxiden", 0.6)
    maxcov = kwargs.get("maxcov", 0.6)
    FRAG_LEN = kwargs.get("fraglen", 2048)
    OVERLAP = kwargs.get("overlap", 1024)
    class_ = kwargs.get("class")
    method = kwargs.get("method", "ANI").upper()
    outtype = kwargs.get("outtype", "CSV").upper()

    # sanity check
    assert abs(trainperc + valperc + testperc - 1.0) < 1e-6, (
        "train+val+test must sum to 1"
    )
    if shutil.which("mmseqs") is None:
        sys.exit(
            "Error: MMseqs2 (`mmseqs`) not found in PATH. Please install MMseqs2 or add it to your PATH."
        )

    out_pref.mkdir(exist_ok=True, parents=True)
    frag_fasta = out_pref / f"{inp.name}.fragments.fasta"
    clusters = out_pref / f"{inp.name}.fragments.clusters"

    with open(frag_fasta, "w") as outfh:
        for name, seq in pyfastx.Fasta(str(inp), build_index=False):
            seq = str(seq)
            start = 0
            frag_id = 0
            L = len(seq)
            if L >= FRAG_LEN:
                while start < L:
                    end = min(start + FRAG_LEN, L)
                    offset = FRAG_LEN - (end - start)
                    start = start if offset == 0 else start - offset
                    frag = seq[start:end]
                    header = f">{name}_frag{frag_id}_start{start}_len{len(frag)}\n"
                    outfh.write(header)
                    # wrap at 70 chars
                    for i in range(0, len(frag), 70):
                        outfh.write(frag[i : i + 70] + "\n")
                    frag_id += 1
                    if end == L:
                        break
                    start = end - OVERLAP
    num_frags = len(
        [name for i, _ in pyfastx.Fasta(str(frag_fasta), build_index=False)]
    )
    # 2) Run MMseqs2 easy-cluster to dereplicate at (ANI or AAI) thresholds
    #    For simplicity, we only implement the nucleotide (ANI) path here.
    assert method == "ANI", "AAI method is not yet implemented"

    tmpdir = out_pref / "tmp"

    # easy-cluster with identity and coverage
    subprocess.run(
        [
            "mmseqs",
            "easy-cluster",
            frag_fasta,
            clusters,
            tmpdir,
            "--min-seq-id",
            str(maxiden),
            "-c",
            str(maxcov),
        ],
        check=True,
    )

    # the representative sequences end up in: {clu}_cluster_rep_seq.fasta
    rep_seq = out_pref / f"{inp.name}.fragments.clusters_rep_seq.fasta"
    if not rep_seq.exists():
        raise FileNotFoundError(f"Expected MMseqs2 rep file: {rep_seq}")

    # 3) Load representatives and split into train/val/test
    reps = list(pyfastx.Fasta(str(rep_seq), build_index=False))
    random.shuffle(reps)
    N = len(reps)
    n_train = int(trainperc * N)
    n_val = int(valperc * N)
    # rest = test
    train = reps[:n_train]
    val = reps[n_train : n_train + n_val]
    test = reps[n_train + n_val :]

    # 4) Write out three FASTA files
    for subset_name, subset in zip(("train", "val", "test"), (train, val, test)):
        if outtype == "FASTA":
            outf = out_pref.with_name(f"{out_pref.name}_{subset_name}.fasta")
            with open(outf, "w") as fh:
                for header, seq in subset:
                    fh.write(f">{header}\n")
                    for i in range(0, len(seq), 70):
                        fh.write(seq[i : i + 70] + "\n")
        else:
            outf = out_pref.with_name(f"{out_pref.name}_{subset_name}.txt")
            with open(outf, "w") as fh:
                for header, seq in subset:
                    fh.write(f"{class_},{seq},{header},class={class_}\n")

    print(
        f"{num_frags} frags →  {len(reps)} reps → "
        f"{len(train)} train, {len(val)} val, {len(test)} test"
    )
