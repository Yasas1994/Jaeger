from jaeger.preprocess.shuffle_dna import dinuc_shuffle
import polars as pl
import pyfastx
import random
import subprocess
import shutil
import sys
import numpy as np
from pathlib import Path
from typing import List
from rich.progress import track
import random
from math import log2, sqrt
from scipy import stats
import csv
from jaeger.utils.logging import get_logger
logger = get_logger(log_file=None, log_path=None, level=3)

def shannon_entropy(seq: str) -> float:
    """
    Calculate Shannon entropy for a sequence.
    """
    counts = {}
    for base in seq:
        counts[base] = counts.get(base, 0) + 1
    entropy = 0.0
    length = len(seq)
    for count in counts.values():
        p = count / length
        entropy -= p * log2(p)
    return entropy

def generate_homopolymer(length: int, base: str = 'A') -> str:
    """
    Generate a homopolymer (single-base repeat) of given length.
    """
    return base * length

def generate_tandem_repeat(motif: str, copies: int) -> str:
    """
    Generate a tandem repeat sequence by repeating a given motif a specified number of times.
    """
    return motif * copies

def generate_random_tandem_repeats(
    num_sequences: int,
    motif_length_range: tuple = (3, 30),
    copy_number: int = 2000,
    alphabet: List[str] = ['A', 'C', 'G', 'T']
) -> List[str]:
    """
    Automatically generate a list of random tandem repeat sequences.
    
    Parameters
    ----------
    num_sequences : int
        Number of sequences to generate.
    motif_length_range : tuple (min_len, max_len)
        Range of lengths for randomly generated motifs.
    copy_number_range : tuple (min_copies, max_copies)
        Range of copy counts to repeat the motif.
    alphabet : list of str
        List of nucleotide characters to sample motifs from.
    
    Returns
    -------
    List[str]
        Generated tandem repeat sequences.
    """
    sequences = []
    for _ in range(num_sequences):
        motif_len = random.randint(*motif_length_range)
        motif = ''.join(random.choices(alphabet, k=motif_len))
        seq = generate_tandem_repeat(motif, copy_number)
        sequences.append(seq[:2048])
    return sequences

def generate_biased_sequence(length: int, freqs: dict = None) -> str:
    """
    Generate a sequence of given length with biased nucleotide frequencies.
    freqs should be a dict like {'A':0.7, 'C':0.1, 'G':0.1, 'T':0.1}.
    """
    if freqs is None:
        freqs = {'A': 0.7, 'C': 0.1, 'G': 0.1, 'T': 0.1}
    bases = list(freqs.keys())
    weights = list(freqs.values())
    return ''.join(random.choices(bases, weights=weights, k=length))

def generate_low_entropy_sequence(length: int, window_size: int, threshold: float,
                                  max_attempts: int = 10000) -> str:
    """
    Generate a random sequence and ensure all sliding windows have entropy below threshold.
    """
    for attempt in range(max_attempts):
        seq = generate_biased_sequence(length)
        # check all windows
        valid = True
        for i in range(length - window_size + 1):
            if shannon_entropy(seq[i:i+window_size]) >= threshold:
                valid = False
                break
        if valid:
            return seq
    raise ValueError(f"Failed to generate low-entropy sequence after {max_attempts} attempts")

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

def kmer_mix_shuffle(seq1: str, seq2: str, k: int):
    """
    Shuffle kmers from 2 sequences from 2 different classes to generate 
    """
    pass

def shuffle_core(**kwargs):
    """
    shuffle sequences while mainintaining the 
    1. dinuc composition or
    2. break a seqeunce into k-mers -> shuffle -> concat
    3. random shuffling 
    """
    if kwargs.get("dinuc"):
        shuffle_fn = dinuc_shuffle
    else:
        def shuffle_fn(x):
            return kmer_shuffle(seq=x, k=kwargs.get('k', 1))
    n_tandem_repeats = kwargs.get("num_tandem_repeats")
    match kwargs.get("itype"):
        case "CSV":
            f = pl.read_csv(
                kwargs.get("input"), truncate_ragged_lines=True, has_header=False
            )
            f = f.select(['column_1', 'column_2', 'column_3'])
            
            f = f.with_columns(
                pl.when((pl.col('column_2').str.count_matches("N")  / pl.col('column_2').str.len_chars()) > 0.3)
                    .then(pl.lit(0).alias("column_1"))
                    .otherwise(pl.lit(1).alias("column_1"))
            )
            fs = f.with_columns(
                pl.col("column_2").map_elements(
                    lambda x: shuffle_fn(x), return_dtype=pl.String
                ),
                pl.lit(0).alias("column_1"),
            )
            ft = pl.DataFrame()
            if n_tandem_repeats > 0:
                ft = pl.from_dict(dict(column_1=np.array([0 for _ in range(n_tandem_repeats)], dtype=np.int32),
                    column_2=generate_random_tandem_repeats(num_sequences=n_tandem_repeats),
                    column_3=[f'tandem_repeat_{i}' for i in range(n_tandem_repeats)]))
            
            logger.info(f"id : {len(f)} ood: {len(fs)} ood_tandem: {0 if ft.is_empty() else len(ft)}")

            f = pl.concat([i for i in [f, fs, ft] if not  i.is_empty()], how='vertical').sample(
                fraction=1.0, shuffle=True, with_replacement=False
            )

            

            match kwargs.get("otype"):
                case "CSV":
                    f.select(['column_1', 'column_2', 'column_3']).write_csv(kwargs.get("output"), include_header=False)
                case "FASTA":
                    with open(kwargs.get("output"), "w") as fh:
                        for row in f.iter_rows(named=True):
                            label = row["col1"]
                            seq = row["col2"]
                            seq_id = row["col3"].split("__class=")[0]
                            # use the 3rd column as header
                            fh.write(f">{seq_id}__class={label}\n")
                            for i in range(0, len(seq), 70):
                                fh.write(seq[i:i+70] + "\n")

        case "FASTA":
            input_path  = kwargs.get("input")
            output_path = kwargs.get("output")
            otype       = kwargs.get("otype")  # "FASTA" or "CSV"
            fasta_iter  = pyfastx.Fasta(input_path, build_index=False)

            # 1) pre‑generate your tandem dict
            tandem = dict(
                column_1=[0 for _ in range(n_tandem_repeats)],
                column_2=generate_random_tandem_repeats(num_sequences=n_tandem_repeats),
                column_3=[f"tandem_repeat_{i}" for i in range(n_tandem_repeats)]
            )

            # 2) collect ALL entries into a list
            entries = []

            # 2a) from your input FASTA
            for name, seq in fasta_iter:
                ncount     = seq.count("N")
                lowcomplex = (len(seq) > 0 and (ncount / len(seq)) > 0.3)
                shuffled   = shuffle_fn(seq)
                id_        = name.split("__class=")[0]
                orig_label = 0 if lowcomplex else 1

                # store tuples of (id, sequence, label)
                entries.append((id_, seq,       orig_label))
                entries.append((id_, shuffled,  0))

            # 2b) from your tandem dict
            for label, tandem_seq, tandem_id in zip(
                tandem["column_1"],
                tandem["column_2"],
                tandem["column_3"]
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


def write_fasta_entry(fh, header, seq, label):
    """
    formats and writes a fasta record to a file
    """
    fh.write(f">{header}class={label}\n")
    for i in range(0, len(seq), 70):
        fh.write(seq[i:i+70] + "\n")

# def split_core(**kwargs):
#     """
#     sequencially sample random fragments from genomes (to mimic metagenome assemblies) for a given size 
#     distribution
#     """


#     input_path = kwargs.get("input")
#     output_path = kwargs.get("output")
#     min_len = kwargs.get("minlen", 2000)
#     max_len = kwargs.get("maxlen", 50000)
#     overlap = kwargs.get("overlap", 0)  # how many bases to overlap
#     shuffle = kwargs.get("shuffle", False)

#     f = pyfastx.Fasta(input_path, build_index=False)

#     with open(output_path, "w") as fh:
#         for name, seq in track(f, description="Processing..."):
#             seq = str(seq)
#             if shuffle:
#                 seq = dinuc_shuffle(seq)

#             start = 0
#             frag_id = 0

#             while start < len(seq):
#                 frag_len = random.randint(min_len, max_len)
#                 end = min(start + frag_len, len(seq))
#                 fragment = seq[start:end]

#                 # Write FASTA record
#                 Ns = fragment.count("N")
#                 if Ns / len(fragment) < 0.3 and len(fragment) >= min_len:
#                     fh.write(f">{name}_frag{frag_id}_start{start}_len{len(fragment)}\n")
#                     for i in range(0, len(fragment), 60):
#                         fh.write(fragment[i : i + 60] + "\n")

#                 # Move to next fragment start (with overlap)
#                 if end == len(seq):
#                     break

#                 start = end - overlap
#                 frag_id += 1


def split_core(**kwargs):
    """
    Sample random fragments from genomes (to mimic metagenome assemblies)
    for a given size distribution.

    Two modes:
    1) Sequentially sample random fragments of varying size (given a size distribution):
       - Used when `coverage` is NOT provided.
       - Walks along each genome with random fragment lengths and fixed overlap.
       - when minlen == maxlen -> sliding window with constant window size

    2) Coverage-based random sampling:
       - Used when `coverage` is provided.
       - For each genome, sample random fragments until
         target_bases = coverage * genome_length is reached (approx).

    Parameters
    ----------
    minlen : int
    minimum fragment length (min of discrete uniform dinstribution)

    maxlen: int
    maximum fragment length (max of discrete uniform dinstribution)

    overlap: int
    number of overlapping nuc between two consecutive fragments

    coverage: int
    switch from sequential sampling to coverage based sampling if not None

    circular: bool
    treats the sequences as circular (conts the ends)

    nax_n_prop: float
    max proportion of Ns allowed in a fragment

    seed: int
    seed for the random number generator

    """

    input_path = kwargs.get("input")
    output_path = kwargs.get("output")

    min_len = kwargs.get("minlen", 2000)
    max_len = kwargs.get("maxlen", 50000)
    overlap = kwargs.get("overlap", 0)       # used only in sequential mode
    shuffle = kwargs.get("shuffle", False)
    coverage = kwargs.get("coverage", None)  # per-genome coverage; if None → sequential
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
        """
        Sample a fragment of length `frag_len` from `seq`.
        If circular=True, allow wrapping around the end.
        Returns (start, fragment_seq).
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
            fragment = seq[start:start + frag_len]
        return start, fragment

    with open(output_path, "w") as fh:
        for name, seq in track(f, description="Processing..."):
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



def read_sequences(input_path: Path, intype: str, seq_col=None, class_col=None, class_id=None):
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
                frag_name = f"{name}_frag{frag_id}_start{start}_len{len(frag)}_cls={cls}"
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
                fh.write(seq[i:i+70] + "\n")


def run_mmseqs_cluster(frag_fasta, out_prefix, tmpdir, min_id, min_cov):
    """Run MMseqs2 easy-cluster."""
    if shutil.which("mmseqs") is None:
        sys.exit("Error: MMseqs2 not found in PATH.")
    subprocess.run(
        [
            "mmseqs", "easy-cluster",
            frag_fasta,
            out_prefix,
            tmpdir,
            "--min-seq-id", str(min_id),
            "-c", str(min_cov)
        ],
        check=True
    )


def split_dataset(records, trainperc, valperc, testperc):
    """Split records into train, val, and test sets."""
    
    random.shuffle(records)
    N = len(records)
    n_train = int(trainperc * N)
    n_val = int(valperc * N)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


def write_output(train, val, test, out_prefix, outtype="CSV"):
    """Write output subsets in FASTA or CSV format."""
    subsets = {"train": train, "val": val, "test": test}
    for name, subset in subsets.items():
        if len(subset) > 0:
            if outtype == "FASTA":
                out_file = out_prefix / out_prefix.with_name(f"{out_prefix.name}_{name}.fasta")
                write_fasta(subset, out_file)
            elif outtype == "CSV":
                out_file = out_prefix / out_prefix.with_name(f"{out_prefix.name}_{name}.csv")
                with open(out_file, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    #writer.writerow(["class", "sequence", "id"])
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

    assert abs(trainperc + valperc + testperc - 1.0) < 1e-6, "train+val+test must sum to 1" 

    def get_class(x):
        return x.split('=')[-1]
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
        reps = [(h, str(s), class_id) for h, s in pyfastx.Fasta(str(rep_seq), build_index=False)]
    else:
        reps = [(h, str(s), get_class(h)) for h, s in pyfastx.Fasta(str(rep_seq), build_index=False)]

    # 5. Split datasets
    train, val, test = split_dataset(reps, trainperc, valperc, testperc)

    # 6. Write outputs
    write_output(train, val, test, out_pref, outtype)

    print(f"{len(fragments)} fragments → {len(reps)} reps → "
          f"{len(train)} train, {len(val)} val, {len(test)} test")
    


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
    input_path = Path(kwargs.get('input'))
    output_path = Path(kwargs.get('output'))
    input_type = kwargs.get('itype')
    if input_type == "CSV":
        # CSV -> FASTA
        df = pd.read_csv(input_path, usecols=[0,1,2], names=['class', 'sequence', 'id'], dtype=str)
        with open(output_path, 'w') as fasta_out:
            for idx, row in df.iterrows():
                seq_id = row['id'].strip()
                cls_id = row['class'].strip()
                seq = row['sequence'].strip()
                fasta_out.write(f">{seq_id}__class={cls_id}\n{seq}\n")
        print(f"[✓] Converted CSV to FASTA: {output_path}")

    elif input_type == "FASTA":
        # FASTA -> CSV
        fasta = pyfastx.Fasta(str(input_path), build_index=False)
        records = []
        for name, seq in fasta:
            seq_id, cls_id = name.split('__class=')
            records.append((cls_id, seq, seq_id))
        df = pd.DataFrame(records, columns=['class', 'sequence', 'id'])
        df.to_csv(output_path, index=False, header=False)
        print(f"[✓] Converted FASTA to CSV: {output_path}")

    else:
        raise ValueError("input_type must be 'CSV' or 'FASTA'")

def significant_top_class(logits_class1, logits_class2, alpha=0.05):
    """
    One-tailed paired t-test to check if top 1 class logits are significantly higher than top 2 class 
    logits.
    """
    # Differences per window
    diffs = np.array(logits_class1) - np.array(logits_class2)

    # Compute t-statistic and p-value
    t_stat, p_two_tailed = stats.ttest_1samp(diffs, 0)
    # Convert to one-tailed (greater-than)
    p_one_tailed = p_two_tailed / 2 if t_stat > 0 else 1 - (p_two_tailed / 2)

    # Decision
    significant = (p_one_tailed < alpha)

    return {
        "t_stat": t_stat,
        "p_value": p_one_tailed,
        "significant": significant
    }

def welch_t_one_tailed(mean1, var1, n1, mean2, var2, n2, alternative="greater"):
    """
    One-tailed Welch's t-test using summary statistics.
    alternative: "greater" tests mean1 > mean2,
                 "less" tests mean1 < mean2.
    """
    # Standard error
    se = sqrt(var1/n1 + var2/n2)

    # t-statistic
    t_stat = (mean1 - mean2) / se

    # Welch–Satterthwaite degrees of freedom
    df_num = (var1/n1 + var2/n2)**2
    df_denom = ((var1/n1)**2 / (n1 - 1)) + ((var2/n2)**2 / (n2 - 1))
    df = df_num / df_denom

    # One-tailed p-value
    if alternative == "greater":
        p = 1 - stats.t.cdf(t_stat, df)
    elif alternative == "less":
        p = stats.t.cdf(t_stat, df)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    return t_stat, df, p

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
    input_path = Path(kwargs.get('input'))
    output_path = Path(kwargs.get('output'))
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
    if len(df) > 1:
        # Create the count plot
        df["above_threshold"] = df["reliability_score"].apply(lambda x : "passed" if x >= 0.8 else "failed")
        ax = sns.countplot(data=df, x="prediction", hue="above_threshold", palette="pastel", stat="percent")
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
        ax = sns.violinplot(df, x='prediction', y='reliability_score')
        sns.stripplot(df, x='prediction', y='reliability_score', s=1, alpha=0.1, color='gray', ax=ax)
        ax.set_ylabel("Reliability score")
        ax.set_xlabel("Class")
        ax.set_title("Per-class distribution of reliability scores")
        sns.despine()
        plt.tight_layout()
        plt.savefig(relscore, dpi=150, bbox_inches="tight")
        plt.close()

        # Calculate per-class distribution of entropy
        ax = sns.violinplot(df, x='prediction', y='entropy')
        sns.stripplot(df, x='prediction', y='entropy', s=1, alpha=0.1, color='gray', ax=ax)
        ax.set_ylabel("Entropy")
        ax.set_xlabel("Class")
        ax.set_title("Per-class distribution of entropy")
        sns.despine()
        plt.tight_layout()
        plt.savefig(ent, dpi=150, bbox_inches="tight")
        plt.close()

        # Calculate per-class distribution of energy
        if "energy" in df.columns:
            ax = sns.violinplot(df, x='prediction', y='energy')
            sns.stripplot(df, x='prediction', y='energy', s=1, alpha=0.1, color='gray', ax=ax)
            ax.set_ylabel("Energy")
            ax.set_xlabel("Class")
            ax.set_title("Per-class distribution of Energy")
            sns.despine()
            plt.tight_layout()
            plt.savefig(eng, dpi=150, bbox_inches="tight")
            plt.close()

        # Calculate perclass score distributions
        # Create the grid
        df_long = pd.melt(df[['contig_id', 'length', 'prediction'] + [i for i in df.columns if i.endswith("_score") and i != "reliability_score"]], 
                        id_vars=['contig_id', 'length', 'prediction'],
                        var_name="score_class",
                        value_name="scores")
        g = sns.FacetGrid(df_long, row="prediction",hue="score_class", margin_titles=False, height=2, aspect=3.5)
        g.map(sns.kdeplot, "scores",fill=True, common_norm=False, alpha=0.2, linewidth=0.5)
        g.add_legend()
        # Add titles and adjust layout
        g.set_axis_labels("Score", "Density")
        #g.set_titles("Per-class score distributions")
        g.savefig(clscores, dpi=150, bbox_inches="tight")
        plt.close()
        try:
            # quantile bins 
            bins = pd.qcut(df["length"], q=5)

            # Extract bin edges
            bin_edges = bins.cat.categories

            # Create labels with numeric min–max
            labels = [f"{int(interval.left):,}–{int(interval.right):,}" for interval in bin_edges]

            # Recreate qcut with readable labels
            df["length_bin"] = pd.qcut(df["length"], q=5, labels=labels)
            # Calculate per-class distribution of reliability scores
            ax = sns.violinplot(df, x='length_bin', y='reliability_score')
            sns.stripplot(df, x='length_bin', y='reliability_score', s=1, alpha=0.1, color='red', ax=ax)
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
    mean_scores = df[[i for i in df.columns if i.endswith("_score") and "reliability" not in i]].to_numpy()
    var_scores = df[[i for i in df.columns if i.endswith("_var")]].to_numpy()
    windows = df[[i for i in df.columns if i.endswith("_windows") and "reliability" not in i]].to_numpy().sum(axis=-1)
    rows = np.arange(mean_scores.shape[0])[:, None]
    sorted_indices = np.flip(np.argsort(mean_scores, axis=-1), axis=-1)
    sorted_means = mean_scores[rows, sorted_indices[:, :2]]
    sorted_vars = var_scores[rows, sorted_indices[:, :2]]
    pvals = []
    for means,vars,n in zip(sorted_means, sorted_vars, windows):
        _,_, p = welch_t_one_tailed(mean1=means[0], var1=vars[0], mean2=means[1], var2=vars[1], n1=n, n2=n)
        pvals.append(p)
    df["pval"] = pvals

    df.to_csv(tsv_with_pvals, index=None, sep="\t", float_format="%.3f" )
        # Create the count plot

    if len(df) > 1:
        df["above_pval_threshold"] = df["pval"].apply(lambda x : "passed" if x <= 0.05 else "failed")
        ax = sns.countplot(data=df, x="prediction", hue="above_pval_threshold", palette="pastel", stat="percent")
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