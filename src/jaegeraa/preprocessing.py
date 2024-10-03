import logging
import pyfastx
import tensorflow as tf
import progressbar
from jaegeraa.utils import signal_l, safe_divide

logger = logging.getLogger("Jaeger")
progressbar.streams.wrap_stderr()

def codon_mapper():
    """
    Creates a static hash table for mapping codons to their corresponding
    values.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for codon mapping.
    """

    trimers = tf.constant(
        [
            "TTT",
            "TTC",
            "TTA",
            "TTG",
            "CTT",
            "CTC",
            "CTA",
            "CTG",
            "ATT",
            "ATC",
            "ATA",
            "ATG",
            "GTT",
            "GTC",
            "GTA",
            "GTG",
            "TCT",
            "TCC",
            "TCA",
            "TCG",
            "CCT",
            "CCC",
            "CCA",
            "CCG",
            "ACT",
            "ACC",
            "ACA",
            "ACG",
            "GCT",
            "GCC",
            "GCA",
            "GCG",
            "TAT",
            "TAC",
            "TAA",
            "TAG",
            "CAT",
            "CAC",
            "CAA",
            "CAG",
            "AAT",
            "AAC",
            "AAA",
            "AAG",
            "GAT",
            "GAC",
            "GAA",
            "GAG",
            "TGT",
            "TGC",
            "TGA",
            "TGG",
            "CGT",
            "CGC",
            "CGA",
            "CGG",
            "AGT",
            "AGC",
            "AGA",
            "AGG",
            "GGT",
            "GGC",
            "GGA",
            "GGG",
        ]
    )
    trimer_vals = tf.constant(
        [
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
            10,
            10,
            11,
            11,
            12,
            12,
            13,
            13,
            14,
            14,
            15,
            15,
            16,
            16,
            17,
            17,
            18,
            18,
            11,
            19,
            20,
            20,
            20,
            20,
            6,
            6,
            20,
            20,
            21,
            21,
            21,
            21,
        ]
    )
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    return tf.lookup.StaticHashTable(trimer_init, default_value=0)


def amino_mapper():
    """
    Creates a static hash table for mapping amino acids to their corresponding
    numerical values.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for amino acid mapping.
    """

    aa = tf.constant(
        [
            "F",
            "L",
            "I",
            "M",
            "V",
            "S",
            "P",
            "T",
            "A",
            "Y",
            "*",
            "H",
            "Q",
            "N",
            "K",
            "D",
            "E",
            "C",
            "W",
            "R",
            "G",
        ]
    )
    aa_num = tf.constant(
        [1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         10,
         21,
         11,
         12,
         13,
         14,
         15,
         16,
         17,
         18,
         19,
         20]
    )
    aa_init = tf.lookup.KeyValueTensorInitializer(aa, aa_num)
    return tf.lookup.StaticHashTable(aa_init, default_value=0)


def c_mapper():
    """
    Creates a static hash table for mapping DNA nucleotides to their reverse
    complements.

    Returns:
        tf.lookup.StaticHashTable: A static hash table for reverse complement
        mapping.
    """

    rc_keys = tf.constant([b"A", b"T", b"G", b"C", b"a", b"t", b"g", b"c"])
    rc_vals = tf.constant([b"T", b"A", b"C", b"G", b"t", b"a", b"c", b"g"])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    return tf.lookup.StaticHashTable(rc_init, default_value="N")


def fasta_gen(file_path,
              fragsize=None,
              stride=None,
              num=None,
              ):
    """
    Generates fragments of DNA sequences from a FASTA file.

    Args:
    ----
        filehandle: File handle for reading DNA sequences.
        fragsize (int, optional): Size of the DNA sequence fragments. Defaults
        to None.
        stride (int, optional): Stride for fragment generation. Defaults to
        None.
        num (int, optional): Total number of sequences to process. Defaults
        to None.
        disable (bool, optional): Flag to disable progress bar. Defaults to
        False.

    Returns:
    -------
        generator: A generator that yields DNA sequence fragments with
        associated information.
    """

    def c():
        # accepts a reference to a file handle
        fa = pyfastx.Fasta(file_path, build_index=False)
        with progressbar.ProgressBar(max_value=num) as pbar:
            for j, record in enumerate(fa):
                pbar.update(j)
                seqlen = len(
                    record[1]
                )  # move size filtering to a separate preprocessing step
                sequence = record[1].strip()
                header = record[0].strip().replace(",", "__")
                # logger.debug(sequence)
                # sequence = str(record[1]).upper()
                # filters the sequence based on size
                if seqlen >= fragsize:
                    # if no fragsize, return the entire sequence
                    if fragsize is None:
                        # sequence and sequence headder
                        yield f"{sequence},{header}"
                    elif fragsize is not None:

                        for i, (l, index) in enumerate(
                            signal_l(
                                range(
                                    0,
                                    seqlen - (fragsize - 1),
                                    fragsize if stride is None else stride,
                                )
                            )
                        ):
                            g = sequence[index: index + fragsize].count("G")
                            c = sequence[index: index + fragsize].count("C")
                            a = sequence[index: index + fragsize].count("A")
                            t = sequence[index: index + fragsize].count("T")
                            gc_skew = safe_divide((g - c), (g + c))
                            # sequnce_fragment, contig_id, index, contig_end, i,
                            # g, c, gc_skew
                            yield f"{sequence[index : index + fragsize]},{header},{index},{l},{i},{seqlen},{g},{c},{a},{t},{gc_skew : .3f}"

    return c


def fasta_gen_lib(filename, fragsize=None, stride=None, num=None):
    """
    Generates fragments of DNA sequences from various input types.

    Args:
    ----
        filehandle: Input source for DNA sequences, can be a file handle,
                    string, list, Seq object, generator, or file object.
        fragsize (int, optional): Size of the DNA sequence fragments.
                                  Defaults to None.
        stride (int, optional): Stride for fragment generation.
                                Defaults to None.
        num (int, optional): Total number of sequences to process.
                             Defaults to None.

    Returns:
    -------
        generator: A generator that yields DNA sequence fragments with
                   associated information.

    Raises:
    ------
        ValueError: If the input type is not supported.
    """

    head = False
    if isinstance(filename, str):
        tmpfn = pyfastx.Fasta(filename, build_index=False)
        head = True
    else:
        raise ValueError("Not a supported input type")

    def c():
        # accepts a reference to a file handle
        for n, record in enumerate(tmpfn):
            if head:
                seqlen = len(
                    record[1]
                )  # move size filtering to a separate preprocessing step
                seq = record[1]
                headder = record[0].replace(",", "__")

            else:
                seqlen = len(record)
                seq = record
                headder = f"seq_{n}"
            # filters the sequence based on size
            if seqlen >= fragsize:
                # if no fragsize, return the entire sequence
                if fragsize is None:
                    yield f"{str(seq)},{str(headder)}"
                elif fragsize is not None:
                    for i, (l, index) in enumerate(
                        signal_l(
                            range(
                                0,
                                seqlen - (fragsize - 1),
                                fragsize if stride is None else stride,
                            )
                        )
                    ):
                        yield f"{str(seq)[index:index + fragsize]},{str(headder)},{str(index)},{str(l)},{str(i)},{str(seqlen)}"

    return c


def process_string(onehot=True, label_onehot=True, crop_size=2048):
    """
    Processes a DNA sequence string by mapping codons and nucleotides.

    Args:
    ----
        onehot (bool, optional): Flag to indicate one-hot encoding. Defaults
                                 to True.
        label_onehot (bool, optional): Flag to indicate one-hot encoding for
                                       labels. Defaults to True.
        crop_size (int, optional): Size for cropping the sequence. Defaults
                                   to 2048.

    Returns:
    -------
        function: A function that processes a DNA sequence string and returns
        mapped codons and nucleotides.
    """

    @tf.function
    def p(string):
        t1, t3 = codon_mapper(), c_mapper()
        x = tf.strings.split(string, sep=",")

        if (crop_size % 3) == 0:
            offset = -2
        elif (crop_size % 3) == 1:
            offset = -1
        elif (crop_size % 3) == 2:
            offset = 0

        forward_strand = tf.strings.bytes_split(x[0])  # split the string
        reverse_strand = t3.lookup(forward_strand[::-1])

        tri_forward = tf.strings.ngrams(forward_strand,
                                        ngram_width=3,
                                        separator="")
        tri_reverse = tf.strings.ngrams(reverse_strand,
                                        ngram_width=3,
                                        separator="")

        f1 = t1.lookup(tri_forward[: -3 + offset: 3])
        f2 = t1.lookup(tri_forward[1: -2 + offset: 3])
        f3 = t1.lookup(tri_forward[2: -1 + offset: 3])

        r1 = t1.lookup(tri_reverse[: -3 + offset: 3])
        r2 = t1.lookup(tri_reverse[1: -2 + offset: 3])
        r3 = t1.lookup(tri_reverse[2: -1 + offset: 3])

        return (
            {
                "forward_1": f1,
                "forward_2": f2,
                "forward_3": f3,
                "reverse_1": r1,
                "reverse_2": r2,
                "reverse_3": r3,
            },
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
            x[9],
            x[10],
        )

    return p


def process_string_textline_protein(label_onehot=True, numclasses=4):
    """
    Processes a protein sequence text line by mapping amino acids and labels.

    Args:
    ----
        label_onehot (bool, optional): Flag to indicate one-hot encoding for
                                       labels. Defaults to True.
        numclasses (int, optional): Number of classes for label encoding.
                                    Defaults to 4.

    Returns:
    -------
        function: A function that processes a protein sequence text line and
                  returns mapped amino acids and labels.
    """

    def p(string):
        t1 = amino_mapper()

        x = tf.strings.split(string, sep=",")

        label = tf.strings.to_number(x[0], tf.int32)
        label = tf.cast(label, dtype=tf.int32)
        # split the string
        prot_strand = tf.strings.bytes_split(x[1])

        protein = t1.lookup(prot_strand)

        if label_onehot:
            label = tf.one_hot(
                label,
                depth=numclasses,
                dtype=tf.float32,
                on_value=1,
                off_value=0
            )

        return protein, label

    return p


def process_string_textline(onehot=True, label_onehot=True, numclasses=4):
    """
    Processes a DNA sequence text line by mapping codons and nucleotides.

    Args:
    ----
        onehot (bool, optional): Flag to indicate one-hot encoding.
                                 Defaults to True.
        label_onehot (bool, optional): Flag to indicate one-hot encoding
                                       for labels. Defaults to True.
        numclasses (int, optional): Number of classes for label encoding.
                                    Defaults to 4.

    Returns:
    -------
        function: A function that processes a DNA sequence text line and
                  returns mapped codons, nucleotides, and labels.
    """

    def p(string):
        t1, t3 = codon_mapper(), c_mapper()

        x = tf.strings.split(string, sep=",")

        label = tf.strings.to_number(x[0], tf.int32)
        label = tf.cast(label, dtype=tf.int32)

        forward_strand = tf.strings.bytes_split(x[1])
        reverse_strand = t3.lookup(forward_strand[::-1])

        tri_forward = tf.strings.ngrams(forward_strand,
                                        ngram_width=3,
                                        separator="")
        tri_reverse = tf.strings.ngrams(reverse_strand,
                                        ngram_width=3,
                                        separator="")

        f1 = t1.lookup(tri_forward[::3])
        f2 = t1.lookup(tri_forward[1::3])
        f3 = t1.lookup(tri_forward[2::3])

        r1 = t1.lookup(tri_reverse[::3])
        r2 = t1.lookup(tri_reverse[1::3])
        r3 = t1.lookup(tri_reverse[2::3])

        if label_onehot:
            label = tf.one_hot(
                label,
                depth=numclasses,
                dtype=tf.float32,
                on_value=1,
                off_value=0
            )

        return {
            "forward_1": f1,
            "forward_2": f2,
            "forward_3": f3,
            "reverse_1": r1,
            "reverse_2": r2,
            "reverse_3": r3,
        }, label

    return p


# Second generation preprocessing code
codons = [
    "TTT",
    "TTC",
    "TTA",
    "TTG",
    "CTT",
    "CTC",
    "CTA",
    "CTG",
    "ATT",
    "ATC",
    "ATA",
    "ATG",
    "GTT",
    "GTC",
    "GTA",
    "GTG",
    "TCT",
    "TCC",
    "TCA",
    "TCG",
    "CCT",
    "CCC",
    "CCA",
    "CCG",
    "ACT",
    "ACC",
    "ACA",
    "ACG",
    "GCT",
    "GCC",
    "GCA",
    "GCG",
    "TAT",
    "TAC",
    "TAA",
    "TAG",
    "CAT",
    "CAC",
    "CAA",
    "CAG",
    "AAT",
    "AAC",
    "AAA",
    "AAG",
    "GAT",
    "GAC",
    "GAA",
    "GAG",
    "TGT",
    "TGC",
    "TGA",
    "TGG",
    "CGT",
    "CGC",
    "CGA",
    "CGG",
    "AGT",
    "AGC",
    "AGA",
    "AGG",
    "GGT",
    "GGC",
    "GGA",
    "GGG",
]

aa = [
    "F",
    "F",
    "L",
    "L",
    "L",
    "L",
    "L",
    "L",
    "I",
    "I",
    "I",
    "M",
    "V",
    "V",
    "V",
    "V",
    "S",
    "S",
    "S",
    "S",
    "P",
    "P",
    "P",
    "P",
    "T",
    "T",
    "T",
    "T",
    "A",
    "A",
    "A",
    "A",
    "Y",
    "Y",
    "*",
    "*",
    "H",
    "H",
    "Q",
    "Q",
    "N",
    "N",
    "K",
    "K",
    "D",
    "D",
    "E",
    "E",
    "C",
    "C",
    "*",
    "W",
    "R",
    "R",
    "R",
    "R",
    "S",
    "S",
    "R",
    "R",
    "G",
    "G",
    "G",
    "G",
]


pc2 = {
    "I": "A",
    "V": "A",
    "L": "A",
    "F": "A",
    "Y": "A",
    "W": "A",
    "H": "B",
    "K": "B",
    "R": "B",
    "D": "B",
    "E": "B",
    "G": "A",
    "A": "A",
    "C": "A",
    "S": "A",
    "T": "A",
    "M": "A",
    "Q": "B",
    "N": "B",
    "P": "A",
}

pc2_num = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    0,
    1,
    2,
    2,
    2,
    2,
    1,
    1,
    2,
    2,
    1,
    1,
    1,
    1,
]

murphy10 = {
    "A": "A",
    "C": "C",
    "G": "G",
    "H": "H",
    "P": "P",
    "L": "L",
    "V": "L",
    "I": "L",
    "M": "L",
    "S": "S",
    "T": "S",
    "F": "F",
    "Y": "F",
    "W": "F",
    "E": "E",
    "D": "E",
    "N": "E",
    "Q": "E",
    "K": "K",
    "R": "K",
}

pc5 = {
    "I": "A",
    "V": "A",
    "L": "A",
    "F": "R",
    "Y": "R",
    "W": "R",
    "H": "R",
    "K": "C",
    "R": "C",
    "D": "C",
    "E": "C",
    "G": "T",
    "A": "T",
    "C": "T",
    "S": "T",
    "T": "D",
    "M": "D",
    "Q": "D",
    "N": "D",
    "P": "D",
}

aa_num = [
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    4,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    10,
    10,
    0,
    0,
    11,
    11,
    12,
    12,
    13,
    13,
    14,
    14,
    15,
    15,
    16,
    16,
    17,
    17,
    0,
    18,
    19,
    19,
    19,
    19,
    6,
    6,
    19,
    19,
    20,
    20,
    20,
    20,
]
cod_num = [
    1,
    2,
    1,
    2,
    3,
    4,
    5,
    6,
    1,
    2,
    3,
    1,
    1,
    2,
    3,
    4,
    1,
    2,
    3,
    4,
    1,
    2,
    3,
    4,
    1,
    2,
    3,
    4,
    1,
    2,
    3,
    4,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    3,
    4,
    5,
    6,
    5,
    6,
    1,
    2,
    3,
    4,
]

murphy10_num = [
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    5,
    5,
    5,
    5,
    1,
    1,
    0,
    0,
    6,
    6,
    7,
    7,
    7,
    7,
    8,
    8,
    7,
    7,
    7,
    7,
    9,
    9,
    0,
    1,
    8,
    8,
    8,
    8,
    3,
    3,
    8,
    8,
    10,
    10,
    10,
    10,
]

pc5_num = [
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    1,
    1,
    0,
    0,
    1,
    1,
    3,
    3,
    3,
    3,
    5,
    5,
    5,
    5,
    5,
    5,
    4,
    4,
    0,
    1,
    5,
    5,
    5,
    5,
    4,
    4,
    5,
    5,
    4,
    4,
    4,
    4,
]

codon_num = list(range(65))


# map codons to amino acids
def codon_mapper_gen2():
    """
    Creates a static hash table for mapping codons to their corresponding
    values.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for codon mapping.
    """

    trimers = tf.constant(codons)
    trimer_vals = tf.constant(murphy10_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    return tf.lookup.StaticHashTable(trimer_init, default_value=0)


def codon_bias_mapper():
    """
    Creates a static hash table for mapping codons to their corresponding
    bias values.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for codon bias mapping.
    """

    trimers = tf.constant(codons)
    trimer_vals = tf.constant(cod_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    return tf.lookup.StaticHashTable(trimer_init, default_value=0)


# convert to complement
def complement_mapper():
    """
    Creates a static hash table for mapping nucleotides to their complements.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for nucleotide
                                   complement mapping.
    """

    rc_keys = tf.constant([b"A", b"T", b"G", b"C", b"a", b"t", b"g", b"c"])
    rc_vals = tf.constant([b"T", b"A", b"C", b"G", b"t", b"a", b"c", b"g"])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    return tf.lookup.StaticHashTable(rc_init, default_value="N")


def nuc_enc_mapper():
    """
    Creates a static hash table for mapping nucleotides to their numerical
    encodings.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for nucleotide encoding
                                   mapping.
    """

    keys_tensor = tf.constant([b"A", b"G", b"C", b"T", b"a", b"g", b"c", b"t"])
    vals_tensor = tf.constant([0, 1, 2, 3, 0, 1, 2, 3])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    return tf.lookup.StaticHashTable(ini, default_value=-1)


def alt_nuc_enc_mapper():
    """
    Creates a static hash table for mapping nucleotides to their alternative
    numerical encodings.

    Returns:
        tf.lookup.StaticHashTable: A static hash table for alternative
                                   nucleotide encoding mapping.
    """

    keys_tensor = tf.constant([b"A", b"G", b"C", b"T", b"a", b"g", b"c", b"t"])
    vals_tensor = tf.constant([0, 1, 1, 0, 0, 1, 1, 0])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    return tf.lookup.StaticHashTable(ini, default_value=-1)


def process_string_gen2(
    onehot=True,
    crop_size=1024,
    maxval=400,
    timesteps=False,
    num_time=None,
    fragsize=200,
    mutate=False,
    mutation_rate=0.1,
):
    """
    Processes a DNA sequence string by mapping codons, nucleotides, and codon
    biases.

    Args:
    ----
        onehot (bool, optional): Flag to indicate one-hot encoding. Defaults
                                 to True.
        crop_size (int, optional): Size for cropping the sequence. Defaults
                                   to 1024.
        maxval (int, optional): Maximum value for mutation. Defaults to 400.
        timesteps (bool, optional): Flag to reshape output for time steps.
                                    Defaults to False.
        num_time (int, optional): Number of time steps. Defaults to None.
        fragsize (int, optional): Size of the DNA sequence fragments. Defaults
                                  to 200.
        mutate (bool, optional): Flag to enable mutation. Defaults to False.
        mutation_rate (float, optional): Probability of mutation. Defaults
                                         to 0.1.

    Returns:
    -------
        function: A function that processes a DNA sequence string and returns
        mapped codons, nucleotides, and codon biases.
    """

    @tf.function
    def p(string):

        t1 = codon_mapper_gen2()
        t3 = complement_mapper()
        # t4 = nuc_enc_mapper()

        x = tf.strings.split(string, sep=",")

        if (crop_size % 3) == 0:
            offset = -2
        elif (crop_size % 3) == 1:
            offset = -1
        elif (crop_size % 3) == 2:
            offset = 0

        forward_strand = tf.strings.bytes_split(x[0])[:crop_size]

        if mutate:
            # Probability of mutation (adjust as needed)
            mutation_prob = mutation_rate
            # Minimum possible value for mutation
            min_value = 0
            # Maximum possible value for mutation
            max_value = 4

            alphabet = tf.constant(["A", "T", "G", "C", "N"], dtype=tf.string)

            mask = (
                tf.random.uniform(
                    shape=tf.shape(forward_strand), minval=0.0, maxval=1.0
                )
                < mutation_prob
            )
            mutation_values = tf.random.uniform(
                shape=tf.shape(forward_strand),
                minval=min_value,
                maxval=max_value,
                dtype=tf.int32,
            )
            selected_strings = tf.gather(alphabet, mutation_values)
            forward_strand = tf.where(mask, selected_strings, forward_strand)

        # generate the reverse strand for the mutated forward strand
        reverse_strand = t3.lookup(forward_strand[::-1])

        # nuc1 = t4.lookup(forward_strand[:])
        # nuc2 = t4.lookup(reverse_strand[:])

        tri_forward = tf.strings.ngrams(forward_strand,
                                        ngram_width=3,
                                        separator="")
        tri_reverse = tf.strings.ngrams(reverse_strand,
                                        ngram_width=3,
                                        separator="")

        f1 = t1.lookup(tri_forward[: -3 + offset: 3])
        f2 = t1.lookup(tri_forward[1: -2 + offset: 3])
        f3 = t1.lookup(tri_forward[2: -1 + offset: 3])

        # fb1=t5.lookup(tri_forward[0:-3+offset:3])
        # fb2=t5.lookup(tri_forward[1:-2+offset:3])
        # fb3=t5.lookup(tri_forward[2:-1+offset:3])

        r1 = t1.lookup(tri_reverse[: -3 + offset: 3])
        r2 = t1.lookup(tri_reverse[1: -2 + offset: 3])
        r3 = t1.lookup(tri_reverse[2: -1 + offset: 3])

        # rb1=t5.lookup(tri_reverse[0:-3+offset:3])
        # rb2=t5.lookup(tri_reverse[1:-2+offset:3])
        # rb3=t5.lookup(tri_reverse[2:-1+offset:3])

        if timesteps:
            f1 = tf.reshape(f1, (num_time, fragsize))
            f2 = tf.reshape(f2, (num_time, fragsize))
            f3 = tf.reshape(f3, (num_time, fragsize))
            r1 = tf.reshape(r1, (num_time, fragsize))
            r2 = tf.reshape(r2, (num_time, fragsize))
            r3 = tf.reshape(r3, (num_time, fragsize))
            seq = tf.stack([f1, f2, f3, r1, r2, r3], 1)
        else:
            seq = tf.stack([f1, f2, f3, r1, r2, r3], 0)
            # nuc = tf.stack([nuc1,nuc2],0)
            # code = tf.stack([fb1,fb2,fb3,rb1,rb2,rb3],0) # codon bias encoder

        return (
            {
                "translated": tf.one_hot(
                    seq, depth=11, dtype=tf.float32, on_value=1, off_value=0
                )
            },
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
            x[9],
            x[10],
        )
        # 'nucleotide': tf.one_hot(nuc, depth=4, dtype=tf.float32, on_value=1,\
        # off_value=0)},

    return p
