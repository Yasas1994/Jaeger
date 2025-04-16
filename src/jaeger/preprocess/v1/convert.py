import logging
import tensorflow as tf
from jaeger.preprocess.v1.maps import (TRIMERS, TRIMER_INT, AMINO_ACIDS, AMINO_ACIDS_INT)
logger = logging.getLogger("Jaeger")


def codon_mapper():
    """
    Creates a static hash table for mapping codons to their corresponding
    values.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for codon mapping.
    """

    trimers = tf.constant(
        TRIMERS
    )
    trimer_vals = tf.constant(
       TRIMER_INT
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
       AMINO_ACIDS
    )
    aa_num = tf.constant(
       AMINO_ACIDS_INT
    )
    aa_init = tf.lookup.KeyValueTensorInitializer(aa, aa_num)
    return tf.lookup.StaticHashTable(aa_init, default_value=0)


def complement_mapper():
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
        t1, t3 = codon_mapper(), complement_mapper()
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
        t1, t3 = codon_mapper(), complement_mapper()

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


