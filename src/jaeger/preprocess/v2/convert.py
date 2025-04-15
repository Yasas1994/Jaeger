from typing import Any
import tensorflow as tf
from preprocess.v2.maps import (CODONS, MURPHY10_INT)


def codon_mapper(mapto: list) -> Any:
    """
    Creates a static hash table for mapping codons to their corresponding
    values.

    Returns:
    -------
        tf.lookup.StaticHashTable: A static hash table for codon mapping.
    """

    trimers = tf.constant(CODONS)
    trimer_vals = tf.constant(mapto)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    return tf.lookup.StaticHashTable(trimer_init, default_value=0)



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


def process_string(
    crop_size:int=1024,
    timesteps:bool=False,
    num_time:int=None,
    fragsize:int=200,
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

        t1 = codon_mapper(MURPHY10_INT)
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
