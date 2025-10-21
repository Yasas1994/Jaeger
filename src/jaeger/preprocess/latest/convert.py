import tensorflow as tf
from jaeger.preprocess.latest.maps import CODON_ID, CODONS


# map codons to amino acids
def _map_codon(codons, codon_num):
    trimers = tf.constant(codons)
    trimer_vals = tf.constant(codon_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    trimer_table = tf.lookup.StaticHashTable(trimer_init, default_value=-1)

    return trimer_table


# convert to complement
def _map_complement():
    rc_keys = tf.constant([b"A", b"T", b"G", b"C", b"a", b"t", b"g", b"c"])
    rc_vals = tf.constant([b"T", b"A", b"C", b"G", b"t", b"a", b"c", b"g"])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    rc_table = tf.lookup.StaticHashTable(rc_init, default_value="N")

    return rc_table


def _remap_labels(original, alternative):
    """
    helper function remaps labels ids to alternative ids
    """
    keys_tensor = tf.constant(original)
    vals_tensor = tf.constant(alternative)  # [0,1,2,3,4,5,6]
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    map_ = tf.lookup.StaticHashTable(ini, default_value=0)
    return map_


def _map_nucleotide():
    keys_tensor = tf.constant([b"A", b"G", b"C", b"T", b"a", b"g", b"c", b"t"])
    vals_tensor = tf.constant([0, 1, 2, 3, 0, 1, 2, 3])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    map_ = tf.lookup.StaticHashTable(ini, default_value=-1)
    return map_


def _map_nucleotide_type():
    """
    maps to purine and pyrimidine
    """
    keys_tensor = tf.constant([b"A", b"G", b"C", b"T"])
    vals_tensor = tf.constant([0, 1, 1, 0])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    map_ = tf.lookup.StaticHashTable(ini, default_value=-1)
    return map_


def process_string_train(
    codons=CODONS,
    codon_num=CODON_ID,
    codon_depth=64,
    label_original=None,
    label_alternative=None,
    class_label_onehot=True,
    seq_onehot=True,
    num_classes=5,
    crop_size=1024,
    timesteps=False,
    num_time=None,
    fragsize=200,
    mutate=False,
    mutation_rate=0.1,
    masking=False,
    ngram_width = 3,
    input_type="translated",  # "translated", "nucleotide", "both"
    shuffle=False,
):
    """
    TensorFlow string processing function for sequence input.

    Args:
        codons: Codon vocabulary list
        codon_num: Number of codons
        label_original: Original labels (optional)
        label_alternative: Alternative labels (optional)
        class_label_onehot: One-hot encode labels
        input_type: 'translated', 'nucleotide', 'both'
    """

    map_codon = _map_codon(codons=codons, codon_num=codon_num)
    map_complement = _map_complement()
    map_nucleotide = _map_nucleotide()

    # Setup label lookup function
    if label_original is not None and label_alternative is not None:
        remap_labels = _remap_labels(
            original=label_original, alternative=label_alternative
        )

        def lookup_label(x):
            return remap_labels.lookup(tf.strings.to_number(x, tf.int32))
    else:

        def lookup_label(x):
            return tf.strings.to_number(x, tf.int32)

    @tf.function
    def p(string):
        x = tf.strings.split(string, sep=",")
        label = tf.cast(lookup_label(x[0]), dtype=tf.int32)

        # Determine offset for codon splitting
        offset = {0: -2, 1: -1, 2: 0}[crop_size % 3]
        forward_strand = tf.strings.bytes_split(x[1])[:crop_size]

        # Apply mutations if requested
        if mutate:
            alphabet = tf.constant(["A", "T", "G", "C", "N"], dtype=tf.string)
            mask = tf.random.uniform(tf.shape(forward_strand)) < mutation_rate
            mutations = tf.random.uniform(
                tf.shape(forward_strand), minval=0, maxval=4, dtype=tf.int32
            )
            forward_strand = tf.where(
                mask, tf.gather(alphabet, mutations), forward_strand
            )
        if shuffle:
            forward_strand = tf.random.shuffle(forward_strand)

        reverse_strand = map_complement.lookup(forward_strand[::-1])

        outputs = {}
        if masking is False:
            forward_strand = tf.strings.upper(forward_strand)
            reverse_strand = tf.strings.upper(reverse_strand)
        # Nucleotide representation
        if input_type in ["nucleotide", "both"]:
            nuc1 = map_nucleotide.lookup(forward_strand)
            nuc2 = map_nucleotide.lookup(reverse_strand)
            nuc = tf.stack([nuc1, nuc2], axis=0)
            outputs["nucleotide"] = tf.one_hot(nuc, depth=4, dtype=tf.float32)

        # Translated representation
        if input_type in ["translated", "both"]:
            tri_forward = tf.strings.ngrams(forward_strand, ngram_width=ngram_width, separator="")
            tri_reverse = tf.strings.ngrams(reverse_strand, ngram_width=ngram_width, separator="")

            f1 = map_codon.lookup(tri_forward[0 : -3 + offset : ngram_width])
            f2 = map_codon.lookup(tri_forward[1 : -2 + offset : ngram_width])
            f3 = map_codon.lookup(tri_forward[2 : -1 + offset : ngram_width])
            r1 = map_codon.lookup(tri_reverse[0 : -3 + offset : ngram_width])
            r2 = map_codon.lookup(tri_reverse[1 : -2 + offset : ngram_width])
            r3 = map_codon.lookup(tri_reverse[2 : -1 + offset : ngram_width])

            if timesteps:
                f1 = tf.reshape(f1, (num_time, fragsize))
                f2 = tf.reshape(f2, (num_time, fragsize))
                f3 = tf.reshape(f3, (num_time, fragsize))
                r1 = tf.reshape(r1, (num_time, fragsize))
                r2 = tf.reshape(r2, (num_time, fragsize))
                r3 = tf.reshape(r3, (num_time, fragsize))
                seq = tf.stack([f1, f2, f3, r1, r2, r3], axis=1)
            else:
                seq = tf.stack([f1, f2, f3, r1, r2, r3], axis=0)
            if seq_onehot:
                outputs["translated"] = tf.one_hot(
                    seq, depth=codon_depth, dtype=tf.float32, on_value=1, off_value=0
                )
            else:
                outputs["translated"] = seq + 1

        if class_label_onehot:
            label = tf.one_hot(
                label, depth=num_classes, dtype=tf.float32, on_value=1, off_value=0
            )
        else:
            label = tf.expand_dims(label, axis=0)

        # return outputs, {'classifier': label,
        #                  'reliability': reliability
        #                 }
        return outputs, label

    return p


def process_string_inference(
    codons=CODONS,
    codon_num=CODON_ID,
    codon_depth=64,
    crop_size=1024,
    seq_onehot=True,
    timesteps=False,
    num_time=None,
    fragsize=200,
    mutate=False,
    mutation_rate=0.1,
    ngram_width = 3,
    input_type="translated",  # "translated", "nucleotide", "both"
):
    """
    TensorFlow string processing function for sequence input.

    Args:
        codons: Codon vocabulary list
        codon_num: Number of codons
        label_original: Original labels (optional)
        label_alternative: Alternative labels (optional)
        class_label_onehot: One-hot encode labels
        input_type: 'translated', 'nucleotide', 'both'
    """

    map_codon = _map_codon(codons=codons, codon_num=codon_num)
    map_complement = _map_complement()
    map_nucleotide = _map_nucleotide()

    @tf.function
    def p(string):
        x = tf.strings.split(string, sep=",")
        # Determine offset for codon splitting
        offset = {0: -2, 1: -1, 2: 0}[crop_size % 3]
        forward_strand = tf.strings.bytes_split(x[0])[:crop_size]

        # Apply mutations if requested
        if mutate:
            alphabet = tf.constant(["A", "T", "G", "C", "N"], dtype=tf.string)
            mask = tf.random.uniform(tf.shape(forward_strand)) < mutation_rate
            mutations = tf.random.uniform(
                tf.shape(forward_strand), minval=0, maxval=4, dtype=tf.int32
            )
            forward_strand = tf.where(
                mask, tf.gather(alphabet, mutations), forward_strand
            )

        reverse_strand = map_complement.lookup(forward_strand[::-1])

        outputs = {}

        # Nucleotide representation
        if input_type in ["nucleotide", "both"]:
            nuc1 = map_nucleotide.lookup(forward_strand)
            nuc2 = map_nucleotide.lookup(reverse_strand)
            nuc = tf.stack([nuc1, nuc2], axis=0)
            outputs["nucleotide"] = tf.one_hot(nuc, depth=4, dtype=tf.float32)

        # Translated representation
        if input_type in ["translated", "both"]:
            tri_forward = tf.strings.ngrams(forward_strand, ngram_width=ngram_width, separator="")
            tri_reverse = tf.strings.ngrams(reverse_strand, ngram_width=ngram_width, separator="")

            f1 = map_codon.lookup(tri_forward[0 : -3 + offset : ngram_width])
            f2 = map_codon.lookup(tri_forward[1 : -2 + offset : ngram_width])
            f3 = map_codon.lookup(tri_forward[2 : -1 + offset : ngram_width])
            r1 = map_codon.lookup(tri_reverse[0 : -3 + offset : ngram_width])
            r2 = map_codon.lookup(tri_reverse[1 : -2 + offset : ngram_width])
            r3 = map_codon.lookup(tri_reverse[2 : -1 + offset : ngram_width])

            if timesteps:
                f1 = tf.reshape(f1, (num_time, fragsize))
                f2 = tf.reshape(f2, (num_time, fragsize))
                f3 = tf.reshape(f3, (num_time, fragsize))
                r1 = tf.reshape(r1, (num_time, fragsize))
                r2 = tf.reshape(r2, (num_time, fragsize))
                r3 = tf.reshape(r3, (num_time, fragsize))
                seq = tf.stack([f1, f2, f3, r1, r2, r3], axis=1)
            else:
                seq = tf.stack([f1, f2, f3, r1, r2, r3], axis=0)

            if seq_onehot:
                outputs["translated"] = tf.one_hot(
                    seq, depth=codon_depth, dtype=tf.float32, on_value=1, off_value=0
                )
            else:
                outputs["translated"] = seq + 1

        # return outputs, {'classifier': label,
        #                  'reliability': reliability
        #                 }
        return (
            outputs,
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
