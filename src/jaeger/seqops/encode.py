"""Sequence encoding utilities.

TensorFlow-based DNA sequence preprocessing for training and inference.
Supports codon translation, nucleotide one-hot encoding, and complement mapping.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.seqops.maps import CODON_ID, CODONS


# ------------------------------------------------------------------
# Lookup table builders
# ------------------------------------------------------------------


def _map_codon(codons, codon_num):
    """Build a TF hash table mapping codon strings to integer IDs."""
    trimers = tf.constant(codons)
    trimer_vals = tf.constant(codon_num)
    trimer_init = tf.lookup.KeyValueTensorInitializer(trimers, trimer_vals)
    return tf.lookup.StaticHashTable(trimer_init, default_value=-1)


def _map_complement():
    """Build a TF hash table for reverse-complement nucleotides."""
    rc_keys = tf.constant([b"A", b"T", b"G", b"C", b"a", b"t", b"g", b"c"])
    rc_vals = tf.constant([b"T", b"A", b"C", b"G", b"t", b"a", b"c", b"g"])
    rc_init = tf.lookup.KeyValueTensorInitializer(rc_keys, rc_vals)
    return tf.lookup.StaticHashTable(rc_init, default_value="N")


def _map_nucleotide():
    """Build a TF hash table mapping nucleotides to integer IDs (A=0, G=1, C=2, T=3)."""
    keys_tensor = tf.constant([b"A", b"G", b"C", b"T", b"a", b"g", b"c", b"t"])
    vals_tensor = tf.constant([0, 1, 2, 3, 0, 1, 2, 3])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    return tf.lookup.StaticHashTable(ini, default_value=-1)


def _map_nucleotide_type():
    """Map nucleotides to purine (0) / pyrimidine (1)."""
    keys_tensor = tf.constant([b"A", b"G", b"C", b"T"])
    vals_tensor = tf.constant([0, 1, 1, 0])
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    return tf.lookup.StaticHashTable(ini, default_value=-1)


def _remap_labels(original, alternative):
    """Build a TF hash table remapping label IDs to alternative IDs."""
    keys_tensor = tf.constant(original)
    vals_tensor = tf.constant(alternative)
    ini = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    return tf.lookup.StaticHashTable(ini, default_value=0)


# ------------------------------------------------------------------
# Training preprocessor
# ------------------------------------------------------------------


def process_string_train(
    codons=CODONS,
    codon_num=CODON_ID,
    codon_depth=None,
    label_original=None,
    label_alternative=None,
    class_label_onehot=True,
    seq_onehot=True,
    num_classes=None,
    crop_size=None,
    timesteps=False,
    num_time=None,
    fragsize=200,
    mutate=False,
    mutation_rate=0.1,
    masking=False,
    ngram_width=None,
    input_type="translated",
    shuffle=False,
    shuffle_frames=False,
):
    """Build a TensorFlow function for training-time sequence preprocessing.

    Returns a callable that takes a CSV string and returns ``(inputs, label)``.
    """
    if codons and codon_num:
        map_codon = _map_codon(codons=codons, codon_num=codon_num)
    map_complement = _map_complement()
    map_nucleotide = _map_nucleotide()

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

        if crop_size:
            mod3 = tf.math.floormod(crop_size, 3)
            forward_strand = tf.strings.bytes_split(x[1])[:crop_size]
            offset_lut = tf.constant([-2, -1, 0], dtype=tf.int32)
            offset = tf.gather(offset_lut, mod3)
        else:
            string_length = tf.strings.length(x[1])
            mod3 = tf.math.floormod(string_length, 3)
            offset_lut = tf.constant([-2, -1, 0], dtype=tf.int32)
            offset = tf.gather(offset_lut, mod3)
            forward_strand = tf.strings.bytes_split(x[1])

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

        if input_type in ["nucleotide", "both"]:
            nuc1 = map_nucleotide.lookup(forward_strand)
            nuc2 = map_nucleotide.lookup(reverse_strand)
            nuc = tf.stack([nuc1, nuc2], axis=0)
            outputs["nucleotide"] = tf.one_hot(nuc, depth=4, dtype=tf.float32)

        if input_type in ["translated", "both"]:
            tri_forward = tf.strings.ngrams(
                forward_strand, ngram_width=ngram_width, separator=""
            )
            tri_reverse = tf.strings.ngrams(
                reverse_strand, ngram_width=ngram_width, separator=""
            )

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

            if shuffle_frames:
                perm = tf.random.shuffle(tf.constant([0, 1, 2, 3, 4, 5]))
                seq = tf.gather(seq, perm, axis=0)

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

        return outputs, label

    return p


# ------------------------------------------------------------------
# Inference preprocessor
# ------------------------------------------------------------------


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
    ngram_width=3,
    shuffle=False,
    masking=False,
    input_type="translated",
):
    """Build a TensorFlow function for inference-time sequence preprocessing.

    Returns a callable that takes a CSV string and returns ``(inputs, metadata)``.
    """
    if codons and codon_num:
        map_codon = _map_codon(codons=codons, codon_num=codon_num)
    map_complement = _map_complement()
    map_nucleotide = _map_nucleotide()

    @tf.function
    def p(string):
        x = tf.strings.split(string, sep=",")

        if crop_size:
            mod3 = tf.math.floormod(crop_size, 3)
            forward_strand = tf.strings.bytes_split(x[0])[:crop_size]
            offset_lut = tf.constant([-2, -1, 0], dtype=tf.int32)
            offset = tf.gather(offset_lut, mod3)
        else:
            string_length = tf.strings.length(x[1])
            mod3 = tf.math.floormod(string_length, 3)
            offset_lut = tf.constant([-2, -1, 0], dtype=tf.int32)
            offset = tf.gather(offset_lut, mod3)
            forward_strand = tf.strings.bytes_split(x[0])

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

        if input_type in ["nucleotide", "both"]:
            nuc1 = map_nucleotide.lookup(forward_strand)
            nuc2 = map_nucleotide.lookup(reverse_strand)
            nuc = tf.stack([nuc1, nuc2], axis=0)
            outputs["nucleotide"] = tf.one_hot(
                nuc, depth=4, dtype=tf.float32, on_value=1, off_value=0
            )

        if input_type in ["translated", "both"]:
            tri_forward = tf.strings.ngrams(
                forward_strand, ngram_width=ngram_width, separator=""
            )
            tri_reverse = tf.strings.ngrams(
                reverse_strand, ngram_width=ngram_width, separator=""
            )

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
                outputs["translated"] = tf.cast(seq + 1, dtype=tf.float32)

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

# ------------------------------------------------------------------
# Raw sequence processors (for numpy_raw / numpy_raw_variable loaders)
# ------------------------------------------------------------------


def _build_codon_lookup_table() -> tf.Tensor:
    """Build a 5x5x5 lookup table for mapping int8 DNA triplets to codon IDs."""
    BASES = ["A", "T", "G", "C", "N"]
    codon_to_id = {c: i for i, c in enumerate(CODONS)}

    lookup = np.zeros((5, 5, 5), dtype=np.int32)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                codon = BASES[i] + BASES[j] + BASES[k]
                lookup[i, j, k] = codon_to_id.get(codon, -1)

    return tf.constant(lookup, dtype=tf.int32)


def _make_process_raw_sequence_fn(
    crop_size: int = 500,
    ngram_width: int = 3,
    num_classes: int = 3,
    shuffle: bool = False,
    mutate: bool = False,
    mutation_rate: float = 0.1,
    shuffle_frames: bool = False,
):
    """Creates a TF function that converts int8 DNA sequences to 6-frame codon embeddings."""
    codon_lookup = _build_codon_lookup_table()
    # Complement mapping for int8: A=0->T=1, T=1->A=0, G=2->C=3, C=3->G=2, N=4->N=4
    comp_lookup = tf.constant([1, 0, 3, 2, 4], dtype=tf.int32)
    seq_len = crop_size // ngram_width - 1  # 165 for crop_size=500

    @tf.function
    def _process_raw_sequence(sequence, label):
        # sequence: int8 tensor of shape (crop_size,)
        # label: float32 tensor of shape (num_classes,)

        # Cast to int32 for indexing
        seq = tf.cast(sequence, tf.int32)

        # Optional: shuffle bases
        if shuffle:
            seq = tf.random.shuffle(seq)

        # Optional: mutate bases
        if mutate:
            mask = tf.random.uniform(tf.shape(seq)) < mutation_rate
            random_bases = tf.random.uniform(
                tf.shape(seq), minval=0, maxval=5, dtype=tf.int32
            )
            seq = tf.where(mask, random_bases, seq)

        # Forward strand: extract n-grams as triplets of indices
        indices = tf.range(crop_size - ngram_width + 1)  # 0 to 497
        tri0 = tf.gather(seq, indices)
        tri1 = tf.gather(seq, indices + 1)
        tri2 = tf.gather(seq, indices + 2)

        # Lookup codon IDs: lookup[tri0, tri1, tri2]
        codon_ids = tf.gather_nd(codon_lookup, tf.stack([tri0, tri1, tri2], axis=1))

        # Reverse complement strand:
        # 1. Reverse the sequence
        rev_seq = tf.reverse(seq, axis=[0])
        # 2. Map complement
        rev_comp_seq = tf.gather(comp_lookup, rev_seq)
        # 3. Extract n-grams from reverse complement
        rev_tri0 = tf.gather(rev_comp_seq, indices)
        rev_tri1 = tf.gather(rev_comp_seq, indices + 1)
        rev_tri2 = tf.gather(rev_comp_seq, indices + 2)
        # 4. Lookup codon IDs
        rev_codon_ids = tf.gather_nd(
            codon_lookup, tf.stack([rev_tri0, rev_tri1, rev_tri2], axis=1)
        )

        # Extract 6 reading frames with stride ngram_width (3)
        def extract_frames(codons):
            f1 = codons[0::ngram_width]
            f2 = codons[1::ngram_width]
            f3 = codons[2::ngram_width]
            return f1, f2, f3

        f1, f2, f3 = extract_frames(codon_ids)
        r1, r2, r3 = extract_frames(rev_codon_ids)

        # Stack frames: shape (6, seq_len)
        frames = tf.stack([f1, f2, f3, r1, r2, r3], axis=0)

        # Trim to exact seq_len
        frames = frames[:, :seq_len]

        # +1 for embedding layer (0 is padding/mask)
        frames = frames + 1

        # Optional: shuffle frames
        if shuffle_frames:
            perm = tf.random.shuffle(tf.range(6))
            frames = tf.gather(frames, perm, axis=0)

        return {"translated": frames}, label

    return _process_raw_sequence


def _make_process_variable_sequence_fn(
    ngram_width: int = 3,
    num_classes: int = 3,
    shuffle: bool = False,
    mutate: bool = False,
    mutation_rate: float = 0.1,
    shuffle_frames: bool = False,
):
    """Creates a TF function for variable-length int8 sequences."""
    codon_lookup = _build_codon_lookup_table()
    comp_lookup = tf.constant([1, 0, 3, 2, 4], dtype=tf.int32)

    @tf.function
    def _process_variable_sequence(sequence, length, label):
        seq = tf.cast(sequence, tf.int32)
        actual_len = tf.cast(length, tf.int32)

        # Crop to actual length
        seq = seq[:actual_len]

        # Optional: shuffle bases
        if shuffle:
            seq = tf.random.shuffle(seq)

        # Optional: mutate bases
        if mutate:
            mask = tf.random.uniform(tf.shape(seq)) < mutation_rate
            random_bases = tf.random.uniform(
                tf.shape(seq), minval=0, maxval=5, dtype=tf.int32
            )
            seq = tf.where(mask, random_bases, seq)

        # Compute number of codons for this sequence
        num_codons = actual_len // ngram_width - 1
        num_codons = tf.maximum(num_codons, 0)

        # Forward strand n-grams
        indices = tf.range(actual_len - ngram_width + 1)
        tri0 = tf.gather(seq, indices)
        tri1 = tf.gather(seq, indices + 1)
        tri2 = tf.gather(seq, indices + 2)

        codon_ids = tf.gather_nd(codon_lookup, tf.stack([tri0, tri1, tri2], axis=1))

        # Reverse complement
        rev_seq = tf.reverse(seq, axis=[0])
        rev_comp_seq = tf.gather(comp_lookup, rev_seq)
        rev_tri0 = tf.gather(rev_comp_seq, indices)
        rev_tri1 = tf.gather(rev_comp_seq, indices + 1)
        rev_tri2 = tf.gather(rev_comp_seq, indices + 2)
        rev_codon_ids = tf.gather_nd(
            codon_lookup, tf.stack([rev_tri0, rev_tri1, rev_tri2], axis=1)
        )

        # Extract frames
        def extract_frames(codons):
            f1 = codons[0::ngram_width]
            f2 = codons[1::ngram_width]
            f3 = codons[2::ngram_width]
            return f1, f2, f3

        f1, f2, f3 = extract_frames(codon_ids)
        r1, r2, r3 = extract_frames(rev_codon_ids)

        # Find minimum frame length to ensure all frames match
        min_len = tf.reduce_min(
            [
                tf.shape(f1)[0],
                tf.shape(f2)[0],
                tf.shape(f3)[0],
                tf.shape(r1)[0],
                tf.shape(r2)[0],
                tf.shape(r3)[0],
            ]
        )

        # Trim all frames to same length
        f1, f2, f3 = f1[:min_len], f2[:min_len], f3[:min_len]
        r1, r2, r3 = r1[:min_len], r2[:min_len], r3[:min_len]

        # Stack frames
        frames = tf.stack([f1, f2, f3, r1, r2, r3], axis=0)

        # +1 for embedding layer
        frames = frames + 1

        # Optional: shuffle frames
        if shuffle_frames:
            perm = tf.random.shuffle(tf.range(6))
            frames = tf.gather(frames, perm, axis=0)

        return {"translated": frames}, label

    return _process_variable_sequence
