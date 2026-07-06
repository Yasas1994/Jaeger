"""Tests for jaeger.seqops.encode."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.seqops import encode


class TestLookupHelpers:
    def test_map_codon(self):
        table = encode._map_codon(["AAA", "AAC"], [0, 1])
        assert table.lookup(tf.constant([b"AAA", b"AAC", b"BBB"])).numpy().tolist() == [0, 1, -1]

    def test_map_complement(self):
        table = encode._map_complement()
        assert table.lookup(tf.constant([b"A", b"T", b"G", b"C"])).numpy().tolist() == [b"T", b"A", b"C", b"G"]

    def test_map_nucleotide(self):
        table = encode._map_nucleotide()
        assert table.lookup(tf.constant([b"A", b"G", b"C", b"T"])).numpy().tolist() == [0, 1, 2, 3]

    def test_remap_labels(self):
        table = encode._remap_labels([0, 1], [10, 20])
        assert table.lookup(tf.constant([0, 1, 2], dtype=tf.int32)).numpy().tolist() == [10, 20, 0]


class TestProcessStringTrain:
    def test_translated_onehot(self):
        processor = encode.process_string_train(
            codon_depth=64,
            num_classes=2,
            crop_size=28,
            fragsize=8,
            ngram_width=3,
            input_type="translated",
            seq_onehot=True,
        )
        csv = "0,ATGCATGCATGCATGCATGCATGCATGC,meta"
        inputs, label = processor(tf.constant(csv))
        assert "translated" in inputs
        assert inputs["translated"].shape.as_list() == [6, 8, 64]
        assert label.shape.as_list() == [2]

    def test_nucleotide_onehot(self):
        processor = encode.process_string_train(
            codon_depth=64,
            num_classes=2,
            crop_size=12,
            ngram_width=3,
            input_type="nucleotide",
        )
        csv = "1,ATGCATGCATGC,meta"
        inputs, label = processor(tf.constant(csv))
        assert "nucleotide" in inputs
        assert inputs["nucleotide"].shape.as_list() == [2, 12, 4]

    def test_label_remapping(self):
        processor = encode.process_string_train(
            codon_depth=64,
            num_classes=2,
            crop_size=12,
            ngram_width=3,
            label_original=[0, 1],
            label_alternative=[1, 0],
        )
        csv = "0,ATGCATGCATGC,meta"
        _, label = processor(tf.constant(csv))
        assert np.argmax(label.numpy()) == 1


class TestProcessStringInference:
    def test_inference_outputs(self):
        processor = encode.process_string_inference(
            codon_depth=64, crop_size=24, fragsize=8, ngram_width=3, input_type="translated"
        )
        csv = "ATGCATGCATGCATGCATGCATGC,h,0,1,0,24,5,5,5,5,0.0"
        outputs = processor(tf.constant(csv))
        inputs, *meta = outputs
        assert "translated" in inputs
        assert len(meta) == 10


class TestRawSequenceProcessors:
    def test_make_process_raw_sequence_fn(self):
        # crop_size must satisfy (crop_size-2) divisible by 3 so all frames match.
        fn = encode._make_process_raw_sequence_fn(
            crop_size=26, ngram_width=3, num_classes=2
        )
        seq = np.array([0, 1, 2, 3] * 7, dtype=np.int8)[:26]
        label = np.array([1.0, 0.0], dtype=np.float32)
        inputs, out_label = fn(tf.constant(seq), tf.constant(label))
        assert "translated" in inputs
        assert inputs["translated"].shape.as_list()[0] == 6
        assert out_label.shape.as_list() == [2]

    def test_make_process_variable_sequence_fn(self):
        fn = encode._make_process_variable_sequence_fn(
            ngram_width=3, num_classes=2
        )
        seq = np.array([0, 1, 2, 3] * 6, dtype=np.int8)
        label = np.array([1.0, 0.0], dtype=np.float32)
        inputs, out_label = fn(tf.constant(seq), tf.constant(24), tf.constant(label))
        assert "translated" in inputs
        assert out_label.shape.as_list() == [2]
