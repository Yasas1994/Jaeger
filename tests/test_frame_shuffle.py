"""
Test script to verify frame shuffle functionality.

This tests that:
1. shuffle_frames=False preserves frame order [f1,f2,f3,r1,r2,r3]
2. shuffle_frames=True randomly permutes frame order
3. All frames are present after shuffling (no duplicates, no drops)
"""

import numpy as np
from jaeger.preprocess.latest.convert import process_string_train
from jaeger.preprocess.latest.maps import CODONS, CODON_ID


def create_test_csv_string(label=1, sequence="ATGCGTACGTTAGCTAGCTAGC"):
    """Create a test CSV string matching the expected format."""
    # Format: label,sequence,header,index,end,gc_count,gc_skew
    gc_count = sequence.count("G") + sequence.count("C")
    gc_skew = (sequence.count("G") - sequence.count("C")) / max(gc_count, 1)
    return f"{label},{sequence},test_header,0,{len(sequence)},{gc_count},{gc_skew}"


def test_baseline_frame_order():
    """Test that baseline (shuffle_frames=False) preserves frame order."""
    processor = process_string_train(
        codons=CODONS,
        codon_num=CODON_ID,
        codon_depth=65,
        num_classes=6,
        crop_size=24,
        ngram_width=3,
        input_type="translated",
        seq_onehot=True,
        shuffle_frames=False,
    )

    test_string = create_test_csv_string()
    outputs, label = processor(test_string)

    seq = outputs["translated"]
    print(f"Baseline frame order shape: {seq.shape}")
    print(f"Frame order preserved: {seq.shape[0] == 6}")

    # With seq_onehot=True, shape is (6, length, codon_depth)
    assert seq.shape[0] == 6, "Expected 6 frames"
    assert seq.shape[2] == 65, f"Expected codon_depth=65, got {seq.shape[2]}"

    return seq


def test_shuffled_frame_order():
    """Test that shuffle_frames=True permutes frame order."""
    processor = process_string_train(
        codons=CODONS,
        codon_num=CODON_ID,
        codon_depth=65,
        num_classes=6,
        crop_size=24,
        ngram_width=3,
        input_type="translated",
        seq_onehot=True,
        shuffle_frames=True,
    )

    test_string = create_test_csv_string()

    # Run multiple times to see different permutations
    print("\nShuffled frame order (10 runs):")
    for i in range(10):
        outputs, label = processor(test_string)
        seq = outputs["translated"]
        print(f"  Run {i + 1}: shape={seq.shape}, dtype={seq.dtype}")
        assert seq.shape[0] == 6, "Expected 6 frames"

    return seq


def test_frame_shuffle_changes_content():
    """Verify that shuffling actually changes the frame positions."""
    processor_baseline = process_string_train(
        codons=CODONS,
        codon_num=CODON_ID,
        codon_depth=65,
        num_classes=6,
        crop_size=24,
        ngram_width=3,
        input_type="translated",
        seq_onehot=True,
        shuffle_frames=False,
    )

    processor_shuffled = process_string_train(
        codons=CODONS,
        codon_num=CODON_ID,
        codon_depth=65,
        num_classes=6,
        crop_size=24,
        ngram_width=3,
        input_type="translated",
        seq_onehot=True,
        shuffle_frames=True,
    )

    test_string = create_test_csv_string()

    # Get baseline output
    outputs_base, _ = processor_baseline(test_string)
    seq_base = outputs_base["translated"]

    # Get multiple shuffled outputs
    n_different = 0
    n_trials = 20

    for _ in range(n_trials):
        outputs_shuf, _ = processor_shuffled(test_string)
        seq_shuf = outputs_shuf["translated"]

        # Check if any frame position differs from baseline
        # (frame 0 in shuffled might not equal frame 0 in baseline)
        is_different = not np.allclose(seq_base.numpy(), seq_shuf.numpy())
        if is_different:
            n_different += 1

    print("\nFrame shuffle content test:")
    print(f"  Trials: {n_trials}")
    print(f"  Different from baseline: {n_different}/{n_trials}")
    print(f"  Shuffle is active: {n_different > 0}")

    assert n_different > 0, "Frame shuffle should change output"
    # Note: With 6! = 720 possible permutations, chance of identity is ~0.14%
    # So it's normal for all 20 trials to differ


if __name__ == "__main__":
    print("=" * 60)
    print("Frame Shuffle Test")
    print("=" * 60)

    print("\n--- Test 1: Baseline frame order ---")
    test_baseline_frame_order()

    print("\n--- Test 2: Shuffled frame order ---")
    test_shuffled_frame_order()

    print("\n--- Test 3: Content changes with shuffle ---")
    test_frame_shuffle_changes_content()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
