from __future__ import annotations

import json

import numpy as np
import pytest

from jaeger.data.loaders import _load_numpy_dataset

CROP_SIZES = [445, 555, 605, 665]
RECORD_LEN = 665
N = 300


def _make_sharded_npz(path, n=N):
    """Minimal sharded NPZ mimicking the real training npz layout."""
    rng = np.random.default_rng(0)
    translated = np.empty(n, dtype=object)
    for i in range(n):
        translated[i] = rng.integers(1, 33, size=(6, RECORD_LEN), dtype=np.int32)
    manifest = {
        "num_shards": 1,
        "keys": ["labels", "lengths", "translated_lengths", "translated"],
        "codon_map": "codon_id",
        "nucleotide_map": None,
    }
    np.savez(
        path,
        _jaeger_manifest=json.dumps(manifest),
        labels_00000=np.arange(n, dtype=np.int32) % 6,
        lengths_00000=np.full(n, RECORD_LEN, dtype=np.int32),
        translated_lengths_00000=np.full(n, RECORD_LEN, dtype=np.int32),
        translated_00000=translated,
    )


def _collect(ds, limit=10000):
    lengths, labels = [], []
    for i, (features, label) in enumerate(ds):
        if i >= limit:
            break
        x = features["translated"].numpy()
        valid = (x != 0).any(axis=0).sum()
        lengths.append(int(valid))
        labels.append(int(np.argmax(label.numpy())))
    return lengths, labels


def test_sample_mode_one_crop_per_record(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="sample",
    )
    lengths, labels = _collect(ds, limit=N * 3)
    # exactly N samples per epoch (one crop per record), not 7*N
    assert len(lengths) == N
    assert set(lengths) <= set(CROP_SIZES)
    assert set(labels) == {0, 1, 2, 3, 4, 5}


def test_sample_mode_variant_distribution_is_uniform(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=1400)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="sample",
    )
    lengths, _ = _collect(ds, limit=1400)
    # 7 variants: 2x445, 2x555, 2x605, 1x665 -> ratios 2/7, 2/7, 2/7, 1/7
    from collections import Counter

    c = Counter(lengths)
    assert c[445] / 1400 == pytest.approx(2 / 7, abs=0.05)
    assert c[555] / 1400 == pytest.approx(2 / 7, abs=0.05)
    assert c[605] / 1400 == pytest.approx(2 / 7, abs=0.05)
    assert c[665] / 1400 == pytest.approx(1 / 7, abs=0.05)


def test_all_mode_fans_out_all_variants(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=50)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="all",
    )
    lengths, _ = _collect(ds, limit=1000)
    # 7 variants per record -> 350 samples
    assert len(lengths) == 350


def test_default_crop_mode_is_all(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=50)
    ds = _load_numpy_dataset(
        str(npz), input_type="translated", num_classes=6, crop_sizes=CROP_SIZES
    )
    lengths, _ = _collect(ds, limit=1000)
    assert len(lengths) == 350


def test_sample_mode_variants_change_across_epochs(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=50)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="sample",
    )
    epoch1, _ = _collect(ds, limit=50)
    epoch2, _ = _collect(ds, limit=50)
    # the variant draws are re-rolled per generator invocation (epoch)
    assert epoch1 != epoch2 or set(epoch1) != {epoch1[0]}


# --- crop_mode: range (continuous variable length) -----------------------------


def test_range_mode_one_crop_per_record_within_bounds(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="range",
    )
    lengths, labels = _collect(ds, limit=N * 3)
    # exactly N samples per epoch (one crop per record)
    assert len(lengths) == N
    assert all(445 <= L <= 665 for L in lengths)
    assert set(labels) == {0, 1, 2, 3, 4, 5}


def test_range_mode_lengths_are_continuous_not_discrete(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=2000)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="range",
    )
    lengths, _ = _collect(ds, limit=2000)
    discrete = {445, 555, 605, 665}
    # most lengths should NOT be one of the four discrete sizes
    off_grid = sum(1 for L in lengths if L not in discrete)
    assert off_grid / len(lengths) > 0.8
    # endpoints reachable
    assert 445 in lengths
    assert 665 in lengths


def test_range_mode_distribution_is_roughly_uniform(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=5000)
    ds = _load_numpy_dataset(
        str(npz),
        input_type="translated",
        num_classes=6,
        crop_sizes=CROP_SIZES,
        crop_mode="range",
    )
    lengths, _ = _collect(ds, limit=5000)
    arr = np.array(lengths)
    # uniform over [445, 665]: mean ~ 555, no length overrepresented
    assert abs(arr.mean() - 555) < 8
    from collections import Counter

    top = Counter(lengths).most_common(1)[0][1]
    assert top / len(lengths) < 0.02  # no single length > 2%


def test_invalid_crop_mode_rejected(tmp_path):
    npz = tmp_path / "train.npz"
    _make_sharded_npz(npz, n=10)
    with pytest.raises(ValueError, match="crop_mode"):
        _load_numpy_dataset(
            str(npz),
            input_type="translated",
            num_classes=6,
            crop_sizes=CROP_SIZES,
            crop_mode="bogus",
        )
