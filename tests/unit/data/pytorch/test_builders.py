import numpy as np
import pytest

from jaeger.data.pytorch.builders import build_datasets


def _make_npz(path, n_samples, length=50, channels=None):
    if channels is None:
        data = np.random.randint(0, 65, size=(n_samples, 6, length)).astype(np.int32)
    else:
        data = np.random.randint(0, 65, size=(n_samples, 6, length, channels)).astype(
            np.int32
        )
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=n_samples)]
    np.savez(path, translated=data, label=labels)


def _build_config(train_paths, val_paths, data_format="numpy_full", batch_size=4):
    return {
        "model": {
            "string_processor": {"data_format": data_format},
        },
        "training": {
            "batch_size": batch_size,
            "fragment_classifier_data": {
                "train": [{"path": [str(p) for p in train_paths]}],
                "validation": [{"path": [str(p) for p in val_paths]}],
            },
        },
    }


def test_build_datasets_numpy_full(tmp_path):
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_npz(train_path, n_samples=8)
    _make_npz(val_path, n_samples=4)

    config = _build_config([train_path], [val_path], batch_size=4)
    loaders = build_datasets(config, branch="classifier")

    assert set(loaders.keys()) == {"train", "validation"}

    batch_x, batch_y, batch_mask = next(iter(loaders["train"]))
    assert batch_x.shape == (4, 6, 50)
    assert batch_y.shape == (4, 3)
    assert batch_mask.shape == (4, 50)

    batch_x, batch_y, batch_mask = next(iter(loaders["validation"]))
    assert batch_x.shape == (4, 6, 50)
    assert batch_y.shape == (4, 3)
    assert batch_mask.shape == (4, 50)


def test_build_datasets_multiple_paths(tmp_path):
    train_path1 = tmp_path / "train1.npz"
    train_path2 = tmp_path / "train2.npz"
    val_path = tmp_path / "val.npz"
    _make_npz(train_path1, n_samples=3)
    _make_npz(train_path2, n_samples=5)
    _make_npz(val_path, n_samples=4)

    config = _build_config([train_path1, train_path2], [val_path], batch_size=8)
    loaders = build_datasets(config, branch="classifier")

    assert len(loaders["train"].dataset) == 8
    batch_x, batch_y, batch_mask = next(iter(loaders["train"]))
    assert batch_x.shape == (8, 6, 50)
    assert batch_y.shape == (8, 3)
    assert batch_mask.shape == (8, 50)


def test_build_datasets_unsupported_format(tmp_path):
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_npz(train_path, n_samples=4)
    _make_npz(val_path, n_samples=4)

    config = _build_config([train_path], [val_path], data_format="csv")
    with pytest.raises(ValueError, match="Unsupported data_format"):
        build_datasets(config, branch="classifier")
