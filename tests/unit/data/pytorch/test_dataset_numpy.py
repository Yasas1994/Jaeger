import numpy as np
import torch
from jaeger.data.pytorch.dataset_numpy import NumpyFullDataset, NumpyRawDataset
from jaeger.seqops.maps import CODON_ID


def test_numpy_full_dataset(tmp_path):
    data = np.random.randint(0, 65, size=(10, 6, 50)).astype(np.int32)
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=10)]
    path = tmp_path / "data.npz"
    np.savez(path, translated=data, label=labels)

    ds = NumpyFullDataset(path, input_key="translated")
    assert len(ds) == 10
    x, y, mask = ds[0]
    assert x.shape == (6, 50)
    assert y.shape == (3,)
    assert mask.shape == (6, 50)
    assert mask.dtype == torch.bool

    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch_x, batch_y, batch_mask = next(iter(loader))
    assert batch_x.shape == (2, 6, 50)
    assert batch_y.shape == (2, 3)
    assert batch_mask.shape == (2, 6, 50)


def test_numpy_full_dataset_3d_input(tmp_path):
    data = np.random.randint(0, 65, size=(5, 6, 50, 4)).astype(np.int32)
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=5)]
    path = tmp_path / "data3d.npz"
    np.savez(path, translated=data, label=labels)

    ds = NumpyFullDataset(path, input_key="translated")
    assert len(ds) == 5
    x, y, mask = ds[0]
    assert x.shape == (6, 50, 4)
    assert y.shape == (3,)
    assert mask.shape == (6, 50)
    assert mask.dtype == torch.bool


def test_numpy_raw_dataset(tmp_path):
    seqs = np.random.randint(0, 4, size=(10, 500)).astype(np.int8)
    labels = np.random.randint(0, 3, size=10).astype(np.int64)
    path = tmp_path / "raw.npz"
    np.savez(path, sequences=seqs, labels=labels)

    codon_table = {c: i for i, c in enumerate(CODON_ID)}
    ds = NumpyRawDataset(
        path,
        seq_key="sequences",
        label_key="labels",
        crop_size=50,
        num_classes=3,
        codon_table=codon_table,
    )
    x, y, mask = ds[0]
    assert x.shape == (6, 50)
    assert mask.shape == (6, 50)
    assert y.shape == (3,)
    assert mask.dtype == torch.bool
    assert y.dtype == torch.float32
