import numpy as np
import torch
from jaeger.data.pytorch.dataset_numpy import NumpyFullDataset, NumpyRawDataset
from jaeger.data.pytorch.transforms import dna_to_indices, translate_to_codons
from jaeger.seqops.maps import CODONS


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
    crop_size = 50
    seqs = np.random.randint(0, 4, size=(10, crop_size)).astype(np.int8)
    labels = np.random.randint(0, 3, size=10).astype(np.int64)
    path = tmp_path / "raw.npz"
    np.savez(path, sequences=seqs, labels=labels)

    codon_table = {c: i + 1 for i, c in enumerate(CODONS)}
    ds = NumpyRawDataset(
        path,
        seq_key="sequences",
        label_key="labels",
        crop_size=crop_size,
        num_classes=3,
        codon_table=codon_table,
    )
    x, y, mask = ds[0]
    assert x.shape[0] == 6
    assert x.shape == mask.shape
    assert y.shape == (3,)
    assert mask.dtype == torch.bool
    assert y.dtype == torch.float32
    assert x.max() <= 64
    assert x.min() >= 0


def test_numpy_raw_dataset_one_hot_labels(tmp_path):
    crop_size = 50
    seqs = np.random.randint(0, 4, size=(10, crop_size)).astype(np.int8)
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=10)]
    path = tmp_path / "raw_onehot.npz"
    np.savez(path, sequences=seqs, labels=labels)

    codon_table = {c: i + 1 for i, c in enumerate(CODONS)}
    ds = NumpyRawDataset(
        path,
        seq_key="sequences",
        label_key="labels",
        crop_size=crop_size,
        num_classes=3,
        codon_table=codon_table,
    )
    x, y, mask = ds[0]
    assert x.shape[0] == 6
    assert x.shape == mask.shape
    assert y.shape == (3,)
    assert torch.allclose(y, torch.tensor(labels[0]))
    assert mask.dtype == torch.bool
    assert y.dtype == torch.float32


def test_translate_to_codons_known_sequence():
    codon_table = {c: i + 1 for i, c in enumerate(CODONS)}
    seq = "ATGAAATTTCCC"
    x = translate_to_codons(dna_to_indices(seq), codon_table)

    assert x.shape[0] == 6
    assert x.shape[1] == 4

    expected = [codon_table["ATG"], codon_table["AAA"], codon_table["TTT"], codon_table["CCC"]]
    assert torch.equal(x[0], torch.tensor(expected, dtype=torch.long))


def test_translate_to_codons_pads_unknown_to_zero():
    codon_table = {c: i + 1 for i, c in enumerate(CODONS)}
    seq = "ATGAAANNNCCC"
    x = translate_to_codons(dna_to_indices(seq), codon_table)

    # The codon containing N should map to 0.
    mask = x != 0
    assert not mask.all()
    assert (x[~mask] == 0).all()
