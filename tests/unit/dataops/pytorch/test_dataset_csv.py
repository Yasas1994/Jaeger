import csv

from jaeger.data.pytorch.dataset_csv import CSVDataset
from jaeger.seqops.maps import CODONS


def test_csv_dataset(tmp_path):
    path = tmp_path / "data.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([1, "ATG" * 200])
        writer.writerow([0, "AAA" * 200])
        writer.writerow([2, "TTT" * 200])

    codon_table = {c: i + 1 for i, c in enumerate(CODONS)}
    ds = CSVDataset(path, crop_size=60, num_classes=3, codon_table=codon_table)
    assert len(ds) == 3
    x, y, mask = ds[0]
    assert x.shape[0] == 6
    assert y.shape == (3,)
    assert mask.shape == x.shape
    assert y[1] == 1.0
