"""Dataset/DataLoader builders from Jaeger YAML training configs."""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import ConcatDataset, DataLoader, Dataset

from jaeger.dataops.pytorch.collate import pad_collate
from jaeger.dataops.pytorch.dataset_csv import CSVDataset
from jaeger.dataops.pytorch.dataset_numpy import NumpyFullDataset, NumpyRawDataset
from jaeger.seqops.maps import CODONS


def build_datasets(
    config: Dict[str, Any], branch: str = "classifier"
) -> Dict[str, DataLoader]:
    """Build training and validation DataLoaders for a config branch.

    The returned DataLoaders use :func:`pad_collate`, so they support
    variable-length samples by padding each batch to the longest sequence
    length in that batch.

    Parameters
    ----------
    config:
        Parsed Jaeger training configuration.
    branch:
        Either ``classifier`` or ``reliability``. Determines which data
        entries are read from ``config["training"]``.

    Returns
    -------
    Dictionary mapping ``"train"`` and ``"validation"`` to
    :class:`torch.utils.data.DataLoader` instances.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    string_cfg = model_cfg.get("string_processor", {})
    batch_size = train_cfg.get("batch_size", 32)
    out_dim_key = (
        "classifier_out_dim" if branch == "classifier" else "reliability_out_dim"
    )
    num_classes = model_cfg.get(out_dim_key, 3)

    data_key = (
        "fragment_classifier_data"
        if branch == "classifier"
        else "fragment_reliability_data"
    )
    data_cfg = train_cfg.get(data_key, {})

    codon_table = {c: i + 1 for i, c in enumerate(CODONS)}

    datasets: Dict[str, Dataset] = {}
    for split in ["train", "validation"]:
        entries = data_cfg.get(split, [])
        paths = [p for entry in entries for p in entry.get("path", [])]
        if not paths:
            raise ValueError(f"No paths found for {branch}/{split}")

        data_format = string_cfg.get("data_format", "numpy_full")
        input_key = string_cfg.get(
            "input_key",
            "nucleotide"
            if model_cfg.get("embedding", {}).get("input_type") == "nucleotide"
            else "translated",
        )
        if data_format == "numpy_full":
            split_datasets: list[Dataset] = [
                NumpyFullDataset(path, input_key=input_key) for path in paths
            ]
            datasets[split] = (
                split_datasets[0]
                if len(split_datasets) == 1
                else ConcatDataset(split_datasets)
            )
        elif data_format == "numpy_raw":
            split_datasets = [
                NumpyRawDataset(
                    path,
                    crop_size=string_cfg.get("crop_size", 500),
                    num_classes=num_classes,
                    codon_table=codon_table,
                    shuffle=string_cfg.get("shuffle", True),
                    mutate=string_cfg.get("mutate", False),
                    mutation_rate=string_cfg.get("mutation_rate", 0.1),
                    shuffle_frames=string_cfg.get("shuffle_frames", False),
                )
                for path in paths
            ]
            datasets[split] = (
                split_datasets[0]
                if len(split_datasets) == 1
                else ConcatDataset(split_datasets)
            )
        elif data_format == "csv":
            split_datasets = [
                CSVDataset(
                    path,
                    crop_size=string_cfg.get("crop_size", 500),
                    num_classes=num_classes,
                    codon_table=codon_table,
                    shuffle=string_cfg.get("shuffle", False),
                    mutate=string_cfg.get("mutate", False),
                    mutation_rate=string_cfg.get("mutation_rate", 0.1),
                    shuffle_frames=string_cfg.get("shuffle_frames", False),
                    label_first=True,
                )
                for path in paths
            ]
            datasets[split] = (
                split_datasets[0]
                if len(split_datasets) == 1
                else ConcatDataset(split_datasets)
            )
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=pad_collate,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate,
        ),
    }
