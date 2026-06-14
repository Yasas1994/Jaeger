"""Dataset/DataLoader builders from Jaeger YAML training configs."""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import DataLoader

from jaeger.data.pytorch.dataset_numpy import NumpyFullDataset


def build_datasets(
    config: Dict[str, Any], branch: str = "classifier"
) -> Dict[str, DataLoader]:
    """Build training and validation DataLoaders for a config branch.

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

    data_key = (
        "fragment_classifier_data"
        if branch == "classifier"
        else "fragment_reliability_data"
    )
    data_cfg = train_cfg.get(data_key, {})

    datasets: Dict[str, NumpyFullDataset] = {}
    for split in ["train", "validation"]:
        entries = data_cfg.get(split, [])
        paths = [p for entry in entries for p in entry.get("path", [])]
        if not paths:
            raise ValueError(f"No paths found for {branch}/{split}")

        data_format = string_cfg.get("data_format", "numpy_full")
        if data_format == "numpy_full":
            datasets[split] = NumpyFullDataset(paths[0])
        else:
            raise ValueError(f"Unsupported data_format in Task 11: {data_format}")

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
