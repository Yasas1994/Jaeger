import torch


def _as_class_indices(y_true: torch.Tensor) -> torch.Tensor:
    """Return class-index labels from one-hot or index labels."""
    if y_true.dim() == 2 and y_true.shape[1] > 1:
        return y_true.argmax(dim=-1)
    return y_true.view(-1).long()


class CategoricalAccuracy:
    """Multi-class accuracy."""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_labels = y_pred.argmax(dim=-1)
        true_labels = _as_class_indices(y_true)
        self.correct += int((pred_labels == true_labels).sum().item())
        self.total += y_true.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class BinaryAccuracy:
    """Binary accuracy."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.shape[-1] == 1:
            pred_labels = (torch.sigmoid(y_pred).squeeze(-1) >= self.threshold).long()
        else:
            pred_labels = y_pred.argmax(dim=-1)
        true_labels = _as_class_indices(y_true)
        self.correct += int((pred_labels == true_labels).sum().item())
        self.total += y_true.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class PrecisionForClass:
    def __init__(self, class_id: int):
        self.class_id = class_id
        self.tp = 0
        self.fp = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_labels = y_pred.argmax(dim=-1)
        true_labels = _as_class_indices(y_true)
        cls = self.class_id
        self.tp += int(((pred_labels == cls) & (true_labels == cls)).sum().item())
        self.fp += int(((pred_labels == cls) & (true_labels != cls)).sum().item())

    def compute(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)


class RecallForClass(PrecisionForClass):
    def __init__(self, class_id: int):
        super().__init__(class_id)
        self.fn = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_labels = y_pred.argmax(dim=-1)
        true_labels = _as_class_indices(y_true)
        cls = self.class_id
        self.tp += int(((pred_labels == cls) & (true_labels == cls)).sum().item())
        self.fn += int(((pred_labels != cls) & (true_labels == cls)).sum().item())

    def compute(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)


class _SklearnMetric:
    """Base for metrics that accumulate predictions and use scikit-learn."""

    def __init__(self):
        self.y_preds: list[torch.Tensor] = []
        self.y_trues: list[torch.Tensor] = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.y_preds.append(y_pred.detach().cpu())
        self.y_trues.append(y_true.detach().cpu())

    def _labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        y_pred = torch.cat(self.y_preds)
        y_true = torch.cat(self.y_trues)
        return y_pred, y_true


class Precision(_SklearnMetric):
    """Overall precision (macro-averaged by default)."""

    def __init__(self, average: str = "macro"):
        super().__init__()
        self.average = average

    def compute(self) -> float:
        from sklearn.metrics import precision_score

        y_pred, y_true = self._labels()
        pred_labels = y_pred.argmax(dim=-1).numpy()
        true_labels = _as_class_indices(y_true).numpy()
        return float(
            precision_score(true_labels, pred_labels, average=self.average, zero_division=0)
        )


class Recall(_SklearnMetric):
    """Overall recall (macro-averaged by default)."""

    def __init__(self, average: str = "macro"):
        super().__init__()
        self.average = average

    def compute(self) -> float:
        from sklearn.metrics import recall_score

        y_pred, y_true = self._labels()
        pred_labels = y_pred.argmax(dim=-1).numpy()
        true_labels = _as_class_indices(y_true).numpy()
        return float(
            recall_score(true_labels, pred_labels, average=self.average, zero_division=0)
        )


class AUC(_SklearnMetric):
    """ROC AUC. Multi-class problems use one-vs-rest macro averaging."""

    def __init__(self, multi_class: str = "ovr", average: str = "macro"):
        super().__init__()
        self.multi_class = multi_class
        self.average = average

    def compute(self) -> float:
        from sklearn.metrics import roc_auc_score

        y_pred, y_true = self._labels()
        y_score = torch.softmax(y_pred, dim=-1).numpy()
        true_labels = _as_class_indices(y_true).numpy()
        num_classes = y_score.shape[-1]
        if num_classes == 2:
            y_score = y_score[:, 1]
        return float(
            roc_auc_score(
                true_labels,
                y_score,
                multi_class=self.multi_class if num_classes > 2 else "raise",
                average=self.average,
            )
        )


class AUCForClass(_SklearnMetric):
    """One-vs-rest ROC AUC for a single class."""

    def __init__(self, class_id: int):
        super().__init__()
        self.class_id = class_id

    def compute(self) -> float:
        from sklearn.metrics import roc_auc_score

        y_pred, y_true = self._labels()
        y_score = torch.softmax(y_pred, dim=-1)[:, self.class_id].numpy()
        true_labels = (_as_class_indices(y_true) == self.class_id).long().numpy()
        return float(roc_auc_score(true_labels, y_score))


_METRIC_BUILDERS = {
    "categorical_accuracy": lambda params, num_classes: CategoricalAccuracy(),
    "binary_accuracy": lambda params, num_classes: BinaryAccuracy(),
    "precision": lambda params, num_classes: Precision(average=params.get("average", "macro")),
    "recall": lambda params, num_classes: Recall(average=params.get("average", "macro")),
    "per_class_precision": lambda params, num_classes: {
        f"precision_class_{cls}": PrecisionForClass(class_id=cls)
        for cls in range(num_classes)
    },
    "per_class_recall": lambda params, num_classes: {
        f"recall_class_{cls}": RecallForClass(class_id=cls)
        for cls in range(num_classes)
    },
    "auc": lambda params, num_classes: AUC(
        multi_class=params.get("multi_class", "ovr"),
        average=params.get("average", "macro"),
    ),
    "per_class_auc": lambda params, num_classes: {
        f"auc_class_{cls}": AUCForClass(class_id=cls)
        for cls in range(num_classes)
    },
}


def build_metrics(metrics_cfg: list[dict], num_classes: int) -> dict[str, object]:
    """Build a dictionary of metric objects from a config list.

    Parameters
    ----------
    metrics_cfg:
        List of dicts with ``name`` and optional ``params`` keys.
    num_classes:
        Number of output classes for the branch.

    Returns
    -------
    Mapping from metric name to metric instance.
    """
    metrics: dict[str, object] = {}
    for entry in metrics_cfg or []:
        name = entry.get("name")
        params = entry.get("params") or {}
        builder = _METRIC_BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"Unsupported metric: {name}")
        result = builder(params, num_classes)
        if isinstance(result, dict):
            metrics.update(result)
        else:
            metrics[name] = result
    return metrics
