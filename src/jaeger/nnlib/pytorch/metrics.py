import torch


class PrecisionForClass:
    def __init__(self, class_id: int):
        self.class_id = class_id
        self.tp = 0
        self.fp = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_labels = y_pred.argmax(dim=-1)
        true_labels = y_true.argmax(dim=-1)
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
        true_labels = y_true.argmax(dim=-1)
        cls = self.class_id
        self.tp += int(((pred_labels == cls) & (true_labels == cls)).sum().item())
        self.fn += int(((pred_labels != cls) & (true_labels == cls)).sum().item())

    def compute(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
