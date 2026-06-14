from typing import Any, Dict

import torch
import torch.nn as nn

from jaeger.nnlib.pytorch.models import (
    ClassificationHead,
    Embedding,
    JaegerModel,
    ReliabilityHead,
    RepresentationModel,
)


class ModelBuilder:
    """Build PyTorch Jaeger models from YAML config."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self.classifier_out_dim = int(self.model_cfg.get("classifier_out_dim", 0))
        self.reliability_out_dim = int(self.model_cfg.get("reliability_out_dim", 0))

    def build_fragment_classifier(self) -> Dict[str, nn.Module]:
        models: Dict[str, nn.Module] = {}

        embedding_cfg = self.model_cfg.get("embedding", {})
        embedding = Embedding(
            input_type=embedding_cfg.get("input_type"),
            vocab_size=embedding_cfg.get("vocab_size"),
            embedding_size=embedding_cfg.get("embedding_size", 4),
            use_embedding_layer=embedding_cfg.get("use_embedding_layer", False),
        )

        rep_cfg = self.model_cfg.get("representation_learner", {})
        rep_model = RepresentationModel(
            embedding=embedding,
            hidden_layers=rep_cfg.get("hidden_layers", []),
            pooling=rep_cfg.get("pooling", "average"),
        )
        models["rep_model"] = rep_model

        if "classifier" in self.model_cfg:
            cls_cfg = self.model_cfg["classifier"]
            head = ClassificationHead(
                input_dim=cls_cfg.get("input_shape", rep_model.output_dim),
                num_classes=self.classifier_out_dim,
                hidden_units=[
                    layer["config"]["units"]
                    for layer in cls_cfg.get("hidden_layers", [])[:-1]
                ],
            )
            models["classification_head"] = head
            models["jaeger_classifier"] = nn.Sequential(rep_model, head)

        if "reliability_model" in self.model_cfg:
            rel_cfg = self.model_cfg["reliability_model"]
            rel_head = ReliabilityHead(
                input_dim=rel_cfg.get("input_shape", rep_model.nmd_dim),
                num_classes=self.reliability_out_dim,
                hidden_units=[
                    layer["config"]["units"]
                    for layer in rel_cfg.get("hidden_layers", [])[:-1]
                ],
            )
            models["reliability_head"] = rel_head

        models["jaeger_model"] = JaegerModel(
            rep_model=rep_model,
            classification_head=models["classification_head"],
            reliability_head=models.get("reliability_head"),
        )
        return models

    def compile_model(
        self, models: Dict[str, nn.Module], train_branch: str = "classifier"
    ) -> tuple:
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        opt_class = self._get_optimizer(opt_name)

        if train_branch == "classifier":
            model = models["jaeger_classifier"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = nn.CrossEntropyLoss()
            return model, optimizer, loss
        if train_branch == "reliability":
            model = models["jaeger_reliability"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = nn.BCEWithLogitsLoss()
            return model, optimizer, loss
        if train_branch == "pretrain":
            model = models["jaeger_projection"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = models["arcface_loss"]
            return model, optimizer, loss
        raise ValueError(f"Unknown train_branch: {train_branch}")

    def get_metrics(self, branch: str = "classifier") -> Dict[str, Any]:
        from jaeger.nnlib.pytorch.metrics import PrecisionForClass, RecallForClass

        metrics = {}
        out_dim = (
            self.classifier_out_dim
            if branch == "classifier"
            else self.reliability_out_dim
        )
        for cls in range(out_dim):
            metrics[f"precision_class_{cls}"] = PrecisionForClass(class_id=cls)
            metrics[f"recall_class_{cls}"] = RecallForClass(class_id=cls)
        return metrics

    def _get_optimizer(self, name: str) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {name}")
        return optimizers[name]
