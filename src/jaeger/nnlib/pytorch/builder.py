from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from jaeger.nnlib.pytorch.models import (
    ClassificationHead,
    Embedding,
    JaegerModel,
    ReliabilityHead,
    RepresentationModel,
    SiameseModel,
)


def _resolve_vocab_size(
    embedding_cfg: Dict[str, Any], string_cfg: Dict[str, Any]
) -> int:
    """Infer vocab_size from config, falling back to the codon table length."""
    vocab_size = embedding_cfg.get("vocab_size")
    if vocab_size is not None:
        return int(vocab_size)

    codon_id = string_cfg.get("codon_id")
    table_len = None
    if isinstance(codon_id, (list, tuple)):
        table_len = len(codon_id)
    elif isinstance(codon_id, str):
        import jaeger.seqops.maps as maps

        table = getattr(maps, codon_id, None)
        if isinstance(table, (list, tuple)):
            table_len = len(table)
    if table_len is not None:
        # Index 0 is reserved for padding/unknown, so the embedding size is
        # one larger than the raw codon table length.
        return table_len + 1
    # Standard 6-frame codon vocabulary size (including padding/unknown index 0)
    return 65


def _extract_hidden_units(hidden_layers: List[Dict[str, Any]]) -> List[int]:
    """Return intermediate dense unit sizes from a Keras-style layer list."""
    units: List[int] = []
    # The final layer produces the output classes, so exclude it.
    for layer in hidden_layers[:-1]:
        cfg = layer.get("config", {})
        if "units" in cfg:
            units.append(int(cfg["units"]))
    return units


def _extract_dropout_rate(hidden_layers: List[Dict[str, Any]]) -> float:
    """Return the first explicit dropout rate found in the layer list, if any."""
    for layer in hidden_layers:
        if layer.get("name", "").lower() == "dropout":
            return float(layer.get("config", {}).get("rate", 0.0))
    return 0.0


class _ClassifierPipeline(nn.Module):
    """Runs the representation model and classification head, propagating masks."""

    def __init__(self, rep_model: nn.Module, head: nn.Module):
        super().__init__()
        self.rep_model = rep_model
        self.head = head

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embedding = self.rep_model(x, mask)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        return self.head(embedding)


class ModelBuilder:
    """Build PyTorch Jaeger models from YAML config."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self.classifier_out_dim = int(self.model_cfg.get("classifier_out_dim", 0))
        self.reliability_out_dim = int(self.model_cfg.get("reliability_out_dim", 0))

    def build_fragment_classifier(self) -> Dict[str, nn.Module]:
        if "classifier" not in self.model_cfg:
            raise ValueError(
                "classifier config is required for build_fragment_classifier"
            )

        models: Dict[str, nn.Module] = {}

        embedding_cfg = self.model_cfg.get("embedding", {})
        string_cfg = self.model_cfg.get("string_processor", {})
        embedding = Embedding(
            input_type=embedding_cfg.get("input_type"),
            vocab_size=_resolve_vocab_size(embedding_cfg, string_cfg),
            embedding_size=embedding_cfg.get("embedding_size", 4),
            use_embedding_layer=embedding_cfg.get("use_embedding_layer", False),
            onehot_dim=embedding_cfg.get("onehot_dim"),
        )

        rep_cfg = self.model_cfg.get("representation_learner", {})
        is_siamese = (
            self.model_cfg.get("model_type") == "siamese"
            or "branch_layers" in rep_cfg
        )
        if is_siamese:
            rep_model = SiameseModel(
                embedding=embedding,
                branch_layers=rep_cfg.get("branch_layers", []),
                pooling=rep_cfg.get("pooling"),
            )
        else:
            rep_model = RepresentationModel(
                embedding=embedding,
                hidden_layers=rep_cfg.get("hidden_layers", []),
                pooling=rep_cfg.get("pooling", "average"),
            )
        models["rep_model"] = rep_model

        cls_cfg = self.model_cfg["classifier"]
        input_dim = cls_cfg.get("input_shape", rep_model.output_dim)
        if isinstance(input_dim, (list, tuple)) and len(input_dim) == 1:
            input_dim = input_dim[0]
        head = ClassificationHead(
            input_dim=input_dim,
            num_classes=self.classifier_out_dim,
            hidden_units=_extract_hidden_units(cls_cfg.get("hidden_layers", [])),
            dropout=_extract_dropout_rate(cls_cfg.get("hidden_layers", []))
            or cls_cfg.get("dropout", 0.0),
        )
        models["classification_head"] = head
        models["jaeger_classifier"] = _ClassifierPipeline(rep_model, head)

        if "reliability_model" in self.model_cfg:
            rel_cfg = self.model_cfg["reliability_model"]
            rel_input_dim = rel_cfg.get("input_shape", rep_model.nmd_dim)
            if isinstance(rel_input_dim, (list, tuple)) and len(rel_input_dim) == 1:
                rel_input_dim = rel_input_dim[0]
            rel_head = ReliabilityHead(
                input_dim=rel_input_dim,
                num_classes=self.reliability_out_dim,
                hidden_units=_extract_hidden_units(rel_cfg.get("hidden_layers", [])),
                dropout=_extract_dropout_rate(rel_cfg.get("hidden_layers", []))
                or rel_cfg.get("dropout", 0.0),
            )
            models["reliability_head"] = rel_head

        models["jaeger_model"] = JaegerModel(
            rep_model=rep_model,
            classification_head=models["classification_head"],
            reliability_head=models.get("reliability_head"),
        )
        return models

    def compile_model(
        self,
        models: Dict[str, nn.Module],
        train_branch: str = "classifier",
        class_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        opt_class = self._get_optimizer(opt_name)

        if train_branch == "classifier":
            model = models["jaeger_classifier"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = nn.CrossEntropyLoss(weight=class_weights)
            return model, optimizer, loss
        if train_branch == "reliability":
            raise NotImplementedError(
                "reliability training branch is not implemented yet"
            )
        if train_branch == "pretrain":
            raise NotImplementedError("pretrain training branch is not implemented yet")
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
