from typing import Dict, List, Optional

import torch
import torch.nn as nn

from jaeger.nnlib.pytorch.layers import (
    AxialAttention,
    CrossFrameAttention,
    GatedFrameGlobalMaxPooling,
    GeLU,
    MaskedBatchNorm,
    MaskedConv1D,
    MaskedGlobalAvgPooling,
    MaskedLayerNorm,
    ResidualBlock,
    TransformerEncoder,
)


class Embedding(nn.Module):
    """Translates DNA into 6-frame codon embeddings or nucleotide one-hot."""

    def __init__(
        self,
        input_type: str,
        vocab_size: Optional[int],
        embedding_size: int,
        use_embedding_layer: bool,
        use_positional_embeddings: bool = False,
        positional_embedding_length: Optional[int] = None,
    ):
        super().__init__()
        self.input_type = input_type
        self.use_embedding_layer = use_embedding_layer
        self.use_positional_embeddings = use_positional_embeddings

        if input_type == "translated":
            if use_embedding_layer:
                self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
                nn.init.orthogonal_(self.embed.weight)
            else:
                self.embed = nn.Linear(vocab_size, embedding_size, bias=False)
                nn.init.orthogonal_(self.embed.weight)
        elif input_type == "nucleotide":
            self.embed = None
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        if use_positional_embeddings:
            from jaeger.nnlib.pytorch.layers import SinusoidalPositionEmbedding

            self.positional = SinusoidalPositionEmbedding(positional_embedding_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape depends on input_type; caller must pass correct shape
        if self.input_type == "translated":
            if self.use_embedding_layer:
                return self.embed(x)
            else:
                return self.embed(x)
        return x


class ClassificationHead(nn.Module):
    """Dense head for class prediction."""

    def __init__(self, input_dim: int, num_classes: int, hidden_units: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.LayerNorm(units))
            layers.append(GeLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReliabilityHead(nn.Module):
    """Dense head for confidence estimation."""

    def __init__(self, input_dim: int, num_classes: int, hidden_units: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.LayerNorm(units))
            layers.append(GeLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """Projection head for ArcFace self-supervised pretraining."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            GeLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepresentationModel(nn.Module):
    def __init__(self, embedding: nn.Module, hidden_layers: List[dict], pooling: str = "average"):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList()
        self.return_nmd = False
        for cfg in hidden_layers:
            name = cfg["name"].lower()
            config = cfg.get("config", {})
            if name == "masked_conv1d":
                self.blocks.append(MaskedConv1D(**config))
            elif name == "masked_batchnorm":
                self.blocks.append(MaskedBatchNorm(**config))
            elif name == "masked_layer_norm":
                self.blocks.append(MaskedLayerNorm(**config))
            elif name == "axial_attention":
                self.blocks.append(AxialAttention(**config))
            elif name == "cross_frame_attention":
                self.blocks.append(CrossFrameAttention(**config))
            elif name == "transformer_encoder":
                self.blocks.append(TransformerEncoder(**config))
            elif name == "residual_block":
                inner = self.blocks.pop() if self.blocks else None
                self.blocks.append(ResidualBlock(inner))
            elif name == "dense":
                if "units" in config:
                    config = dict(config)
                    config["out_features"] = config.pop("units")
                self.blocks.append(nn.Linear(**config))
            elif name == "activation":
                self.blocks.append(GeLU() if config.get("activation") == "gelu" else nn.ReLU())
            elif name == "dropout":
                self.blocks.append(nn.Dropout(config.get("rate", 0.0)))
            else:
                raise ValueError(f"Unknown layer type: {name}")
        self.pooler = self._build_pooler(pooling)

    def _build_pooler(self, pooling: str):
        if pooling == "average":
            return MaskedGlobalAvgPooling()
        elif pooling == "gatedframe":
            return GatedFrameGlobalMaxPooling(return_gate=False)
        raise ValueError(f"Unknown pooling: {pooling}")

    def _accepts_mask(self, block):
        return "mask" in block.forward.__code__.co_varnames

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        nmd = None
        for block in self.blocks:
            if self._accepts_mask(block):
                if hasattr(block, "return_nmd") and block.return_nmd:
                    x, nmd = block(x, mask)
                else:
                    x, mask = block(x, mask)
            else:
                x = block(x)
        pooled = self.pooler(x, mask)
        if nmd is not None:
            return pooled, nmd
        return pooled


class JaegerModel(nn.Module):
    """Combined model exposing all outputs for inference."""

    def __init__(
        self,
        rep_model: nn.Module,
        classification_head: nn.Module,
        reliability_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.rep_model = rep_model
        self.classification_head = classification_head
        self.reliability_head = reliability_head

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.rep_model(x, mask)
        if isinstance(outputs, tuple):
            embedding = outputs[0]
            nmd = outputs[1] if len(outputs) > 1 else None
            gate = outputs[2] if len(outputs) > 2 else None
        else:
            embedding = outputs
            nmd = None
            gate = None

        prediction = self.classification_head(embedding)
        result: Dict[str, torch.Tensor] = {"prediction": prediction, "embedding": embedding}
        if nmd is not None:
            result["nmd"] = nmd
        if gate is not None:
            result["gate"] = gate
        if self.reliability_head is not None and nmd is not None:
            result["reliability"] = self.reliability_head(nmd)
        return result
