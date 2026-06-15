from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        onehot_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_type = input_type
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_embedding_layer = use_embedding_layer
        self.use_positional_embeddings = use_positional_embeddings
        self.onehot_dim = onehot_dim

        if input_type == "translated":
            if use_embedding_layer:
                self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
                nn.init.orthogonal_(self.embed.weight)
            else:
                self.embed = nn.Linear(vocab_size, embedding_size, bias=False)
                nn.init.orthogonal_(self.embed.weight)
        elif input_type == "nucleotide":
            # Nucleotide input can be provided as a float one-hot tensor or as
            # integer indices. Integer indices are one-hot encoded and then
            # optionally projected to ``embedding_size``. Padding (token 0) is
            # always mapped to an all-zero vector.
            effective_onehot_dim = (
                onehot_dim if onehot_dim is not None else (vocab_size - 1)
            )
            self.effective_onehot_dim = effective_onehot_dim
            if use_embedding_layer:
                self.embed = None
            elif effective_onehot_dim == embedding_size:
                self.embed = None
            else:
                self.embed = nn.Linear(effective_onehot_dim, embedding_size, bias=False)
                nn.init.orthogonal_(self.embed.weight)

            # Mapping matrix from integer tokens to one-hot vectors. Row 0
            # (padding) is all-zero; token i maps to channel i-1 when it fits.
            mapping = torch.zeros(vocab_size, effective_onehot_dim)
            for token in range(1, vocab_size):
                channel = token - 1
                if channel < effective_onehot_dim:
                    mapping[token, channel] = 1.0
            self.register_buffer("_nucleotide_onehot_mapping", mapping)
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        if use_positional_embeddings:
            from jaeger.nnlib.pytorch.layers import SinusoidalPositionEmbedding

            self.positional = SinusoidalPositionEmbedding(positional_embedding_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape depends on input_type; caller must pass correct shape
        if self.input_type == "translated":
            if self.use_embedding_layer:
                x = self.embed(x)
            else:
                if not torch.is_floating_point(x):
                    x = F.one_hot(x, num_classes=self.embed.in_features).to(
                        self.embed.weight.dtype
                    )
                x = self.embed(x)
        elif self.input_type == "nucleotide":
            if not torch.is_floating_point(x):
                x = F.embedding(
                    x, self._nucleotide_onehot_mapping.to(x.device)
                ).to(torch.float32)
            if self.embed is not None:
                x = self.embed(x)
        if self.use_positional_embeddings:
            x = self.positional(x)
        return x


class ClassificationHead(nn.Module):
    """Dense head for class prediction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_units: List[int],
        dropout: float = 0.0,
    ):
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

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_units: List[int],
        dropout: float = 0.0,
    ):
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
    def __init__(
        self, embedding: nn.Module, hidden_layers: List[dict], pooling: str = "average"
    ):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList()
        self.return_nmd = False
        current_dim = self._embedding_output_dim(embedding)
        for cfg in hidden_layers:
            name = cfg["name"].lower()
            config = cfg.get("config", {})
            if name == "masked_conv1d":
                block = MaskedConv1D(**config)
                current_dim = block.filters
                self.blocks.append(block)
            elif name == "masked_batchnorm":
                config = dict(config)
                config.setdefault("num_features", current_dim)
                block = MaskedBatchNorm(**config)
                current_dim = block.num_features
                self.blocks.append(block)
            elif name == "masked_layer_norm":
                config = dict(config)
                config.setdefault("num_features", current_dim)
                block = MaskedLayerNorm(**config)
                current_dim = block.num_features
                self.blocks.append(block)
            elif name == "axial_attention":
                config = dict(config)
                config.setdefault("embed_dim", current_dim)
                block = AxialAttention(**config)
                current_dim = block.embed_dim
                self.blocks.append(block)
            elif name == "cross_frame_attention":
                config = dict(config)
                config.setdefault("embed_dim", current_dim)
                block = CrossFrameAttention(**config)
                current_dim = block.embed_dim
                self.blocks.append(block)
            elif name == "transformer_encoder":
                config = dict(config)
                config.setdefault("embed_dim", current_dim)
                block = TransformerEncoder(**config)
                current_dim = self._block_output_dim(block)
                self.blocks.append(block)
            elif name == "residual_block":
                # Config-driven residual block: ``config`` defines the internal
                # stack (e.g. ``block_size`` MaskedConv1D layers). If an older
                # style config passes the previous layer directly, wrap it.
                if "layer" in config:
                    block = ResidualBlock(layer=config["layer"])
                else:
                    cfg_copy = dict(config)
                    cfg_copy.setdefault("filters", current_dim)
                    block = ResidualBlock(config=cfg_copy)
                    current_dim = cfg_copy.get("filters", current_dim)
                self.blocks.append(block)
            elif name == "dense":
                config = dict(config)
                if "units" in config:
                    config["out_features"] = config.pop("units")
                config.setdefault("in_features", current_dim)
                block = nn.Linear(**config)
                current_dim = block.out_features
                self.blocks.append(block)
            elif name == "activation":
                self.blocks.append(
                    GeLU() if config.get("activation") == "gelu" else nn.ReLU()
                )
            elif name == "dropout":
                self.blocks.append(nn.Dropout(config.get("rate", 0.0)))
            else:
                raise ValueError(f"Unknown layer type: {name}")
        self.pooler = self._build_pooler(pooling)
        self.output_dim, self.nmd_dim = self._infer_dims()

    @staticmethod
    def _embedding_output_dim(embedding: nn.Module) -> int:
        if isinstance(embedding, Embedding):
            return int(embedding.embedding_size)
        if hasattr(embedding, "embed") and embedding.embed is not None:
            if hasattr(embedding.embed, "embedding_dim"):
                return int(embedding.embed.embedding_dim)
            if hasattr(embedding.embed, "out_features"):
                return int(embedding.embed.out_features)
        return 4

    def _infer_dims(self):
        output_dim = None
        for block in reversed(self.blocks):
            dim = self._block_output_dim(block)
            if dim is not None:
                output_dim = dim
                break

        # Match the forward loop: the last block that returns NMD wins.
        nmd_dim = None
        for block in reversed(self.blocks):
            dim = self._block_nmd_dim(block)
            if dim is not None:
                nmd_dim = dim
                break
        return output_dim, nmd_dim

    @staticmethod
    def _block_output_dim(block: nn.Module) -> Optional[int]:
        if isinstance(block, MaskedConv1D):
            return int(block.filters)
        if isinstance(block, (MaskedBatchNorm, MaskedLayerNorm)):
            return int(block.num_features)
        if isinstance(block, (AxialAttention, CrossFrameAttention)):
            return int(block.embed_dim)
        if isinstance(block, TransformerEncoder):
            return int(block.encoder.layers[0].self_attn.embed_dim)
        if isinstance(block, nn.Linear):
            return int(block.out_features)
        if isinstance(block, ResidualBlock):
            return RepresentationModel._block_output_dim(block.layer)
        return None

    @staticmethod
    def _block_nmd_dim(block: nn.Module) -> Optional[int]:
        if isinstance(block, ResidualBlock):
            return RepresentationModel._block_nmd_dim(block.layer)
        if getattr(block, "return_nmd", False):
            return RepresentationModel._block_output_dim(block)
        return None

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
                out = block(x, mask)
                if isinstance(out, tuple):
                    x, maybe_mask_or_nmd = out
                    if maybe_mask_or_nmd is None:
                        # mask unchanged
                        pass
                    elif getattr(block, "return_nmd", False):
                        nmd = maybe_mask_or_nmd
                    elif (
                        isinstance(maybe_mask_or_nmd, torch.Tensor)
                        and maybe_mask_or_nmd.dtype == torch.bool
                    ):
                        mask = maybe_mask_or_nmd
                else:
                    x = out
            else:
                x = block(x)
        pooled = self.pooler(x, mask)
        if nmd is not None:
            return pooled, nmd
        return pooled


class SiameseModel(nn.Module):
    """Siamese representation model with two identical native-layer branches.

    The input is expected to be a Jaeger nucleotide tensor of shape
    ``(B, 2, L, D)`` (two strands after the embedding layer). Each strand is
    processed by the same branch network and the resulting feature vectors are
    averaged.
    """

    def __init__(
        self,
        embedding: nn.Module,
        branch_layers: List[Dict[str, Any]],
        pooling: Optional[str] = None,
    ):
        super().__init__()
        from jaeger.nnlib.pytorch.layers import NativeSequential

        self.embedding = embedding
        self.branch = NativeSequential(branch_layers)
        self.output_dim = self._infer_branch_output_dim()
        self.nmd_dim = self.output_dim

    def _infer_branch_output_dim(self) -> int:
        """Return the output dimension of the last Linear layer, if known."""
        for layer in reversed(self.branch.layers):
            if isinstance(layer, nn.Linear):
                return int(layer.out_features)
        raise ValueError(
            "SiameseModel branch_layers must end with a linear layer "
            "so the output dimension can be inferred."
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(x)  # (B, 2, L, D)
        x0 = x[:, 0]  # (B, L, D)
        x1 = x[:, 1]  # (B, L, D)

        # Optional: zero padded positions before the branch.
        if mask is not None:
            mask0 = mask[:, 0].unsqueeze(-1).to(x.dtype)
            mask1 = mask[:, 1].unsqueeze(-1).to(x.dtype)
            x0 = x0 * mask0
            x1 = x1 * mask1

        f0, _ = self.branch(x0)
        f1, _ = self.branch(x1)
        return (f0 + f1) * 0.5


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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
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
        result: Dict[str, torch.Tensor] = {
            "prediction": prediction,
            "embedding": embedding,
        }
        if nmd is not None:
            result["nmd"] = nmd
        if gate is not None:
            result["gate"] = gate
        if self.reliability_head is not None and nmd is not None:
            result["reliability"] = self.reliability_head(nmd)
        return result
