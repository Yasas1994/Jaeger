from typing import List, Optional

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


class RepresentationModel(nn.Module):
    def __init__(self, embedding: nn.Module, hidden_layers: List[dict], pooling: str = "average"):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList()
        self.return_nmd = False
        self.output_dim = None
        self.nmd_dim = None
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        nmd = None
        for block in self.blocks:
            if hasattr(block, "return_nmd") and block.return_nmd:
                x, nmd = block(x, mask)
            else:
                x, mask = block(x, mask)
        pooled = self.pooler(x, mask)
        if nmd is not None:
            return pooled, nmd
        return pooled
