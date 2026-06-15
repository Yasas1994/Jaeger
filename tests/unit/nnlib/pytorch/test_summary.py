import pytest
import torch
from jaeger.nnlib.pytorch.models import (
    ClassificationHead,
    Embedding,
    JaegerModel,
    RepresentationModel,
)
from jaeger.nnlib.pytorch.summary import ModelSummary

try:
    import torchinfo
except ImportError:
    torchinfo = None


def _make_tiny_jaeger_model():
    """Build a minimal JaegerModel for lightweight summary tests."""
    embedding = Embedding(
        input_type="nucleotide",
        vocab_size=5,
        embedding_size=8,
        use_embedding_layer=False,
        onehot_dim=4,
    )
    rep_model = RepresentationModel(
        embedding=embedding,
        hidden_layers=[
            {"name": "masked_conv1d", "config": {"filters": 8, "kernel_size": 3}},
            {"name": "masked_batchnorm", "config": {}},
        ],
        pooling="average",
    )
    classifier = ClassificationHead(
        input_dim=rep_model.output_dim,
        num_classes=2,
        hidden_units=[],
    )
    return JaegerModel(rep_model=rep_model, classification_head=classifier)


@pytest.mark.skipif(torchinfo is None, reason="torchinfo not installed")
def test_model_summary_returns_non_empty_string():
    model = _make_tiny_jaeger_model()
    dummy = (torch.zeros((1, 2, 32), dtype=torch.long), torch.ones((1, 2, 32), dtype=torch.bool))
    summary = ModelSummary(model, input_data=dummy)
    text = summary.summary(branch_label="classifier")
    assert isinstance(text, str)
    assert "Total params" in text
    assert len(text) > 0
