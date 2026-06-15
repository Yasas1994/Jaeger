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


@pytest.fixture
def dummy_input():
    return (
        torch.zeros((1, 2, 32), dtype=torch.long),
        torch.ones((1, 2, 32), dtype=torch.bool),
    )


@pytest.mark.skipif(torchinfo is None, reason="torchinfo not installed")
def test_model_summary_returns_non_empty_string(dummy_input):
    model = _make_tiny_jaeger_model()
    summary = ModelSummary(model, input_data=dummy_input)
    text = summary.summary(branch_label="classifier")
    assert isinstance(text, str)
    assert "Total params" in text
    assert len(text) > 0


def test_model_summary_warns_when_torchinfo_missing(
    monkeypatch, caplog, dummy_input, propagate_jaeger_logger
):
    import jaeger.nnlib.pytorch.summary as summary_mod

    monkeypatch.setattr(summary_mod, "torchinfo", None)
    model = _make_tiny_jaeger_model()
    with caplog.at_level("WARNING"):
        text = ModelSummary(model, input_data=dummy_input).summary(branch_label="classifier")
    assert text == ""
    assert "torchinfo is not installed" in caplog.text


def test_model_summary_warns_when_torchinfo_raises(
    monkeypatch, caplog, dummy_input, propagate_jaeger_logger
):
    import jaeger.nnlib.pytorch.summary as summary_mod

    def _broken_summary(*args, **kwargs):
        raise RuntimeError("boom")

    fake_torchinfo = type("FakeTorchinfo", (), {"summary": staticmethod(_broken_summary)})()
    monkeypatch.setattr(summary_mod, "torchinfo", fake_torchinfo)

    model = _make_tiny_jaeger_model()
    with caplog.at_level("WARNING"):
        text = ModelSummary(model, input_data=dummy_input).summary(branch_label="classifier")
    assert text == ""
    assert "Failed to generate classifier model summary" in caplog.text
