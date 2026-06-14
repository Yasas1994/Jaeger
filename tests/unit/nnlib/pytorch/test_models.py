import torch
from jaeger.nnlib.pytorch.models import ClassificationHead, Embedding, JaegerModel, ReliabilityHead, RepresentationModel


def test_embedding_translated_with_embedding_layer():
    emb = Embedding(input_type="translated", vocab_size=65, embedding_size=32, use_embedding_layer=True)
    x = torch.randint(0, 65, (2, 6, 50))
    out = emb(x)
    assert out.shape == (2, 6, 50, 32)


def test_embedding_translated_without_embedding_layer():
    emb = Embedding(input_type="translated", vocab_size=65, embedding_size=32, use_embedding_layer=False)
    x = torch.randn(2, 6, 50, 65)
    out = emb(x)
    assert out.shape == (2, 6, 50, 32)


def test_classification_head_output_shape():
    head = ClassificationHead(input_dim=64, num_classes=3, hidden_units=[128, 64])
    x = torch.randn(2, 64)
    out = head(x)
    assert out.shape == (2, 3)


def test_reliability_head_output_shape():
    head = ReliabilityHead(input_dim=64, num_classes=1, hidden_units=[32])
    x = torch.randn(2, 64)
    out = head(x)
    assert out.shape == (2, 1)


def test_jaeger_model_forward():
    embedding = Embedding(input_type="translated", vocab_size=65, embedding_size=32, use_embedding_layer=True)
    rep_model = RepresentationModel(
        embedding=embedding,
        hidden_layers=[
            {"name": "masked_conv1d", "config": {"filters": 16, "kernel_size": 3, "padding": "same"}},
            {"name": "masked_batchnorm", "config": {"num_features": 16, "return_nmd": True}},
        ],
        pooling="average",
    )
    classification_head = ClassificationHead(input_dim=16, num_classes=3, hidden_units=[8])
    reliability_head = ReliabilityHead(input_dim=16, num_classes=1, hidden_units=[8])
    model = JaegerModel(rep_model=rep_model, classification_head=classification_head, reliability_head=reliability_head)

    x = torch.randint(0, 65, (2, 6, 50))
    mask = torch.ones(2, 6, 50, dtype=torch.bool)
    out = model(x, mask)
    assert out["prediction"].shape == (2, 3)
    assert out["embedding"].shape == (2, 16)
    assert out["nmd"].shape == (2, 16)
    assert out["reliability"].shape == (2, 1)
