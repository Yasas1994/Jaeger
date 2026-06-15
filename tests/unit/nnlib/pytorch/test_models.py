import pytest
import torch
from jaeger.nnlib.pytorch.models import (
    ClassificationHead,
    Embedding,
    JaegerModel,
    ProjectionHead,
    ReliabilityHead,
    RepresentationModel,
    SiameseModel,
)


def test_embedding_translated_with_embedding_layer():
    emb = Embedding(
        input_type="translated",
        vocab_size=65,
        embedding_size=32,
        use_embedding_layer=True,
    )
    x = torch.randint(0, 65, (2, 6, 50))
    out = emb(x)
    assert out.shape == (2, 6, 50, 32)


def test_embedding_translated_without_embedding_layer():
    emb = Embedding(
        input_type="translated",
        vocab_size=65,
        embedding_size=32,
        use_embedding_layer=False,
    )
    x = torch.randn(2, 6, 50, 65)
    out = emb(x)
    assert out.shape == (2, 6, 50, 32)


def test_embedding_with_positional_embeddings():
    emb_pos = Embedding(
        input_type="translated",
        vocab_size=65,
        embedding_size=32,
        use_embedding_layer=True,
        use_positional_embeddings=True,
        positional_embedding_length=10000,
    )
    emb_no_pos = Embedding(
        input_type="translated",
        vocab_size=65,
        embedding_size=32,
        use_embedding_layer=True,
        use_positional_embeddings=False,
    )
    x = torch.randint(0, 65, (2, 6, 50))
    out_pos = emb_pos(x)
    out_no_pos = emb_no_pos(x)
    assert out_pos.shape == (2, 6, 50, 32)
    assert out_no_pos.shape == (2, 6, 50, 32)
    assert not torch.allclose(out_pos, out_no_pos)


def test_projection_head_output_shape():
    head = ProjectionHead(input_dim=64, projection_dim=128)
    x = torch.randn(2, 64)
    out = head(x)
    assert out.shape == (2, 128)


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
    embedding = Embedding(
        input_type="translated",
        vocab_size=65,
        embedding_size=32,
        use_embedding_layer=True,
    )
    rep_model = RepresentationModel(
        embedding=embedding,
        hidden_layers=[
            {
                "name": "masked_conv1d",
                "config": {"filters": 16, "kernel_size": 3, "padding": "same"},
            },
            {
                "name": "masked_batchnorm",
                "config": {"num_features": 16, "return_nmd": True},
            },
        ],
        pooling="average",
    )
    classification_head = ClassificationHead(
        input_dim=16, num_classes=3, hidden_units=[8]
    )
    reliability_head = ReliabilityHead(input_dim=16, num_classes=1, hidden_units=[8])
    model = JaegerModel(
        rep_model=rep_model,
        classification_head=classification_head,
        reliability_head=reliability_head,
    )

    x = torch.randint(0, 65, (2, 6, 50))
    mask = torch.ones(2, 6, 50, dtype=torch.bool)
    out = model(x, mask)
    assert out["prediction"].shape == (2, 3)
    assert out["embedding"].shape == (2, 16)
    assert out["nmd"].shape == (2, 16)
    assert out["reliability"].shape == (2, 1)


def test_embedding_nucleotide_onehot_dim_4():
    """Padding and N must map to an all-zero one-hot vector."""
    emb = Embedding(
        input_type="nucleotide",
        vocab_size=6,
        embedding_size=4,
        use_embedding_layer=False,
        onehot_dim=4,
    )
    # tokens: pad=0, A=1, T=2, G=3, C=4, N=5
    x = torch.arange(6).view(6, 1)
    out = emb(x)
    assert out.shape == (6, 1, 4)
    assert torch.allclose(out[0], torch.zeros(4))   # padding
    assert torch.allclose(out[5], torch.zeros(4))   # N
    assert torch.allclose(out[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(out[4], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_embedding_nucleotide_default_onehot_dim_excludes_padding():
    """Without onehot_dim, default to vocab_size-1 and keep padding zero."""
    emb = Embedding(
        input_type="nucleotide",
        vocab_size=6,
        embedding_size=5,
        use_embedding_layer=False,
    )
    x = torch.arange(6).view(6, 1)
    out = emb(x)
    assert out.shape == (6, 1, 5)
    assert torch.allclose(out[0], torch.zeros(5))


def test_siamese_model_forward_shape():
    """SiameseModel averages two branch outputs."""
    embedding = Embedding(
        input_type="nucleotide",
        vocab_size=6,
        embedding_size=4,
        use_embedding_layer=False,
        onehot_dim=4,
    )
    branch_layers = [
        {"name": "permute", "config": {"dims": [0, 2, 1]}},
        {"name": "conv1d", "config": {"in_channels": 4, "out_channels": 8, "kernel_size": 3}},
        {"name": "relu"},
        {"name": "adaptive_max_pool1d", "config": {"output_size": 1}},
        {"name": "squeeze_last"},
        {"name": "linear", "config": {"in_features": 8, "out_features": 16}},
    ]
    model = SiameseModel(embedding=embedding, branch_layers=branch_layers)
    x = torch.randint(0, 6, (2, 2, 20))
    out = model(x)
    assert out.shape == (2, 16)


def test_siamese_model_branch_must_end_with_linear():
    """SiameseModel needs a final linear layer to infer output_dim."""
    embedding = Embedding(
        input_type="nucleotide",
        vocab_size=6,
        embedding_size=4,
        use_embedding_layer=False,
        onehot_dim=4,
    )
    with pytest.raises(ValueError, match="linear layer"):
        SiameseModel(embedding=embedding, branch_layers=[{"name": "relu"}])
