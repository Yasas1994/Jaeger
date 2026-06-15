import torch

from jaeger.nnlib.pytorch.losses import ArcFaceLoss


def test_arcface_loss_shape():
    loss = ArcFaceLoss(num_classes=3, embedding_dim=64, margin=0.5, scale=30.0)
    labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    embeddings = torch.randn(3, 64)
    out = loss(labels, embeddings)
    assert out.ndim == 0


def test_arcface_loss_class_index_labels():
    loss = ArcFaceLoss(
        num_classes=3, embedding_dim=64, margin=0.5, scale=30.0, onehot=False
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    embeddings = torch.randn(3, 64)
    out = loss(labels, embeddings)
    assert out.ndim == 0


def test_arcface_loss_changes_with_margin():
    labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    embeddings = torch.randn(3, 64)

    loss_a = ArcFaceLoss(num_classes=3, embedding_dim=64, margin=0.1, scale=30.0)
    loss_b = ArcFaceLoss(num_classes=3, embedding_dim=64, margin=0.9, scale=30.0)

    out_a = loss_a(labels, embeddings)
    out_b = loss_b(labels, embeddings)
    assert out_a.item() != out_b.item()
