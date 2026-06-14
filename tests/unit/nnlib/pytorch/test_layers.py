import torch
import pytest
from jaeger.nnlib.pytorch.layers import GeLU, MaskedConv1D


def test_gelu_matches_torch_nn_gelu():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    layer = GeLU()
    out = layer(x)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(out, expected, atol=1e-6)


def test_masked_conv1d_shape_and_mask():
    # (B=2, F=6, L=10, C=4)
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    mask[0, :, 5:] = False
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="same")
    out, out_mask = layer(x, mask)
    assert out.shape == (2, 6, 10, 8)
    assert out_mask.shape == (2, 6, 10)
    assert not out_mask[0, :, 5:].any()
