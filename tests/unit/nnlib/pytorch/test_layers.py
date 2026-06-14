import torch
from jaeger.nnlib.pytorch.layers import (
    AxialAttention,
    GeLU,
    GatedFrameGlobalMaxPooling,
    MaskedBatchNorm,
    MaskedConv1D,
    MaskedLayerNorm,
)


def test_gelu_matches_torch_nn_gelu():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    layer = GeLU()
    out = layer(x)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(out, expected, atol=1e-6)


def test_gelu_preserves_shape():
    x = torch.randn(2, 4, 8)
    layer = GeLU()
    out = layer(x)
    assert out.shape == x.shape


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


def test_masked_conv1d_valid_padding():
    x = torch.randn(2, 3, 10, 4)
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="valid")
    out, out_mask = layer(x, mask=None)
    assert out.shape == (2, 3, 8, 8)


def test_masked_conv1d_stride_and_dilation():
    x = torch.randn(2, 3, 10, 4)
    layer = MaskedConv1D(
        filters=8, kernel_size=3, padding="same", strides=2, dilation_rate=2
    )
    out, out_mask = layer(x, mask=None)
    assert out.shape == (2, 3, 5, 8)
    assert out_mask is None


def test_masked_conv1d_no_mask():
    x = torch.randn(2, 3, 10, 4)
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="same")
    out, out_mask = layer(x, mask=None)
    assert out.shape == (2, 3, 10, 8)
    assert out_mask is None


def test_masked_conv1d_activation():
    x = torch.abs(torch.randn(2, 3, 10, 4)) + 1.0
    layer = MaskedConv1D(
        filters=8, kernel_size=3, padding="same", activation="relu"
    )
    out, _ = layer(x, mask=None)
    assert (out >= 0).all()


def test_masked_conv1d_interior_mask_hole():
    x = torch.randn(1, 1, 10, 4)
    mask = torch.ones(1, 1, 10, dtype=torch.bool)
    mask[0, 0, 5] = False
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="same")
    _, out_mask = layer(x, mask)
    assert not out_mask[0, 0, 5]
    assert out_mask[0, 0, 2]
    assert out_mask[0, 0, 7]


def test_masked_conv1d_parameters_before_forward():
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="same")
    params = list(layer.parameters())
    assert len(params) > 0


def test_masked_conv1d_state_dict_before_forward():
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="same")
    state = layer.state_dict()
    assert "conv.weight" in state
    assert "conv.bias" in state


def test_masked_batchnorm_output_shape_and_nmd():
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    mask[0, :, 5:] = False
    layer = MaskedBatchNorm(num_features=4, return_nmd=True)
    out, nmd = layer(x, mask)
    assert out.shape == (2, 6, 10, 4)
    assert nmd.shape == (2, 4)


def test_masked_batchnorm_no_mask():
    x = torch.randn(2, 6, 10, 4)
    layer = MaskedBatchNorm(num_features=4)
    out, _ = layer(x, mask=None)
    assert out.shape == (2, 6, 10, 4)
    # running stats should have updated in train mode
    assert not torch.allclose(layer.running_mean, torch.zeros(4))
    assert not torch.allclose(layer.running_var, torch.ones(4))


def test_masked_batchnorm_eval_uses_running_stats():
    x = torch.randn(2, 6, 10, 4)
    layer = MaskedBatchNorm(num_features=4)
    layer.train()
    _ = layer(x)
    running_mean = layer.running_mean.clone()
    running_var = layer.running_var.clone()
    layer.eval()
    out, _ = layer(x, mask=None)
    assert out.shape == (2, 6, 10, 4)
    # manual normalization using running stats should match
    expected = (x - running_mean.view(1, 1, 1, -1)) / torch.sqrt(
        running_var.view(1, 1, 1, -1) + layer.eps
    )
    expected = expected * layer.gamma.view(1, 1, 1, -1) + layer.beta.view(1, 1, 1, -1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_masked_layer_norm_shape():
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    mask[0, :, 5:] = False
    layer = MaskedLayerNorm(num_features=4)
    out = layer(x, mask)
    assert out.shape == (2, 6, 10, 4)
    # masked positions should be zeroed
    assert torch.allclose(out[0, :, 5:, :], torch.zeros_like(out[0, :, 5:, :]))


def test_masked_layer_norm_no_mask():
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    layer = MaskedLayerNorm(num_features=4)
    out_masked = layer(x, mask)
    out_unmasked = layer(x, mask=None)
    assert torch.allclose(out_masked, out_unmasked, atol=1e-5)


def test_masked_layer_norm_numerical():
    # Simple 1 position, 4 channels, all valid.
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    layer = MaskedLayerNorm(num_features=4)
    with torch.no_grad():
        layer.gamma.fill_(1.0)
        layer.beta.fill_(0.0)
    out = layer(x)
    mean = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + layer.eps)
    assert torch.allclose(out, expected, atol=1e-5)


def test_gated_frame_pooling_shape():
    x = torch.randn(2, 6, 10, 4)
    layer = GatedFrameGlobalMaxPooling(return_gate=False)
    out = layer(x)
    assert out.shape == (2, 4)


def test_axial_attention_shape():
    x = torch.randn(2, 6, 10, 4)
    layer = AxialAttention(embed_dim=4, num_heads=2)
    out, mask = layer(x)
    assert out.shape == (2, 6, 10, 4)
