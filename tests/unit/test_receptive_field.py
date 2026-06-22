from jaeger.utils.receptive_field import (
    compute_receptive_field,
    receptive_field_summary,
)


class TestComputeReceptiveField:
    def test_single_conv(self):
        layers = [
            {"name": "masked_conv1d", "config": {"kernel_size": 7, "dilation_rate": 1}}
        ]
        rf, trace = compute_receptive_field(layers)
        assert rf == 7
        assert trace == [("input", 1), ("masked_conv1d", 7)]

    def test_residual_block(self):
        layers = [
            {
                "name": "residual_block",
                "config": {
                    "block_size": 2,
                    "kernel_size": 5,
                    "dilation_rate": 3,
                },
            }
        ]
        rf, _ = compute_receptive_field(layers)
        # Two convolutions each adding (5 - 1) * 3 = 12
        assert rf == 1 + 2 * 12

    def test_mixed_layers(self):
        layers = [
            {"name": "masked_conv1d", "config": {"kernel_size": 7, "dilation_rate": 1}},
            {"name": "nmd"},
            {"name": "masked_layernorm"},
            {
                "name": "residual_block",
                "config": {
                    "block_size": 2,
                    "kernel_size": 5,
                    "dilation_rate": 3,
                },
            },
            {"name": "activation", "config": {"activation": "gelu"}},
        ]
        rf, trace = compute_receptive_field(layers)
        assert rf == 7 + 2 * 12
        # Layers that do not change RF keep the previous value.
        assert trace[2] == ("nmd", 7)
        assert trace[3] == ("masked_layernorm", 7)
        assert trace[4] == ("residual_block", 7 + 2 * 12)
        assert trace[5] == ("activation", 7 + 2 * 12)

    def test_brain_like_architecture(self):
        layers = [
            {"name": "masked_conv1d", "config": {"kernel_size": 7, "dilation_rate": 1}},
            {"name": "nmd"},
            {"name": "masked_layernorm"},
            {"name": "activation", "config": {"activation": "gelu"}},
            {
                "name": "residual_block",
                "config": {
                    "block_size": 2,
                    "kernel_size": 5,
                    "dilation_rate": 3,
                },
            },
            {"name": "nmd"},
            {"name": "masked_layernorm"},
            {"name": "activation", "config": {"activation": "gelu"}},
            {
                "name": "residual_block",
                "config": {
                    "block_size": 2,
                    "kernel_size": 5,
                    "dilation_rate": 3,
                },
            },
            {"name": "nmd"},
            {"name": "masked_layernorm"},
            {"name": "activation", "config": {"activation": "gelu"}},
            {
                "name": "residual_block",
                "config": {
                    "block_size": 2,
                    "kernel_size": 5,
                    "dilation_rate": 3,
                },
            },
            {"name": "nmd"},
            {"name": "masked_layernorm"},
            {"name": "activation", "config": {"activation": "gelu"}},
        ]
        rf, _ = compute_receptive_field(layers)
        # 1 + (7-1)*1 + 3 blocks * 2 convs * (5-1)*3
        assert rf == 1 + 6 + 3 * 2 * 12


class TestReceptiveFieldSummary:
    def test_summary_includes_crop_size(self):
        layers = [
            {"name": "masked_conv1d", "config": {"kernel_size": 7, "dilation_rate": 1}}
        ]
        summary = receptive_field_summary(layers, crop_size=500)
        assert "Receptive field: 7" in summary
        assert "crop size: 500" in summary

    def test_summary_without_crop_size(self):
        layers = [
            {"name": "masked_conv1d", "config": {"kernel_size": 7, "dilation_rate": 1}}
        ]
        summary = receptive_field_summary(layers)
        assert "Receptive field: 7" in summary
        assert "crop size" not in summary
