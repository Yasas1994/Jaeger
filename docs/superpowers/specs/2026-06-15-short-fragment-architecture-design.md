# Design: Short-Fragment Architecture Improvements for Jaeger

## Status

Approved — ready for implementation planning.

## Context

Jaeger's default fragment classifier is trained and evaluated on 500 bp crops (and similar lengths). When fragments are shorter — in the **300 bp–2000 bp** range — standard global average pooling dilutes the signal over padded positions, fixed-size convolutions miss fine-grained local patterns, and full axial attention may attend across mostly-padding regions. The goal of this work is to add opt-in architectural components that improve performance on these shorter fragments without breaking existing models or configs.

## Goals

1. Improve feature extraction for 300 bp–2000 bp DNA fragments.
2. Respect actual sequence length during pooling so padding does not dilute signals.
3. Provide local/windowed attention as an alternative to full axial attention.
4. Keep existing configs and SavedModels fully backward compatible.
5. Deliver the changes through new, explicit YAML training configs.

## Non-Goals

- Length-conditional routing or separate short/long pathways.
- Changing default behavior of existing layers.
- Data-augmentation changes (e.g., k-mer masking, random frame selection).
- New loss functions such as focal loss.

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Multi-scale conv + masked pooling + local attention | User-selected architecture changes for 300 bp–2000 bp. |
| Integration style | Add new layer classes, register in builder, expose via YAML | Matches Jaeger's existing config-driven `DynamicModelBuilder` pattern; explicit and composable. |
| Routing | Exclude length-conditioned routing | Explicitly out of scope per user. |
| Config delivery | New training config(s) | Keeps existing 500 bp configs untouched. |
| Backward compatibility | Preserve existing layer signatures and default pooler | Existing SavedModels and `.weights.h5` checkpoints continue to load. |

## Architecture

### New Layer Components

All new layers operate on the same 4D tensor shape used throughout Jaeger v2: `(batch, frames, length, channels)`.

#### `MultiScaleConv1D`

A wrapper layer that runs several parallel `MaskedConv1D` branches and merges their outputs.

- **Input:** `(B, F, L, C)`
- **Branches:** Each branch is configured with `filters`, `kernel_size`, `dilation_rate`, `activation`, and optional `use_bias` / initializers / regularizers. All branches use `padding: same` and `strides: 1` so sequence lengths align.
- **Merge:** `concat` (default) or `add`. For `concat`, output channels equal the sum of branch filters. For `add`, all branches must have the same number of filters.
- **Mask propagation:** The layer forwards a mask. Because all branches preserve length and use valid-only convolution, the input mask can be propagated directly.
- **Registration name in builder:** `multi_scale_conv`

Example YAML:

```yaml
- name: multi_scale_conv
  config:
    branches:
      - filters: 32
        kernel_size: 3
        dilation_rate: 1
      - filters: 32
        kernel_size: 5
        dilation_rate: 1
      - filters: 32
        kernel_size: 3
        dilation_rate: 3
    merge: concat
```

#### `MaskedGlobalAvgPooling2D`

A mask-aware global average pooler for 4D inputs.

- **Input:** `(B, F, L, C)`
- **Behavior:** Multiplies the input by the propagated Keras mask, sums over the frame and length axes, and divides by the number of valid (non-masked) positions.
- **Output:** `(B, C)`
- **Registration name in builder poolers:** `masked_average`

This layer replaces the unmasked `tf.keras.layers.GlobalAveragePooling2D` when `pooling: masked_average` is specified, fixing the signal-dilution problem for padded short fragments.

#### `LocalAttention`

Windowed self-attention along the sequence-length axis.

- **Input:** `(B, F, L, C)`
- **Behavior:** Each position attends only to a local window of `±window_size/2` neighbors along the length axis. The implementation reshapes the tensor so that `MultiHeadAttention` can be applied per window, then restores the original rank-4 shape.
- **Output:** `(B, F, L, C)`
- **Config knobs:** `embed_dim`, `num_heads`, `feed_forward_dim`, `window_size`, `dropout_rate`, `num_blocks`.
- **Registration name in builder:** `local_attention`

This can replace or precede `axial_attention` blocks in short-fragment configs.

### Builder Integration

In `src/jaeger/nnlib/builder.py`:

- Add `"multi_scale_conv": MultiScaleConv1D` to `DynamicModelBuilder._layers`.
- Add `"local_attention": LocalAttention` to `DynamicModelBuilder._layers`.
- Add `"masked_average": MaskedGlobalAvgPooling2D` to `DynamicModelBuilder._get_pooler()`.

The existing `pooling: average` continues to map to `tf.keras.layers.GlobalAveragePooling2D`, so old configs are unaffected.

### New Training Config

Create `train_config/nn_config_300-2000bp_multiscale.yaml`:

- Reuses the same embedding, class map, and training hyperparameters as `nn_config_500bp_axial.yaml`.
- Representation learner uses the new layers:
  - Optional initial `masked_conv1d`.
  - `multi_scale_conv` block.
  - Batch normalization + activation.
  - `local_attention` block.
  - `pooling: masked_average`.
- Classifier head remains unchanged.

### Data Flow

```text
Input: (B, 6, L, 64)
  │
  ▼
multi_scale_conv ──► (B, 6, L, C_total)  where C_total = sum(branch.filters)
  │
  ▼
masked_batchnorm + activation
  │
  ▼
local_attention ──► (B, 6, L, C_total)
  │
  ▼
masked_average pooling ──► (B, C_total)
  │
  ▼
classifier head ──► (B, 3) logits
```

## Error Handling and Invariants

- `MultiScaleConv1D` validates at build time that all branches keep sequence length constant. Any deviation raises `ValueError`.
- `LocalAttention` requires `window_size > 0` and `embed_dim % num_heads == 0`.
- `MaskedGlobalAvgPooling2D` safely handles the case where an entire example is masked by adding epsilon before division.
- Existing configs that do not reference the new layers compile and run exactly as before.

## Backward Compatibility

- No existing layer signatures, names, or default behaviors change.
- Existing SavedModel graphs and `.weights.h5` checkpoints load unchanged.
- The new config is opt-in; old configs do not need modification.

## Testing Plan

### Unit tests

Add tests under `tests/unit/`:

- `test_multi_scale_conv1d.py`
  - Output shape with `concat` and `add` merge modes.
  - Mask propagation through the layer.
  - Mismatched branch filters raises error for `add` merge.
- `test_masked_global_avg_pooling2d.py`
  - Average over valid positions equals expected value.
  - Padded positions do not affect the average.
  - All-masked example returns near-zero without NaN.
- `test_local_attention.py`
  - Output shape matches input shape.
  - Gradient flows through the layer.
  - Invalid `window_size` / `embed_dim` raises `ValueError`.

### Integration test

- `test_builder_short_fragment_config.py`: load `nn_config_300-2000bp_multiscale.yaml` with `DynamicModelBuilder`, build the fragment classifier, compile it, and run a forward pass on synthetic data.

### Smoke test

- Add a short training smoke script or extend an existing one to train the new config for a few steps on synthetic 300 bp–500 bp data and verify loss decreases.

## Files to Create / Modify

### New files

- `src/jaeger/nnlib/v2/layers.py` additions:
  - `MultiScaleConv1D`
  - `MaskedGlobalAvgPooling2D`
  - `LocalAttention`
- `train_config/nn_config_300-2000bp_multiscale.yaml`
- `tests/unit/test_multi_scale_conv1d.py`
- `tests/unit/test_masked_global_avg_pooling2d.py`
- `tests/unit/test_local_attention.py`
- `tests/integration/test_builder_short_fragment_config.py`

### Modified files

- `src/jaeger/nnlib/builder.py`
  - Register `multi_scale_conv` and `local_attention` in `self._layers`.
  - Register `masked_average` in `_get_pooler()`.

## Open Questions

None — design approved by user.

## Appendix: Example Config Snippet

```yaml
representation_learner:
  hidden_layers:
    - name: multi_scale_conv
      config:
        branches:
          - filters: 32
            kernel_size: 3
            dilation_rate: 1
          - filters: 32
            kernel_size: 5
            dilation_rate: 1
          - filters: 32
            kernel_size: 3
            dilation_rate: 3
        merge: concat
    - name: masked_batchnorm
      config:
        return_nmd: false
    - name: activation
      config:
        activation: gelu
    - name: local_attention
      config:
        embed_dim: 96
        num_heads: 4
        feed_forward_dim: 128
        window_size: 32
        dropout_rate: 0.1
        num_blocks: 2
    - name: masked_layernorm
  pooling: masked_average
```
