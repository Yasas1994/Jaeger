# Generalized Neural Mean Discrepancy (NMD) Layer Design

## Goal

Improve the reliability head by giving it discrepancy signals from **multiple depths** in the representation learner. Today, NMD is produced only by `MaskedBatchNorm(return_nmd=True)` inside a residual block. We want a standalone `NMDLayer` that can be attached to any intermediate feature map, and a configurable merge step that combines all NMD vectors before they enter the reliability head.

## Background

- `MaskedBatchNorm` in `src/jaeger/nnlib/v2/layers.py` already computes an NMD vector: the per-example channel mean (mask-aware) minus the learned/reference channel mean. NMD here refers to Neural Mean Discrepancy (see arXiv:2104.11408).
- `ResidualBlock_wrapper` and `_build_block` in `src/jaeger/nnlib/builder.py` collect a single NMD tensor and expose it as `models["rep_model"].output[1]`.
- The reliability head currently receives that single NMD vector directly.

## Design

### High-level architecture

```
Input
  │
  ▼
Representation Learner
  │    ┌─────────────┐
  ├───►│ NMDLayer    │──► nmd_1
  │    └─────────────┘
  │    ┌─────────────┐
  ├───►│ NMDLayer    │──► nmd_2
  │    └─────────────┘
  │         ...
  │    ┌─────────────┐
  └───►│ NMDLayer    │──► nmd_n
       └─────────────┘
  │
  ▼
Global Pooling
  │
  ├───► Classification Head ──► prediction
  │
  └───► Merge NMDs (concat/sum/mean/max/weighted)
           │
           ▼
        Reliability Head ──► reliability
```

### Components

1. **`NMDLayer`**
   - A new Keras layer in `src/jaeger/nnlib/v2/layers.py`.
   - Computes the same per-channel mean-difference statistic as `MaskedBatchNorm(return_nmd=True)`.
   - Inputs: a feature map of shape `(batch, length, channels)` or `(batch, length, frames, channels)`.
   - Outputs: a vector of shape `(batch, channels)`.
   - Supports masking.
   - Configurable `epsilon` and `momentum` for the running mean.
   - Serializes/deserializes correctly via `get_config`.

2. **`NMDMerge` layer / builder helper**
   - Takes a list of NMD tensors (each `(batch, channels_i)`) and combines them.
   - Supported modes:
     - `concat` (default): concatenate along the channel axis.
     - `sum`: element-wise sum after projecting each NMD to a common dimension.
     - `mean`: element-wise mean after projecting each NMD to a common dimension.
     - `max`: element-wise max after projecting each NMD to a common dimension.
     - `weighted`: learned per-layer scalar weights applied before summing.
   - For fixed-dimension modes (`sum`, `mean`, `max`, `weighted`), each NMD vector is first passed through a learned linear projection to `reliability_model.input_shape`.

3. **Builder changes**
   - `_build_block` already collects NMD tensors in a list.
   - Extend layer lookup to recognize the new `nmd` layer name and route it to `NMDLayer`.
   - When `reliability_model` is configured, collect all NMD tensors and apply the configured merge before feeding the reliability head.
   - Validate that the merged output dimension matches `reliability_model.input_shape`.
   - Expose merged reliability output in `jaeger_reliability` and `jaeger_model`.

### Config schema

```yaml
representation_learner:
  hidden_layers:
  - name: masked_conv1d
    config:
      filters: 32
      kernel_size: 7
  - name: nmd
    config:
      epsilon: 1e-5
      momentum: 0.9
  - name: residual_block
    config:
      block_size: 2
      filters: 32
  - name: nmd
    config:
      epsilon: 1e-5
  pooling: max

reliability_model:
  merge:
    mode: concat        # concat | sum | mean | max | weighted
    axis: -1
  input_shape: 64       # must match merged dimension
  hidden_layers:
  - name: dense
    config:
      units: 8
      activation: gelu
  - name: dense
    config:
      units: 1
      activation: null
      dtype: float32
```

### Backward compatibility

- Existing `MaskedBatchNorm(return_nmd=True)` and `ResidualBlock(return_nmd=True)` keep working unchanged.
- Existing configs without `nmd` layers produce the same model graph as before.
- The new merge logic is only active when:
  - `reliability_model` is present, **and**
  - at least one `nmd` layer is present, **or**
  - a legacy `return_nmd` tensor is present.
- If only a legacy single NMD exists, the merge step becomes a no-op pass-through.

## Error handling

- If `merge.mode` is unsupported, raise `ValueError` at build time.
- If the merged tensor shape does not match `reliability_model.input_shape`, raise a clear `ValueError` with both shapes.
- If `reliability_model` is configured but no NMD tensor is produced, raise a clear `ValueError` telling the user to add an `nmd` layer or set `return_nmd: true`.

## Testing

- **Unit tests**
  - `NMDLayer` forward pass on 3D and 4D inputs with and without mask.
  - Each merge mode (`concat`, `sum`, `mean`, `max`, `weighted`) produces the expected output shape.
  - `get_config` round-trip serialization.
- **Integration tests**
  - Builder produces a `jaeger_reliability` model with the correct input shape for each merge mode.
  - Builder raises the expected errors for invalid merge configs.
  - Existing configs without NMD layers still build identical graphs (snapshot or shape comparison).

## Open questions

None at this time. All clarifying questions were resolved during brainstorming.
