# Grouped (padding-free) batching for NumPy training data

## Goal

Eliminate wasted compute from padding variable-length sequence crops during training and validation by batching crops of the **same exact length** together. The user can switch between the existing padded batch strategy and the new grouped strategy from the YAML config.

## Scope

- Applies to **NumPy `.npz` data only** (classifier and reliability training).
- CSV pipelines keep the existing `padded_batch` behavior.
- Both training and validation NumPy datasets use the selected strategy.
- Backward compatible: missing `batching` config defaults to `padded`.

## Config schema

New optional block under `string_processor`:

```yaml
string_processor:
  batching:
    strategy: grouped          # "padded" (default, backward compatible) | "grouped"
    length_batch_sizes:        # exact crop length -> batch size
      200: 300
      300: 400
      400: 500
      500: 600
    default_batch_size: 64     # used for any length not listed above
```

When `strategy` is omitted or set to `padded`, the current `padded_batch` path is used unchanged.

## Loader changes

A new boolean parameter `pad_to_max` is added to:

- `_load_numpy_dataset(..., pad_to_max: bool = True)`
- `_load_cropped_numpy_dataset(..., pad_to_max: bool = True)`

### When `pad_to_max=True` (default)

Behavior is identical to today: every crop is padded to the global `max(crop_sizes)` so all batches have a fixed sequence length.

### When `pad_to_max=False`

Crops keep their natural length. Output tensor specs use `None` on the sequence-length axis instead of `max_crop_size`.

Two code paths inside `_load_cropped_numpy_dataset` are updated:

1. **Dense slicing path** (`_slice_crop`): skips the `tf.pad` step.
2. **Object-array path** (`_sample_features`): skips `np.pad` and leaves length dynamic.

In both paths, the post-processing `tf.ensure_shape` calls use `None` for the length dimension.

## Grouped batching helper

A new helper in `src/jaeger/commands/train.py`:

```python
def _apply_grouped_batching(
    ds: tf.data.Dataset,
    batching_cfg: dict[str, Any],
    num_replicas: int,
    feature_key: str,
) -> tf.data.Dataset:
    ...
```

Implementation uses `tf.data.Dataset.group_by_window` with a per-key window size:

- `key_func`: returns the length of the sequence on axis 1 of `features[feature_key]` as a `tf.int64` scalar.
- `window_size_func`: maps each length key to its configured batch size via a `tf.lookup.StaticHashTable` built from `length_batch_sizes` (with `default_batch_size` as the default value).
- `reduce_func`: receives one length group and returns `dataset.batch(effective_batch_size)`.

For multi-GPU training, each per-length batch size is rounded down to the nearest multiple of `num_replicas`:

```python
effective_batch_size = (batch_size // num_replicas) * num_replicas
```

If the rounded value is zero, the length falls back to the rounded `default_batch_size`. A final filter removes any undersized remainder windows so `MirroredStrategy` always sees full, replica-divisible batches.

## Train wiring

Only the NumPy branches in `train.py` branch on `batching.strategy`:

```python
batching_cfg = string_processor_config.get("batching", {})
if batching_cfg.get("strategy") == "grouped":
    ds = _apply_grouped_batching(
        ds,
        batching_cfg,
        num_replicas=strategy.num_replicas_in_sync,
        feature_key=next(iter(ds.element_spec[0].keys())),
    )
else:
    ds = ds.padded_batch(batch_size=..., padded_shapes=..., drop_remainder=multi_gpu)
```

This is applied to:

- Classifier training NumPy dataset
- Classifier validation NumPy dataset
- Reliability training NumPy dataset
- Reliability validation NumPy dataset

CSV branches continue to use `padded_batch`.

## Error handling and validation

| Condition | Behavior |
|-----------|----------|
| Unknown `strategy` value | `ValueError` |
| `length_batch_sizes` key is not an integer | `ValueError` |
| `strategy: grouped` with missing `default_batch_size` | `ValueError` |
| Length group has fewer examples than effective batch size | Dropped by the remainder filter |
| `num_replicas > 1` and a configured batch size is not divisible by replica count | Silently rounded down; fallback used if zero |

## Testing plan

1. **Unit test — unique length per batch**
   - Build a synthetic `.npz` with mixed-length crops.
   - Load with `pad_to_max=False` and `_apply_grouped_batching`.
   - Assert every produced batch contains a single unique length.

2. **Unit test — multi-GPU rounding**
   - Mock `num_replicas=4` with `length_batch_sizes: {500: 127, 600: 200}`.
   - Assert effective batch sizes are `124` and `200`.

3. **Smoke test — training parity**
   - Run a few steps with `strategy=padded` and `strategy=grouped`.
   - Verify no shape errors and that model weights update.

4. **Integration test — Zeus config**
   - Add the new `batching` block to the Zeus 1500 bp config.
   - Run a short build/smoke test before the full Slurm submission.

## Open questions / future work

- CSV grouped batching: not included in this design because CSV preprocessing is length-agnostic and would require parsing each line before batching.
- Partial crops at sequence ends are grouped by their exact length using `default_batch_size`, which may create small groups.
