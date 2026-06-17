# Streaming, optionally-unpadded `optimize-data` design

## Context

`jaeger utils optimize-data` currently reads the whole CSV into memory, encodes all crops (including all sliding-window crops introduced by `--overlap`), concatenates everything, and only then writes the final `.npz`. With large inputs or high overlap this can exhaust RAM before the file is written.

The training pipeline already uses `tf.data.Dataset.padded_batch`, so it does not require globally-padded inputs. This lets us make two related changes:

1. Stream encoding in batches so memory use is bounded by a budget.
2. Store each crop at its actual length by default, letting the loader pad per-batch at training time.

## Goals

- Add `--max-memory-mb` to `optimize-data` so the converter streams when the encoded output would exceed a memory budget.
- Add `--pad` to `optimize-data`; default is **no padding**.
- Keep producing a single `.npz` file as the final artifact.
- Maintain backward compatibility: old dense `.npz` files must still load.
- Minimize changes to the rest of the training pipeline.

## Non-goals

- Change the final file format away from `.npz`.
- Eliminate the final save spike entirely (assembling the final `.npz` still needs the final arrays in memory).
- Support arbitrary ragged shapes for outputs that are not produced by `optimize-data`.

## Design

### 1. CLI

Two new options on `jaeger utils optimize-data`:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--max-memory-mb` | `int` | auto ≈ 75 % of available RAM | Budget in MB for encoded output buffers. `0` disables streaming and keeps the legacy all-in-memory path. |
| `--pad` | flag | `False` | Produce globally padded dense arrays. When omitted, each crop is stored at its actual length. |

`--overlap` (already implemented) continues to drive per-crop strides.

### 2. Memory estimate and streaming trigger

The converter already knows:

- `crop_sizes`
- `strides` (one per crop size)
- `fmt` (`nucleotide`, `translated`, `both`)
- `one_hot`
- `codon_map_len`

A safe per-output-row byte estimate is computed from the largest crop size, assuming worst-case padding. With overlap active the row multiplier is `ceil(max_crop / max_stride)`; otherwise it is `1`.

The budget is:

- `--max-memory-mb` if the user supplied a positive value.
- 75 % of available RAM if `--max-memory-mb` was not supplied.
- Unlimited (legacy path) if `--max-memory-mb` is `0`.

If the estimated output memory is less than or equal to the budget, the existing `_convert_to_npz` fast path is used. Otherwise the converter streams.

### 3. Streaming fast path

When streaming:

1. Compute `batch_rows = max(1, int(budget * 0.8 / per_row_bytes))`.
2. Open the input CSV as a line iterator and accumulate at most `batch_rows` lines per batch.
3. Encode the batch with the existing `_process_chunk_npz` worker function.
4. If `--pad` is `False`, trim each crop to its actual length (`lengths[i]`, `translated_lengths[i]`) and store the batch as 1-D object arrays (`nucleotide`, `translated`).
   If `--pad` is `True`, pad the batch arrays to the global maxima as the legacy converter does.
5. Write the batch arrays to temporary `.npy` shards in a temp directory next to the output file.
6. Repeat until the CSV is exhausted.
7. Concatenate all shards for each output key into the final arrays.
   Object-array shards concatenate into a longer object array; dense shards concatenate normally.
8. Write the final `.npz` and remove the temp directory in a `finally` block.

### 4. Output schema

All `.npz` files continue to contain:

- `labels` — `(N,)` int32
- `lengths` — `(N,)` int32
- `translated_lengths` — `(N,)` int32
- `crop_sizes` — `(C,)` int32
- `strides` — `(C,)` int32
- `pad_int` — scalar int32
- `padded` — **new** scalar bool
- `nucleotide_map` — JSON string, when `nucleotide` is present
- `codon_map` — string, when `translated` is present

The representation of `nucleotide` and `translated` depends on `--pad`:

- `--pad` (`padded=True`): dense arrays as today.
- No `--pad` (`padded=False`): 1-D object arrays of shape `(N,)` where each element is the encoded crop at its true length:
  - nucleotide integer: `(2, L)` int32
  - nucleotide one-hot: `(2, L, 4)` float32
  - translated integer codon: `(6, L)` int32
  - translated integer dicodon: `(6, L)` int32
  - translated one-hot: `(6, L, depth)` float32

### 5. Loader changes

`jaeger.data.loaders._load_numpy_dataset` branches on the dtype of `nucleotide` and/or `translated`:

- **Dense path:** existing logic is unchanged.
- **Ragged path:** when an array has object dtype, build a generator that yields `(features, label)` tuples with variable-length tensors. One-hot conversion is applied per sample inside the generator if `seq_onehot` is requested.

The function returns an unbatched `tf.data.Dataset`. `train.py` already calls `.padded_batch()` using the dataset's `element_spec`, so variable shapes are padded to the per-batch maximum automatically.

### 6. Training integration

`train.py` currently calls `.cache()` on numpy datasets. For ragged generator datasets this would rematerialize the data in memory, so the cache call will be skipped when the loaded dataset is ragged.

No other training code changes are required; the model's `Masking` layer already ignores zero-padded positions.

### 7. Error handling and cleanup

- The temp shard directory is removed on both success and failure.
- A partially written output file is deleted if the final `.npz` save fails.
- `--max-memory-mb` must allow at least one input row per batch; otherwise the converter raises a clear error.

### 8. Testing

- **Round-trip parity:** convert a small CSV with `--pad` and without `--pad`; assert the loaded padded samples are identical.
- **Streaming parity:** force a tiny `--max-memory-mb` and compare the result to the non-streaming path.
- **Loader shapes:** verify `padded_batch` produces correct shapes for ragged integer and one-hot crops.
- **Backward compatibility:** existing dense `.npz` files still load and batch correctly.
- **Regression:** all existing `tests/unit/test_dataops_convert.py` tests pass.

## Trade-offs

- Object arrays require `np.load(..., allow_pickle=True)`. This is acceptable because the elements are NumPy arrays, not arbitrary Python objects.
- The final `.npz` save still holds the full arrays in memory, but the peak is now final-arrays-only rather than final + all intermediate batches.
- Defaulting to unpadded output changes the shape of newly produced `.npz` files. Old files continue to work, but external scripts that read the raw arrays must be updated.

## Open questions

None remaining; the design has been approved by the user.
