# Refactor `jaeger utils optimize-data` Design

## Goal

Replace the current `optimize-data` command's format zoo (`tfrecord`, `numpy_raw`, `numpy_full`, `numpy_raw_variable`) with a single, always-NPZ output that supports three sequence representations: `nucleotide`, `translated`, or `both`. The output is variable-length and padded to the largest crop size. Users can supply multiple crop sizes and a shared stride, and can optionally request one-hot float tensors instead of integer indices.

## Background

- `src/jaeger/cli.py` defines the current `optimize-data` Click command.
- `src/jaeger/commands/utils.py::optimize_data_core` delegates to `src/jaeger/dataops/convert.py::convert_dataset`.
- `dataops/convert.py` already has a Numba-optimized 6-frame codon converter (`_process_batch_numba`) used by the `numpy_full` path.
- There is no existing Numba path for nucleotide-only or `both` outputs.

## CLI interface

```bash
jaeger utils optimize-data -i train.csv -o train.npz --format translated
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-i, --input` | PATH | required | Input CSV file (`label,sequence`) |
| `-o, --output` | TEXT | required | Output `.npz` file |
| `--format` | choice | required | `nucleotide`, `translated`, or `both` |
| `--crop-size` | multiple int | `500` | One or more crop lengths |
| `--stride` | int | `0` | Step between crops; `0` means one crop per sequence |
| `--num-classes` | int | `3` | Number of classes |
| `--num-workers` | int | all CPUs | Parallel workers |
| `--one-hot` | flag | false | Output float one-hot tensors |
| `--pad-int` | int | `0` | Padding value for integer outputs; stored as `pad_int` |
| `--codon-map` | choice | `codon_id` | Codon map: `codon_id`, `aa_id`, `pc5_id`, `murphy10_id`, `cod_id`, `pc2_id` |
| `--nucleotide-map` | JSON | `{"A":1,"G":2,"T":3,"C":4,"N":0}` | Base-to-int mapping |
| `--max-length` | int | `5000` | Deprecated and ignored |
| `--compress` | choice | `fast` | NPZ compression: `fast`, `default`, or `none` |

Old options `--use-embedding-layer` and the old `--format` choices are removed.

## Data flow

1. Read the CSV once and parse `label,sequence` lines.
2. For each sequence, generate crops for every `--crop-size` using the shared `--stride`.
   - If `stride == 0`, the whole sequence (truncated to the largest crop size) is used once.
   - If `stride > 0`, sliding windows of each crop size are extracted along the sequence.
3. Encode each crop:
   - `nucleotide` → 2-strand integer or one-hot encoding.
   - `translated` → 6-frame codon integer or one-hot encoding.
   - `both` → store both arrays.
4. Collect all encoded crops, determine the maximum actual length, and pad shorter crops to that length using `--pad-int`.
5. Save an NPZ file with the arrays, per-sample lengths, integer class labels, and metadata.

## Encoding details

### Nucleotide

- Default map: `A=1, G=2, T=3, C=4, N=0`.
- User-supplied map must cover at least `A, C, G, T, N` and is validated.
- Integer output shape: `(num_samples, 2, max_length)`.
- One-hot output shape: `(num_samples, 2, max_length, 4)`; `N` and padding are all-zero vectors.

### Translated

- Codons are mapped with the chosen codon map from `jaeger.seqops.maps`.
- For single-codon maps (`codon_id`, `aa_id`, `pc5_id`, `murphy10_id`, `pc2_id`), the output uses 3-base codons; the maximum number of codons per frame for a crop of size `C` is `C // 3 - 1`.
- For `cod_id`, overlapping dicodons (6-mers) are encoded; the maximum number of dicodons per frame for a crop of size `C` is `C // 3 - 2`.
- Integer output uses codon-map IDs offset by `+1` so that `0` is reserved for padding/unknown; shape `(num_samples, 6, max_codons)`.
- One-hot output shape: `(num_samples, 6, max_codons, codon_map_size + 1)`, where index `0` is the all-zero padding vector.

### Both

- Both `nucleotide` and `translated` arrays are computed and saved under their respective keys.

## NPZ output schema

| Key | Present when | Shape / Type |
|-----|--------------|--------------|
| `nucleotide` | format is `nucleotide` or `both` | `(N, 2, L)` int32 or `(N, 2, L, 4)` float32 |
| `translated` | format is `translated` or `both` | `(N, 6, L)` int32 or `(N, 6, L, D)` float32 |
| `lengths` | always | `(N,)` int32 |
| `labels` | always | `(N,)` int32, class indices starting from 0 |
| `pad_int` | always | scalar int |
| `nucleotide_map` | format includes `nucleotide` | JSON string |
| `codon_map` | format includes `translated` | string name |
| `crop_sizes` | always | list of ints used |
| `stride` | always | int |

## Error handling

- `--format` must be one of `nucleotide`, `translated`, `both`.
- `--nucleotide-map` must be valid JSON and contain entries for `A, C, G, T, N`.
- `--codon-map` must map to an existing name in `jaeger.seqops.maps`.
- Output path extension is warned if not `.npz`.
- Empty input file raises `ValueError`.

## Backward compatibility

This refactor intentionally removes the old `tfrecord`, `numpy_raw`, `numpy_full`, and `numpy_raw_variable` formats. The command now only emits NPZ files. Any scripts or configs relying on the old formats will need to be updated.

## Testing

- **Unit tests**
  - Nucleotide Numba encoder produces correct integer and one-hot outputs.
  - Translated Numba encoder matches existing `_process_batch_numba` output.
  - Custom nucleotide map is parsed and applied correctly.
- **Integration tests**
  - CLI builds the help text matching the specification.
  - All three formats (`nucleotide`, `translated`, `both`) produce valid NPZ files.
  - `--one-hot` changes dtype/shape correctly.
  - Labels are stored as integer class indices starting from 0.
  - Multiple crop sizes and stride generate the expected number of samples.
  - Invalid nucleotide map and invalid codon map raise clear errors.

## Open questions

None. All clarifying questions were resolved during brainstorming.
