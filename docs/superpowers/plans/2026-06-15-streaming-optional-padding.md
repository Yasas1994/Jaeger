# Streaming, optionally-unpadded `optimize-data` implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `jaeger utils optimize-data` so it can stream encoding in memory-bounded batches and store crops at their actual length by default, while the loader pads batches at training time.

**Architecture:** Add a streaming branch to `jaeger.dataops.convert.convert_dataset` that shards encoded batches to temporary `.npy` files and concatenates them at the end. Centralize the "finalize one batch" logic so both the legacy fast path and the streaming path can produce either padded dense arrays or unpadded object arrays. Update the numpy loader to detect object arrays and yield variable-length tensors via `tf.data.Dataset.from_generator`, then rely on `train.py`'s existing `.padded_batch()` call.

**Tech Stack:** Python, NumPy, TensorFlow / `tf.data`, Click, pytest.

---

## Files that will change

| File | Responsibility |
|---|---|
| `src/jaeger/cli.py` | Add `--max-memory-mb` and `--pad` options to `optimize-data`. |
| `src/jaeger/commands/utils.py` | Pass the new options through `optimize_data_core` to `convert_dataset`. |
| `src/jaeger/dataops/convert.py` | Memory estimation, batch finalization, streaming converter, dispatcher. |
| `src/jaeger/data/loaders.py` | Detect and load ragged (object-array) NPZ files. |
| `src/jaeger/commands/train.py` | Skip `.cache()` for ragged datasets. |
| `tests/unit/test_dataops_convert.py` | Tests for no-pad, streaming, and overlap. |
| `tests/unit/test_data_loaders.py` | Tests for ragged loader path. |

---

## Task 1: Add per-row memory helper

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Write the helper and update imports**

Add `itertools` to the imports at the top of `src/jaeger/dataops/convert.py` if it is not already imported, then append these helpers after `_estimate_onehot_memory`:

```python
def _estimate_output_bytes_per_row(
    crop_size: int,
    fmt: str,
    one_hot: bool,
    codon_map_len: int | None,
) -> int:
    """Safe upper-bound bytes for one output crop, assuming worst-case length."""
    crop_size = max(0, crop_size)
    per_row = 0
    if fmt in ("nucleotide", "both"):
        if one_hot:
            per_row += (
                2 * crop_size * 4 * np.dtype(np.float32).itemsize
            )
        else:
            per_row += 2 * crop_size * np.dtype(np.int32).itemsize

    if fmt in ("translated", "both") and codon_map_len is not None:
        # Use codon length (crop_size // 3 - 1) because it is >= dicodon length.
        seq_len = max(0, crop_size // 3 - 1)
        if one_hot:
            per_row += (
                6 * seq_len * (codon_map_len + 1) * np.dtype(np.float32).itemsize
            )
        else:
            per_row += 6 * seq_len * np.dtype(np.int32).itemsize

    return max(1, per_row)
```

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestMemoryEstimate:
    def test_per_row_nucleotide_integer(self):
        b = convert._estimate_output_bytes_per_row(
            12, "nucleotide", one_hot=False, codon_map_len=None
        )
        assert b == 2 * 12 * 4

    def test_per_row_nucleotide_onehot(self):
        b = convert._estimate_output_bytes_per_row(
            12, "nucleotide", one_hot=True, codon_map_len=None
        )
        assert b == 2 * 12 * 4 * 4

    def test_per_row_both(self):
        b = convert._estimate_output_bytes_per_row(
            12, "both", one_hot=False, codon_map_len=64
        )
        seq_len = 12 // 3 - 1
        expected = 2 * 12 * 4 + 6 * seq_len * 4
        assert b == expected
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/test_dataops_convert.py::TestMemoryEstimate -v
```
Expected: `AttributeError: module 'jaeger.dataops.convert' has no attribute '_estimate_output_bytes_per_row'`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_dataops_convert.py::TestMemoryEstimate -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat(convert): add per-row output memory estimate helper"
```

---

## Task 2: Add batch finalization helper

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Write the helper**

Insert `_finalize_batch_arrays` in `src/jaeger/dataops/convert.py` right after `_process_chunk_npz`. It takes the raw chunk result and returns a dict ready to be saved (without scalar metadata).

```python
def _finalize_batch_arrays(
    result: dict,
    fmt: str,
    crop_sizes: list[int],
    one_hot: bool,
    codon_map_len: int | None,
    pad: bool,
    pad_int: int,
    nucleotide_map: dict[str, int],
    codon_map_name: str,
) -> dict[str, np.ndarray]:
    """Turn a processed chunk into save-ready arrays.

    ``result`` is the dict returned by ``_process_chunk_npz``. If ``pad`` is
    True the arrays are padded to the global maximum length (legacy behavior).
    If ``pad`` is False each crop is trimmed to its actual length and stored
    in a 1-D object array.
    """
    save_dict: dict[str, np.ndarray] = {
        "labels": result["labels"],
        "lengths": result["lengths"],
        "translated_lengths": result["translated_lengths"],
    }

    def _to_object_array(items: list[np.ndarray]) -> np.ndarray:
        arr = np.empty(len(items), dtype=object)
        arr[:] = items
        return arr

    if fmt in ("nucleotide", "both"):
        arrays = result["nucleotide"]
        if arrays:
            if pad:
                max_len = max(a.shape[-1] for a in arrays)
                if one_hot:
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=0.0) for a in arrays
                    ]
                    save_dict["nucleotide"] = np.concatenate(padded, axis=0)
                else:
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=pad_int)
                        for a in arrays
                    ]
                    save_dict["nucleotide"] = np.concatenate(padded, axis=0)
            else:
                items: list[np.ndarray] = []
                offset = 0
                for arr, crop_size in zip(arrays, crop_sizes):
                    n = arr.shape[0]
                    lens = result["lengths"][offset : offset + n]
                    for i in range(n):
                        L = int(lens[i])
                        if one_hot:
                            items.append(arr[i, :, :L, :])
                        else:
                            items.append(arr[i, :, :L])
                    offset += n
                save_dict["nucleotide"] = _to_object_array(items)
        else:
            save_dict["nucleotide"] = (
                np.empty((0,), dtype=np.float32)
                if one_hot
                else np.empty((0,), dtype=np.int32)
            )
        save_dict["nucleotide_map"] = np.array(json.dumps(nucleotide_map))

    if fmt in ("translated", "both"):
        arrays = result["translated"]
        if arrays:
            if pad:
                max_len = max(a.shape[-1] for a in arrays)
                if one_hot and codon_map_len is not None:
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=0) for a in arrays
                    ]
                    stacked = np.concatenate(padded, axis=0)
                    save_dict["translated"] = _one_hot_integer(
                        stacked, codon_map_len + 1
                    )
                else:
                    padded = [
                        _pad_axis(a, max_len, axis=-1, pad_value=0) for a in arrays
                    ]
                    save_dict["translated"] = np.concatenate(padded, axis=0)
            else:
                items: list[np.ndarray] = []
                offset = 0
                for arr, crop_size in zip(arrays, crop_sizes):
                    n = arr.shape[0]
                    if one_hot and codon_map_len is not None:
                        arr = _one_hot_integer(arr, codon_map_len + 1)
                    lens = result["translated_lengths"][offset : offset + n]
                    for i in range(n):
                        L = int(lens[i])
                        if one_hot:
                            items.append(arr[i, :, :L, :])
                        else:
                            items.append(arr[i, :, :L])
                    offset += n
                save_dict["translated"] = _to_object_array(items)
        else:
            save_dict["translated"] = (
                np.empty((0,), dtype=np.float32)
                if one_hot
                else np.empty((0,), dtype=np.int32)
            )
        save_dict["codon_map"] = np.array(codon_map_name)

    return save_dict
```

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestFinalizeBatchArrays:
    def test_unpadded_nucleotide_object_array(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGCATGC\n1,GGGG\n")
        result = convert._process_chunk_npz(
            lines=csv.read_text().splitlines(),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            one_hot=False,
            pad_int=0,
            nucleotide_lookups=convert._build_nucleotide_lookups(
                {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
            ),
            codon_lut=np.empty(0, dtype=np.int32),
            codon_map_len=64,
            standard_codon_lut3=np.empty(0, dtype=np.int32),
            dicodon_lut=np.empty(0, dtype=np.int32),
            ascii_lut=np.zeros(256, dtype=np.int8),
            comp_lut=np.zeros(256, dtype=np.int8),
        )
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="nucleotide",
            crop_sizes=[12],
            one_hot=False,
            codon_map_len=None,
            pad=False,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["nucleotide"].dtype == object
        assert finalized["nucleotide"][0].shape == (2, 12)
        assert finalized["nucleotide"][1].shape == (2, 4)
```

- [ ] **Step 3: Run test**

```bash
pytest tests/unit/test_dataops_convert.py::TestFinalizeBatchArrays -v
```
Expected: PASS (after the helper exists).

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat(convert): add batch finalization helper for padded and unpadded output"
```

---

## Task 3: Update the legacy fast path to use the helper

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Replace the final save logic in `_convert_to_npz`**

Replace lines 962–1010 of `src/jaeger/dataops/convert.py` (the construction of `save_dict`) with:

```python
    save_dict = _finalize_batch_arrays(
        result,
        fmt=fmt,
        crop_sizes=crop_sizes,
        one_hot=one_hot,
        codon_map_len=codon_map_len,
        pad=pad,
        pad_int=pad_int,
        nucleotide_map=nucleotide_map,
        codon_map_name=codon_map_name,
    )
    save_dict.update(
        {
            "pad_int": np.int32(pad_int),
            "crop_sizes": np.array(crop_sizes, dtype=np.int32),
            "strides": np.array(strides, dtype=np.int32),
            "padded": np.bool_(pad),
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _save_npz(output_path, save_dict, compress)
```

- [ ] **Step 2: Add the `pad` parameter to `_convert_to_npz`**

Change the signature of `_convert_to_npz` to include `pad: bool = False`.

- [ ] **Step 3: Add a no-pad round-trip test**

Append to `tests/unit/test_dataops_convert.py`:

```python
    def test_nucleotide_unpadded(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGC", "1,GGGG"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            pad=False,
        )
        data = np.load(out, allow_pickle=True)
        assert data["padded"].item() is False
        assert data["nucleotide"].dtype == object
        assert data["nucleotide"][1].shape == (2, 4)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_dataops_convert.py::TestConvertToNpz -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat(convert): support unpadded output in legacy fast path"
```

---

## Task 4: Implement the streaming converter

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Add helper imports**

Add to the top of `src/jaeger/dataops/convert.py`:

```python
import itertools
import shutil
import tempfile
```

- [ ] **Step 2: Add `_convert_to_npz_streaming`**

Insert this function after `_convert_to_npz`:

```python
def _convert_to_npz_streaming(
    input_path: str,
    output_path: str,
    fmt: str,
    crop_sizes: list[int],
    strides: list[int],
    num_classes: int,
    num_workers: int | None,
    one_hot: bool,
    pad_int: int,
    codon_map_name: str,
    nucleotide_map: dict[str, int],
    compress: str,
    max_memory_bytes: int,
    pad: bool,
) -> None:
    """Memory-bounded CSV -> NPZ converter that shards encoded batches to disk."""
    max_crop = max(crop_sizes) if crop_sizes else 500
    codon_map_len: int | None = None
    if fmt in ("translated", "both"):
        codon_map_len = len(_get_codon_map(codon_map_name))

    per_row = _estimate_output_bytes_per_row(
        max_crop, fmt, one_hot, codon_map_len
    )
    batch_rows = max(1, int(max_memory_bytes * 0.8 / per_row))

    nucleotide_lookups = _build_nucleotide_lookups(nucleotide_map)
    _, ascii_lut, comp_lut = _build_numba_lookups()

    codon_map = _get_codon_map(codon_map_name)
    if len(codon_map) == 4096:
        standard_codon_lut3 = _build_standard_codon_lut3()
        dicodon_lut = _build_dicodon_lut(codon_map)
        codon_lut = np.empty(0, dtype=np.int32)
    else:
        codon_lut = _build_codon_lut(codon_map)
        standard_codon_lut3 = np.empty(0, dtype=np.int32)
        dicodon_lut = np.empty(0, dtype=np.int32)

    worker_kwargs = {
        "fmt": fmt,
        "crop_sizes": crop_sizes,
        "strides": strides,
        "one_hot": one_hot,
        "pad_int": pad_int,
        "nucleotide_lookups": nucleotide_lookups,
        "codon_lut": codon_lut,
        "codon_map_len": codon_map_len if codon_map_len is not None else 0,
        "standard_codon_lut3": standard_codon_lut3,
        "dicodon_lut": dicodon_lut,
        "ascii_lut": ascii_lut,
        "comp_lut": comp_lut,
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="jaeger_optimize_", dir=Path(output_path).parent))
    try:
        keys: list[str] = []
        batch_idx = 0
        with open(input_path) as f:
            while True:
                batch_lines = list(itertools.islice(f, batch_rows))
                if not batch_lines:
                    break

                result = _process_chunk_npz(batch_lines, **worker_kwargs)
                batch_dict = _finalize_batch_arrays(
                    result,
                    fmt=fmt,
                    crop_sizes=crop_sizes,
                    one_hot=one_hot,
                    codon_map_len=codon_map_len,
                    pad=pad,
                    pad_int=pad_int,
                    nucleotide_map=nucleotide_map,
                    codon_map_name=codon_map_name,
                )

                if not keys:
                    keys = [k for k in batch_dict.keys() if k not in ("nucleotide_map", "codon_map")]

                for key in keys:
                    shard_path = tmp_dir / f"batch_{batch_idx:05d}_{key}.npy"
                    np.save(shard_path, batch_dict[key])
                batch_idx += 1

        if batch_idx == 0:
            raise ValueError(f"Input file is empty: {input_path}")

        save_dict: dict[str, np.ndarray] = {}
        for key in keys:
            shard_paths = sorted(tmp_dir.glob(f"batch_*_{key}.npy"))
            shards = [np.load(p, allow_pickle=True) for p in shard_paths]
            save_dict[key] = np.concatenate(shards, axis=0)

        save_dict.update(
            {
                "pad_int": np.int32(pad_int),
                "crop_sizes": np.array(crop_sizes, dtype=np.int32),
                "strides": np.array(strides, dtype=np.int32),
                "padded": np.bool_(pad),
            }
        )
        if fmt in ("nucleotide", "both"):
            save_dict["nucleotide_map"] = np.array(json.dumps(nucleotide_map))
        if fmt in ("translated", "both"):
            save_dict["codon_map"] = np.array(codon_map_name)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _save_npz(output_path, save_dict, compress)
    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 3: Write the streaming parity test**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestStreamingConvert:
    def _csv(self, tmp_path: Path, lines: list[str]) -> str:
        path = tmp_path / "input.csv"
        path.write_text("\n".join(lines))
        return str(path)

    def test_streaming_matches_fast_path(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0," + "A" * 25, "1," + "G" * 25])
        fast = tmp_path / "fast.npz"
        stream = tmp_path / "stream.npz"
        convert.convert_dataset(
            input_path=csv,
            output_path=str(fast),
            format="nucleotide",
            crop_size=20,
            stride=10,
            num_classes=2,
            num_workers=1,
            max_memory_mb=0,
            pad=True,
        )
        convert.convert_dataset(
            input_path=csv,
            output_path=str(stream),
            format="nucleotide",
            crop_size=20,
            stride=10,
            num_classes=2,
            num_workers=1,
            max_memory_mb=1,
            pad=True,
        )
        fast_data = np.load(fast)
        stream_data = np.load(stream)
        assert np.array_equal(fast_data["labels"], stream_data["labels"])
        assert np.array_equal(fast_data["nucleotide"], stream_data["nucleotide"])
```

- [ ] **Step 4: Run the streaming test**

```bash
pytest tests/unit/test_dataops_convert.py::TestStreamingConvert -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat(convert): implement memory-bounded streaming NPZ converter"
```

---

## Task 5: Wire the dispatcher to choose the streaming path

**Files:**
- Modify: `src/jaeger/dataops/convert.py`

- [ ] **Step 1: Update `convert_dataset` signature and logic**

Change the signature of `convert_dataset` to add `max_memory_mb: int | None = None` and `pad: bool = False`.

Replace the memory-guard block in `convert_dataset` (around lines 1115–1137) with:

```python
    nucleotide_map_dict = _parse_nucleotide_map(nucleotide_map)

    logger.info(f"Converting {input_path} -> {output_path}")
    logger.info(
        f"Format: {format}, Crop sizes: {crop_sizes}, Strides: {strides}, "
        f"Num classes: {num_classes}, Pad: {pad}"
    )

    codon_map_len: int | None = None
    if format in ("translated", "both"):
        codon_map_arr = _get_codon_map(codon_map)
        codon_map_len = len(codon_map_arr)

    total_lines = _count_lines(input_path)
    max_crop_idx = int(np.argmax(crop_sizes))
    max_crop = crop_sizes[max_crop_idx]
    max_stride = strides[max_crop_idx]
    multiplier = 1
    if max_stride > 0 and max_crop > 0:
        multiplier = math.ceil(max_crop / max_stride)
    total_rows = total_lines * multiplier

    # Streaming decision.
    if max_memory_mb is not None and max_memory_mb > 0:
        budget = max_memory_mb * 1024 * 1024
    elif max_memory_mb is None:
        budget = int(psutil.virtual_memory().available * 0.75)
    else:
        budget = None

    stream = False
    if budget is not None:
        per_row = _estimate_output_bytes_per_row(
            max_crop, format, one_hot, codon_map_len
        )
        stream = total_rows * per_row > budget

    if one_hot and not stream:
        # Legacy guard for the non-streaming path.
        estimated = _estimate_onehot_memory(
            total_rows=total_lines,
            crop_size=max_crop,
            fmt=format,
            one_hot=one_hot,
            codon_map_len=codon_map_len,
            stride=max_stride,
        )
        _check_onehot_memory(estimated, psutil.virtual_memory().available)

    if stream:
        _convert_to_npz_streaming(
            input_path=input_path,
            output_path=output_path,
            fmt=format,
            crop_sizes=crop_sizes,
            strides=strides,
            num_classes=num_classes,
            num_workers=num_workers,
            one_hot=one_hot,
            pad_int=pad_int,
            codon_map_name=codon_map,
            nucleotide_map=nucleotide_map_dict,
            compress=compress,
            max_memory_bytes=budget,
            pad=pad,
        )
    else:
        _convert_to_npz(
            input_path=input_path,
            output_path=output_path,
            fmt=format,
            crop_sizes=crop_sizes,
            strides=strides,
            num_classes=num_classes,
            num_workers=num_workers,
            one_hot=one_hot,
            pad_int=pad_int,
            codon_map_name=codon_map,
            nucleotide_map=nucleotide_map_dict,
            compress=compress,
            pad=pad,
        )
```

- [ ] **Step 2: Run existing tests**

```bash
pytest tests/unit/test_dataops_convert.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/dataops/convert.py
git commit -m "feat(convert): choose streaming vs fast path based on memory budget"
```

---

## Task 6: Add CLI options and pass them through

**Files:**
- Modify: `src/jaeger/cli.py`
- Modify: `src/jaeger/commands/utils.py`

- [ ] **Step 1: Add options in `src/jaeger/cli.py`**

Insert two new options in the `optimize-data` command, near the existing `--overlap` option (around line 1000):

```python
@click.option(
    "--max-memory-mb",
    type=int,
    default=None,
    show_default=True,
    help=(
        "Memory budget in MB for encoded output buffers. "
        "If omitted, auto-budget is ~75%% of available RAM. "
        "Set to 0 to disable streaming."
    ),
)
@click.option(
    "--pad",
    is_flag=True,
    default=False,
    show_default=True,
    help="Pad all crops to the global maximum length (legacy behavior).",
)
```

- [ ] **Step 2: Pass the options in the `optimize_data` function**

Update the `optimize_data_core(...)` call to include:

```python
        max_memory_mb=kwargs.get("max_memory_mb"),
        pad=kwargs.get("pad"),
```

- [ ] **Step 3: Update `optimize_data_core` in `src/jaeger/commands/utils.py`**

Change the signature to include `max_memory_mb: int | None = None` and `pad: bool = False`, then pass them to `convert_dataset`.

- [ ] **Step 4: Add a CLI help test**

Append to `tests/unit/test_cli.py`:

```python
from click.testing import CliRunner
from jaeger.cli import main


def test_optimize_data_help_includes_new_flags():
    runner = CliRunner()
    result = runner.invoke(main, ["utils", "optimize-data", "--help"])
    assert result.exit_code == 0
    assert "--max-memory-mb" in result.output
    assert "--pad" in result.output
```

- [ ] **Step 5: Run the CLI test**

```bash
pytest tests/unit/test_cli.py::test_optimize_data_help_includes_new_flags -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/jaeger/cli.py src/jaeger/commands/utils.py tests/unit/test_cli.py
git commit -m "feat(cli): expose --max-memory-mb and --pad in optimize-data"
```

---

## Task 7: Update the numpy loader for ragged object arrays

**Files:**
- Modify: `src/jaeger/data/loaders.py`
- Test: `tests/unit/test_data_loaders.py`

- [ ] **Step 1: Change `np.load` to allow pickling for object arrays**

In `src/jaeger/data/loaders.py`, change line 108 from:

```python
    data = np.load(path, allow_pickle=False)
```

to:

```python
    data = np.load(path, allow_pickle=True)
```

- [ ] **Step 2: Add a ragged loader helper**

Insert this helper after `_load_numpy_dataset` or inside it:

```python
def _is_ragged(data: np.lib.npyio.NpzFile, input_type: str) -> bool:
    for key in ("nucleotide", "translated"):
        if input_type in (key, "both") and key in data:
            arr = data[key]
            if arr.ndim == 1 and arr.dtype == object:
                return True
    return False


def _load_ragged_numpy_dataset(
    data: np.lib.npyio.NpzFile,
    input_type: str,
    seq_onehot: bool,
    codon_depth: int | None,
    nucleotide_onehot_map: dict[str, list[float]] | None,
    num_classes: int | None,
    one_hot_labels: bool,
) -> tf.data.Dataset:
    """Load an NPZ where feature arrays are 1-D object arrays of crops."""
    lookup = None
    if input_type in ("nucleotide", "both") and seq_onehot:
        nucleotide_map = json.loads(str(data["nucleotide_map"]))
        lookup = _build_nucleotide_onehot_lookup(nucleotide_map, nucleotide_onehot_map)

    n = len(data["labels"])

    def _sample_features(i: int) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        if input_type in ("nucleotide", "both"):
            nuc = data["nucleotide"][i]
            if seq_onehot and nuc.ndim == 3 and lookup is not None:
                nuc = lookup[nuc]
            features["nucleotide"] = nuc
        if input_type in ("translated", "both"):
            trans = data["translated"][i]
            if seq_onehot and trans.ndim == 2 and codon_depth is not None:
                t = tf.cast(trans, tf.int32)
                mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
                oh = tf.one_hot(t, depth=codon_depth, dtype=tf.float32)
                trans = (oh * mask).numpy()
            features["translated"] = trans
        return features

    sample = _sample_features(0)
    output_signature = (
        {
            key: tf.TensorSpec(shape=[None] * arr.ndim, dtype=tf.as_dtype(arr.dtype))
            for key, arr in sample.items()
        },
        tf.TensorSpec(
            shape=(num_classes,)
            if one_hot_labels and num_classes and num_classes > 1
            else (1,) if num_classes == 1 else (),
            dtype=tf.float32,
        ),
    )

    def generator():
        for i in range(n):
            features = _sample_features(i)
            label = int(data["labels"][i])
            if one_hot_labels and num_classes is not None and num_classes > 1:
                label_arr = np.eye(num_classes, dtype=np.float32)[label]
            elif num_classes == 1:
                label_arr = np.array([float(label)], dtype=np.float32)
            else:
                label_arr = np.float32(label)
            yield features, label_arr

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return ds
```

- [ ] **Step 3: Branch `_load_numpy_dataset` at the top**

After loading `data`, add:

```python
    if _is_ragged(data, input_type):
        return _load_ragged_numpy_dataset(
            data,
            input_type=input_type,
            seq_onehot=seq_onehot,
            codon_depth=codon_depth,
            nucleotide_onehot_map=nucleotide_onehot_map,
            num_classes=num_classes,
            one_hot_labels=one_hot_labels,
        )
```

- [ ] **Step 4: Add loader tests**

Append to `tests/unit/test_data_loaders.py`:

```python
@pytest.fixture
def ragged_nucleotide_npz(tmp_path: Path) -> str:
    path = tmp_path / "ragged_nuc.npz"
    crops = [
        np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32),
        np.array([[1, 2], [2, 1]], dtype=np.int32),
    ]
    arr = np.empty(len(crops), dtype=object)
    arr[:] = crops
    labels = np.array([0, 1], dtype=np.int32)
    np.savez(
        path,
        nucleotide=arr,
        labels=labels,
        lengths=np.array([4, 2], dtype=np.int32),
        translated_lengths=np.array([0, 0], dtype=np.int32),
        nucleotide_map='{"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}',
        crop_sizes=np.array([4], dtype=np.int32),
        strides=np.array([0], dtype=np.int32),
        pad_int=np.int32(0),
        padded=np.bool_(False),
    )
    return str(path)


class TestRaggedLoaders:
    def test_ragged_nucleotide_integer(self, ragged_nucleotide_npz: str):
        ds = loaders._load_numpy_dataset(
            ragged_nucleotide_npz,
            input_type="nucleotide",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        features, label = next(iter(ds))
        assert features["nucleotide"].shape == (2, 4)
        assert label.shape == (NUM_CLASSES,)

    def test_ragged_padded_batch(self, ragged_nucleotide_npz: str):
        ds = loaders._load_numpy_dataset(
            ragged_nucleotide_npz,
            input_type="nucleotide",
            seq_onehot=False,
            num_classes=NUM_CLASSES,
        )
        batched = ds.padded_batch(2, padded_shapes=({"nucleotide": [2, None]}, [NUM_CLASSES]))
        features, labels = next(iter(batched))
        assert features["nucleotide"].shape == (2, 2, 4)
```

- [ ] **Step 5: Run loader tests**

```bash
pytest tests/unit/test_data_loaders.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/jaeger/data/loaders.py tests/unit/test_data_loaders.py
git commit -m "feat(loaders): support ragged object-array NPZ files"
```

---

## Task 8: Skip `.cache()` for ragged datasets in training

**Files:**
- Modify: `src/jaeger/commands/train.py`

- [ ] **Step 1: Add a ragged detector helper**

Inside `src/jaeger/commands/train.py` (near the numpy data-format block), add:

```python
def _is_ragged_dataset(ds: tf.data.Dataset) -> bool:
    feat_spec = ds.element_spec[0]
    for spec in tf.nest.flatten(feat_spec):
        if any(d is None for d in spec.shape.as_list()):
            return True
    return False
```

- [ ] **Step 2: Skip cache for ragged numpy datasets**

Replace the cache block in the training data path (around line 257):

```python
                ds = _data
                if _onehot_buffer is None and not _is_ragged_dataset(_data):
                    ds = ds.cache()
```

Do the same for the validation block if it has an identical cache call.

- [ ] **Step 3: Run training import smoke test**

```bash
python -c "from jaeger.commands import train; print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/commands/train.py
git commit -m "feat(train): do not cache ragged numpy datasets"
```

---

## Task 9: Add end-to-end integration tests

**Files:**
- Modify: `tests/unit/test_dataops_convert.py`
- Modify: `tests/unit/test_data_loaders.py`

- [ ] **Step 1: Add a no-pad round-trip loader test**

Append to `tests/unit/test_data_loaders.py`:

```python
def test_convert_and_load_unpadded(simple_csv_path: str, tmp_path: Path):
    from jaeger.dataops import convert

    out = tmp_path / "unpadded.npz"
    convert.convert_dataset(
        input_path=simple_csv_path,
        output_path=str(out),
        format="nucleotide",
        crop_size=24,
        num_classes=2,
        num_workers=1,
        pad=False,
    )
    ds = loaders._load_numpy_dataset(
        str(out),
        input_type="nucleotide",
        seq_onehot=False,
        num_classes=2,
    )
    features, label = next(iter(ds))
    assert features["nucleotide"].shape[1] <= 24
    assert label.shape == (2,)
```

- [ ] **Step 2: Add a streaming no-pad test in `test_dataops_convert.py`**

Append:

```python
    def test_streaming_unpadded(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0," + "A" * 25, "1," + "G" * 25])
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=csv,
            output_path=str(out),
            format="nucleotide",
            crop_size=20,
            stride=10,
            num_classes=2,
            num_workers=1,
            max_memory_mb=1,
            pad=False,
        )
        data = np.load(out, allow_pickle=True)
        assert data["padded"].item() is False
        assert data["nucleotide"].dtype == object
```

- [ ] **Step 3: Run all convert and loader tests**

```bash
pytest tests/unit/test_dataops_convert.py tests/unit/test_data_loaders.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_dataops_convert.py tests/unit/test_data_loaders.py
git commit -m "test: add integration tests for streaming and unpadded output"
```

---

## Task 10: Lint, format, and full unit test run

- [ ] **Step 1: Run ruff**

```bash
ruff check src/jaeger/cli.py src/jaeger/commands/utils.py src/jaeger/dataops/convert.py src/jaeger/data/loaders.py src/jaeger/commands/train.py tests/unit/test_dataops_convert.py tests/unit/test_data_loaders.py tests/unit/test_cli.py
ruff format src/jaeger/cli.py src/jaeger/commands/utils.py src/jaeger/dataops/convert.py src/jaeger/data/loaders.py src/jaeger/commands/train.py tests/unit/test_dataops_convert.py tests/unit/test_data_loaders.py tests/unit/test_cli.py
```

- [ ] **Step 2: Run unit tests**

```bash
pytest tests/unit/test_dataops_convert.py tests/unit/test_data_loaders.py tests/unit/test_cli.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit formatting fixes**

```bash
git add -A
git commit -m "style: ruff format for streaming optional-padding feature"
```

---

## Self-review checklist

- **Spec coverage:** every section of the design spec has at least one task.
- **No placeholders:** every step contains concrete code or commands.
- **Type consistency:** `_finalize_batch_arrays`, `_convert_to_npz`, `_convert_to_npz_streaming`, and `convert_dataset` all accept the same `pad` and `max_memory_mb` semantics.
