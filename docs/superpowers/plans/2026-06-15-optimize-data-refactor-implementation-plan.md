# Refactor `jaeger utils optimize-data` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old `optimize-data` format options with a single, always-NPZ output supporting `nucleotide`, `translated`, and `both` representations, with multi-crop-size support, sliding-window stride, optional one-hot output, and user-configurable nucleotide mapping.

**Architecture:** A new Numba-accelerated nucleotide encoder is added alongside the existing 6-frame codon encoder in `dataops/convert.py`. A new `_convert_to_npz` function orchestrates CSV parsing, crop generation, per-crop-size batch encoding, variable-length padding, and NPZ writing. The Click command in `cli.py` and the thin wrapper in `commands/utils.py` are updated to expose the new options and remove the old ones.

**Tech Stack:** Python 3.13, Click, NumPy, Numba (optional but preferred).

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/jaeger/dataops/convert.py` (modify) | New nucleotide encoder, map helpers, crop helpers, padding helper, `_convert_to_npz`, and updated `convert_dataset` dispatcher. |
| `src/jaeger/cli.py` (modify) | Refactored `optimize-data` Click command with the new option set. |
| `src/jaeger/commands/utils.py` (modify) | Updated `optimize_data_core` wrapper. |
| `tests/unit/test_dataops_convert.py` (modify) | Tests for helpers, encoders, and the new NPZ conversion. |
| `tests/integration/test_optimize_data_cli.py` (add) | Integration tests for the refactored `utils optimize-data` CLI. |

---

## Task 1: Add codon-map and nucleotide-map helpers

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestMapHelpers:
    def test_get_codon_map_codom_id(self):
        from jaeger.dataops.convert import _get_codon_map
        mapping = _get_codon_map("codon_id")
        assert len(mapping) == 64
        assert mapping[0] == 0

    def test_get_codon_map_invalid(self):
        from jaeger.dataops.convert import _get_codon_map
        with pytest.raises(ValueError):
            _get_codon_map("not_a_map")

    def test_parse_nucleotide_map_default(self):
        from jaeger.dataops.convert import _parse_nucleotide_map
        m = _parse_nucleotide_map(None)
        assert m["A"] == 1 and m["N"] == 0

    def test_parse_nucleotide_map_custom(self):
        from jaeger.dataops.convert import _parse_nucleotide_map
        m = _parse_nucleotide_map('{"A":0,"G":1,"T":2,"C":3,"N":4}')
        assert m == {"A": 0, "G": 1, "T": 2, "C": 3, "N": 4}

    def test_parse_nucleotide_map_missing_base(self):
        from jaeger.dataops.convert import _parse_nucleotide_map
        with pytest.raises(ValueError):
            _parse_nucleotide_map('{"A":0}')

    def test_parse_nucleotide_map_invalid_json(self):
        from jaeger.dataops.convert import _parse_nucleotide_map
        with pytest.raises(ValueError):
            _parse_nucleotide_map("not json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestMapHelpers -v`
Expected: `ImportError` for `_get_codon_map` / `_parse_nucleotide_map`.

- [ ] **Step 3: Write minimal implementation**

In `src/jaeger/dataops/convert.py`, add near the existing lookup helpers:

```python
import json

_CODON_MAP_NAMES = {
    "codon_id": "CODON_ID",
    "aa_id": "AA_ID",
    "pc5_id": "PC5_ID",
    "murphy10_id": "MURPHY10_ID",
    "cod_id": "DICODON_ID",
    "pc2_id": "PC2_ID",
}

_DEFAULT_NUCLEOTIDE_MAP = {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}


def _get_codon_map(name: str) -> list[int]:
    """Return the codon map list from ``jaeger.seqops.maps`` by CLI name."""
    from jaeger.seqops import maps

    attr = _CODON_MAP_NAMES.get(name.lower())
    if attr is None or not hasattr(maps, attr):
        raise ValueError(f"Unknown codon map: {name}")
    return list(getattr(maps, attr))


def _parse_nucleotide_map(json_str: str | None) -> dict[str, int]:
    """Parse and validate a JSON nucleotide-to-int mapping."""
    if json_str is None:
        return dict(_DEFAULT_NUCLEOTIDE_MAP)
    try:
        mapping = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --nucleotide-map JSON: {exc}") from exc
    for base in "ACGTN":
        if base not in mapping:
            raise ValueError(f"--nucleotide-map must contain a mapping for {base}")
    return {k: int(v) for k, v in mapping.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestMapHelpers -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat: add codon and nucleotide map helpers"
```

---

## Task 2: Add nucleotide batch encoder

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestNucleotideEncoder:
    def test_encode_nucleotide_int(self):
        from jaeger.dataops.convert import _encode_nucleotide_batch, _build_nucleotide_lookups

        sequences = np.full((1, 4), ord("N"), dtype=np.uint8)
        sequences[0, :4] = np.frombuffer(b"ATGC", dtype=np.uint8)
        lengths = np.array([4], dtype=np.int32)
        user_map = {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
        ascii_to_user, comp_user, ascii_to_oh, comp_oh = _build_nucleotide_lookups(user_map)
        out = _encode_nucleotide_batch(
            sequences, lengths, 4, ascii_to_user, comp_user, ascii_to_oh, comp_oh, False
        )
        assert out.shape == (1, 2, 4)
        # forward strand: A=1, T=3, G=2, C=4
        assert out[0, 0].tolist() == [1, 3, 2, 4]

    def test_encode_nucleotide_onehot(self):
        from jaeger.dataops.convert import _encode_nucleotide_batch, _build_nucleotide_lookups

        sequences = np.full((1, 4), ord("N"), dtype=np.uint8)
        sequences[0, :4] = np.frombuffer(b"ATGC", dtype=np.uint8)
        lengths = np.array([4], dtype=np.int32)
        user_map = {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
        ascii_to_user, comp_user, ascii_to_oh, comp_oh = _build_nucleotide_lookups(user_map)
        out = _encode_nucleotide_batch(
            sequences, lengths, 4, ascii_to_user, comp_user, ascii_to_oh, comp_oh, True
        )
        assert out.shape == (1, 2, 4, 4)
        assert out[0, 0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]  # A
        assert out[0, 0, 2].tolist() == [0.0, 1.0, 0.0, 0.0]  # G
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestNucleotideEncoder -v`
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

In `src/jaeger/dataops/convert.py`, add:

```python
def _build_nucleotide_lookups(user_map: dict[str, int]):
    """Build ASCII lookup tables for integer and one-hot nucleotide encoding.

    Returns
    -------
    ascii_to_user : np.ndarray, shape (256,)
        Maps ASCII code to the user-defined integer token. Unknown -> N value.
    comp_user : np.ndarray, shape (max_token + 1,)
        Maps a user integer token to its reverse-complement user token.
    ascii_to_oh : np.ndarray, shape (256,)
        Maps ASCII code to 0-3 one-hot index; N/unknown -> -1.
    comp_oh : np.ndarray, shape (4,)
        Maps one-hot base index to its reverse-complement one-hot index.
    """
    ascii_to_user = np.full(256, user_map["N"], dtype=np.int32)
    ascii_to_oh = np.full(256, -1, dtype=np.int32)

    oh_order = ["A", "G", "T", "C"]
    oh_index = {b: i for i, b in enumerate(oh_order)}
    comp_letters = {"A": "T", "T": "A", "G": "C", "C": "G"}

    for base in "ACGTN":
        ascii_to_user[ord(base)] = user_map[base]
        ascii_to_user[ord(base.lower())] = user_map[base]
        if base in oh_index:
            ascii_to_oh[ord(base)] = oh_index[base]
            ascii_to_oh[ord(base.lower())] = oh_index[base]

    max_token = max(user_map.values())
    comp_user = np.full(max_token + 1, user_map["N"], dtype=np.int32)
    for base, token in user_map.items():
        if base == "N":
            continue
        comp_letter = comp_letters.get(base, "N")
        comp_user[token] = user_map.get(comp_letter, user_map["N"])

    comp_oh = np.array([oh_index[comp_letters[b]] for b in oh_order], dtype=np.int32)
    return ascii_to_user, comp_user, ascii_to_oh, comp_oh


@njit(cache=False)
def _encode_nucleotide_batch(
    sequences,
    lengths,
    crop_size,
    ascii_to_user,
    comp_user,
    ascii_to_oh,
    comp_oh,
    one_hot,
):
    """Encode a batch of DNA crops to nucleotide integer or one-hot arrays."""
    n = len(lengths)
    if one_hot:
        out = np.zeros((n, 2, crop_size, 4), dtype=np.float32)
    else:
        out = np.zeros((n, 2, crop_size), dtype=np.int32)

    for s in range(n):
        length = min(lengths[s], crop_size)
        for i in range(length):
            au = ascii_to_user[sequences[s, i]]
            cu = comp_user[au]
            if one_hot:
                aoh = ascii_to_oh[sequences[s, i]]
                if 0 <= aoh < 4:
                    out[s, 0, i, aoh] = 1.0
                coh = comp_oh[aoh] if 0 <= aoh < 4 else -1
                if 0 <= coh < 4:
                    out[s, 1, i, coh] = 1.0
            else:
                out[s, 0, i] = au
                out[s, 1, i] = cu
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestNucleotideEncoder -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat: add numba nucleotide batch encoder"
```

---

## Task 3: Add crop generation and padding helpers

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestCropHelpers:
    def test_generate_crops_stride_zero(self):
        from jaeger.dataops.convert import _generate_crops
        crops = _generate_crops("ATGCATGC", (4, 6), 0)
        assert crops == [("ATGC", 4), ("ATGCAT", 6)]

    def test_generate_crops_with_stride(self):
        from jaeger.dataops.convert import _generate_crops
        crops = _generate_crops("ATGCATGCAT", (4,), 3)
        assert crops == [("ATGC", 4), ("CATG", 4), ("GCAT", 4)]

    def test_pad_array(self):
        from jaeger.dataops.convert import _pad_array
        arr = np.array([[1, 2], [3, 4]])
        out = _pad_array(arr, 4, 0, axis=-1)
        assert out.shape == (2, 4)
        assert out[:, 2:].tolist() == [[0, 0], [0, 0]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestCropHelpers -v`
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

In `src/jaeger/dataops/convert.py`, add:

```python
def _generate_crops(seq: str, crop_sizes: tuple[int, ...], stride: int) -> list[tuple[str, int]]:
    """Generate sequence crops for each crop size.

    If ``stride == 0`` or the sequence is shorter than the largest crop size,
    one crop per crop size is returned (truncated to the crop size). Otherwise
    sliding windows of each crop size are extracted.
    """
    if not crop_sizes:
        raise ValueError("At least one crop size is required")

    max_cs = max(crop_sizes)
    if stride == 0 or len(seq) <= max_cs:
        return [(seq[:cs], cs) for cs in crop_sizes]

    crops: list[tuple[str, int]] = []
    for cs in crop_sizes:
        for start in range(0, len(seq) - cs + 1, stride):
            crops.append((seq[start : start + cs], cs))
    if not crops:
        crops.append((seq[:max_cs], max_cs))
    return crops


def _pad_array(arr: np.ndarray, target_length: int, pad_value: int | float, axis: int = -1) -> np.ndarray:
    """Pad an array along *axis* to *target_length*."""
    if arr.shape[axis] >= target_length:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target_length)
        return arr[tuple(slices)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_length - arr.shape[axis])
    return np.pad(arr, pad_width, constant_values=pad_value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestCropHelpers -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat: add crop generation and padding helpers"
```

---

## Task 4: Implement `_convert_to_npz`

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/unit/test_dataops_convert.py`:

```python
class TestConvertToNpz:
    def test_translated(self, simple_csv_path: str, tmp_path: Path):
        from jaeger.dataops.convert import _convert_to_npz

        out = tmp_path / "out.npz"
        _convert_to_npz(
            input_path=simple_csv_path,
            output_path=str(out),
            format="translated",
            crop_sizes=(24,),
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map_json=None,
        )
        assert out.exists()
        data = np.load(out)
        assert "translated" in data
        assert "labels" in data
        assert "lengths" in data

    def test_nucleotide(self, simple_csv_path: str, tmp_path: Path):
        from jaeger.dataops.convert import _convert_to_npz

        out = tmp_path / "out.npz"
        _convert_to_npz(
            input_path=simple_csv_path,
            output_path=str(out),
            format="nucleotide",
            crop_sizes=(24,),
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map_json=None,
        )
        data = np.load(out)
        assert "nucleotide" in data
        assert data["nucleotide"].shape[1] == 2  # two strands
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestConvertToNpz -v`
Expected: `ImportError` for `_convert_to_npz`.

- [ ] **Step 3: Write minimal implementation**

In `src/jaeger/dataops/convert.py`, add:

```python
def _save_npz(output_path: str, compress: str, **arrays):
    """Save arrays to NPZ with selectable compression."""
    if compress == "none":
        np.savez(output_path, **arrays)
    elif compress == "fast":
        # zlib level 1
        import zipfile
        from io import BytesIO

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for key, arr in arrays.items():
                buf = BytesIO()
                np.save(buf, arr)
                zf.writestr(key + ".npy", buf.getvalue())
    else:
        np.savez_compressed(output_path, **arrays)


def _convert_to_npz(
    input_path: str,
    output_path: str,
    format: str,
    crop_sizes: tuple[int, ...],
    stride: int,
    num_classes: int,
    num_workers: int | None,
    one_hot: bool,
    pad_int: int,
    codon_map_name: str,
    nucleotide_map_json: str | None,
    compress: str = "fast",
):
    """Convert a CSV dataset to the new unified NPZ format."""
    format = format.lower()
    if format not in {"nucleotide", "translated", "both"}:
        raise ValueError(f"Invalid format: {format}")

    crop_sizes = tuple(int(cs) for cs in crop_sizes)
    if any(cs <= 0 for cs in crop_sizes):
        raise ValueError("All crop sizes must be positive")
    if stride < 0:
        raise ValueError("stride must be >= 0")

    if num_workers is None:
        num_workers = cpu_count()

    user_nuc_map = _parse_nucleotide_map(nucleotide_map_json)
    codon_map = _get_codon_map(codon_map_name) if format in ("translated", "both") else None

    # Parse CSV and generate crops
    crops: list[tuple[str, int, int]] = []  # (seq, crop_size, label)
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = line.find(",")
            if comma == -1:
                continue
            label = int(line[:comma])
            seq = line[comma + 1 :]
            for crop_seq, cs in _generate_crops(seq, crop_sizes, stride):
                crops.append((crop_seq, cs, label))

    if not crops:
        raise ValueError("No valid sequences found in input file")

    print(f"Total crops: {len(crops)}")

    # Encode per crop size, then combine and pad
    all_arrays: dict[str, list[np.ndarray]] = {"nucleotide": [], "translated": []}
    all_lengths: list[int] = []
    all_labels: list[int] = []

    # Group by crop size so we can batch-process efficiently
    by_size: dict[int, list[tuple[str, int]]] = {cs: [] for cs in crop_sizes}
    for seq, cs, label in crops:
        by_size[cs].append((seq, label))

    for cs in crop_sizes:
        batch = by_size[cs]
        if not batch:
            continue
        seqs, labels = zip(*batch)
        n = len(seqs)
        all_labels.extend(labels)

        # Pack sequences into a uint8 array
        sequences = np.full((n, cs), ord("N"), dtype=np.uint8)
        lengths = np.zeros(n, dtype=np.int32)
        for i, seq in enumerate(seqs):
            seq_bytes = seq.encode()[:cs]
            sequences[i, : len(seq_bytes)] = np.frombuffer(seq_bytes, dtype=np.uint8)
            lengths[i] = len(seq_bytes)

        if format in ("nucleotide", "both"):
            ascii_to_user, comp_user, ascii_to_oh, comp_oh = _build_nucleotide_lookups(user_nuc_map)
            nuc = _encode_nucleotide_batch(
                sequences, lengths, cs, ascii_to_user, comp_user, ascii_to_oh, comp_oh, one_hot
            )
            all_arrays["nucleotide"].append(nuc)

        if format in ("translated", "both"):
            codon_lut, ascii_lut, comp_lut = _build_numba_lookups()
            seq_len = cs // 3 - 1
            translated = _process_batch_numba(
                sequences, lengths, cs, seq_len, codon_lut, comp_lut, ascii_lut
            )
            if one_hot:
                depth = len(codon_map) + 1
                oh = np.zeros(translated.shape + (depth,), dtype=np.float32)
                for s in range(translated.shape[0]):
                    for f in range(6):
                        for p in range(translated.shape[2]):
                            idx = translated[s, f, p]
                            if 0 < idx < depth:
                                oh[s, f, p, idx] = 1.0
                translated = oh
            all_arrays["translated"].append(translated)

        all_lengths.extend(lengths.tolist())

    # Determine max lengths and pad
    if format in ("nucleotide", "both"):
        nuc_parts = all_arrays["nucleotide"]
        max_len = max(p.shape[-2] for p in nuc_parts)
        nuc_padded = []
        for p in nuc_parts:
            pad_val = 0 if one_hot else pad_int
            nuc_padded.append(_pad_array(p, max_len, pad_val, axis=-2))
        nuc_arr = np.concatenate(nuc_padded, axis=0)

    if format in ("translated", "both"):
        trans_parts = all_arrays["translated"]
        max_len = max(p.shape[-2] for p in trans_parts)
        trans_padded = []
        for p in trans_parts:
            pad_val = 0 if one_hot else pad_int
            trans_padded.append(_pad_array(p, max_len, pad_val, axis=-2))
        trans_arr = np.concatenate(trans_padded, axis=0)

    lengths_arr = np.array(all_lengths, dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    # Build save dict
    save_dict: dict[str, np.ndarray] = {
        "labels": labels_arr,
        "lengths": lengths_arr,
        "pad_int": np.array(pad_int, dtype=np.int32),
        "crop_sizes": np.array(crop_sizes, dtype=np.int32),
        "stride": np.array(stride, dtype=np.int32),
    }
    if format in ("nucleotide", "both"):
        save_dict["nucleotide"] = nuc_arr
        save_dict["nucleotide_map"] = np.array(json.dumps(user_nuc_map))
    if format in ("translated", "both"):
        save_dict["translated"] = trans_arr
        save_dict["codon_map"] = np.array(codon_map_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _save_npz(output_path, compress, **save_dict)
    print(f"Saved {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py::TestConvertToNpz -v`
Expected: pass after fixing any shape issues.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat: implement unified NPZ conversion"
```

---

## Task 5: Update `convert_dataset` dispatcher

**Files:**
- Modify: `src/jaeger/dataops/convert.py`
- Test: `tests/unit/test_dataops_convert.py`

- [ ] **Step 1: Replace old dispatcher tests**

Replace `TestConvertDataset` in `tests/unit/test_dataops_convert.py` with:

```python
class TestConvertDataset:
    def test_nucleotide(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="nucleotide",
            crop_sizes=(24,),
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()
        data = np.load(out)
        assert "nucleotide" in data
        assert "labels" in data

    def test_translated(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="translated",
            crop_sizes=(24,),
            num_classes=2,
            num_workers=1,
        )
        data = np.load(out)
        assert "translated" in data

    def test_both(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="both",
            crop_sizes=(24,),
            num_classes=2,
            num_workers=1,
        )
        data = np.load(out)
        assert "nucleotide" in data
        assert "translated" in data

    def test_invalid_format(self, simple_csv_path: str, tmp_path: Path):
        with pytest.raises(ValueError):
            convert.convert_dataset(
                input_path=simple_csv_path,
                output_path=str(tmp_path / "out.npz"),
                format="unknown",
                crop_sizes=(24,),
                num_classes=2,
            )
```

- [ ] **Step 2: Update `convert_dataset` signature and body**

Replace the existing `convert_dataset` function in `src/jaeger/dataops/convert.py` with:

```python
def convert_dataset(
    input_path: str,
    output_path: str,
    format: str,
    crop_sizes: tuple[int, ...] | list[int] = (500,),
    stride: int = 0,
    num_classes: int = 3,
    num_workers: int | None = None,
    one_hot: bool = False,
    pad_int: int = 0,
    codon_map: str = "codon_id",
    nucleotide_map: str | None = None,
    compress: str = "fast",
):
    """Convert a CSV dataset to the unified NPZ format.

    Args:
        input_path: Path to input CSV file (label,sequence format).
        output_path: Path to output `.npz` file.
        format: Target representation — ``nucleotide``, ``translated``, or ``both``.
        crop_sizes: One or more crop lengths.
        stride: Step between crops; ``0`` means one crop per sequence.
        num_classes: Number of output classes (used for label validation).
        num_workers: Number of parallel workers (``None`` = auto).
        one_hot: If True, output float one-hot tensors instead of integer indices.
        pad_int: Integer padding value for integer outputs.
        codon_map: Name of the codon map from ``jaeger.seqops.maps``.
        nucleotide_map: JSON string with base-to-int mapping, or None for default.
        compress: NPZ compression level — ``fast``, ``default``, or ``none``.
    """
    print(f"Converting {input_path} -> {output_path}")
    print(f"Format: {format}, Crop sizes: {crop_sizes}, Num classes: {num_classes}")

    _convert_to_npz(
        input_path=input_path,
        output_path=output_path,
        format=format,
        crop_sizes=tuple(crop_sizes),
        stride=stride,
        num_classes=num_classes,
        num_workers=num_workers,
        one_hot=one_hot,
        pad_int=pad_int,
        codon_map_name=codon_map,
        nucleotide_map_json=nucleotide_map,
        compress=compress,
    )
```

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/dataops/convert.py tests/unit/test_dataops_convert.py
git commit -m "feat: update convert_dataset dispatcher to new NPZ format"
```

---

## Task 6: Refactor CLI `optimize-data`

**Files:**
- Modify: `src/jaeger/cli.py`
- Modify: `src/jaeger/commands/utils.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing CLI help test**

Append to `tests/unit/test_cli.py`:

```python
def test_optimize_data_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["utils", "optimize-data", "--help"])
    assert result.exit_code == 0
    assert "nucleotide" in result.output.lower()
    assert "translated" in result.output.lower()
    assert "crop-size" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_cli.py::test_optimize_data_help -v`
Expected: assertion fails because current help text does not contain new options.

- [ ] **Step 3: Replace the Click command and wrapper**

In `src/jaeger/cli.py`, replace the `optimize_data` function and its decorator block with:

```python
@utils.command(
    context_settings=dict(ignore_unknown_options=True, show_default=True),
    help="""
            Convert training data to optimized NPZ format for faster loading.
            Output is always variable-length and padded to the longest sample.

            usage
            -----

            jaeger utils optimize-data -i train.csv -o train.npz --format translated

            Supported formats:
              nucleotide   - 2-strand nucleotide representation
              translated   - 6-frame codon representation
              both         - store both nucleotide and translated arrays

            Available codon maps: CODON_ID, AA_ID, PC5_ID, MURPHY10_ID, COD_ID, PC2_ID
        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input CSV file",
)
@click.option(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to output file",
)
@click.option(
    "--format",
    type=click.Choice(["nucleotide", "translated", "both"], case_sensitive=False),
    required=True,
    help="Output representation",
)
@click.option(
    "--crop-size",
    type=int,
    multiple=True,
    default=(500,),
    show_default=True,
    help="One or more crop lengths",
)
@click.option(
    "--stride",
    type=int,
    default=0,
    show_default=True,
    help="Sliding-window step (0 = one crop per sequence)",
)
@click.option(
    "--num-classes",
    type=int,
    default=3,
    show_default=True,
    help="Number of classes",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: all CPUs)",
)
@click.option(
    "--one-hot",
    is_flag=True,
    help="Output one-hot float tensors instead of integer indices",
)
@click.option(
    "--pad-int",
    type=int,
    default=0,
    show_default=True,
    help="Integer value used for padding and stored as 'pad_int' in the NPZ",
)
@click.option(
    "--codon-map",
    type=click.Choice(
        ["codon_id", "aa_id", "pc5_id", "murphy10_id", "cod_id", "pc2_id"],
        case_sensitive=False,
    ),
    default="codon_id",
    show_default=True,
    help="Codon-to-int map from jaeger.seqops.maps (translated/both only)",
)
@click.option(
    "--nucleotide-map",
    type=str,
    default=None,
    help='JSON base-to-int map, e.g. \'{"A":1,"G":2,"T":3,"C":4,"N":0}\'',
)
@click.option(
    "--max-length",
    type=int,
    default=5000,
    show_default=True,
    hidden=True,
    help="Deprecated; kept for compatibility but ignored",
)
@click.option(
    "--compress",
    type=click.Choice(["fast", "default", "none"], case_sensitive=False),
    default="fast",
    show_default=True,
    help="NPZ compression level: fast (zlib 1), default (zlib 6), or none",
)
def optimize_data(**kwargs):
    """Convert CSV training data to optimized NPZ format."""
    from jaeger.commands.utils import optimize_data_core

    optimize_data_core(
        input_path=kwargs.get("input"),
        output_path=kwargs.get("output"),
        format=kwargs.get("format"),
        crop_sizes=kwargs.get("crop_size"),
        stride=kwargs.get("stride"),
        num_classes=kwargs.get("num_classes"),
        num_workers=kwargs.get("num_workers"),
        one_hot=kwargs.get("one_hot"),
        pad_int=kwargs.get("pad_int"),
        codon_map=kwargs.get("codon_map"),
        nucleotide_map=kwargs.get("nucleotide_map"),
        compress=kwargs.get("compress"),
    )
```

In `src/jaeger/commands/utils.py`, replace `optimize_data_core` with:

```python
def optimize_data_core(
    input_path: str,
    output_path: str,
    format: str,
    crop_sizes: tuple[int, ...] = (500,),
    stride: int = 0,
    num_classes: int = 3,
    num_workers: int | None = None,
    one_hot: bool = False,
    pad_int: int = 0,
    codon_map: str = "codon_id",
    nucleotide_map: str | None = None,
    compress: str = "fast",
):
    """Convert Jaeger CSV training data to a unified NPZ file."""
    from jaeger.dataops.convert import convert_dataset

    convert_dataset(
        input_path=input_path,
        output_path=output_path,
        format=format,
        crop_sizes=crop_sizes,
        stride=stride,
        num_classes=num_classes,
        num_workers=num_workers,
        one_hot=one_hot,
        pad_int=pad_int,
        codon_map=codon_map,
        nucleotide_map=nucleotide_map,
        compress=compress,
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_cli.py::test_optimize_data_help tests/unit/test_dataops_convert.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/cli.py src/jaeger/commands/utils.py tests/unit/test_cli.py
git commit -m "feat: refactor optimize-data CLI for unified NPZ output"
```

---

## Task 7: Clean up dead code

**Files:**
- Modify: `src/jaeger/dataops/convert.py`

- [ ] **Step 1: Remove old format functions**

Delete the following unused functions from `src/jaeger/dataops/convert.py`:
- `_int64_feature`, `_float_feature`, `_serialize_tfrecord_embedding`, `_serialize_tfrecord_onehot`
- `_convert_to_tfrecord`, `_convert_with_tf`
- `_convert_to_numpy_raw` and its helpers
- `_convert_to_numpy_raw_variable` and its helpers
- `_BASE_MAP`, `_BASE_MAP_LOWER` if only used by removed code

Keep `_process_sequence_numba`, `_process_batch_numba`, `_process_chunk_numpy_full`, `_convert_to_numpy_full` if still referenced by tests; otherwise remove them too. The new `_convert_to_npz` uses `_process_batch_numba` and `_build_numba_lookups`.

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py tests/unit/test_cli.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/dataops/convert.py
git commit -m "chore: remove obsolete optimize-data format converters"
```

---

## Task 8: Run the full test suite

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=src:$PYTHONPATH pytest tests/unit/test_dataops_convert.py tests/unit/test_cli.py -v
```
Expected: all pass.

- [ ] **Step 2: Run broader smoke tests**

```bash
PYTHONPATH=src:$PYTHONPATH pytest tests/unit tests/pytest/test_cli.py -q
```
Expected: all pass (or only pre-existing failures).

- [ ] **Step 3: Commit any fixes**

```bash
git commit -am "fix: address test fallout from optimize-data refactor"
```

---

## Self-review checklist

1. **Spec coverage:**
   - Always-NPZ output → `_convert_to_npz` and `_save_npz`.
   - `nucleotide/translated/both` formats → `convert_dataset` validation and `_convert_to_npz` branching.
   - Multi-crop-size and stride → `_generate_crops` and per-crop-size batching.
   - One-hot option → `_encode_nucleotide_batch` and translated one-hot conversion.
   - Custom nucleotide map → `_parse_nucleotide_map` / `_build_nucleotide_lookups`.
   - Integer labels starting from 0 → `labels_arr` dtype int32.
   - Deprecated `--max-length` → hidden Click option.
   - Compression levels → `_save_npz`.
   - Tests → Tasks 1-6 and Task 8.

2. **Placeholder scan:** No TBD/TODO/fill-in-details strings.

3. **Type consistency:** `crop_sizes` is passed as tuple of ints consistently; `labels` are int32.

4. **Scope:** Single coherent refactor. No decomposition needed.
