# Chimeric / mix OOD perturbation for reliability data

## Background

The Jaeger reliability pipeline generates synthetic out-of-distribution (OOD) examples by corrupting in-distribution (ID) sequences that the classifier already predicts confidently. Existing perturbations (`shuffle`, `subseq_repeat`, `tandem_repeat`) operate on a single sequence. Assembly errors, however, can also produce chimeric contigs that join pieces of sequences from different classes. This spec adds a `mix` perturbation that creates such chimeras.

## Goal

Add a `mix` perturbation to `reliability_data_generation.perturbations` that builds a synthetic OOD sequence by splicing together contiguous segments from multiple ID sequences that belong to **different classes**.

## Design

### 1. User-facing configuration

The new perturbation follows the same shape as the existing ones:

```yaml
reliability_data_generation:
  perturbations:
    mix: true
    # or
    mix:
      enabled: true
      n_segments: 2   # number of source sequences / classes to splice
```

Optional `count` or `multiplier` keys also work, reusing the existing `_compute_perturbation_counts` machinery:

```yaml
mix:
  enabled: true
  n_segments: 3
  multiplier: 0.5
```

### 2. New sequence operator: `apply_mix`

Location: `src/jaeger/seqops/synthetic.py`

```python
def apply_mix(
    sequences: list[str],
    output_length: int | None = None,
    pad_value: str = "N",
) -> str:
    ...
```

Behaviour:

1. **Segment allocation** — If `output_length` is given, split it into `len(sequences)` random contiguous segment lengths using `n_segments - 1` random cut points, then sort the cut points. This produces a natural distribution of breakpoints.
2. **Segment extraction** — For each source sequence and its allocated length, pick a random start position and extract a contiguous substring. If the source is shorter than the allocated length, use the whole source.
3. **Concatenate and normalize** — Join the segments in the order the source sequences were supplied. If `output_length` is given, truncate or pad with `pad_value` to that length.
4. If `output_length` is `None`, simply concatenate the full source sequences.

This keeps the helper reusable outside the reliability generator while still producing a true chimera when `output_length` is the crop size.

### 3. Reliability generator integration

Location: `src/jaeger/dataops/reliability_generator.py`

- Extend `_normalize_perturbation_cfg` to parse the `mix` key and create a spec named `mix`.
- The mix spec needs to know the resolved crop size so that `apply_mix` can target the correct output length. `_generate_synthetic_sequences` therefore gains an optional `crop_size` argument:

  ```python
  def _generate_synthetic_sequences(
      records: list[tuple[int, str]],
      multiplier: float,
      perturbations_cfg: dict[str, Any],
      crop_size: int | None = None,
  ) -> list[str]:
      ...
  ```

- When generating mix samples, the generator groups `records` by class label, samples `n_segments` **distinct** labels without replacement, picks one sequence per label, and passes the list of sequences to `apply_mix(..., output_length=crop_size)`.
- Callers of `_generate_synthetic_sequences` (both train and validation synthetic OOD generation) pass the resolved `crop_size` in nucleotides.

### 4. Error handling

- **Insufficient classes:** If the number of distinct labels in the ID records is smaller than `n_segments`, raise `ValueError` with a clear message (e.g. "mix requires at least 3 distinct classes, found 2").
- **Empty records:** If `records` is empty, `_generate_synthetic_sequences` already returns early; no change needed.
- **Segment length larger than source:** Clamp the segment to the source length and pad the final chimera to `output_length`.

### 5. Testing

- **Unit tests for `apply_mix`** in a new or existing test file for `jaeger.seqops.synthetic`:
  - Output length equals `output_length`.
  - Output contains material from each input when segments are non-empty.
  - Padding is applied when combined segments are too short.
  - Truncation is applied when combined segments are too long.
- **Unit tests for `_normalize_perturbation_cfg`**:
  - `mix: true` produces a `mix` spec with default `n_segments=2`.
  - Structured config with `n_segments: 3` is parsed correctly.
- **Unit tests for generator-level class enforcement**:
  - Mixing records with fewer distinct labels than `n_segments` raises `ValueError`.
  - Generated mix sequences are label `0` (OOD) and have length equal to `crop_size`.
- **Smoke test:** Enable `mix: true` in the existing reliability generator smoke test and ensure it still passes.

## Open questions / future work

- Should we allow same-class mixing with a configurable probability? The initial implementation forces distinct classes, which is the stricter and more defensible OOD definition.
- Should breakpoints be sampled uniformly along the crop or biased toward the middle? Random cut points are simple and sufficient for a first version.
