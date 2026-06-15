# PyTorch Model Summary Design

## Goal

Provide a Keras `model.summary()`-style display of the Jaeger PyTorch model before training starts. The summary is printed automatically when `jaeger train` runs and is not suppressible.

## Decision log

- **Where it appears:** automatic print before `jaeger train` starts, for both classifier and reliability stages.
- **Content:** layers, output shapes, params per layer, total/trainable params.
- **Layout:** one table per training stage, with branch prefixes on layer names (`rep_model/...`, `classification_head/...`, `reliability_head/...`).
- **Implementation approach:** use the `torchinfo` third-party library.
- **Missing-library behavior:** warn and continue training.

## Context

Jaeger's PyTorch model is built by `jaeger.nnlib.pytorch.builder.ModelBuilder`. The top-level object used for classifier training is `JaegerModel`, which contains:

- `rep_model` — `RepresentationModel` with embedding, blocks, and pooler
- `classification_head` — `ClassificationHead`
- `reliability_head` — optional `ReliabilityHead`

Reliability training reuses `rep_model` through `_ReliabilityPipeline(rep_model, reliability_head)`.

`src/jaeger/commands/train.py` already runs `_initialize_lazy_layers(models, config)` to materialize lazy layers before training.

## Architecture

### New module: `jaeger.nnlib.pytorch.summary`

Create `src/jaeger/nnlib/pytorch/summary.py` containing a `ModelSummary` class.

```python
class ModelSummary:
    def __init__(self, model: nn.Module, input_data: Tuple[torch.Tensor, ...]):
        self.model = model
        self.input_data = input_data

    def summary(self, branch_label: str = "") -> str:
        ...
```

Responsibilities:

1. Import `torchinfo` at module level with a `None` fallback so the dependency remains optional and the module can be monkeypatched in tests.
2. Build a `torchinfo.summary(...)` call with:
   - `input_data=self.input_data`
   - `col_names=["input_size", "output_size", "num_params"]`
   - `row_settings=["var_names"]`
   - `depth=8`
3. Prepend a branch label header (e.g., `"=== Classifier model summary ==="`).
4. Return the formatted string.

If `torchinfo` is unavailable or `torchinfo.summary` raises, log a warning and return an empty string so training continues.

### Integration point

Modify `src/jaeger/commands/train.py`:

1. After `_initialize_lazy_layers(models, config)` succeeds and models are on the target device, construct a fresh dummy input matching the config's `input_type` and `input_shape`. To avoid duplication, factor the dummy-input construction out of `_initialize_lazy_layers` into a shared helper (e.g., `_make_dummy_input(config, device)`) that both `_initialize_lazy_layers` and `ModelSummary` use.
2. **Classifier stage:** before creating the `Trainer`, print the summary for `models["jaeger_classifier"]`.
3. **Reliability stage:** before creating the reliability `Trainer`, print the summary for `_ReliabilityPipeline(rep_model, models["reliability_head"])`.

Both summaries are logged with `logger.info` so they appear in stdout and log files.

### Data flow

```text
config -> ModelBuilder -> models["jaeger_classifier"]
                              |
                              v
                  _initialize_lazy_layers (dummy forward pass)
                              |
                              v
                     ModelSummary(dummy input)
                              |
                              v
                  torchinfo.summary(model, input_data)
                              |
                              v
                       logger.info(output)
                              |
                              v
                         Trainer.fit()
```

## Error handling

| Situation | Behavior |
|---|---|
| `torchinfo` not installed | Log warning, skip summary, continue training |
| `torchinfo.summary` raises (e.g., unsupported layer) | Catch exception, log warning with message, continue training |
| Dummy input shape mismatch | Already guarded by `_initialize_lazy_layers`; reuse same shape-building logic |
| Model has no parameters | `torchinfo` will show zero params; still printed |

The summary step must not mutate model parameters, buffers, or optimizer state.

## Dependencies

Add `torchinfo>=1.8.0` to the main dependencies in `pyproject.toml`.

## Testing

### Unit tests

1. **Happy path:** Build a minimal `JaegerModel`, call `ModelSummary(...).summary()`, assert the returned string contains `"Total params"` and is non-empty.
2. **Missing `torchinfo`:** Patch `jaeger.nnlib.pytorch.summary.torchinfo` to `None`, call `summary()`, assert a warning is logged and no exception is raised.
3. **Failing `torchinfo.summary`:** Patch `torchinfo.summary` to raise, assert a warning is logged and no exception propagates.

### Smoke test

Run a short `jaeger train` invocation (e.g., `--only_save` or `--epochs 0` if supported) with a tiny config and assert the summary header appears in the captured log output.

## Open questions / future work

- Should the summary also be exposed as a standalone CLI command (e.g., `jaeger summary --config ...`)? Out of scope for this design; can be added later without changing the API.
- Should the summary depth be configurable? For now fixed at `depth=8`; can become a config key if users request it.

## Implementation notes

- `torchinfo` is imported at module level in `src/jaeger/nnlib/pytorch/summary.py` (with a `None` fallback) rather than purely lazily inside `summary()`. This makes the module testable via monkeypatching while keeping the public API unchanged.
- `_initialize_lazy_layers` calls `_make_dummy_input(config, device)` without an explicit `length` argument, preserving the original per-input-type default lengths (64 for nucleotide, `crop_size // 3 - 1` for translated).
- Tests that assert on Jaeger log output use a shared `propagate_jaeger_logger` fixture in `tests/conftest.py`, because the Jaeger logger disables propagation by default.
