# Design: Opt-in saving of embedding and NMD vectors in `jaeger predict`

## Goal

`jaeger predict` currently writes `_embedding.npz` and `_nmd.npz` files whenever the loaded model exposes `embedding` and `nmd` outputs. These files are not needed by most users and can waste disk space / I/O time. This design makes both outputs opt-in via CLI flags.

## Decision log

| Question | Decision |
|----------|----------|
| Separate or combined flag? | **Separate flags** (`--save-embedding`, `--save-nmd`). |
| Flag names | `--save-embedding` and `--save-nmd`. |
| Default behaviour | **Do not save** either file unless explicitly requested. |
| Scope | Only the modern `jaeger predict` command (`src/jaeger/commands/predict.py`). Legacy predict and training exports are unchanged. |

## Proposed change

### 1. CLI interface (`src/jaeger/cli.py`)

Add two boolean options to the `predict` click command:

```python
@click.option(
    "--save-embedding/--no-save-embedding",
    default=False,
    help="Save the per-window embedding vectors to <sample>_embedding.npz.",
)
@click.option(
    "--save-nmd/--no-save-nmd",
    default=False,
    help="Save the per-window NMD (novelty) vectors to <sample>_nmd.npz.",
)
```

Both flags default to `False`. The `--no-*` variants follow the existing Click convention used elsewhere in the CLI.

### 2. Command implementation (`src/jaeger/commands/predict.py`)

The existing unconditional `np.savez` calls for `embedding` and `nmd` are wrapped with the corresponding flag:

```python
if kwargs.get("save_embedding") and "embedding" in y_pred:
    # write <sample>_embedding.npz

if kwargs.get("save_nmd") and "nmd" in y_pred:
    # write <sample>_nmd.npz
```

When an output is available but the flag is not set, an info log is emitted so users can see why the file is absent:

```python
logger.info(
    "Skipping %s output; pass %s to save it.",
    output_name,
    flag_name,
)
```

The model graph, signature, output keys, and file contents remain unchanged. Only the persistence step is gated.

### 3. Testing

Add/update a smoke test under `tests/smoke/` or `tests/pytest/` that:

1. Runs `jaeger predict` on a tiny model without the new flags.
2. Asserts that `<sample>_embedding.npz` and `<sample>_nmd.npz` are **not** created.
3. Re-runs with `--save-embedding --save-nmd`.
4. Asserts that both files **are** created and contain the expected keys.

## Backwards compatibility

This is a breaking change for workflows that relied on the files being produced automatically. Those workflows can restore the previous behaviour by adding `--save-embedding --save-nmd`. No command-line positional arguments or existing flag behaviour changes.

## Out of scope

- Legacy `predict_legacy` command.
- Embedding-only SavedModel export (`save_embedding_model`).
- Removing `embedding`/`nmd` from the model graph signature or avoiding their computation.
- Any config-file control for these outputs.
