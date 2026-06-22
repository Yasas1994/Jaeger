# Design: Add `--precision` flag to `jaeger train`

## Objective

Unify the training precision options in `jaeger train` so users can choose between
full `float32`, `mixed_float16`, and `mixed_bfloat16`, instead of only having an
on/off switch for fp16.

## Background

Currently `jaeger train` only supports `--mixed_precision`, which hard-codes
`mixed_float16`. Other commands (`predict`, `taxonomy`) already expose a
`--precision` flag with choices `fp32`, `fp16`, `bf16`. This change brings the
train command in line with those commands and lets users experiment with
bfloat16 on supported hardware.

## Design

1. Add a new `--precision` option to the `train_fragment` CLI:
   - choices: `fp32`, `fp16`, `bf16`
   - default: `fp32`
2. Keep `--mixed_precision` as a hidden, deprecated alias:
   - when used, set `precision = "fp16"`
   - emit a `DeprecationWarning`
   - if both flags are provided, raise a `click.UsageError`
3. In `train_fragment_core`, map the resolved precision string to the Keras
   mixed-precision policy:
   - `fp32` → no policy (or `Policy("float32")`)
   - `fp16` → `Policy("mixed_float16")`
   - `bf16` → `Policy("mixed_bfloat16")`
4. Store `config["precision"] = precision` (replacing the old boolean
   `config["mix_precision"]`). The old key is not read anywhere else in the
   codebase, so this is safe.
5. Update help text so `--help` documents the new flag.

## Migration / Backwards Compatibility

Existing scripts that use `--mixed_precision` will continue to work and produce
the same behavior as `--precision fp16`, but will print a deprecation warning.

## Testing

- Unit test that `--precision fp16` sets `mixed_float16`.
- Unit test that `--precision bf16` sets `mixed_bfloat16`.
- Unit test that `--mixed_precision` still works and emits a deprecation warning.
- Unit test that passing both flags raises a usage error.
