# Jaeger — Agent Instructions

## Project background

Jaeger (`jaeger-bio`) is a homology-free deep-learning command-line tool for identifying bacteriophage genome sequences. The repository lives at `/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger` and uses Python, PyTorch, NumPy, and YAML training configs.

## Memory protocol — MemPalace

This project uses [MemPalace](https://mempalaceofficial.com/) for durable, searchable memory across Kimi Code sessions. You MUST follow this protocol on every turn:

1. **On wake-up / session start**: call `mcp__mempalace__mempalace_status` to load the palace overview and orient yourself.
2. **Before answering** any question about past plans, decisions, people, or project facts: call `mcp__mempalace__mempalace_search` or `mcp__mempalace__mempalace_kg_query`. Do not guess.
3. **After creating or updating a plan**: save it immediately with `mcp__mempalace__mempalace_add_drawer`:
   - `wing`: `jaeger`
   - `room`: `plans`
   - `topic`: `plan`
   - `content`: the full plan text, including goals, tasks, exact file paths, and decisions.
4. **After making a decision**: record it with `mcp__mempalace__mempalace_kg_add`:
   - Example: `subject="Jaeger project"`, `predicate="decided_to"`, `object="adopt AGENTS.md memory protocol"`, `valid_from="YYYY-MM-DD"`.
5. **After each session**: write a diary entry with `mcp__mempalace__mempalace_diary_write`:
   - `agent_name`: `kimi-code`
   - `wing`: `jaeger`
   - `topic`: `session-summary`
   - `entry`: a concise, factual summary of what was discussed, decided, implemented, and left unfinished.
6. **When facts change**: invalidate the old fact with `mcp__mempalace__mempalace_kg_invalidate` and add the corrected fact with `mcp__mempalace__mempalace_kg_add`.
7. **Do not bulk-delete drawers or knowledge-graph facts** without explicit user approval.

## Tool permission note

Read/search MemPalace tools (`mempalace_status`, `mempalace_search`, `mempalace_get_*`, `mempalace_kg_query`) are pre-approved in `~/.kimi-code/config.toml`. Write tools (`add_drawer`, `kg_add`, `diary_write`) will trigger approval prompts unless the user enables YOLO mode.

---

# Agent working notes

The sections below supplement the README and `docs/_source/` with conventions and pitfalls that are useful when editing code or answering questions about Jaeger.

## Project basics

| Item | Value |
|---|---|
| PyPI package | `jaeger-bio` |
| Import package | `jaeger` |
| Current version | `1.27.1` |
| Python support | `>=3.11, <3.14` |
| Build backend | `pdm-backend` |
| Source layout | `src/jaeger/` |
| CLI entry point | `jaeger = jaeger.cli:main` |
| License | MIT |

## Development workflow

Install in editable mode for development:

```bash
pip install -e ".[gpu,test]"    # or [cpu,test], [darwin-arm,test]
```

Run the test suite:

```bash
pytest
```

`pyproject.toml` configures `testpaths = ["tests/unit", "tests/integration"]`. Smoke and CLI tests also live under `tests/smoke/`, `tests/pytest/`, and `test_cli/`.

Build a release artifact:

```bash
pdm build
```

Check the installation health:

```bash
jaeger health
```

## Code organization

| Path | Purpose |
|---|---|
| `src/jaeger/cli.py` | Click CLI definition, PyTorch/legacy model routing, and TF import suppression for legacy workflows |
| `src/jaeger/commands/` | Command implementations: `predict`, `predict_legacy`, `train`, `tune`, `health`, `quantize`, `convert_graph`, `taxonomy`, `downloads`, `test`, plus `utils*.py` |
| `src/jaeger/nnlib/` | Neural-network library: `v1/` and `v2/` layers, losses, metrics, inference, conversion, builder |
| `src/jaeger/dataops/` | Data operations: `convert.py`, `dataset.py`, `ood.py`, `split.py` |
| `src/jaeger/data/` | Bundled data (`config.json`), TFRecord helpers, dataset loaders |
| `src/jaeger/seqops/` | Sequence operations: encode, maps, stats, synthetic, transform, validate, io |
| `src/jaeger/postprocess/` | Post-processing: `collect.py`, `helpers.py`, `prophages.py` |
| `src/jaeger/preprocess/` | Preprocessing maps/converters (`v1/`, `v2/`) |
| `src/jaeger/utils/` | Utilities: logging, filesystem, misc, stats, termini, GPU, test helpers |
| `train_config/` | YAML training configuration templates |
| `scripts/` | Standalone conversion scripts for NumPy/TFRecord formats |
| `recipes/jaeger-bio/` | Bioconda recipe |
| `.github/scripts/` | `bump-version.sh` and dependency helpers |
| `.github/workflows/` | PyPI publish, GitHub Release, and Bioconda update workflows |

## CLI conventions

- Commands are built with **Click** and usually set `context_settings={"show_default": True}`.
- Verbosity is a **count flag**: default is warning, `-v` is info, `-vv` is debug.
- Modern SavedModels (e.g., `jaeger_38341_1.4M_fragment`) are recommended; the bundled `default` model and the `experimental_*` models use the **legacy prediction workflow** and are deprecated.
- Model discovery uses `src/jaeger/data/config.json` under the `model_paths` key. Register a new model directory with:
  ```bash
  jaeger register-models --path /path/to/my_model
  ```
- Download additional models with:
  ```bash
  jaeger download --list
  jaeger download --model_name jaeger_38341_1.4M --path /path/to/store/models
  ```

## PyTorch backend and environment notes

Starting with v1.27.1, Jaeger uses **PyTorch** as its primary training and inference backend. The `jaeger train` and `jaeger predict` commands run through the PyTorch path by default.

TensorFlow is still imported only for the deprecated legacy prediction workflow (`jaeger predict-legacy`) and bundled `default`/`experimental_*` models. When TensorFlow is imported, Jaeger aggressively suppresses TF and low-level C++ logging because the CLI is user-facing:

- `TF_CPP_MIN_LOG_LEVEL=3`
- `TF_ENABLE_ONEDNN_OPTS=0`
- `GRPC_VERBOSITY=ERROR`
- `GLOG_minloglevel=2`
- `ABSL_MIN_LOG_LEVEL=2`

These variables are set in `src/jaeger/__init__.py` and again at the top of `src/jaeger/cli.py` **before** the first `import tensorflow`.

`src/jaeger/cli.py` also contains a compatibility patch for TensorFlow 2.18 + Python 3.12 that wraps `tensorflow.python.framework.tensor_util.is_tf_type`. Do not move `import tensorflow` before this patch.

On Linux, `XLA_FLAGS` is set to `--xla_gpu_cuda_data_dir=/usr/lib/cuda` to support XLA JIT (`--xla`) for the legacy TensorFlow path.

## Training data and model formats

### Data formats

Training data is loaded from CSV by default and preprocessed on-the-fly. For large training jobs, convert to an optimized format and set `model.string_processor.data_format`:

| Format | Speedup | Best for |
|---|---|---|
| `csv` | 1.0× | Small datasets, quick experiments |
| `tfrecord` | ~12× | Datasets too large for RAM |
| `numpy_raw` | ~17× | Fast loading + runtime augmentations |
| `numpy_raw_variable` | ~3× | Variable-length sequences |
| `numpy_full` | ~9× | Maximum throughput, no augmentations |

Convert CSV to any optimized format with:

```bash
jaeger utils optimize-data -i train.csv -o train.npz --format numpy_full
```

Standalone conversion scripts are also available in `scripts/`.

### Inference backends

`jaeger predict` uses PyTorch by default. Several optional backends and precision modes are available for legacy TensorFlow SavedModels and converted graphs:

- `--precision fp16|bf16` — mixed-precision inference (legacy TensorFlow path)
- `--xla` — XLA JIT compilation (legacy TensorFlow path)
- `--onnx` — ONNX Runtime inference (requires `jaeger utils convert-graph --mode onnx` first)
- `--onnx --int8` — INT8 quantized ONNX
- `--quantized dynamic|float16|full_int8` — TFLite quantization

## Release workflow

Jaeger uses a calendar-style version format `1.<year>.<update>` (e.g., `1.26.4`).

1. Bump the version:
   ```bash
   .github/scripts/bump-version.sh
   ```
   This updates `pyproject.toml`, `.cz.toml`, `recipes/jaeger-bio/meta.yaml`, `singularity/jaeger_singularity.def`, `README.md`, `AGENTS.md`, and `docs/_source/usage.md`.
2. Commit and tag:
   ```bash
   git add -A
   git commit -m "chore(release): bump version to X.Y.Z"
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   ```
   Or use `cz bump` to update `CHANGELOG.md` and create the tag.
3. Push to the canonical repository `Yasas1994/Jaeger`:
   ```bash
   git push origin main
   git push origin --tags
   ```
4. CI workflows run in sequence:
   - `publish-to-pypi.yaml` — builds and publishes to PyPI via Trusted Publishing (OIDC)
   - `release.yaml` — creates a GitHub Release from the tag and `CHANGELOG.md`
   - `bioconda-update.yaml` — opens a Bioconda PR after the GitHub Release is published

Only `Yasas1994/Jaeger` can publish to PyPI and Bioconda; forks build artifacts for validation only.

Manual fallback:

```bash
pdm build
pdm publish
```

## Testing conventions

- Use `pytest` for all tests.
- Unit and integration tests live in `tests/unit/` and `tests/integration/`.
- Smoke and CLI-specific tests are in `tests/smoke/`, `tests/pytest/`, and `test_cli/`.
- Shared fixtures are in `tests/conftest.py`.
- CLI tests use `click.testing.CliRunner` and import from `jaeger.cli`.
- Keep TensorFlow quiet in tests by setting `TF_CPP_MIN_LOG_LEVEL=2` and `GRPC_VERBOSITY=ERROR` in fixtures or test setup when needed.

## Common agent pitfalls

- Do not change the PyPI package name (`jaeger-bio`); the import package remains `jaeger`.
- Do not move `import tensorflow` above the environment-variable setup in `cli.py` or `__init__.py`.
- Do not delete the legacy `predict_legacy` path or the `default` model config; mark deprecated workflows as deprecated instead.
- When adding new CLI commands, follow the existing Click patterns and add corresponding help tests.
- When updating the version, use `.github/scripts/bump-version.sh` so all versioned files stay in sync.

---

# Cluster Agent Instructions

## Cluster : zeus

**Project directory:** /mnt/beegfs/bioinf/wijesekara/jaeger

## File Structure

```
project-root/
├── data/            # Input datasets
├── configs/         # Jaeger training configs
├── experiments/     # Training outputs: checkpoints, metrics, logs
├── slurm/
│   ├── logs/        # Job stdout/stderr
│   └── scripts/     # Batch scripts
├── tmp/             # Scratch files
├── container/       # Singularity container image(s)
├── Jaeger/          # Jaeger source code (bound into container at runtime)
```

## Partitions & Resources

| Partition | Nodes     | GPU     | Max time  | CPUs | Mem  |
|-----------|-----------|---------|-----------|------|------|
| `gpu`     | 030       | 2× L40S | 72:00:00  | 128  | 500G |
| `batch`   | 010 – 016 | —       | 72:00:00  | 40   | 220G |
| `batch`   | 001 – 008 | —       | 72:00:00  | 8    | 220G |

## Agent Behaviour

- Do not modify `AGENTS.md`.
- When uncertain about something related to running a  jobs on zeus, search the mempalace first.
- Update mempalace when you make progress, a new decision, or a plan.

## Rules

- Always run jobs via the Singularity container in `container/`, binding the `Jaeger/` source directory into it.
- Always test on `gpu` with a small number of steps or a small dataset before full runs.
- Save checkpoints regularly; resume from the latest checkpoint on restart.
- Do not use or configure conda environments to run jobs.
- If you encounter a bug in Jaeger, fix it locally, test the fix, then sync the updated source code to zeus.
