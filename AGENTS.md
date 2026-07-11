# AGENTS.md — Jaeger Developer Guide

This file is written for AI coding agents working on **Jaeger** (`jaeger-bio`). It assumes no prior knowledge of the project and only describes what is actually present in the repository.

---

## Project Overview

Jaeger is a command-line bioinformatics tool that uses homology-free deep learning to identify bacteriophage and prophage sequences in metagenomic assemblies. It is distributed as the Python package `jaeger-bio` on PyPI and Bioconda.

- **Package name on PyPI / Bioconda:** `jaeger-bio`
- **CLI entry point:** `jaeger`
- **Current version:** `1.26.4` (defined in `pyproject.toml`, `.cz.toml`, `recipes/jaeger-bio/meta.yaml`, and the release tooling)
- **License:** MIT
- **Python support:** `>=3.11, <3.14`
- **Repository:** https://github.com/Yasas1994/Jaeger
- **Documentation:** https://jaeger.readthedocs.io/

The CLI is implemented with `click` and exposes subcommands such as `predict`, `train`, `download`, `health`, `register-models`, and nested groups `utils` and `taxonomy`.

---

## Technology Stack

- **Language:** Python 3.11–3.13
- **Build backend:** `pdm-backend` (configured in `pyproject.toml`)
- **Package layout:** `src/jaeger/`
- **Deep-learning framework:** TensorFlow 2.21–2.22 + Keras 3.12+
- **CLI framework:** `click`
- **Progress / logging:** `rich` progress bars, standard `logging`
- **Sequence I/O:** `pyfastx`, `pydustmasker`
- **Numerical / scientific:** `numpy`, `scipy`, `pandas`, `polars`, `scikit-learn`, `h5py`
- **Alignment:** `parasail` (pinned to `==1.3.4`)
- **Visualization:** `matplotlib`, `seaborn`, `pycirclize`
- **Change-point / stats:** `ruptures`, `kneed`
- **Configuration templating:** `jinja2` + `pyyaml`
- **Documentation:** Sphinx with `myst-parser` and the `furo` theme
- **Versioning / releases:** Commitizen (`cz`) with conventional commits
- **Linting / formatting:** `ruff` (no configuration in `pyproject.toml`; uses defaults)

Platform-specific TensorFlow extras are declared in `pyproject.toml`:

- `jaeger-bio[cpu]` — CPU-only TensorFlow
- `jaeger-bio[gpu]` — TensorFlow with CUDA
- `jaeger-bio[darwin-arm]` — TensorFlow + `tensorflow-metal` for Apple Silicon
- `jaeger-bio[onnx]` — ONNX Runtime, `tf2onnx`, and dependencies for ONNX inference
- `jaeger-bio[test]` — `pytest`, `pytest-mock`
- `jaeger-bio[taxonomy]` — `taxopy`, `faiss-cpu`

---

## Repository Layout

```
.
├── src/jaeger/                 # Main Python package
│   ├── cli.py                  # Click CLI definition and entry point
│   ├── commands/               # Implementation of CLI subcommands
│   │   ├── predict.py          # Modern SavedModel inference pipeline
│   │   ├── predict_legacy.py   # Legacy .h5 / pickled-model pipeline
│   │   ├── train.py            # Model training
│   │   ├── tune.py             # Fine-tuning
│   │   ├── health.py           # Installation health checks
│   │   ├── downloads.py        # Model download from CKAN
│   │   ├── utils.py            # Utility commands (mask, convert, stats, ...)
│   │   ├── utils_models.py     # Model combination / ensemble helpers
│   │   ├── quantize.py         # TFLite quantization
│   │   ├── convert_graph.py    # XLA / TFLite / ONNX / TensorRT conversion
│   │   ├── taxonomy.py         # Experimental taxonomy pipeline
│   │   └── configs/            # Default bundled config
│   ├── nnlib/                  # Neural-network library
│   │   ├── builder.py          # DynamicModelBuilder from YAML configs
│   │   ├── inference.py        # Inference builders and engines
│   │   ├── conversion.py       # Graph-format conversions
│   │   ├── metrics.py          # Per-class Keras metrics
│   │   ├── v1/                 # Legacy model layers
│   │   └── v2/                 # Current model layers, losses, maps
│   ├── seqops/                 # Sequence operations
│   │   ├── io.py               # FASTA reading, fragment generation
│   │   ├── encode.py           # One-hot / codon / frame encoding
│   │   ├── maps.py             # Codon / amino-acid lookup tables
│   │   ├── transform.py        # Sequence transformations
│   │   ├── stats.py            # Sequence statistics
│   │   ├── synthetic.py        # Synthetic sequence generation
│   │   └── validate.py         # Sequence validation
│   ├── dataops/                # Dataset operations
│   │   ├── dataset.py          # Non-redundant fragment database creation
│   │   ├── convert.py          # Format conversions
│   │   ├── split.py            # Train/val/test splitting
│   │   └── ood.py              # Out-of-distribution / shuffle data
│   ├── preprocess/             # Training preprocessing pipelines
│   │   ├── v1/                 # Legacy preprocessing
│   │   └── v2/                 # Current preprocessing
│   ├── postprocess/            # Prediction aggregation
│   │   ├── collect.py          # Collect predictions into tables
│   │   ├── helpers.py          # Helper transforms
│   │   └── prophages.py        # Prophage extraction
│   ├── data/                   # Bundled data
│   │   ├── config.json         # Default model registry
│   │   ├── models/             # Bundled models (default, experimental, test)
│   │   ├── test/               # Small FASTA fixtures
│   │   ├── loaders.py          # NumPy dataset loaders
│   │   └── tfrecord.py         # TFRecord serialization
│   └── utils/                  # Shared utilities
│       ├── misc.py             # Common helpers, config loading, model discovery
│       ├── fs.py               # Filesystem helpers
│       ├── logging.py          # Logging setup
│       ├── gpu.py              # GPU utilities
│       ├── stats.py            # General statistics
│       ├── termini.py          # Terminal-repeat detection
│       └── test.py             # TensorFlow smoke-test helper
├── tests/                      # Test suite
│   ├── conftest.py             # Shared pytest fixtures
│   ├── pytest/                 # pytest-style tests run by CI
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── smoke/                  # Standalone smoke scripts
├── test_cli/                   # End-to-end CLI smoke-test runner
│   ├── run_cli_tests.sh
│   └── README.md
├── train_config/               # YAML training configurations
├── scripts/                    # Standalone evaluation / conversion scripts
├── singularity/                # Apptainer/Singularity definitions
├── slurm/                      # SLURM job scripts for training
├── recipes/jaeger-bio/         # Bioconda recipe
├── docs/                       # Sphinx documentation
├── pyproject.toml              # Project metadata and build config
├── .cz.toml                    # Commitizen config
├── install.sh                  # One-liner install script
└── CHANGELOG.md
```

---

## Build, Install, and Run

### Editable install for development

```bash
# CPU-only TensorFlow (recommended for local development / CI)
pip install -e ".[cpu,test]"

# GPU install
pip install -e ".[gpu,test]"
```

### Build artifacts

The project uses `pdm-backend`. To build wheels and source distributions:

```bash
pip install pdm
pdm build
```

This produces `dist/` containing an sdist and a wheel.

### Running the CLI

After installation:

```bash
jaeger --version
jaeger health
jaeger predict --help
jaeger predict -i input.fasta -o output_dir
```

---

## Testing

### Test organization

Tests are split into four directories:

- `tests/pytest/` — pytest-discoverable tests that CI runs with `python -m pytest tests/pytest/ -v --tb=short`.
- `tests/unit/` — additional unit tests discovered via `tool.pytest.ini_options` (`testpaths = ["tests/unit", "tests/integration"]`).
- `tests/integration/` — integration tests.
- `tests/smoke/` — standalone scripts that exercise layers, data pipelines, training, etc.

### Running tests locally

```bash
# Run the pytest-discoverable suite
python -m pytest tests/pytest/ -v --tb=short

# Run unit and integration tests
python -m pytest tests/unit tests/integration -v --tb=short

# Smoke tests (each is a standalone script)
python tests/smoke/convert_test.py
python tests/smoke/layers_test.py
python tests/smoke/loss_test.py
python tests/smoke/train_test.py

# End-to-end CLI smoke tests (requires models to be downloaded/registered)
./test_cli/run_cli_tests.sh
```

### CI testing

The `.github/workflows/tests.yml` workflow runs on pushes and PRs to `main` and `dev`:

1. **Lint:** `ruff check src/jaeger` and `ruff format --check src/jaeger`.
2. **Test matrix:** Python 3.11, 3.12, 3.13 on Ubuntu.
   - Installs `-e ".[cpu]"`.
   - Runs `python -m pytest tests/pytest/ -v --tb=short`.
   - Runs the four standalone smoke scripts listed above.
   - Verifies `jaeger health --help`.
   - Builds the wheel and runs `tests/pytest/test_pyproject.py`.

### Test conventions

- Use `from __future__ import annotations` in new test modules.
- Use the shared fixtures in `tests/conftest.py` for common inputs (FASTA paths, CSV paths, small one-hot arrays, random logits).
- Use `click.testing.CliRunner` for CLI entry-point tests (see `tests/unit/test_cli.py`).
- TensorFlow-dependent tests should guard against a missing `tensorflow` install (see `tests/pytest/test_health.py` for `pytest.mark.skipif` usage).
- Smoke tests are self-contained scripts; keep them runnable without pytest.

---

## Code Style Guidelines

- **Formatter / linter:** `ruff` with default rules.
- **Import style:** Use `from __future__ import annotations` in new modules; prefer absolute imports.
- **Type hints:** Encouraged; the codebase uses `dict[str, Any]`, `Path`, and `|` union syntax.
- **String formatting:** Mixed use of f-strings and `.format()`; match surrounding code when editing.
- **Docstrings:** Modules and public functions use Google-style / descriptive docstrings.
- **Logging:** Use the project logger (`logging.getLogger("Jaeger")`) or `jaeger.utils.logging.get_logger`.
- **Progress bars:** Use `jaeger.utils.misc.track_ms` for Rich-based progress tracking.
- **TensorFlow imports:** Package-level environment-variable suppression happens in `src/jaeger/__init__.py` and `src/jaeger/cli.py`. Do not remove these; they prevent noisy native logs.

---

## Configuration and Model Files

### Training configs

Training is driven by YAML files under `train_config/`. The config format supports Jinja2 templating and is parsed by `jaeger.utils.misc.load_model_config` using a two-pass render so nested references resolve.

Key sections in a training config:

- `model.name`, `model.experiment`, `model.seed`
- `model.embedding` — input type (`translated`, etc.), frames, strands, embedding size
- `model.string_processor` — data format (`csv`, `numpy`), crop size, augmentation flags. `crop_size` is canonically in **codons**; `crop_units` (default `codon`, or `nucleotide`) selects the unit. The nucleotide window is `3 * crop_size + 5` (see `jaeger.seqops.crop`), which is the only length where the TensorFlow and numba frame extractors agree on the codon count.
- `model.representation_learner` — stack of `masked_conv1d`, `residual_block`, `transformer`, etc.
- `model.classifier` — classification head
- `model.reliability` — optional reliability / OOD head
- `training.*` — optimizer, batch size, epochs, paths

### Model registry

Models are discovered via `src/jaeger/data/config.json`, which contains:

- `default`: path to the legacy bundled model
- `model_paths`: list of user-registered directories where downloaded or custom models live
- Per-model metadata is generated at runtime by `jaeger.utils.misc.AvailableModels`

To register a new model directory:

```bash
jaeger register-models --path /new/model/path
```

### Bundled models

- `src/jaeger/data/models/default/` — legacy `.h5` model + normalization arrays
- `src/jaeger/data/models/experimental_1/` and `experimental_2/` — experimental legacy models
- `src/jaeger/data/models/test/` — tiny SavedModel used by tests

Modern models (e.g., `jaeger_38341_1.4M_fragment`) are downloaded separately via `jaeger download` because they are too large to bundle.

---

## Runtime Architecture

### Inference pipeline (`jaeger predict`)

1. Parse CLI options and resolve the model path from `config.json`.
2. Validate the input FASTA and count entries with `jaeger.utils.fs.validate_fasta_entries`.
3. Configure TensorFlow device (CPU/GPU, memory limit, mixed precision, XLA).
4. Generate sequence fragments with `jaeger.seqops.io.fragment_generator`.
5. Run inference through either:
   - `jaeger.nnlib.inference.InferModel` / `TFLiteInferModel` / `ONNXEngine` for modern SavedModels
   - `jaeger.commands.predict_legacy.run_core` for legacy `.h5` models
6. Aggregate window-level predictions in `jaeger.postprocess.collect`.
7. Optionally extract prophage regions and generate circular/linear plots.

### Training pipeline (`jaeger train`)

1. Load and render the YAML config with Jinja2.
2. Build the model with `jaeger.nnlib.builder.DynamicModelBuilder`.
3. Load data via the configured `string_processor.data_format` (CSV or NumPy).
4. Train representation learner + classifier + optional reliability head.
5. Save checkpoints and final SavedModel graphs.

---

## Deployment and Distribution

### PyPI

Publishing is handled by `.github/workflows/publish-to-pypi.yaml`:

- Triggered on pushes to `main` and tags matching `v*`.
- Builds with `pdm build`.
- Publishes to TestPyPI from `main`.
- Publishes to production PyPI only from tags on the canonical repo `Yasas1994/Jaeger` using Trusted Publishing (OIDC).

### GitHub Releases

`.github/workflows/release.yml` creates a GitHub Release from tags matching `v*` and pulls release notes from `CHANGELOG.md`.

### Bioconda

`.github/workflows/bioconda-update.yml` is triggered by a published GitHub Release. It updates `recipes/jaeger-bio/meta.yaml` (version + SHA256) and opens a PR against `bioconda/bioconda-recipes`.

The recipe file `recipes/jaeger-bio/meta.yaml` is maintained in-repo and must be kept in sync with `pyproject.toml` dependencies.

### Singularity / Apptainer

Definitions live in `singularity/`:

- `jaeger_singularity.def` — production container using the released PyPI package.
- `jaeger_dev_singularity.def` — development container installing from the `dev` branch.

### SLURM

Example job scripts for training and data conversion are in `slurm/`.

### Documentation

Docs are built with Sphinx. Read the Docs config is in `.readthedocs.yaml`. Install docs dependencies with:

```bash
pip install -r docs/requirements.txt
```

Build locally from `docs/`:

```bash
make html
```

---

## Release Workflow

Jaeger uses a calendar-style version scheme: `<major>.<year>.<update>` (e.g., `1.26.4`).

1. Bump the version:
   ```bash
   .github/scripts/bump-version.sh        # auto-bump
   .github/scripts/bump-version.sh 1 26 5 # explicit
   ```
   The script updates `pyproject.toml`, `.cz.toml`, `recipes/jaeger-bio/meta.yaml`, `singularity/jaeger_singularity.def`, `README.md`, `AGENTS.md`, and docs.

2. Review and commit:
   ```bash
   git diff
   git add -A
   git commit -m "chore(release): bump version to X.Y.Z"
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin main
   git push origin --tags
   ```

   Alternatively, `cz bump` can update `CHANGELOG.md` and create the tag.

3. CI publishes to PyPI, creates a GitHub Release, and opens the Bioconda PR.

For full details, see `docs/_source/releasing.md`.

---

## Security Considerations

- The install script (`install.sh`) downloads and executes code from GitHub; it is intended for end users, not development.
- Model download URLs come from a CKAN catalog in `jaeger.commands.downloads`; verify URLs when changing them.
- PyPI publishing uses OIDC Trusted Publishing; no long-lived API tokens are stored in repository secrets.
- The Bioconda workflow uses a `BIOCONDA_BOT_TOKEN` secret.
- Avoid logging full file paths or environment details beyond what `jaeger health` already reports.
- Imported model graphs (TensorFlow SavedModels, ONNX, TFLite) are executed by native libraries; only load models from trusted sources.

---

## Common Commands Reference

```bash
# Install editable (CPU)
pip install -e ".[cpu,test]"

# Lint
ruff check src/jaeger
ruff format --check src/jaeger

# Format
ruff format src/jaeger

# Tests
python -m pytest tests/pytest/ -v --tb=short
python -m pytest tests/unit tests/integration -v --tb=short

# Build
pdm build

# CLI health check
jaeger health

# Run prediction with a downloaded modern model
jaeger download --model_name jaeger_38341_1.4M --path /path/to/models
jaeger predict -i contigs.fasta -o results --model jaeger_38341_1.4M_fragment

# Train from config
jaeger train -c train_config/nn_config.yaml

# Docs
pip install -r docs/requirements.txt
cd docs && make html
```

---

## Notes for Agents

- Do not remove or weaken the TensorFlow log-suppression code in `src/jaeger/__init__.py` and `src/jaeger/cli.py`; users rely on quiet startup.
- When adding new CLI options, prefer `click.Choice` for enumerated values and keep help text consistent with existing commands.
- Any change to `pyproject.toml` dependencies likely needs a matching update in `recipes/jaeger-bio/meta.yaml`.
- If you change the version, run `.github/scripts/bump-version.sh` rather than editing version strings manually; this keeps all files in sync.
- The `default` model uses the legacy prediction path and is deprecated; new inference work should target modern SavedModels under `jaeger_38341_1.4M_fragment`.
- Crop length is canonicalized in `src/jaeger/seqops/crop.py`: `crop_size` is in codons and the nucleotide window is `3 * codons + 5` (never `3 * codons`). Use `codons_to_nucleotides` / `resolve_crop` instead of multiplying by 3. Existing `665`-codon models correspond to a `2000`-nt window.
