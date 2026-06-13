# Jaeger CLI smoke tests

A configurable, end-to-end smoke-test runner for the `jaeger` command-line
interface.

## Quick start

```bash
./test_cli/run_cli_tests.sh
```

Each invocation creates a timestamped run directory under `./test_cli/outputs/`
(e.g. `./test_cli/outputs/run_20260613_192330/`). Results, logs, and all
command outputs for that run are kept inside that directory.

## Configuration

All inputs are configurable through environment variables. The most common ones
are listed below.

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_FASTA` | `src/jaeger/data/test/test_contigs.fasta` | FASTA used for `health`, `predict`, and `utils` |
| `UTILS_INPUT_FASTA` | same as `INPUT_FASTA` | FASTA used specifically for utility commands |
| `OUTPUT_DIR` | `test_cli/outputs` | Directory for all outputs and logs |
| `PREDICT_MODEL` | `jaeger_38341_1.4M_fragment` | Model passed to `predict` |
| `QUANTIZE_MODEL` | same as `PREDICT_MODEL` | Model passed to `utils quantize` |
| `CONVERT_MODEL` | same as `PREDICT_MODEL` | Model passed to `utils convert-graph` |
| `TAXDUMP_DIR` | unset | Path to an NCBI `taxdump` directory (`nodes.dmp`, `names.dmp`, `merged.dmp`) |
| `ACC2TAX_PATH` | unset | Path to an NCBI `accession2taxid` file |
| `TAXONOMY_MODEL` | `jaeger_848498a0_1.2M_fragment` | Embedding model used by `taxonomy` |
| `TRAIN_CONFIG` | `train_config/nn_config.yaml` | Training config (only used if `RUN_TRAIN=1`) |
| `RUN_TRAIN` | `0` | Set to `1` to also run `jaeger train` |
| `REGISTER_MODEL_PATH` | unset | Path to a model directory for `jaeger register-models` |

### Examples

Run with the bundled test data:

```bash
./test_cli/run_cli_tests.sh
```

Use a custom FASTA:

```bash
INPUT_FASTA=/path/to/contigs.fasta ./test_cli/run_cli_tests.sh
```

Run the full taxonomy pipeline (requires `taxdump` and `accession2taxid`):

```bash
TAXDUMP_DIR=/path/to/taxdump \
ACC2TAX_PATH=/path/to/prot.accession2taxid.gz \
./test_cli/run_cli_tests.sh
```

Enable the optional training smoke test:

```bash
RUN_TRAIN=1 TRAIN_CONFIG=/path/to/nn_config.yaml ./test_cli/run_cli_tests.sh
```

## What it does

1. Verifies `--help` for the CLI and every subcommand.
2. Runs real commands with tiny inputs:
   - `jaeger download -l`
   - `jaeger health`
   - `jaeger predict`
   - `jaeger utils {fragment,mask,convert,optimize-data,ood-data}`
   - `jaeger utils stats` (on the `predict` output TSV)
   - `jaeger utils quantize`
   - `jaeger utils convert-graph`
   - `jaeger taxonomy {build,predict}` (only if taxonomy data is provided)
   - `jaeger train` (only if `RUN_TRAIN=1`)
   - `jaeger register-models` (only if `REGISTER_MODEL_PATH` is set)
3. Prints a pass/fail/skip summary and writes a detailed log to
   `$OUTPUT_DIR/cli_smoke_test.log`.

The runner continues after individual failures and exits with a non-zero status
if any command failed.
