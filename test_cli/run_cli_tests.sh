#!/usr/bin/env bash
# Jaeger CLI smoke-test runner
#
# Usage:
#   ./test_cli/run_cli_tests.sh
#
# All file/output paths and toggles can be overridden through environment
# variables. The defaults below use the bundled test FASTA and write results to
# ./test_cli/outputs/.
#
# Example:
#   TAXDUMP_DIR=/path/to/taxdump ACC2TAX_PATH=/path/to/acc2taxid.gz \
#     ./test_cli/run_cli_tests.sh

set -u
set -o pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----------------------------- configurable inputs --------------------------- #
INPUT_FASTA="${INPUT_FASTA:-$SOURCE_ROOT/src/jaeger/data/test/test_contigs.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_ROOT/test_cli/outputs}"

# Auxiliary utils use the same FASTA by default but can be pointed elsewhere.
UTILS_INPUT_FASTA="${UTILS_INPUT_FASTA:-$INPUT_FASTA}"
FRAGMENT_MINLEN="${FRAGMENT_MINLEN:-1000}"
FRAGMENT_MAXLEN="${FRAGMENT_MAXLEN:-5000}"
FRAGMENT_OVERLAP="${FRAGMENT_OVERLAP:-1000}"

PREDICT_MODEL="${PREDICT_MODEL:-jaeger_38341_1.4M_fragment}"
QUANTIZE_MODEL="${QUANTIZE_MODEL:-$PREDICT_MODEL}"
CONVERT_MODEL="${CONVERT_MODEL:-$PREDICT_MODEL}"

TAXONOMY_MODEL="${TAXONOMY_MODEL:-jaeger_848498a0_1.2M_fragment}"
TAXONOMY_FASTA="${TAXONOMY_FASTA:-$INPUT_FASTA}"
TAXDUMP_DIR="${TAXDUMP_DIR:-}"
ACC2TAX_PATH="${ACC2TAX_PATH:-}"

TRAIN_CONFIG="${TRAIN_CONFIG:-$SOURCE_ROOT/train_config/nn_config.yaml}"
RUN_TRAIN="${RUN_TRAIN:-0}"

REGISTER_MODEL_PATH="${REGISTER_MODEL_PATH:-}"

# --------------------------------------------------------------------------- #

mkdir -p "$OUTPUT_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_DIR/run_$RUN_ID"
mkdir -p "$RUN_DIR"
LOGFILE="$RUN_DIR/cli_smoke_test.log"
PASSED=0
FAILED=0
SKIPPED=0

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "$ts" "$1" | tee -a "$LOGFILE"
}

run_cmd() {
    local name="$1"
    shift
    log "==> $name"
    if "$@" >>"$LOGFILE" 2>&1; then
        log "    PASS: $name"
        ((PASSED += 1))
        return 0
    else
        log "    FAIL: $name"
        ((FAILED += 1))
        return 1
    fi
}

skip_cmd() {
    log "==> $1"
    log "    SKIP: $1"
    ((SKIPPED += 1))
}

# Ensure jaeger is available.
if ! command -v jaeger >/dev/null 2>&1; then
    log "ERROR: 'jaeger' command not found in PATH."
    exit 1
fi

log "Starting Jaeger CLI smoke tests"
log "jaeger: $(command -v jaeger)"
log "INPUT_FASTA: $INPUT_FASTA"
log "OUTPUT_DIR: $OUTPUT_DIR"
log "PREDICT_MODEL: $PREDICT_MODEL"

# ------------------------------- help smoke tests -------------------------- #
HELP_COMMANDS=(
    "jaeger --help"
    "jaeger health --help"
    "jaeger predict --help"
    "jaeger train --help"
    "jaeger download --help"
    "jaeger register-models --help"
    "jaeger utils --help"
    "jaeger utils fragment --help"
    "jaeger utils mask --help"
    "jaeger utils stats --help"
    "jaeger utils convert --help"
    "jaeger utils optimize-data --help"
    "jaeger utils ood-data --help"
    "jaeger utils quantize --help"
    "jaeger utils convert-graph --help"
    "jaeger taxonomy --help"
    "jaeger taxonomy build --help"
    "jaeger taxonomy predict --help"
)

for hcmd in "${HELP_COMMANDS[@]}"; do
    run_cmd "help: $hcmd" bash -c "$hcmd"
done

# --------------------------- real command execution ------------------------- #

run_cmd "jaeger download -l" jaeger download -l

if [[ -f "$INPUT_FASTA" ]]; then
    run_cmd "jaeger health" jaeger health

    PREDICT_OUT="$RUN_DIR/predict"
    run_cmd "jaeger predict" \
        jaeger predict \
            -i "$INPUT_FASTA" \
            -o "$PREDICT_OUT" \
            -m "$PREDICT_MODEL" \
            --cpu \
            --batch 1 \
            --workers 1

    PREDICT_TSV="$(find "$PREDICT_OUT" -maxdepth 2 -name '*.tsv' | head -n 1)"
    if [[ -n "$PREDICT_TSV" && -f "$PREDICT_TSV" ]]; then
        run_cmd "jaeger utils stats" \
            jaeger utils stats \
                -i "$PREDICT_TSV" \
                -o "$RUN_DIR/stats"
    else
        skip_cmd "jaeger utils stats (no TSV produced by predict)"
    fi
else
    skip_cmd "jaeger health (INPUT_FASTA not found)"
    skip_cmd "jaeger predict (INPUT_FASTA not found)"
    skip_cmd "jaeger utils stats (predict not run)"
fi

# ----------------------------- utils subcommands ---------------------------- #

if [[ -f "$UTILS_INPUT_FASTA" ]]; then
    run_cmd "jaeger utils fragment" \
        jaeger utils fragment \
            -i "$UTILS_INPUT_FASTA" \
            -o "$RUN_DIR/fragmented.fasta" \
            --minlen "$FRAGMENT_MINLEN" \
            --maxlen "$FRAGMENT_MAXLEN" \
            --overlap "$FRAGMENT_OVERLAP"

    run_cmd "jaeger utils mask" \
        jaeger utils mask \
            -i "$UTILS_INPUT_FASTA" \
            -o "$RUN_DIR/masked.fasta" \
            --maxperc 0.1

    # jaeger utils convert expects FASTA headers to contain a class annotation
    # (e.g. >seqname__class=0). Build an annotated FASTA on the fly so the
    # bundled test data can still exercise this command.
    ANNOTATED_FASTA="$RUN_DIR/annotated.fasta"
    python3 - "$UTILS_INPUT_FASTA" "$ANNOTATED_FASTA" <<'PY'
import sys
import pyfastx
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w") as out:
    for seq in pyfastx.Fasta(src):
        out.write(f">{seq.name}__class=0\n{seq.seq}\n")
PY

    CONVERTED_CSV="$RUN_DIR/converted.csv"
    if [[ -f "$ANNOTATED_FASTA" ]]; then
        run_cmd "jaeger utils convert (fasta -> csv)" \
            jaeger utils convert \
                -i "$ANNOTATED_FASTA" \
                -o "$CONVERTED_CSV" \
                --itype fasta
    else
        skip_cmd "jaeger utils convert (annotated FASTA not created)"
    fi

    if [[ -f "$CONVERTED_CSV" ]]; then
        run_cmd "jaeger utils optimize-data" \
            jaeger utils optimize-data \
                -i "$CONVERTED_CSV" \
                -o "$RUN_DIR/optimized.npz" \
                --format numpy_full \
                --crop-size 500
    else
        skip_cmd "jaeger utils optimize-data (converted CSV missing)"
    fi

    run_cmd "jaeger utils ood-data" \
        jaeger utils ood-data \
            -i "$UTILS_INPUT_FASTA" \
            -o "$RUN_DIR/ood_shuffled.fasta" \
            --itype fasta \
            --otype fasta \
            -k 1
else
    skip_cmd "jaeger utils fragment (UTILS_INPUT_FASTA not found)"
    skip_cmd "jaeger utils mask (UTILS_INPUT_FASTA not found)"
    skip_cmd "jaeger utils convert (UTILS_INPUT_FASTA not found)"
    skip_cmd "jaeger utils optimize-data (UTILS_INPUT_FASTA not found)"
    skip_cmd "jaeger utils ood-data (UTILS_INPUT_FASTA not found)"
fi

# --------------------------- model conversion / quantization ----------------- #

run_cmd "jaeger utils quantize" \
    jaeger utils quantize \
        -m "$QUANTIZE_MODEL" \
        -o "$RUN_DIR/quantized" \
        --mode dynamic

run_cmd "jaeger utils convert-graph (xla)" \
    jaeger utils convert-graph \
        -m "$CONVERT_MODEL" \
        -o "$RUN_DIR/converted_graph" \
        --mode xla

# -------------------------------- taxonomy ---------------------------------- #

if [[ -n "$TAXDUMP_DIR" && -d "$TAXDUMP_DIR" && -n "$ACC2TAX_PATH" && -f "$ACC2TAX_PATH" && -f "$TAXONOMY_FASTA" ]]; then
    TAXDB="$RUN_DIR/taxonomy_db"
    run_cmd "jaeger taxonomy build" \
        jaeger taxonomy build \
            -i "$TAXONOMY_FASTA" \
            -t "$TAXDUMP_DIR" \
            -a "$ACC2TAX_PATH" \
            -o "$TAXDB" \
            -m "$TAXONOMY_MODEL" \
            --cpu \
            --batch 1

    if [[ -d "$TAXDB" ]]; then
        run_cmd "jaeger taxonomy predict" \
            jaeger taxonomy predict \
                -i "$TAXONOMY_FASTA" \
                -d "$TAXDB" \
                -o "$RUN_DIR/taxonomy_predict" \
                -m "$TAXONOMY_MODEL" \
                --cpu \
                --batch 1
    else
        skip_cmd "jaeger taxonomy predict (taxonomy database not built)"
    fi
else
    skip_cmd "jaeger taxonomy build/predict (TAXDUMP_DIR and/or ACC2TAX_PATH not provided)"
fi

# ---------------------------------- train ----------------------------------- #

if [[ "$RUN_TRAIN" == "1" && -f "$TRAIN_CONFIG" ]]; then
    run_cmd "jaeger train" jaeger train -c "$TRAIN_CONFIG"
elif [[ "$RUN_TRAIN" == "1" ]]; then
    skip_cmd "jaeger train (TRAIN_CONFIG not found: $TRAIN_CONFIG)"
else
    skip_cmd "jaeger train (set RUN_TRAIN=1 to enable)"
fi

# ------------------------------ register-models ----------------------------- #

if [[ -n "$REGISTER_MODEL_PATH" && -d "$REGISTER_MODEL_PATH" ]]; then
    run_cmd "jaeger register-models" \
        jaeger register-models -p "$REGISTER_MODEL_PATH"
else
    skip_cmd "jaeger register-models (REGISTER_MODEL_PATH not set or missing)"
fi

# -------------------------------- summary ----------------------------------- #

TOTAL=$((PASSED + FAILED + SKIPPED))
log "================================================================"
log "CLI smoke-test summary"
log "  Passed:  $PASSED"
log "  Failed:  $FAILED"
log "  Skipped: $SKIPPED"
log "  Total:   $TOTAL"
log "  Run dir: $RUN_DIR"
log "  Log:     $LOGFILE"
log "================================================================"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
exit 0
