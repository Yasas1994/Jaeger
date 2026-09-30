#!/usr/bin/env python
# ruff: noqa: E402

"""
Copyright 2024 R. Y. Wijesekara - University Medicine Greifswald, Germany

Identifying phage genome sequences concealed in metagenomes is a
long standing problem in viral metagenomics and ecology.
The Jaeger approach uses homology-free machine learning to identify
both phages and prophages in metagenomic assemblies.
"""

import os
import sys
import time

# Suppress TensorFlow and C++ warnings before any TF imports.
# These must be set before the first tensorflow import.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom ops warning
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC logs
os.environ["GLOG_minloglevel"] = "2"  # Suppress glog INFO messages
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

if sys.platform.startswith("linux"):
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

# Capture wall-clock start time as early as possible for the final summary.
_START_TIME = time.perf_counter()


# Patch TensorFlow 2.18 + Python 3.12 compatibility bug before any TF imports.
# TF's internal _DictWrapper objects fail inspect.getattr_static during
# isinstance checks with typing generics, causing SavedModel loading to crash.
# This bug is fixed in TensorFlow 2.21+.
# See: https://github.com/tensorflow/tensorflow/issues/78784
# We also temporarily silence native stderr during this first TF import to
# suppress harmless but noisy startup messages (oneDNN, CUDA plugin checks,
# absl pre-init warnings, etc.).
def _silence_tf_import():
    import os as _os

    stderr_fd = _os.dup(2)
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    try:
        _os.dup2(devnull, 2)
        import tensorflow as tf  # noqa: F401
        from tensorflow.python.framework import tensor_util

        _original_is_tf_type = tensor_util.is_tf_type

        def _patched_is_tf_type(x):
            try:
                return _original_is_tf_type(x)
            except TypeError:
                return False

        # Detect the bug by trying the actual _DictWrapper class
        from tensorflow.python.trackable.data_structures import _DictWrapper

        try:
            _original_is_tf_type(_DictWrapper())
        except TypeError:
            tensor_util.is_tf_type = _patched_is_tf_type
    finally:
        _os.dup2(stderr_fd, 2)
        _os.close(devnull)
        _os.close(stderr_fd)


try:
    _silence_tf_import()
except Exception:
    pass  # TF not installed, incompatible version, or bug already fixed
finally:
    del _silence_tf_import

import click
import logging
from pathlib import Path
from importlib.metadata import version
from importlib.resources import files
from jaeger.utils.misc import json_to_dict, add_data_to_json, AvailableModels
import warnings
from jaeger.cli_utils import utils

warnings.filterwarnings("ignore")

if sys.platform.startswith("linux"):
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"


USER_MODEL_PATHS = json_to_dict(files("jaeger.data") / "config.json").get("model_paths")
DEFAULT_MODEL_PATH = [files("jaeger.data")]
AVAIL_MODELS = [
    i
    for i in AvailableModels(path=DEFAULT_MODEL_PATH + USER_MODEL_PATHS).info.keys()
    if i != "jaeger_fragment"
] + ["default"]

# Models that still use the legacy prediction workflow.
LEGACY_PREDICT_MODELS = {"default", "experimental_1", "experimental_2"}

logger = logging.getLogger("Jaeger")


@click.group()
@click.version_option(version("jaeger-bio"), prog_name="jaeger")
def main():
    "Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences."


@click.command()
@click.option(
    "-v", "--verbose", count=True, help="Verbosity level: -vv debug, -v info", default=1
)
def health(**kwargs):
    """Runs tests to check health of the installation."""
    from jaeger.commands.health import health_core

    health_core(**kwargs)


@click.command(context_settings={"show_default": True})
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "--fsize",
    type=int,
    default=2000,
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=1500,
    help="The gap between two the sliding windows",
)
@click.option(
    "--dynamic-stride",
    is_flag=True,
    help="For short contigs, distribute windows evenly so the final window reaches the contig end.",
)
@click.option(
    "--dynamic-stride-threshold",
    type=float,
    default=10.0,
    help="Multiplier of --fsize below which dynamic stride is applied (default: 10.0).",
)
@click.option(
    "--crf",
    is_flag=True,
    help="(experimental) Decode per-window predictions jointly with a linear-chain "
    "CRF (Viterbi) instead of independent per-window argmax.",
)
@click.option(
    "--crf-switch-cost",
    type=float,
    default=2.0,
    help="(experimental) Global CRF transition cost lambda in log-probability units. "
    "Higher values smooth more aggressively.",
)
@click.option(
    "--crf-prior",
    type=click.Choice(["biological", "uniform"]),
    default="biological",
    help="(experimental) CRF transition-cost prior. 'biological' encodes class "
    "co-occurrence priors (cheap bacteria<->phage switches, expensive "
    "eukarya<->phage); 'uniform' uses the same cost for every class switch.",
)
@click.option(
    "--crf-transition-matrix",
    type=click.Path(exists=True),
    default=None,
    help="(experimental) Path to a JSON file with a custom class-name-keyed "
    'transition-cost matrix (e.g. {"bacteria": {"phage": 0.5}}). '
    "Overrides --crf-prior.",
)
@click.option(
    "--dustmask/--no-dustmask",
    default=True,
    help="Mask low-complexity regions with pydustmasker before fragmenting. "
    "Use --no-dustmask to disable.",
)
@click.option(
    "--min-len",
    type=int,
    default=None,
    help="Minimum contig length to classify (default: --fsize). "
    "Contigs between --min-len and --fsize are classified as a single whole-contig window.",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--model_path",
    default=None,
    help=("Give the path to a model. overrides --model"),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (e.g., when using Apptainer or Docker)",
)
@click.option(
    "-p",
    "--prophage",
    is_flag=True,
    help="Extract and report prophage-like regions",
)
@click.option(
    "-s",
    "--sensitivity",
    type=float,
    default=1.5,
    help="Sensitivity of the prophage extraction algorithm (0-4)",
)
@click.option(
    "--lc",
    type=int,
    default=500_000,
    help="Minimum contig length for prophage extraction",
)
@click.option(
    "--plot-type",
    type=click.Choice(["circular", "linear", "both", "none"]),
    default="none",
    help="Prophage plot type: circular, linear, both, or none (default)",
)
@click.option(
    "--interactive-plot",
    is_flag=True,
    help="Generate interactive HTML genome plots for prophage regions",
)
@click.option(
    "--filter-low-gene-density",
    is_flag=True,
    help="Filter out prophage predictions with gene density lower than host (only for complete genomes)",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions",
)
@click.option(
    "--uc",
    type=float,
    default=0.5,
    help="Reliability score below which contigs are labeled 'uncertain' "
    "instead of the argmax class (default: 0.5)",
)
@click.option(
    "--pc",
    type=float,
    default=0.5,
    help="Minimum phage probability required to accept predictions (0-1)",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option("--workers", type=int, default=4, help="Number of threads to use")
@click.option(
    "--chunk-size",
    type=int,
    default=0,
    help="Process the input in chunks of N contigs to limit memory (0 = disabled)",
)
@click.option(
    "--no-banner",
    is_flag=True,
    hidden=True,
    help="Suppress the Jaeger banner in logs (used internally for chunk subprocesses)",
)
@click.option(
    "--window-scores",
    is_flag=True,
    help="Writes window-wise prediction scores and per-window metadata to a .npz file",
)
@click.option(
    "--getsequences",
    is_flag=True,
    help="Writes the putative phage sequences to a .fasta file",
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "--getalllabels",
    is_flag=True,
    help="Get predicted labels for Non-Viral contigs",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files")
@click.option(
    "--quantized",
    type=click.Choice(["dynamic", "float16", "full_int8"]),
    help="Use quantized TFLite model for inference (dynamic|float16|full_int8)",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp32",
    help="GPU inference precision: fp32 (default), fp16, or bf16. fp16/bf16 reduce memory and may speed up inference on compatible GPUs.",
)
@click.option(
    "--xla",
    is_flag=True,
    help="Enable XLA JIT compilation for inference. May provide 2-3x speedup on GPU after initial compilation overhead.",
)
@click.option(
    "--onnx",
    is_flag=True,
    help="Use ONNX Runtime for inference. Requires converting the model first with 'jaeger utils convert-graph --mode onnx'.",
)
@click.option(
    "--int8",
    is_flag=True,
    help="Use INT8 quantized ONNX model (use with --onnx). Requires 'jaeger utils convert-graph --mode onnx --int8'.",
)
@click.option(
    "--save-embedding",
    is_flag=True,
    help="Save per-window embedding vectors to <sample>_embedding.npz",
)
@click.option(
    "--save-nmd",
    is_flag=True,
    help="Save per-window NMD (novelty) vectors to <sample>_nmd.npz",
)
@click.option(
    "--refine",
    is_flag=True,
    help="Apply post-hoc refinement using the model's shipped calibration file.",
)
@click.option(
    "--refine-mode",
    type=click.Choice(["gated", "weighted", "unweighted"]),
    default="gated",
    help="Contig aggregation mode when --refine is active.",
)
@click.option(
    "--refine-min-windows",
    type=int,
    default=3,
    help="Minimum informative windows required for a refined contig call.",
)
@click.option(
    "--refine-merge-split",
    type=click.Choice(["half", "full"]),
    default="half",
    help="How merged-label windows split weight between constituent classes.",
)
@click.option(
    "--refine-allow-merged-contig-call",
    is_flag=True,
    help="Allow contig-level hedged calls (bacteria_or_plasmid / virus_any).",
)
@click.option(
    "--refine-contig-hedge-margin",
    type=float,
    default=1.0,
    help="Margin threshold for contig-level hedge (when --refine-allow-merged-contig-call).",
)
def predict(**kwargs):
    """
    Runs Jaeger on a dataset
    """
    model = kwargs.get("model")
    model_path = kwargs.get("model_path")
    config = kwargs.get("config")

    if model_path:
        model = "from_path"
    else:
        if model is None:
            model = "default"
        elif config is None and model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )

    """Run jaeger inference pipeline."""
    if model in LEGACY_PREDICT_MODELS:
        import click as _click

        _click.echo(
            _click.style(
                f"Warning: model '{model}' uses the legacy prediction workflow and is deprecated. "
                "It will be removed in a future release. Please migrate to a modern model "
                "(e.g., 'jaeger_38341_1.4M_fragment').",
                fg="yellow",
            ),
            err=True,
        )

    chunk_size = kwargs.pop("chunk_size", 0)

    try:
        if model == "default":
            from jaeger.commands.predict_legacy import run_core

            run_core(**kwargs)
        else:
            from jaeger.commands.predict import run_core

            run_core(chunk_size=chunk_size, argv=sys.argv, **kwargs)
    finally:
        elapsed = time.perf_counter() - _START_TIME
        click.echo(
            click.style(
                f"Jaeger finished in {elapsed:.2f}s",
                fg="green",
                bold=True,
            )
        )


@click.command(context_settings={"show_default": True})
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input FASTA file",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available choices: {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--model_path",
    default=None,
    help=("Give the path to a model. overrides --model"),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (e.g., when using Apptainer or Docker)",
)
@click.option(
    "--fsize",
    type=int,
    default=2000,
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=1500,
    help="The gap between two the sliding windows",
)
@click.option(
    "--min-len",
    type=int,
    default=None,
    help="Minimum contig length to process (default: --fsize). "
    "Shorter contigs are skipped.",
)
@click.option(
    "--batch",
    type=int,
    default=16,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option("--workers", type=int, default=4, help="Number of threads to use")
@click.option(
    "--class-index",
    type=int,
    default=None,
    help="Class index to compute saliency for (default: the predicted class per window)",
)
@click.option(
    "--flavour",
    type=click.Choice(["raw", "gradxinput", "integrated"]),
    default="raw",
    help="Attribution method: raw (sum of |gradient| over embedding dims), "
    "gradxinput (gradient x (embedding - null baseline), signed), or "
    "integrated (integrated gradients from the null baseline; slow, eager "
    "mode). The null baseline is the embedding of input id 0 (N/pad codon).",
)
@click.option(
    "--ig-steps",
    type=int,
    default=50,
    help="Number of interpolation steps for --flavour integrated",
)
@click.option(
    "--all-classes",
    is_flag=True,
    help="Compute saliency/attribution for every class (not just the "
    "predicted class); stored as saliency_by_class in the npz",
)
@click.option(
    "--store-gradients",
    is_flag=True,
    help="Also store raw per-embedding-dimension gradients in the npz file "
    "(only with --flavour raw)",
)
@click.option(
    "--whole-sequence",
    is_flag=True,
    help="Use each contig as a single (padded) window instead of sliding windows",
)
@click.option(
    "--dustmask/--no-dustmask",
    default=True,
    help="Mask low-complexity regions with pydustmasker before fragmenting. "
    "Use --no-dustmask to disable.",
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp32",
    help="GPU precision: fp32 (default), fp16, or bf16.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files")
def interpret(**kwargs):
    """Generate gradient-based saliency maps for model interpretation.

    Rebuilds the Keras model from the model's project recipe and weights
    (unlike `jaeger predict`, which runs the frozen SavedModel graph), then
    computes the gradient of the predicted-class probability with respect to
    the codon embeddings for every sliding window.
    """
    model = kwargs.get("model")
    if kwargs.get("model_path") is None and model not in AVAIL_MODELS:
        raise click.BadParameter(
            f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
        )

    from jaeger.commands.interpret import run_core

    run_core(**kwargs)


@click.command(context_settings={"show_default": True})
@click.option(
    "-c",
    "--config",
    type=click.Path(
        exists=True,
        file_okay=True,
    ),
    required=None,
    help="Path to tuning configuration file (YAML)",
)
@click.option("-n", required=True, help="Model name")
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
def tune(**kwargs):
    """Fine-tune existing models on custom databases."""
    from jaeger.commands.tune import tune_core

    tune_core(**kwargs)


@click.command(context_settings={"show_default": True})
@click.option(
    "-c",
    "--config",
    type=click.Path(
        exists=True,
        file_okay=True,
    ),
    required=True,
    help="Path to training configuration file (YAML)",
)
@click.option(
    "--only_classification_head",
    is_flag=True,
    required=False,
    help="Only train the classification head without the updating the representation learner's weights",
)
@click.option(
    "--only_reliability_head",
    is_flag=True,
    required=False,
    help="Only train the reliability model head",
)
@click.option(
    "--self_supervised_pretraining",
    is_flag=True,
    required=False,
    help="retrain representation model with self supervised learning",
)
@click.option(
    "--only_heads",
    is_flag=True,
    required=False,
    help="Only train the reliability model head",
)
@click.option(
    "--from_last_checkpoint",
    is_flag=True,
    required=False,
    help="continue training from the last checkpoints",
)
@click.option(
    "--force",
    is_flag=True,
    required=False,
    help="delete existing checkpoints and continue",
)
@click.option(
    "--ignore_convergence",
    is_flag=True,
    required=False,
    help="Ignore convergence markers and re-train from the last checkpoint",
)
@click.option(
    "--save_model",
    is_flag=True,
    required=False,
    help="save a model with weights from the last checkpoint",
)
@click.option(
    "--masking/--no-masking",
    "masking",
    default=None,
    help="Enable/disable sequence masking in convolutional/normalization layers. "
    "Defaults to the config value (model.use_masking) or True.",
)
@click.option(
    "--only_save",
    is_flag=True,
    required=False,
    help="save the model with weights from last checkpoints without training",
)
@click.option(
    "--generate_reliability_data",
    is_flag=True,
    default=False,
    help="Generate reliability training data after classifier training",
)
@click.option(
    "--tmp",
    type=click.Path(file_okay=False),
    default=None,
    help="Fast temporary directory for intermediate files (e.g. local scratch). "
    "Reliability NPZ files are written here first and then copied to the output path.",
)
@click.option(
    "--id_threshold",
    type=float,
    default=None,
    show_default=True,
    help="Confidence threshold for selecting ID / high-confidence OOD samples (default: 0.8)",
)
@click.option(
    "--synthetic_ood_threshold",
    type=float,
    default=None,
    show_default=True,
    help="Confidence threshold for synthetic OOD samples (default: 0.8)",
)
@click.option(
    "--synthetic_ood_multiplier",
    type=float,
    default=None,
    show_default=True,
    help="Number of synthetic OOD sequences to generate per real training sequence (default: 1.0)",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"], case_sensitive=False),
    default="fp32",
    show_default=True,
    help="Numeric precision: fp32, fp16 (mixed_float16), or bf16 (mixed_bfloat16).",
)
@click.option(
    "--mixed_precision",
    is_flag=True,
    required=False,
    hidden=True,
    help="Deprecated: use --precision fp16 instead.",
)
@click.option(
    "--xla",
    is_flag=True,
    required=False,
    help="Enable XLA JIT compilation for training",
)
@click.option(
    "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of CPU threads for TensorFlow intra- and inter-op parallelism",
)
@click.option(
    "--meta",
    type=click.Path(
        exists=False,
        file_okay=True,
    ),
    required=False,
    help="Path to write metadata from the container",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
def train(**kwargs):
    """Train new models on custom databases from scratch.

    # to start training from scratch
    jaeger train -c training/test8.yaml

    # to resume training (representation learner and heads) from latest checkpoints
    jaeger train -c training/test8.yaml --from_last_checkpoint

    # to resume training the classification head from the latest classification model checkpoint
    jaeger train -c training/test8.yaml --from_last_checkpoint --only_classification_head

    # to resume training the reliability model head from the latest reliability model checkpoint
    jaeger train -c training/test8.yaml --from_last_checkpoint --only_reliability_head


    # to resume training the reliability model head and classification model head from the latest model checkpoints
    jaeger train -c training/test8.yaml --from_last_checkpoint --only_heads

    """

    # Collect which flags are True
    selected = [
        flag
        for flag, value in {
            "only_classification_head": kwargs.get("only_classification_head", False),
            "only_reliability_head": kwargs.get("only_reliability_head", False),
            "only_heads": kwargs.get("only_heads", False),
        }.items()
        if value
    ]

    # Check mutual exclusivity
    if len(selected) > 1:
        raise click.UsageError(
            f"{', '.join('--' + flag for flag in selected)} are mutually exclusive. Please specify only one."
        )

    if kwargs.get("from_last_checkpoint") and kwargs.get("force"):
        raise click.UsageError(
            "--from_last_checkpoint and --force are mutually exclusive. Please specify only one."
        )

    precision = kwargs.get("precision", "fp32")
    if kwargs.get("mixed_precision", False):
        if precision != "fp32":
            raise click.UsageError(
                "--mixed_precision and --precision are mutually exclusive. "
                "Use --precision fp16 instead of --mixed_precision."
            )
        warnings.warn(
            "--mixed_precision is deprecated; use --precision fp16 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs["precision"] = "fp16"

    from jaeger.commands.train import train_fragment_core

    train_fragment_core(**kwargs)


@click.command(context_settings={"show_default": True})
@click.option(
    "-p",
    "--path",
    type=click.Path(file_okay=False),
    required=True,
    help="Path to model weights, graph and configuration files",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=False,
    help="Path to jager config file (usefule when using Apptainer or Docker)",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
def register_models(**kwargs):
    """Appends newly trained and fine-tuned models to the model path"""

    if kwargs.get("config") is not None:
        # update an external config file
        path = Path(kwargs.get("config"))
    else:
        # update internal config file
        path = Path(files("jaeger.data")) / "config.json"
    model_path = Path(kwargs.get("path")).resolve()
    add_data_to_json(path, str(model_path), list_key="model_paths")


@click.command(context_settings={"show_default": True})
@click.option(
    "-p",
    "--path",
    type=click.Path(file_okay=False),
    required=False,
    default=None,
    help="Path to save model weights, graph and configuration files",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=False,
    help="Path to jager config file (useful when using Apptainer or Docker)",
)
@click.option(
    "-m",
    "--model_name",
    required=False,
    default=None,
    help="identifier of the model to download",
)
@click.option(
    "-l",
    "--list",
    "list_models",
    is_flag=True,
    default=False,
    help="List all available models",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
def download(path, model_name, list_models, **kwargs):
    """Downloads model weights and updated model paths, or lists available models."""
    from jaeger.commands.downloads import list_ckan_model_download_links, download_file

    # Enforce mutual exclusivity
    if list_models and (model_name or path):
        raise click.UsageError(
            "The '--list' option cannot be used with '--model' or '--path'."
        )

    if not list_models and (not model_name or not path):
        raise click.UsageError(
            "You must provide both '--model' and '--path', or just '--list'."
        )

    model_links = list_ckan_model_download_links()

    if list_models:
        click.echo("Available models:")
        for name in sorted(model_links.keys()):
            click.echo(f"- {name}")
        return

    if model_name not in model_links:
        raise click.UsageError(
            f"Model '{model_name}' not found. Use '--list' to see available models."
        )

    # to avoid jaeger from scanning huge dirs for models
    model_path = Path(path).resolve() / "jaeger_models"
    model_path.mkdir(parents=True, exist_ok=True)

    download_file((model_name, model_links[model_name]), output_dir=model_path)

    if kwargs.get("config", False):
        # update an external config file
        config_path = Path(kwargs.get("config"))
    else:
        # update internal config file
        config_path = Path(files("jaeger.data")) / "config.json"
    try:
        add_data_to_json(config_path, str(model_path), list_key="model_paths")
    except Exception:
        logger.warning(
            "failed to add model path to jaeger config. Seems like you are running jaeger inside a container. please explicitly define the model path in a config file `jaeger predict --config "
            "wget -O config.json https://raw.githubusercontent.com/Yasas1994/Jaeger/dev/src/jaeger/data/config.json"
        )


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def taxonomy(obj):
    """
    experimental taxonomy prediction pipeline
    """
    pass


@taxonomy.command(
    help="""
            Build a taxonomy database

            usage                                                                      
            -----                                                                      

            jaeger taxonomy build [OPTIONS] -i contigs.fasta -o taxonomy_db

        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option(
    "-t",
    "--tax",
    type=click.Path(exists=True),
    required=True,
    help="Path to taxdump",
)
@click.option(
    "-a",
    "--acc2tax",
    type=click.Path(exists=True),
    required=True,
    help="Path to accession2taxid file",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "--fsize",
    type=int,
    default=2048,
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="The length of the gap between two sliding windows (stride==fsize)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--model_path",
    default=None,
    help=("Give the path to a model. overrides --model"),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (useful when using Apptainer or Docker)",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option("--workers", type=int, default=4, help="Number of threads to use")
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp32",
    help="GPU inference precision: fp32 (default), fp16, or bf16. fp16/bf16 reduce memory and may speed up inference on compatible GPUs.",
)
@click.option(
    "--xla",
    is_flag=True,
    help="Enable XLA JIT compilation for inference. May provide 2-3x speedup on GPU after initial compilation overhead.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing database")
def build(**kwargs):
    model = kwargs.get("model")
    if kwargs.get("config") is None:
        if model is None:
            model = "default"
        elif model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )
    else:
        if model is None:
            model = "default"

    """Run jaeger taxonomy database generation pipeline"""
    if model == "default":
        raise click.BadParameter(f"Model '{model}' is not supported")

    else:
        from jaeger.commands.taxonomy import build_taxdb

        build_taxdb(**kwargs)


@taxonomy.command(
    "predict",
    help="""
            Use experimental taxonomy prediction workflow

            usage                                                                      
            -----                                                                      

            jaeger taxonomy predict [OPTIONS] -i contigs.fasta -d taxonomy_db -o output_dir

        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option(
    "-d",
    "--db",
    type=click.Path(exists=True),
    required=True,
    help="Path to taxonomy database",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "--fsize",
    type=int,
    default=2048,
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="The gap between two sliding of the sliding window (fsize = stride)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available models: (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (e.g., when using Apptainer or Docker)",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option("--workers", type=int, default=4, help="Number of threads to use")
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp32",
    help="GPU inference precision: fp32 (default), fp16, or bf16. fp16/bf16 reduce memory and may speed up inference on compatible GPUs.",
)
@click.option(
    "--xla",
    is_flag=True,
    help="Enable XLA JIT compilation for inference. May provide 2-3x speedup on GPU after initial compilation overhead.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing database")
def predict_tax(**kwargs):  # noqa: F811
    model = kwargs.get("model")
    if kwargs.get("config") is None:
        if model is None:
            model = "default"
        elif model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )
    else:
        if model is None:
            model = "default"

    """Run jaeger taxonomy database generation pipeline"""
    if model == "default":
        raise click.BadParameter(f"Model '{model}' is not supported")

    else:
        from jaeger.commands.taxonomy import predict_taxonomy

        predict_taxonomy(**kwargs)


main.add_command(health)
main.add_command(predict)
main.add_command(interpret)
main.add_command(train)
main.add_command(register_models)
main.add_command(download)
main.add_command(utils)
main.add_command(taxonomy)


if __name__ == "__main__":
    main()
