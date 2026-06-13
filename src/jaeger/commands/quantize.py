"""Model quantization utilities for Jaeger.

Supports:
- Dynamic range quantization (INT8 weights, FP32 activations)
- Full integer quantization (INT8 weights and activations)
- Float16 quantization (FP16 weights, FP32 activations)

Note: Quantization is performed via TensorFlow Lite conversion.
The quantized model is saved as a TFLite model alongside metadata.

Important:
    TFLite models produced by this tool are primarily intended for
    edge/mobile deployment where model size matters. On desktop GPUs,
    TFLite inference may not be faster than the original SavedModel
    due to fallback TF ops and interpreter overhead.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from jaeger.utils.misc import AvailableModels

logger = logging.getLogger("Jaeger")


def quantize_model(model: str, output: Path, mode: str, verbose: int):
    """Core quantization logic (non-CLI)."""
    log = logging.getLogger("Jaeger")
    log.info(f"Quantizing model '{model}' with mode '{mode}'")

    # Resolve model path
    model_info = _resolve_model(model)
    if model_info is None:
        log.error(f"Model '{model}' not found")
        sys.exit(1)

    graph_dir = Path(model_info["graph"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Create output subdirectory
    quantized_dir = output / f"{model}_{mode}"
    if quantized_dir.exists():
        shutil.rmtree(quantized_dir)
    quantized_dir.mkdir(parents=True)

    # Copy metadata files
    for key in ("classes", "project", "weights"):
        if key in model_info and model_info[key]:
            src = Path(model_info[key])
            dst = quantized_dir / src.name
            shutil.copy2(src, dst)
            log.info(f"Copied {key}: {src.name}")

    # Perform quantization via frozen graph -> TFLite
    # Frozen graph eliminates resource variables that TFLite cannot handle.
    log.info(f"Loading SavedModel from {graph_dir}")
    loaded = tf.saved_model.load(str(graph_dir))
    infer = loaded.signatures["serving_default"]

    log.info("Freezing graph (converting variables to constants)...")
    frozen_func = convert_variables_to_constants_v2(infer)

    log.info(f"Converting to TFLite (mode={mode})...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
    # Jaeger models use ops (e.g., Gelu, BatchMatMul) that are not in the
    # baseline TFLite builtin set; allow TensorFlow fallback ops.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    if mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "full_int8":
        log.warning(
            "full_int8 mode is experimental and may reduce accuracy. "
            "Use a representative dataset from your target domain for best results."
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Keep input/output as FLOAT32 so the TFLite runtime handles
        # quantization/dequantization automatically. This avoids needing
        # to manually quantize inputs in the inference pipeline.
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        log.error(f"Quantization failed: {e}")
        sys.exit(1)

    # Save TFLite model
    tflite_path = quantized_dir / f"{model}_{mode}.tflite"
    tflite_path.write_bytes(tflite_model)
    log.info(
        f"Saved quantized model: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)"
    )

    # Save quantization info
    info = {
        "original_graph": str(graph_dir),
        "quantization_mode": mode,
        "tflite_model": str(tflite_path.name),
        "model_format": "tflite",
        "note": (
            "TFLite model for edge deployment. "
            "Resize interpreter input tensor at runtime for variable sequence lengths."
        ),
    }
    info_path = quantized_dir / "quantization_info.yaml"
    info_path.write_text(yaml.safe_dump(info))

    log.info(f"Quantization complete. Output: {quantized_dir}")


@click.command()
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="Model name to quantize (e.g., jaeger_57341_1.5M_fragment)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the quantized model",
)
@click.option(
    "--mode",
    type=click.Choice(["dynamic", "float16", "full_int8"]),
    default="dynamic",
    help=(
        "Quantization mode. dynamic (default) is recommended. "
        "full_int8 is experimental and may reduce accuracy."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level",
    default=1,
)
def quantize(model: str, output: Path, mode: str, verbose: int):
    """Quantize a Jaeger model for edge deployment.

    Converts a SavedModel to a quantized TensorFlow Lite model. The primary
    benefit is reduced model size (e.g., ~6 MB -> ~1.6 MB with dynamic
    quantization). Note that TFLite inference speed depends heavily on the
    target hardware and may not exceed the original SavedModel on desktop GPUs.

    Examples:
        jaeger utils quantize -m default -o ./quantized_models
        jaeger utils quantize -m jaeger_57341_1.5M_fragment -o ./quantized --mode float16
    """
    quantize_model(model, output, mode, verbose)


def _resolve_model(model_name: str) -> dict | None:
    """Resolve model name to path dict."""
    from importlib.resources import files

    from jaeger.utils.misc import json_to_dict

    CONFIG_PATH = files("jaeger.data") / "config.json"
    model_paths = json_to_dict(CONFIG_PATH).get("model_paths", [])
    models = AvailableModels(path=model_paths)
    return models.info.get(model_name)


def _make_rep_dataset():
    """Create a representative dataset generator for INT8 quantization."""

    def dataset_gen():
        for _ in range(100):
            yield [np.random.randn(1, 6, 100, 64).astype(np.float32)]

    return dataset_gen
