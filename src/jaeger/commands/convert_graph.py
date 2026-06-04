"""Graph conversion utilities for Jaeger.

Supports optimizing SavedModel graphs for inference via:
- XLA (JIT compilation)
- TensorFlow Lite (mobile/edge deployment)
- ONNX (cross-platform deployment with TensorRT/CUDA support)
- TensorRT (NVIDIA GPU acceleration — requires TF built with TensorRT support)

Usage:
    jaeger utils convert-graph -m default -o ./optimized --mode xla
    jaeger utils convert-graph -m default -o ./optimized --mode tflite
    jaeger utils convert-graph -m default -o ./optimized --mode onnx
    jaeger utils convert-graph -m default -o ./optimized --mode tensorrt
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
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from jaeger.utils.misc import AvailableModels

logger = logging.getLogger("Jaeger")


def convert_graph(model: str, output: Path, mode: str, verbose: int):
    """Core graph conversion logic (non-CLI)."""
    log = logging.getLogger("Jaeger")
    log.info(f"Converting model '{model}' with mode '{mode}'")

    # Resolve model path
    model_info = _resolve_model(model)
    if model_info is None:
        log.error(f"Model '{model}' not found")
        sys.exit(1)

    graph_dir = Path(model_info["graph"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Create output subdirectory
    converted_dir = output / f"{model}_{mode}"
    if converted_dir.exists():
        shutil.rmtree(converted_dir)
    converted_dir.mkdir(parents=True)

    # Copy metadata files
    for key in ("classes", "project", "weights"):
        if key in model_info and model_info[key]:
            src = Path(model_info[key])
            dst = converted_dir / src.name
            shutil.copy2(src, dst)
            log.info(f"Copied {key}: {src.name}")

    if mode == "xla":
        _convert_xla(graph_dir, converted_dir, model, log)
    elif mode == "tflite":
        _convert_tflite(graph_dir, converted_dir, model, log)
    elif mode == "onnx":
        _convert_onnx(graph_dir, converted_dir, model, log)
    elif mode == "tensorrt":
        _convert_tensorrt(graph_dir, converted_dir, model, log)
    else:
        log.error(f"Unknown conversion mode: {mode}")
        sys.exit(1)

    # Save conversion info
    info = {
        "original_graph": str(graph_dir),
        "conversion_mode": mode,
        "output_dir": str(converted_dir),
    }
    info_path = converted_dir / "conversion_info.yaml"
    info_path.write_text(yaml.safe_dump(info))

    log.info(f"Conversion complete. Output: {converted_dir}")


def _convert_xla(graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger):
    """Convert SavedModel to XLA-optimized SavedModel.

    XLA (Accelerated Linear Algebra) uses JIT compilation to fuse operations
    and optimize memory access patterns. This can provide 2-3x speedup on GPU
    for compute-bound models after initial compilation overhead.
    """
    log.info("Loading SavedModel for XLA optimization...")
    loaded = tf.saved_model.load(str(graph_dir))
    infer = loaded.signatures["serving_default"]

    log.info("Wrapping with XLA JIT compilation...")

    # Create an XLA-optimized signature
    @tf.function(jit_compile=True)
    def xla_serving(inputs):
        return infer(inputs=inputs)

    # Trace with a representative input to build the graph
    log.info("Tracing XLA graph (this may take a while)...")
    rep_input = tf.constant(np.random.randn(1, 6, 500, 64).astype(np.float32))
    _ = xla_serving(rep_input)

    # Save as a new SavedModel
    log.info(f"Saving XLA-optimized model to {output_dir}...")

    class XLAModel(tf.Module):
        def __init__(self, serving_fn):
            super().__init__()
            self.serving_fn = serving_fn

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, 6, None, 64), dtype=tf.float32, name="inputs")
            ]
        )
        def serving_default(self, inputs):
            return self.serving_fn(inputs)

    xla_module = XLAModel(xla_serving)
    tf.saved_model.save(xla_module, str(output_dir))
    log.info("XLA model saved")


def _convert_tflite(graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger):
    """Convert SavedModel to TFLite (same as quantize but without quantization)."""
    log.info("Loading SavedModel for TFLite conversion...")
    loaded = tf.saved_model.load(str(graph_dir))
    infer = loaded.signatures["serving_default"]

    log.info("Converting to frozen graph...")
    frozen_func = convert_variables_to_constants_v2(infer)

    log.info("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_path = output_dir / f"{model_name}.tflite"
    tflite_path.write_bytes(tflite_model)
    log.info(f"Saved TFLite model: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")


def _convert_onnx(graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger):
    """Convert SavedModel to ONNX format.

    ONNX models can be run with ONNX Runtime, which supports multiple
    execution providers (TensorRT, CUDA, CPU) without requiring
    TensorFlow to be built with those backends.

    This is the recommended approach for TensorRT acceleration when
    using pip-installed TensorFlow.
    """
    try:
        import tf2onnx
    except ImportError:
        log.error(
            "tf2onnx is not installed. Install it with:\n"
            "  pip install tf2onnx onnxruntime-gpu"
        )
        sys.exit(1)

    log.info("Converting SavedModel to ONNX...")

    onnx_path = output_dir / f"{model_name}.onnx"

    # Use tf2onnx to convert
    import subprocess
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", str(graph_dir),
        "--output", str(onnx_path),
        "--opset", "13",
        "--signature", "serving_default",
        "--tag", "serve",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ONNX conversion failed:\n{result.stderr}")
        sys.exit(1)

    log.info(f"Saved ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        log.info("ONNX model validation passed")
    except Exception as e:
        log.warning(f"ONNX validation warning: {e}")


def _convert_tensorrt(graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger):
    """Convert SavedModel to TensorRT-optimized SavedModel.

    Requires TensorFlow built with TensorRT support. Most pip-installed
    TensorFlow packages do NOT include TensorRT. Use NVIDIA's NGC containers
    or build from source with --config=cuda and TensorRT headers.

    Alternative: Use --mode onnx to create an ONNX model, then run with
    ONNX Runtime's TensorRT execution provider.

    When available, TensorRT can provide 2-5x speedup on NVIDIA GPUs by:
    - Fusing layers (conv+bias+relu)
    - Selecting optimal CUDA kernels
    - Using FP16/INT8 tensor cores
    """
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
    except ImportError:
        log.error(
            "TensorRT is not available. TensorFlow must be built with TensorRT support.\n"
            "Options:\n"
            "  1. Use NVIDIA NGC Docker container: nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3\n"
            "  2. Install tensorflow[and-cuda] with TensorRT headers present\n"
            "  3. Build TensorFlow from source with --config=cuda and TensorRT\n"
            "  4. Use --mode onnx instead, then run with ONNX Runtime + TensorRT"
        )
        sys.exit(1)

    # Verify TRT is actually usable
    try:
        trt._check_trt_version_compatibility()
    except RuntimeError as e:
        log.error(f"TensorRT compatibility check failed: {e}")
        sys.exit(1)

    log.info("Loading SavedModel for TensorRT optimization...")

    # Convert with TensorRT
    params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP32,
        max_workspace_size_bytes=8000000000,  # 8GB
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(graph_dir),
        conversion_params=params,
    )

    log.info("Converting to TensorRT graph...")
    converter.convert()

    log.info("Building TensorRT engines (this may take several minutes)...")

    def input_fn():
        for _ in range(10):
            yield [np.random.randn(1, 6, 500, 64).astype(np.float32)]

    converter.build(input_fn=input_fn)

    log.info(f"Saving TensorRT-optimized model to {output_dir}...")
    converter.save(str(output_dir))
    log.info("TensorRT model saved")


@click.command()
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="Model name to convert (e.g., jaeger_57341_1.5M_fragment)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the converted model",
)
@click.option(
    "--mode",
    type=click.Choice(["xla", "tflite", "onnx", "tensorrt"]),
    default="xla",
    help="Conversion mode: xla (default), tflite, onnx, or tensorrt",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level",
    default=1,
)
def convert_graph_cmd(model: str, output: Path, mode: str, verbose: int):
    """Convert a Jaeger SavedModel to an optimized inference graph.

    Supports four optimization backends:

    \b
    xla (default):
        JIT-compiles the graph for GPU. Provides 2-3x speedup after
        initial compilation. Best for large datasets with repeated shapes.

    \b
    tflite:
        Converts to TensorFlow Lite for mobile/edge deployment.
        Produces a smaller model (~3.5x size reduction).

    \b
    onnx:
        Converts to ONNX format for cross-platform deployment.
        Recommended for TensorRT acceleration with pip-installed TF.
        Use with ONNX Runtime's TensorRT execution provider.

    \b
    tensorrt:
        NVIDIA TensorRT optimization for maximum GPU performance.
        Requires TensorFlow built with TensorRT support (not available
        in standard pip packages). Use NVIDIA NGC containers.

    Examples:
        jaeger utils convert-graph -m default -o ./optimized --mode xla
        jaeger utils convert-graph -m default -o ./optimized --mode onnx
    """
    convert_graph(model, output, mode, verbose)


def _resolve_model(model_name: str) -> dict | None:
    """Resolve model name to path dict."""
    from importlib.resources import files

    from jaeger.utils.misc import json_to_dict

    CONFIG_PATH = files("jaeger.data") / "config.json"
    model_paths = json_to_dict(CONFIG_PATH).get("model_paths", [])
    models = AvailableModels(path=model_paths)
    return models.info.get(model_name)
