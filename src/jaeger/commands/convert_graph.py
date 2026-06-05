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
    jaeger utils convert-graph -m default -o ./optimized --mode onnx --int8
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
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from jaeger.utils.misc import AvailableModels

logger = logging.getLogger("Jaeger")


def convert_graph(
    model: str,
    output: Path,
    mode: str,
    verbose: int,
    int8: bool = False,
):
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
    suffix = "_int8" if (mode == "onnx" and int8) else ""
    converted_dir = output / f"{model}_{mode}{suffix}"
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
        _convert_onnx(graph_dir, converted_dir, model, log, int8=int8)
    elif mode == "tensorrt":
        _convert_tensorrt(graph_dir, converted_dir, model, log)
    else:
        log.error(f"Unknown conversion mode: {mode}")
        sys.exit(1)

    # Save conversion info
    info = {
        "original_graph": str(graph_dir),
        "conversion_mode": mode,
        "int8_quantization": bool(int8) if mode == "onnx" else None,
        "output_dir": str(converted_dir),
    }
    info_path = converted_dir / "conversion_info.yaml"
    info_path.write_text(yaml.safe_dump(info))

    log.info(f"Conversion complete. Output: {converted_dir}")


def _convert_xla(
    graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger
):
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
                tf.TensorSpec(
                    shape=(None, 6, None, 64), dtype=tf.float32, name="inputs"
                )
            ]
        )
        def serving_default(self, inputs):
            return self.serving_fn(inputs)

    xla_module = XLAModel(xla_serving)
    tf.saved_model.save(xla_module, str(output_dir))
    log.info("XLA model saved")


def _convert_tflite(
    graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger
):
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


def _convert_onnx(
    graph_dir: Path,
    output_dir: Path,
    model_name: str,
    log: logging.Logger,
    int8: bool = False,
):
    """Convert SavedModel to ONNX format.

    ONNX models can be run with ONNX Runtime, which supports multiple
    execution providers (TensorRT, CUDA, CPU) without requiring
    TensorFlow to be built with those backends.

    This is the recommended approach for TensorRT acceleration when
    using pip-installed TensorFlow.

    Args:
        int8: If True, apply static INT8 quantization using ONNX Runtime's
            quantization tools. This produces a smaller model that can run
            on TensorRT's INT8 tensor cores. Requires a calibration step.
    """
    try:
        import tf2onnx  # noqa: F401
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
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        str(graph_dir),
        "--output",
        str(onnx_path),
        "--opset",
        "13",
        "--signature",
        "serving_default",
        "--tag",
        "serve",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ONNX conversion failed:\n{result.stderr}")
        sys.exit(1)

    log.info(
        f"Saved ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    # Verify the model
    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        log.info("ONNX model validation passed")
    except Exception as e:
        log.warning(f"ONNX validation warning: {e}")

    if int8:
        _quantize_onnx_int8(onnx_path, graph_dir, output_dir, model_name, log)


def _quantize_onnx_int8(
    onnx_path: Path,
    graph_dir: Path,
    output_dir: Path,
    model_name: str,
    log: logging.Logger,
):
    """Apply static INT8 quantization to an ONNX model.

    Uses ONNX Runtime's quantization tools with a calibration dataset
    derived from real DNA sequences. The quantized model is saved in the
    output directory as ``{model_name}_int8.onnx``.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        log.error(
            "ONNX Runtime quantization tools are not available. "
            f"Install with: pip install onnxruntime-gpu sympy\n{e}"
        )
        sys.exit(1)

    import onnx

    log.info("Preparing ONNX model for INT8 quantization...")

    # Run ONNX shape inference first. This is required for both quantization
    # and for TensorRT/CUDA providers to resolve tensor shapes.
    try:
        inferred_path = onnx_path.with_suffix(".inferred.onnx")
        onnx_model = onnx.load(str(onnx_path))
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(inferred_model, str(inferred_path))
        input_model_path = str(inferred_path)
        preprocessed_path = inferred_path
    except Exception as e:
        log.warning(f"ONNX shape inference failed, using original model: {e}")
        input_model_path = str(onnx_path)
        preprocessed_path = None

    # Determine input shape from the model
    onnx_model = onnx.load(input_model_path)
    input_tensor = onnx_model.graph.input[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    # Replace dynamic axes with representative values
    batch = input_shape[0] if input_shape[0] > 0 else 1
    frames = input_shape[1] if input_shape[1] > 0 else 6
    seq_len = input_shape[2] if input_shape[2] > 0 else 500
    depth = input_shape[3] if len(input_shape) > 3 and input_shape[3] > 0 else 64
    input_name = input_tensor.name

    log.info(
        f"Calibration input shape: ({batch}, {frames}, {seq_len}, {depth}); "
        f"name={input_name}"
    )

    # Build a calibration dataset from real DNA fragments so the INT8
    # ranges reflect the actual input distribution.
    calibration_inputs = _build_calibration_inputs(
        graph_dir=graph_dir,
        target_shape=(batch, frames, seq_len, depth),
        num_samples=100,
        log=log,
    )

    class _Calibrator(CalibrationDataReader):
        def __init__(self, input_name: str, inputs: list[np.ndarray]):
            self.input_name = input_name
            self.inputs = inputs
            self._idx = 0

        def get_next(self):
            if self._idx >= len(self.inputs):
                return None
            item = {self.input_name: self.inputs[self._idx]}
            self._idx += 1
            return item

    calibrator = _Calibrator(input_name, calibration_inputs)

    quantized_path = output_dir / f"{model_name}_int8.onnx"

    log.info("Running INT8 static quantization (this may take a while)...")
    try:
        quantize_static(
            model_input=input_model_path,
            model_output=str(quantized_path),
            calibration_data_reader=calibrator,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
            },
        )
    except Exception as e:
        log.error(f"INT8 quantization failed: {e}")
        if preprocessed_path and preprocessed_path.exists():
            preprocessed_path.unlink()
        sys.exit(1)

    # Clean up preprocessed intermediate
    if preprocessed_path and preprocessed_path.exists():
        preprocessed_path.unlink()

    log.info(
        f"Saved INT8 ONNX model: {quantized_path} "
        f"({quantized_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )


def _build_calibration_inputs(
    graph_dir: Path,
    target_shape: tuple[int, int, int, int],
    num_samples: int,
    log: logging.Logger,
) -> list[np.ndarray]:
    """Generate calibration inputs by running real DNA through the SavedModel.

    Falls back to Gaussian noise if the real-data path fails.
    """
    from importlib.resources import files

    batch, frames, seq_len, depth = target_shape
    inputs: list[np.ndarray] = []

    # Try to use bundled test FASTA for realistic calibration data
    test_fasta = files("jaeger.data") / "test" / "test_contigs.fasta"
    if not test_fasta.exists():
        log.warning(
            "No bundled test FASTA found; falling back to synthetic one-hot calibration data. "
            "INT8 accuracy may be degraded."
        )
        return _synthetic_one_hot_samples(target_shape, num_samples)

    try:
        log.info(f"Building calibration dataset from {test_fasta}")

        # Load model metadata needed for preprocessing
        from jaeger.preprocess.fasta import fragment_generator
        from jaeger.preprocess.latest.convert import process_string_inference
        from jaeger.nnlib.inference import InferModel

        # Use InferModel only for its metadata-loading helper; avoid creating
        # a full session since we only need the string_processor_config.
        _tmp_info = {"graph": graph_dir, "classes": None, "project": None}
        _tmp_infer = InferModel.__new__(InferModel)
        _tmp_infer.class_map = {}
        _tmp_infer.string_processor_config = _tmp_infer._load_string_processor_config(
            None
        )

        # We need the project YAML to get the real config. Try to find it next
        # to the SavedModel directory.
        project_yaml = graph_dir.parent / f"{graph_dir.parent.name}_project.yaml"
        if not project_yaml.exists():
            candidates = list(graph_dir.parent.glob("*_project.yaml"))
            if candidates:
                project_yaml = candidates[0]
        if project_yaml.exists():
            import yaml

            with project_yaml.open() as fh:
                _tmp_infer.string_processor_config = yaml.safe_load(fh)

        spc = _tmp_infer.string_processor_config

        # Build a tf.data pipeline from the test FASTA
        dataset = tf.data.Dataset.from_generator(
            fragment_generator(
                file_path=str(test_fasta),
                no_progress=True,
                fragsize=spc.get("fragsize", 200),
                stride=spc.get("stride", spc.get("fragsize", 200) // 2),
            ),
            output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
        )

        process_fn = process_string_inference(
            codons=spc.get("codon"),
            codon_num=spc.get("codon_id"),
            codon_depth=spc.get("codon_depth", 1),
            ngram_width=spc.get("ngram_width", 3),
            seq_onehot=spc.get("seq_onehot"),
            crop_size=spc.get("crop_size"),
            input_type=spc.get("input_type", "translated"),
            masking=spc.get("masking"),
            mutate=spc.get("mutate"),
            mutation_rate=spc.get("mutation_rate"),
            shuffle=spc.get("shuffle"),
        )

        dataset = dataset.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)

        input_type = spc.get("input_type", "translated")

        for inputs_dict in dataset.take(num_samples):
            x = inputs_dict.get(input_type)
            if isinstance(x, tf.Tensor):
                x = x.numpy()
            # Pad or crop to target seq_len
            if x.shape[2] < seq_len:
                pad = seq_len - x.shape[2]
                x = np.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0)), mode="constant")
            elif x.shape[2] > seq_len:
                x = x[:, :, :seq_len, :]
            # Ensure depth matches
            if x.shape[3] != depth:
                if x.shape[3] < depth:
                    pad_d = depth - x.shape[3]
                    x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad_d)), mode="constant")
                else:
                    x = x[:, :, :, :depth]
            inputs.append(x.astype(np.float32))

        if not inputs:
            raise RuntimeError("No calibration inputs generated from FASTA")

        log.info(f"Generated {len(inputs)} real calibration samples")
        return inputs

    except Exception as e:
        log.warning(
            f"Failed to build real calibration dataset ({e}); "
            "falling back to synthetic one-hot data. INT8 accuracy may be degraded."
        )
        return _synthetic_one_hot_samples(target_shape, num_samples)


def _synthetic_one_hot_samples(
    target_shape: tuple[int, int, int, int], num_samples: int
) -> list[np.ndarray]:
    """Generate random one-hot tensors matching codon-encoded DNA inputs.

    Real Jaeger inputs are one-hot codon tensors (shape: B, 6, L, 64). Using
    random one-hot samples for calibration is closer to the true distribution
    than Gaussian noise and usually yields better INT8 accuracy.
    """
    batch, frames, seq_len, depth = target_shape
    samples = []
    for _ in range(num_samples):
        indices = np.random.randint(0, depth, size=(batch, frames, seq_len))
        one_hot = np.eye(depth)[indices].astype(np.float32)
        samples.append(one_hot)
    return samples


def _convert_tensorrt(
    graph_dir: Path, output_dir: Path, model_name: str, log: logging.Logger
):
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
    "--int8",
    is_flag=True,
    help=(
        "Apply static INT8 quantization (ONNX mode only). "
        "Produces a smaller model that can use TensorRT INT8 tensor cores."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level",
    default=1,
)
def convert_graph_cmd(model: str, output: Path, mode: str, int8: bool, verbose: int):
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
        Use --int8 to additionally quantize to INT8 for smaller models
        and INT8 tensor-core inference.

    \b
    tensorrt:
        NVIDIA TensorRT optimization for maximum GPU performance.
        Requires TensorFlow built with TensorRT support (not available
        in standard pip packages). Use NVIDIA NGC containers.

    Examples:
        jaeger utils convert-graph -m default -o ./optimized --mode xla
        jaeger utils convert-graph -m default -o ./optimized --mode onnx
        jaeger utils convert-graph -m default -o ./optimized --mode onnx --int8
    """
    convert_graph(model, output, mode, verbose, int8=int8)


def _resolve_model(model_name: str) -> dict | None:
    """Resolve model name to path dict."""
    from importlib.resources import files

    from jaeger.utils.misc import json_to_dict

    CONFIG_PATH = files("jaeger.data") / "config.json"
    model_paths = json_to_dict(CONFIG_PATH).get("model_paths", [])
    models = AvailableModels(path=model_paths)
    return models.info.get(model_name)
