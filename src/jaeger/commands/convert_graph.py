"""Graph conversion CLI for Jaeger.

Thin Click wrapper around `jaeger.nnlib.conversion`.
"""

from __future__ import annotations

from pathlib import Path

import click

from jaeger.nnlib.conversion import convert_graph


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
