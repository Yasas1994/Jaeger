# Performance optimizations

Jaeger provides several optional inference backends and precision modes.
This page explains how to use them, ordered from **least effort** to **most effort**.

All commands below assume Jaeger is installed with GPU support
(`pip install jaeger-bio[gpu]` or equivalent).

---

## Quick comparison

| Optimization | Effort | Speedup | Model size | Best for |
|--------------|--------|---------|------------|----------|
| [Mixed precision](#1-mixed-precision) | One flag | 1–1.3× | No change | Any GPU with FP16/BF16 support |
| [XLA JIT](#2-xla-jit-compilation) | One flag | 1.5–3× after warmup | No change | Large datasets with repeated shapes |
| [ONNX Runtime](#3-onnx-runtime) | One conversion | 1.5–2× | No change | Reliable cross-platform GPU inference |
| [TFLite quantization](#4-tflite-quantization) | One conversion | Similar | ~3.5× smaller | Edge / mobile / low-storage deployments |
| [ONNX INT8](#5-onnx-int8-quantization) | Conversion + quantization | 1–1.5× | ~2.5× smaller | Smallest GPU-deployable model |
| [TensorRT (TF-TRT)](#6-tensorrt-tf-trt) | Custom TF build | 2–5× | No change | Maximum GPU performance in specialized containers |

---

## 1. Mixed precision

**Effort:** lowest — add one flag to `jaeger predict`.

Run inference with FP16 or BF16 instead of FP32. This reduces memory
bandwidth and can speed up math-bound layers on modern NVIDIA GPUs
(Compute Capability ≥ 7.0 for FP16, ≥ 8.0 for BF16).

```bash
# FP16 (widely supported)
jaeger predict -i contigs.fasta -o output_dir --precision fp16

# BF16 (Ampere/Ada and newer)
jaeger predict -i contigs.fasta -o output_dir --precision bf16
```

### When to use

- You want a quick, risk-free speedup with no preprocessing.
- Your GPU supports FP16/BF16 tensor cores.
- You are not memory-limited (model size stays the same).

### Caveats

- Very small batches may not see a speedup because the overhead of
casting can dominate.
- Some older GPUs (pre-Volta) have reduced FP16 throughput.

---

## 2. XLA JIT compilation

**Effort:** low — add one flag to `jaeger predict`.

XLA (Accelerated Linear Algebra) JIT-compiles the TensorFlow graph
for each unique input shape. After the first compilation, repeated
shapes run significantly faster.

```bash
jaeger predict -i contigs.fasta -o output_dir --xla
```

You can combine XLA with mixed precision:

```bash
jaeger predict -i contigs.fasta -o output_dir --xla --precision fp16
```

### When to use

- Large datasets where many windows have the same length.
- Benchmarking / repeated inference on the same file.

### Caveats

- The first batch for each unique shape is slow (~10–30 s compilation).
- For small or highly variable-length datasets, compilation overhead
can exceed the speedup.

---

## 3. ONNX Runtime

**Effort:** medium — convert the model once, then run with `--onnx`.

ONNX Runtime decouples Jaeger from TensorFlow's execution stack and
supports multiple GPU providers (TensorRT, CUDA, CPU) without requiring
TensorFlow to be built with those backends.

### 3.1 Install dependencies

```bash
pip install onnxruntime-gpu tf2onnx sympy onnx
```

**Important:** ONNX Runtime 1.26 requires **TensorRT 10**.
If you have TensorRT 11 installed, downgrade:

```bash
pip install tensorrt==10.16.1.11
```

### 3.2 Convert the model

```bash
jaeger utils convert-graph \
  -m jaeger_57341_1.5M_fragment \
  -o ./optimized \
  --mode onnx
```

This creates:

```
optimized/
└── jaeger_57341_1.5M_fragment_onnx/
    ├── jaeger_57341_1.5M_fragment.onnx
    ├── jaeger_57341_1.5M_fragment_classes.yaml
    └── jaeger_57341_1.5M_fragment_project.yaml
```

### 3.3 Run inference

```bash
jaeger predict -i contigs.fasta -o output_dir \
  -m jaeger_57341_1.5M_fragment \
  --onnx
```

### Provider selection

`ONNXEngine` automatically picks the best available provider:

1. `TensorrtExecutionProvider` (NVIDIA GPUs)
2. `CUDAExecutionProvider`
3. `CPUExecutionProvider`

The first inference with TensorRT is slow because it builds the
TensorRT engine; subsequent runs with the same shape reuse the cached
engine.

### When to use

- You want 1.5–2× speedup on NVIDIA GPUs without custom TensorFlow builds.
- You need a portable model that can also run on CPU.

---

## 4. TFLite quantization

**Effort:** medium — quantize the model once, then run with `--quantized`.

TFLite is primarily useful for reducing model size for edge or mobile
deployment. Speed on desktop GPUs is usually comparable to the original
SavedModel.

### 4.1 Quantize the model

```bash
# Dynamic range quantization (recommended)
jaeger utils quantize \
  -m jaeger_57341_1.5M_fragment \
  -o ./quantized \
  --mode dynamic

# Float16 weights
jaeger utils quantize \
  -m jaeger_57341_1.5M_fragment \
  -o ./quantized \
  --mode float16

# Full INT8 (experimental)
jaeger utils quantize \
  -m jaeger_57341_1.5M_fragment \
  -o ./quantized \
  --mode full_int8
```

This creates:

```
quantized/
└── jaeger_57341_1.5M_fragment_dynamic/
    ├── jaeger_57341_1.5M_fragment_dynamic.tflite
    └── ... metadata ...
```

### 4.2 Run inference

```bash
jaeger predict -i contigs.fasta -o output_dir \
  -m jaeger_57341_1.5M_fragment \
  --quantized dynamic
```

### When to use

- Model size matters more than speed (e.g., ~6 MB → ~1.6 MB with dynamic).
- Edge devices, mobile apps, or low-bandwidth deployments.

### Caveats

- `full_int8` is experimental and can reduce accuracy on real data;
`dynamic` is recommended.
- Desktop GPU speed is usually not faster than the original model.

---

## 5. ONNX INT8 quantization

**Effort:** higher — convert to ONNX and then apply static INT8
quantization. Gives the smallest GPU-runnable model.

### 5.1 Install dependencies

Same as [ONNX Runtime](#31-install-dependencies):

```bash
pip install onnxruntime-gpu tf2onnx sympy onnx
```

### 5.2 Convert and quantize

```bash
jaeger utils convert-graph \
  -m jaeger_57341_1.5M_fragment \
  -o ./optimized \
  --mode onnx \
  --int8
```

This creates:

```
optimized/
└── jaeger_57341_1.5M_fragment_onnx_int8/
    ├── jaeger_57341_1.5M_fragment_int8.onnx   # quantized
    ├── jaeger_57341_1.5M_fragment.onnx        # original FP32
    └── ... metadata ...
```

### 5.3 Run inference

```bash
jaeger predict -i contigs.fasta -o output_dir \
  -m jaeger_57341_1.5M_fragment \
  --onnx --int8
```

### When to use

- You need the smallest possible model that still runs on GPU
(~6 MB → ~2.4 MB).
- You want faster-than-SavedModel inference without a custom TF build.

### Caveats

- INT8 ONNX models use the **CUDA** execution provider, not TensorRT,
because TensorRT's ONNX parser has strict requirements for quantized
subgraphs.
- Calibration is performed with synthetic one-hot codon tensors.
Accuracy is usually very close to FP32, but you should validate on
your target data.

---

## 6. TensorRT (TF-TRT)

**Effort:** highest — requires TensorFlow built with TensorRT support.

Standard pip-installed TensorFlow does **not** include TensorRT.
This path is only practical if you use NVIDIA's NGC containers or
build TensorFlow from source.

### 6.1 Use an NGC container

```bash
docker run --gpus all -it nvcr.io/nvidia/tensorflow:24.10-tf2-py3
```

Inside the container, install Jaeger and run:

```bash
jaeger utils convert-graph \
  -m jaeger_57341_1.5M_fragment \
  -o ./optimized \
  --mode tensorrt
```

### 6.2 Easier alternative

For most users, [ONNX Runtime with TensorRT](#3-onnx-runtime) is the
better path: it does not require a custom TensorFlow build and gives
most of the speedup.

### When to use

- Maximum GPU performance is required.
- You already run workloads in NVIDIA NGC containers.

---

## Environment configuration

### For ONNX Runtime + TensorRT

If you see errors like:

```
libnvinfer.so.10: cannot open shared object file
libcudnn.so.9: cannot open shared object file
```

Install the matching pip packages:

```bash
# Required for ONNX Runtime 1.26
pip install tensorrt==10.16.1.11
pip install nvidia-cudnn-cu12  # provides libcudnn.so.9
```

Jaeger's `ONNXEngine` automatically preloads these libraries via
`ctypes` so you usually do **not** need to set `LD_LIBRARY_PATH`.
If you still have issues, you can export the library paths manually:

```bash
export LD_LIBRARY_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')/tensorrt_libs:$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
```

### Verify providers

```python
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output on a working NVIDIA system:

```python
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Choosing an optimization

| Situation | Recommended option |
|-----------|--------------------|
| Just want a quick speedup | `--precision fp16` |
| Large dataset, repeated shapes | `--xla --precision fp16` |
| Reliable GPU speedup without custom builds | Convert to ONNX, then `--onnx` |
| Need smallest model on GPU | Convert to ONNX INT8, then `--onnx --int8` |
| Need smallest model for edge/mobile | `jaeger utils quantize --mode dynamic` |
| Maximum performance in NGC containers | `jaeger utils convert-graph --mode tensorrt` |
