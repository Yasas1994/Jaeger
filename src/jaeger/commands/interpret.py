"""Gradient-based saliency interpretation for Jaeger models.

Unlike ``jaeger predict`` (which runs a frozen SavedModel graph), ``jaeger
interpret`` rebuilds the Keras model from the model's ``*_project.yaml``
recipe and ``*.weights.h5`` weights, then computes gradient saliency maps
with ``tf.GradientTape``: the gradient of the predicted-class probability
with respect to the codon-embedding activations (the same quantity the
original Jaeger saliency analysis used).
"""

import sys
import time
import traceback
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tensorflow as tf

from jaeger.commands.predict import _build_prediction_dataset
from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.logging import description, get_logger
from jaeger.utils.misc import (
    AvailableModels,
    get_model_id,
    json_to_dict,
    load_model_config,
)
from jaeger.utils.misc import track_ms as track

GB_BYTES = 1024**3


def _strip_bias_initializers(config: dict[str, Any]) -> None:
    """Remove ``calculate_from_train_data`` bias initializers from the config.

    They read the training dataset at build time, which is unavailable on
    inference machines. Interpret loads the trained weights afterwards, so
    the initial bias values are irrelevant.
    """
    model_cfg = config.get("model", {})
    for section in ("classifier", "reliability_model", "projection"):
        for layer in (model_cfg.get(section, {}) or {}).get("hidden_layers", []) or []:
            layer.get("config", {}).pop("bias_initializer", None)


def _sanitize_training_config(config: dict[str, Any]) -> None:
    """Neutralize training-time paths so building touches no files.

    The project recipe records checkpoint/data directories from the training
    machine (possibly a cluster filesystem that does not exist locally).
    """
    training = config.get("training") or {}
    for key in (
        "classifier_dir",
        "reliability_dir",
        "projection_dir",
        "experiment_root",
        "data_dir",
        "model_saving",
    ):
        training.pop(key, None)
    callbacks = training.get("callbacks") or {}
    callbacks.pop("directories", None)
    config["from_last_checkpoint"] = False
    config["force"] = False
    config["use_xla"] = False
    config["precision"] = "fp32"
    config["mix_precision"] = False
    _strip_bias_initializers(config)


def load_interpret_model(
    model_info: dict[str, Any], logger: Any
) -> tuple[tf.keras.Model, dict[str, Any], DynamicModelBuilder]:
    """Rebuild the Keras model from the project recipe and load its weights."""
    project_path = model_info.get("project")
    weights_path = model_info.get("weights")
    if project_path is None or weights_path is None:
        raise SystemExit(
            "interpret requires a model directory with *_project.yaml and "
            "*.weights.h5 files (the same files shipped with SavedModels)."
        )

    config = load_model_config(Path(project_path))
    _sanitize_training_config(config)

    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    model = models["jaeger_model"]
    model.load_weights(weights_path)
    logger.info(f"loaded weights from {weights_path}")
    return model, config, builder


def find_embedding_layer(model: tf.keras.Model) -> tf.keras.layers.Embedding:
    """Return the model's Embedding layer (id 0 is the N/pad codon)."""
    emb_layer = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Embedding)
        ),
        None,
    )
    if emb_layer is None:
        raise SystemExit(
            "gradient saliency requires an embedding layer; this model was "
            "built with use_embedding_layer: false"
        )
    return emb_layer


def build_saliency_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a model mapping inputs to (codon embeddings, class logits).

    The gradient of the predicted-class probability with respect to the
    embedding activations is the saliency signal; exposing the embedding
    tensor as an output lets ``GradientTape`` differentiate through it.
    """
    emb_layer = find_embedding_layer(model)

    outputs = model.output
    if isinstance(outputs, dict):
        pred = outputs["prediction"]
    else:
        raise SystemExit("could not locate the prediction output of the model")

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[emb_layer.output, pred],
        name="saliency_model",
    )


def _probs_and_target(
    logits: tf.Tensor, class_index: int | None
) -> tuple[tf.Tensor, tf.Tensor]:
    """Softmax probabilities and the scalar target (predicted-class prob)."""
    probs = tf.nn.softmax(logits, axis=-1)
    if class_index is None:
        idx = tf.argmax(probs, axis=-1)
    else:
        idx = tf.fill([tf.shape(probs)[0]], tf.cast(class_index, tf.int64))
    onehot = tf.one_hot(idx, depth=tf.shape(probs)[-1], dtype=probs.dtype)
    target = tf.reduce_sum(probs * onehot, axis=-1)
    return probs, target


def compute_saliency_batch(
    saliency_model: tf.keras.Model,
    x: tf.Tensor,
    class_index: int | None = None,
    all_classes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Gradient saliency for one batch.

    Returns ``(saliency, probs, grads, saliency_by_class)`` where ``saliency``
    is the sum of absolute gradients over the embedding dimension (frames,
    codons), ``probs`` the class probabilities, and ``grads`` the raw
    gradients (frames, codons, embedding). With ``all_classes=True``,
    ``saliency_by_class`` holds the per-class saliency (batch, classes,
    frames, codons) — the gradient of each class probability, not just the
    predicted class.
    """
    with tf.GradientTape(persistent=all_classes) as tape:
        emb, logits = saliency_model(x, training=False)
        tape.watch(emb)
        probs, target = _probs_and_target(logits, class_index)
        # slicing must happen inside the tape context, otherwise the slice op
        # is not recorded and gradients w.r.t. it come back None
        class_probs = (
            [probs[:, c] for c in range(probs.shape[-1])] if all_classes else None
        )
    grads = tape.gradient(target, emb)
    saliency = tf.reduce_sum(tf.abs(grads), axis=-1)
    saliency_bc = None
    if all_classes:
        per_class = [
            tf.reduce_sum(tf.abs(tape.gradient(pc, emb)), axis=-1) for pc in class_probs
        ]
        saliency_bc = tf.stack(per_class, axis=1).numpy()
        del tape
    return saliency.numpy(), probs.numpy(), grads.numpy(), saliency_bc


def compute_attribution_batch(
    saliency_model: tf.keras.Model,
    emb_layer: tf.keras.layers.Embedding,
    x: tf.Tensor,
    class_index: int | None = None,
    flavour: str = "gradxinput",
    ig_steps: int = 50,
    all_classes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Baseline-referenced attribution for one batch (signed, per codon).

    Both flavours attribute the predicted-class probability relative to the
    null baseline ``emb0`` — the embedding of input id 0 (N/pad codon, which
    is masked everywhere downstream):

    - ``gradxinput``: ``sum_dims grad * (emb - emb0)`` — a one-step
      approximation of integrated gradients.
    - ``integrated``: integrated gradients — the average gradient along the
      straight path from ``emb0`` to ``emb`` (midpoint rule, ``ig_steps``
      steps), times ``(emb - emb0)``. Runs eagerly: the embedding layer's
      ``call`` is patched per step to emit the interpolated embedding while
      the id-derived mask (``mask_zero``) stays intact.

    Returns ``(attribution, probs, attribution_by_class)`` with shapes
    (batch, frames, codons), (batch, classes) and — only when
    ``all_classes=True`` — (batch, classes, frames, codons) attributions for
    every class probability. ``probs`` always comes from the unperturbed
    input.
    """
    base_vec = emb_layer.weights[0][0]  # (embedding_size,)

    def _forward() -> tuple[tf.Tensor, tf.Tensor]:
        emb, logits = saliency_model(x, training=False)
        return emb, logits

    # fix the target class from the clean input so interpolation steps never
    # switch to a different argmax branch. Keep the clean embedding on the
    # host: retaining it on the GPU across the IG loop adds ~100s of MB to
    # the per-step peak for long genomes.
    emb_clean, logits_clean = _forward()
    probs_clean, _ = _probs_and_target(logits_clean, class_index)
    probs_clean_np = probs_clean.numpy()
    if class_index is None:
        class_index = int(tf.argmax(probs_clean, axis=-1)[0])
    emb_clean_np = tf.cast(emb_clean, tf.float32).numpy()
    del emb_clean, logits_clean, probs_clean

    def _grads_at_current_embeddings() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor | None]:
        with tf.GradientTape(persistent=all_classes) as tape:
            emb, logits = _forward()
            tape.watch(emb)
            probs, target = _probs_and_target(logits, class_index)
            # slicing must happen inside the tape context, otherwise the
            # slice op is not recorded and gradients come back None
            class_probs = (
                [probs[:, c] for c in range(probs.shape[-1])] if all_classes else None
            )
        grads = tape.gradient(target, emb)
        grads_bc = None
        if all_classes:
            grads_bc = tf.stack([tape.gradient(pc, emb) for pc in class_probs], axis=1)
            del tape
        return grads, emb, grads_bc

    def _attribute(grads: tf.Tensor, emb: tf.Tensor) -> tf.Tensor:
        delta = emb - tf.cast(base_vec, emb.dtype)
        return tf.reduce_sum(tf.cast(grads * delta, tf.float32), axis=-1)

    def _attribute_bc(grads_bc: tf.Tensor, emb: tf.Tensor) -> tf.Tensor:
        delta = emb - tf.cast(base_vec, emb.dtype)
        return tf.reduce_sum(tf.cast(grads_bc * delta[:, None], tf.float32), axis=-1)

    if flavour == "gradxinput":
        grads, emb, grads_bc = _grads_at_current_embeddings()
        attribution_bc = _attribute_bc(grads_bc, emb).numpy() if all_classes else None
        return _attribute(grads, emb).numpy(), probs_clean_np, attribution_bc

    if flavour != "integrated":
        raise ValueError(f"unknown attribution flavour: {flavour}")

    orig_call = emb_layer.call
    base_np = np.asarray(base_vec, dtype=np.float32)
    delta_np = emb_clean_np - base_np  # (batch, frames, codons, embedding)
    grad_sum_np = None
    grad_sum_bc_np = None
    try:
        for step in range(ig_steps):
            alpha = (step + 0.5) / ig_steps  # midpoint rule

            def _interp_call(inputs, _a=alpha, _orig=orig_call):
                emb = _orig(inputs)
                base = tf.cast(base_vec, emb.dtype)
                return base + _a * (emb - base)

            emb_layer.call = _interp_call
            grads, _, grads_bc = _grads_at_current_embeddings()
            # accumulate on the host to keep the GPU footprint per-step only
            g_np = tf.cast(grads, tf.float32).numpy()
            grad_sum_np = g_np if grad_sum_np is None else grad_sum_np + g_np
            if all_classes:
                gbc_np = tf.cast(grads_bc, tf.float32).numpy()
                grad_sum_bc_np = (
                    gbc_np if grad_sum_bc_np is None else grad_sum_bc_np + gbc_np
                )
    finally:
        emb_layer.call = orig_call
    attribution = ((grad_sum_np / ig_steps) * delta_np).sum(axis=-1)
    attribution_bc = (
        ((grad_sum_bc_np / ig_steps) * delta_np[:, None]).sum(axis=-1)
        if all_classes
        else None
    )
    return attribution, probs_clean_np, attribution_bc


def run_core(**kwargs: Any) -> None:
    current_process = psutil.Process()

    USER_MODEL_PATH = kwargs.get("model_path")
    CONFIG_PATH = kwargs.get("config") or files("jaeger.data") / "config.json"

    if not USER_MODEL_PATH:
        model_name = kwargs.get("model")
        model_id = get_model_id(model_name)
        model_paths = json_to_dict(CONFIG_PATH).get("model_paths")
        info = AvailableModels(path=model_paths).info
        if model_name not in info:
            raise SystemExit(
                f"model '{model_name}' not found in {model_paths}. "
                "Use --model_path to point at a model directory directly."
            )
        model_info = info[model_name]
    else:
        info = AvailableModels(path=USER_MODEL_PATH).info
        model_paths = USER_MODEL_PATH
        classification_models = {
            name: meta
            for name, meta in info.items()
            if meta.get("project") is not None and meta.get("weights") is not None
        }
        if not classification_models:
            raise SystemExit(
                f"No model with *_project.yaml and *.weights.h5 found in {model_paths}"
            )
        model_name = next(iter(classification_models))
        model_id = get_model_id(model_name)
        model_info = classification_models[model_name]

    input_file_path = Path(kwargs.get("input"))
    input_file = input_file_path.name
    file_base = input_file_path.stem

    OUTPUT_DIR = Path(kwargs.get("output")) / model_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_file = Path(f"{file_base}_interpret.log")
    logger = get_logger(OUTPUT_DIR, log_file, level=kwargs.get("verbose"))
    logger.info(
        description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters")
    )

    if kwargs.get("whole_sequence"):
        # One window per contig: set the window to the longest contig so every
        # contig takes the whole-contig path (padded with 'M' in the fragment
        # generator). Useful to test whether windowing artifacts affect
        # saliency patterns.
        import pyfastx

        fa = pyfastx.Fasta(str(input_file_path), build_index=False)
        max_len = max(len(record[1]) for record in fa)
        kwargs["fsize"] = max_len
        kwargs["stride"] = max_len
        logger.info(f"whole-sequence mode: window set to longest contig ({max_len} nt)")
        min_len = kwargs.get("min_len") or 1
    else:
        min_len = kwargs.get("min_len") or kwargs.get("fsize")
    try:
        num = validate_fasta_entries(str(input_file_path), min_len=min_len)
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_saliency_path = OUTPUT_DIR / f"{file_base}_saliency.npz"
    if output_saliency_path.exists() and not kwargs.get("overwrite"):
        logger.error(
            "output file exists. enable --overwrite option to overwrite the output file."
        )
        sys.exit(1)

    THREADS = kwargs.get("workers")
    MEMORY_LIMIT = 1024 * kwargs.get("mem", 4)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    except RuntimeError:
        logger.warning(
            f"TensorFlow runtime already initialized; using default threading (requested {THREADS})"
        )
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    mode = None

    if kwargs.get("cpu"):
        mode = "CPU"
        tf.config.set_visible_devices([], "GPU")
        logger.info("CPU only mode selected")
    elif gpus:
        mode = "GPU"
        tf.config.set_visible_devices([gpus[kwargs.get("physicalid")]], "GPU")
        precision = kwargs.get("precision", "fp32")
        if precision == "fp16":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("GPU precision: mixed_float16 (FP16 compute, FP32 variables)")
        elif precision == "bf16":
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            logger.info("GPU precision: mixed_bfloat16 (BF16 compute, FP32 variables)")
        else:
            logger.info("GPU precision: float32 (FP32)")
        try:
            tf.config.set_logical_device_configuration(
                gpus[kwargs.get("physicalid")],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=MEMORY_LIMIT, experimental_device_ordinal=10
                    )
                ],
            )
        except Exception as e:
            logger.error(f"an error {e} occurred during virtual device initialization ")
            logger.debug(traceback.format_exc())
    else:
        mode = "CPU"
        logger.warning(
            "could not find a GPU on the system. For optimal performance run Jaeger on a GPU."
        )

    logger.info(f"tensorflow: {version('tensorflow')}")
    logger.info(f"input file: {input_file}")
    logger.info(f"log file: {log_file.name}")
    logger.info(f"outpath: {OUTPUT_DIR.resolve()}")
    logger.info(f"fragment size: {kwargs.get('fsize')}")
    logger.info(f"stride: {kwargs.get('stride')}")
    logger.info(f"batch size: {kwargs.get('batch')}")
    logger.info(f"mode: {mode}")
    logger.info(f"model: {model_id}")

    model, config, builder = load_interpret_model(model_info, logger)
    saliency_model = build_saliency_model(model)

    flavour = kwargs.get("flavour", "raw")
    ig_steps = kwargs.get("ig_steps", 50)
    emb_layer = find_embedding_layer(model) if flavour != "raw" else None
    logger.info(
        f"attribution flavour: {flavour}"
        + (f" (ig_steps={ig_steps})" if flavour == "integrated" else "")
    )

    string_processor_config = builder._get_string_processor_config()
    dataset = _build_prediction_dataset(
        input_file_path,
        num,
        string_processor_config,
        kwargs.get("fsize"),
        kwargs.get("stride"),
        kwargs.get("batch"),
        min_len,
        max_len=None,
        dynamic_stride=False,
        dynamic_stride_threshold=10.0,
        dustmask=kwargs.get("dustmask", True),
    )

    class_index = kwargs.get("class_index")
    all_classes = kwargs.get("all_classes", False)
    store_gradients = kwargs.get("store_gradients", False)
    if store_gradients and flavour != "raw":
        logger.warning("--store-gradients only applies to --flavour raw; ignoring it")
        store_gradients = False
    if all_classes:
        logger.info("computing saliency for all classes (saliency_by_class)")

    saliency_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    grads_all: list[np.ndarray] = []
    saliency_bc_all: list[np.ndarray] = []
    starts_all: list[int] = []
    headers_all: list[str] = []

    for batch in track(dataset, description="[cyan]Computing saliency..."):
        x = batch[0]["translated"]
        if flavour == "raw":
            saliency, probs, grads, saliency_bc = compute_saliency_batch(
                saliency_model, x, class_index=class_index, all_classes=all_classes
            )
            if store_gradients:
                grads_all.append(grads.astype(np.float16))
        else:
            saliency, probs, saliency_bc = compute_attribution_batch(
                saliency_model,
                emb_layer,
                x,
                class_index=class_index,
                flavour=flavour,
                ig_steps=ig_steps,
                all_classes=all_classes,
            )
        saliency_all.append(saliency)
        probs_all.append(probs)
        if saliency_bc is not None:
            saliency_bc_all.append(saliency_bc)
        headers_all.extend(h.decode() for h in batch[1].numpy())
        starts_all.extend(int(s) for s in batch[2].numpy())

    arrays: dict[str, Any] = {
        "saliency": np.concatenate(saliency_all).astype(np.float32),
        "probs": np.concatenate(probs_all).astype(np.float32),
        "starts": np.array(starts_all, dtype=np.int64),
        "headers": np.array(headers_all),
        "flavour": np.array(flavour),
    }
    if store_gradients:
        arrays["grads"] = np.concatenate(grads_all)
    if saliency_bc_all:
        arrays["saliency_by_class"] = np.concatenate(saliency_bc_all).astype(np.float32)

    np.savez_compressed(output_saliency_path, **arrays)
    logger.info(f"processed {len(headers_all)} windows from {num} sequences")
    logger.info(f"wrote {output_saliency_path}")
    logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
    logger.info(
        f"memory usage : {current_process.memory_full_info().rss / GB_BYTES:.2f}GB "
        f"({current_process.memory_percent():.2f}%)"
    )


if __name__ == "__main__":
    run_core()
