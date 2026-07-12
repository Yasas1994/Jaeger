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

import polars as pl

from jaeger.nnlib.inference import InferModel, TFLiteInferModel, ONNXEngine
from jaeger.postprocess.refinement import (
    SCORE_COLS,
    add_score_features,
    aggregate_contig,
    load_refinement,
    refine,
)
from jaeger.seqops.crop import nucleotides_to_codons
from jaeger.seqops.io import fragment_generator
from jaeger.utils.gpu import get_device_name
from jaeger.utils.termini import scan_for_terminal_repeats
from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.misc import json_to_dict, AvailableModels, get_model_id
from jaeger.utils.logging import description, get_logger

# from jaeger.utils.tandem import split_fasta_with_pyfastx, run_batch, merge_masked_files
GB_BYTES = 1024**3


def _crop_length_warning(
    trained_codons: int | None, trained_nt: int | None, fsize: int
) -> str | None:
    """Return a warning if ``fsize`` does not match the model's trained
    fragment length, else ``None``.

    Codon-based models compare codon-frame counts; nucleotide models (no codon
    count) compare the nucleotide length directly.
    """
    if trained_codons is not None:
        runtime_codons = nucleotides_to_codons(int(fsize))
        if runtime_codons == trained_codons:
            return None
        nt_hint = f" ({trained_nt} nt)" if trained_nt is not None else ""
        prefer = trained_nt if trained_nt is not None else "used at training"
        return (
            f"runtime --fsize {fsize} maps to {runtime_codons} codon frames, but the "
            f"model was trained on {trained_codons} codons{nt_hint}. Fixed-length "
            f"architectures (e.g. hyena) may degrade or collapse to a single class at "
            f"a different length; prefer --fsize {prefer} for this model."
        )
    if trained_nt is not None and int(fsize) != int(trained_nt):
        return (
            f"runtime --fsize {fsize} differs from the model's trained fragment "
            f"length ({trained_nt} nt). Fixed-length architectures (e.g. hyena) may "
            f"degrade at a different length; prefer --fsize {trained_nt} for this model."
        )
    return None


def _save_auxiliary_outputs(
    y_pred: dict[str, np.ndarray],
    output_dir: Path,
    file_base: str,
    save_embedding: bool,
    save_nmd: bool,
    logger=None,
) -> None:
    """Write optional embedding and NMD vector files.

    The main ``jaeger predict`` outputs (TSV tables, window scores, FASTA
    sequences, etc.) are handled elsewhere. This helper only persists the
    ``embedding`` and ``nmd`` tensors when the user explicitly requests them.

    Args:
        y_pred: Dictionary of model outputs, must contain ``meta_0`` headers
            unless both vector outputs are absent.
        output_dir: Directory where the .npz files will be written.
        file_base: Base filename prefix (e.g., sample name).
        save_embedding: If True, write ``<file_base>_embedding.npz``.
        save_nmd: If True, write ``<file_base>_nmd.npz``.
        logger: Optional logger for diagnostic messages.
    """
    headers = y_pred.get("meta_0", np.array([], dtype=object))

    if save_embedding and "embedding" in y_pred:
        np.savez(
            output_dir / f"{file_base}_embedding.npz",
            embedding=y_pred["embedding"],
            headers=headers,
        )
        if logger is not None:
            logger.info(f"{file_base}_embedding.npz created")
    elif "embedding" in y_pred and logger is not None:
        logger.info("Skipping embedding output; pass --save-embedding to save it.")

    if save_nmd and "nmd" in y_pred:
        # Preserving legacy NPZ key name "embedding" for NMD vectors.
        np.savez(
            output_dir / f"{file_base}_nmd.npz",
            embedding=y_pred["nmd"],
            headers=headers,
        )
        if logger is not None:
            logger.info(f"{file_base}_nmd.npz created")
    elif "nmd" in y_pred and logger is not None:
        logger.info("Skipping nmd output; pass --save-nmd to save it.")


def _build_refined_contig_df(
    data_full: dict,
    taus: dict[str, dict[str, float]],
    mode: str,
    min_windows: int,
    merge_split: str,
    allow_merged_contig_call: bool,
    contig_hedge_margin: float,
) -> pl.DataFrame | None:
    """Build a refined per-contig DataFrame from raw window logits."""
    predictions = data_full.get("predictions")
    headers = data_full.get("headers")
    if predictions is None or headers is None:
        return None

    rows: list[dict] = []
    for contig_id, logits in zip(headers, predictions):
        if logits.ndim != 2:
            continue
        for window_idx, window_logits in enumerate(logits):
            row: dict[str, Any] = {
                "contig_id": contig_id,
                "window_idx": window_idx,
            }
            for score_col, value in zip(SCORE_COLS, window_logits):
                row[score_col] = value
            rows.append(row)

    if not rows:
        return None

    window_df = pl.DataFrame(rows)
    window_df = add_score_features(window_df)
    window_df = refine(window_df, taus)
    return aggregate_contig(
        window_df,
        mode=mode,
        min_windows=min_windows,
        merge_split=merge_split,
        allow_merged_contig_call=allow_merged_contig_call,
        contig_hedge_margin=contig_hedge_margin,
    )


def _make_padded_batch_specs(
    string_processor_config: dict,
) -> tuple[tuple, tuple]:
    """Return (padded_shapes, padding_values) for process_string_inference output."""
    input_type = string_processor_config.get("input_type")
    codon_depth = string_processor_config.get("codon_depth")
    padded_features: dict[str, list[int | None]] = {}
    padding_features: dict[str, float] = {}

    seq_onehot = string_processor_config.get("seq_onehot", True)
    if input_type in ("translated", "both"):
        padded_features["translated"] = (
            [6, None, codon_depth] if seq_onehot else [6, None]
        )
        padding_features["translated"] = 0.0
    if input_type in ("nucleotide", "both"):
        padded_features["nucleotide"] = [2, None, 4]
        padding_features["nucleotide"] = 0.0

    # Ten trailing metadata strings from process_string_inference.
    metadata_shape = ()
    metadata_pad = ""
    padded_shapes = (padded_features, *([metadata_shape] * 10))
    padding_values = (padding_features, *([metadata_pad] * 10))
    return padded_shapes, padding_values


def _build_prediction_dataset(
    input_file_path: Path,
    num: int,
    string_processor_config: dict,
    fragsize: int,
    stride: int,
    batch: int,
    min_len: int,
    max_len: int | None,
    dynamic_stride: bool,
    dynamic_stride_threshold: float,
    use_padded_batch: bool,
    dustmask: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data dataset for one prediction pass."""
    from jaeger.seqops.encode import process_string_inference

    input_dataset = tf.data.Dataset.from_generator(
        lambda: fragment_generator(
            file_path=str(input_file_path),
            no_progress=False,
            fragsize=fragsize,
            stride=stride,
            num=num,
            dynamic_stride=dynamic_stride,
            dynamic_stride_threshold=dynamic_stride_threshold,
            min_len=min_len,
            max_len=max_len,
            dustmask=dustmask,
        ),
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
    )

    mapped = input_dataset.map(
        process_string_inference(
            codons=string_processor_config.get("codon"),
            codon_num=string_processor_config.get("codon_id"),
            codon_depth=string_processor_config.get("codon_depth"),
            ngram_width=string_processor_config.get("ngram_width"),
            seq_onehot=string_processor_config.get("seq_onehot"),
            crop_size=fragsize,
            input_type=string_processor_config.get("input_type"),
            masking=string_processor_config.get("masking"),
            mutate=string_processor_config.get("mutate"),
            mutation_rate=string_processor_config.get("mutation_rate"),
            shuffle=string_processor_config.get("shuffle"),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if use_padded_batch:
        padded_shapes, padding_values = _make_padded_batch_specs(
            string_processor_config
        )
        return mapped.padded_batch(
            batch,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
        ).prefetch(25)
    return mapped.batch(batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(25)


def _concat_predictions(a: dict, b: dict) -> dict:
    """Concatenate two model prediction dicts along the batch axis.

    In two-pass inference either pass may match no contigs and return an
    empty dict; the other pass's predictions are then returned as-is.
    """
    if not a:
        return b
    if not b:
        return a
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def _write_prediction_outputs(
    y_pred: dict[str, np.ndarray],
    *,
    model,
    model_name: str,
    model_info: dict,
    input_file_path: Path,
    output_table_path: Path,
    output_phage_table_path: Path,
    file_base: str,
    OUTPUT_DIR: Path,
    term_repeats: Any,
    num: int,
    logger: Any,
    **kwargs: Any,
) -> None:
    """Run post-processing (summary, refinement, prophages, aux files) once.

    This used to live inside the single-pass branch of ``run_core``; moving it
    into a helper lets both single-pass and two-pass inference share the same
    output pipeline.
    """
    from jaeger.postprocess.collect import pred_to_dict, write_output

    if kwargs.get("getalllabels"):
        pass

    data, data_full = pred_to_dict(
        y_pred,
        class_map=model.class_map,
        fsize=kwargs.get("fsize"),
        term_repeats=term_repeats,
    )

    refined_contig = None
    if kwargs.get("refine", False):
        graph_dir = Path(model_info["graph"])
        refine_path = graph_dir.parent / f"{model_name}_refine.yaml"
        if refine_path.exists():
            try:
                refine_cfg = load_refinement(refine_path, expect_model=model_name)
                refined_contig = _build_refined_contig_df(
                    data_full,
                    refine_cfg["taus"],
                    mode=kwargs.get("refine_mode", "gated"),
                    min_windows=kwargs.get("refine_min_windows", 3),
                    merge_split=kwargs.get("refine_merge_split", "half"),
                    allow_merged_contig_call=kwargs.get(
                        "refine_allow_merged_contig_call", False
                    ),
                    contig_hedge_margin=kwargs.get("refine_contig_hedge_margin", 1.0),
                )
                logger.info(f"Applied refinement calibration from {refine_path}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Refinement failed: {e}; using default predictions")
        else:
            logger.warning(
                f"No refinement calibration found at {refine_path}; "
                "using default predictions"
            )

    num_written = write_output(
        data,
        labels=model.class_map.get("class"),
        indices=model.class_map.get("index"),
        output_table_path=output_table_path,
        output_phage_table_path=output_phage_table_path,
        reliability_cutoff=kwargs.get("rc", 0.5),
        phage_score=kwargs.get("pc", 1),
        refined_contig=refined_contig.to_pandas()
        if refined_contig is not None
        else None,
    )

    logger.info(f"processed {num_written}/{num} sequences")

    # --- Prophage extraction ---
    if kwargs.get("prophage"):
        from jaeger.postprocess.prophages import (
            logits_to_df_v2,
            plot_scores,
            plot_scores_linear,
            prophage_report,
            segment,
        )

        try:
            if logits_df := logits_to_df_v2(
                class_map=model.class_map,
                cmdline_kwargs=kwargs,
                headers=data_full["headers"],
                predictions=data_full["predictions"],
                lengths=data_full["lengths"],
                gc_skews=data_full["gc_skews"],
                gcs=data_full["gcs"],
            ):
                logger.info("identifying prophages")
                pro_dir = OUTPUT_DIR / f"{file_base}_prophages"
                plots_dir = pro_dir / "plots"
                for d in [pro_dir, plots_dir]:
                    d.mkdir(parents=True, exist_ok=True)

                phage_cord = segment(
                    logits_df,
                    outdir=plots_dir,
                    cutoff_length=kwargs.get("lc"),
                    sensitivity=kwargs.get("sensitivity"),
                    identifier="phage",
                )

                from jaeger.postprocess.prophage_boundaries import (
                    refine_prophage_boundaries,
                )

                refined_boundaries = refine_prophage_boundaries(
                    prophage_cordinates=phage_cord,
                    fasta_path=input_file_path,
                    fsize=kwargs.get("fsize"),
                )

                plot_type = kwargs.get("plot_type", "circular")
                if plot_type in ("circular", "both"):
                    plot_scores(
                        logits_df,
                        config={
                            "all_labels": {
                                i: c
                                for i, c in enumerate(model.class_map.get("class", []))
                            }
                        },
                        model=model_name,
                        fsize=kwargs.get("fsize"),
                        infile_base=file_base,
                        outdir=plots_dir,
                        phage_cordinates=phage_cord,
                    )
                if plot_type in ("linear", "both"):
                    plot_scores_linear(
                        logits_df,
                        config={
                            "all_labels": {
                                i: c
                                for i, c in enumerate(model.class_map.get("class", []))
                            }
                        },
                        model=model_name,
                        fsize=kwargs.get("fsize"),
                        infile_base=file_base,
                        outdir=plots_dir,
                        phage_cordinates=phage_cord,
                    )
                prophage_report(
                    fsize=kwargs.get("fsize"),
                    filehandle=str(input_file_path),
                    prophage_cordinates=phage_cord,
                    outdir=pro_dir,
                    refined_boundaries=refined_boundaries,
                )
            else:
                logger.info("no prophage regions found")
        except Exception as e:
            logger.error(f"an error {e} occurred during the prophage prediction step")
            logger.debug(traceback.format_exc())

    # --- Write phage sequences to FASTA ---
    if kwargs.get("getsequences"):
        from jaeger.postprocess.collect import write_fasta_from_results

        output_fasta_file = f"{file_base}_phages_jaeger.fasta"
        output_fasta_file_path = OUTPUT_DIR / output_fasta_file
        write_fasta_from_results(
            input_fasta=input_file_path,
            output_tsv=output_phage_table_path,
            output_fasta=output_fasta_file_path,
        )
        logger.info(f"{output_fasta_file} created")

    # --- Write window-wise scores + metadata to NPZ ---
    if kwargs.get("window_scores"):
        output_scores = f"{file_base}_window_scores.npz"
        output_scores_path = OUTPUT_DIR / output_scores
        logger.info(f"writing window-wise scores and metadata to {output_scores_path}")
        np.savez(
            output_scores_path,
            headers=data_full["headers"],
            lengths=data_full["lengths"],
            predictions=np.array(data_full["predictions"], dtype=object),
            gc_skews=np.array(data_full["gc_skews"], dtype=object),
            gcs=np.array(data_full["gcs"], dtype=object),
        )
        logger.info(f"{output_scores} created")

    logger.info(f"CPU time(s) : {psutil.Process().cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - psutil.Process().create_time():.2f}")
    logger.info(
        f"memory usage : {psutil.Process().memory_full_info().rss / GB_BYTES:.2f}GB "
        f"({psutil.Process().memory_percent():.2f}%)"
    )
    _save_auxiliary_outputs(
        y_pred,
        OUTPUT_DIR,
        file_base,
        save_embedding=kwargs.get("save_embedding", False),
        save_nmd=kwargs.get("save_nmd", False),
        logger=logger,
    )


def run_core(**kwargs):
    current_process = psutil.Process()

    USER_MODEL_PATH = kwargs.get("model_path")
    CONFIG_PATH = kwargs.get("config") or files("jaeger.data") / "config.json"

    if not USER_MODEL_PATH:
        # Use default model from config
        model_name = kwargs.get("model")
        model_id = get_model_id(model_name)

        model_paths = json_to_dict(CONFIG_PATH).get("model_paths")
        info = AvailableModels(path=model_paths).info

        model_info = info[model_name]
    else:
        # Use provided model path
        info = AvailableModels(path=USER_MODEL_PATH).info
        model_paths = USER_MODEL_PATH

        if not info:
            print(f"No model found in {model_paths}", file=sys.stderr)
            sys.exit(1)

        # A model directory may contain both a classification graph
        # (e.g. *_fragment_graph) and an embedding-only graph
        # (e.g. *_fragment_embedding_graph). Predict needs the classification
        # model, which is the one with classes, project, and weights files.
        classification_models = {
            name: meta
            for name, meta in info.items()
            if meta.get("graph") is not None and meta.get("classes") is not None
        }
        if not classification_models:
            print(
                f"No classification model found in {model_paths}. "
                "Expected a *_graph directory, *_classes.yaml, "
                "*_project.yaml, and *.weights.h5 files.",
                file=sys.stderr,
            )
            sys.exit(1)

        if len(classification_models) > 1:
            # Prefer the non-embedding model if both are present.
            non_embedding = {
                name: meta
                for name, meta in classification_models.items()
                if not name.endswith("_embedding")
            }
            if non_embedding:
                classification_models = non_embedding

        model_name = next(iter(classification_models))
        model_id = get_model_id(model_name)
        model_info = classification_models[model_name]

    MEMORY_LIMIT = 1024 * kwargs.get("mem", 4)
    THREADS = kwargs.get("workers")
    input_file_path = Path(kwargs.get("input"))
    input_file = input_file_path.name
    file_base = input_file_path.stem
    # INPUT_FILE_MASKED = input_file_path.with_name(file_base + "_masked" + input_file_path.suffix)

    OUTPUT_DIR = Path(kwargs.get("output")) / model_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # CHUNKS_DIR = OUTPUT_DIR / "chunks"
    # MAKSED_DIR = OUTPUT_DIR / "masked"

    log_file = Path(f"{file_base}_jaeger.log")
    logger = get_logger(OUTPUT_DIR, log_file, level=kwargs.get("verbose"))
    logger.info(
        description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters")
    )
    logger.debug(info)
    logger.debug(model_info)
    try:
        min_len = kwargs.get("min_len") or kwargs.get("fsize")
        num = validate_fasta_entries(str(input_file_path), min_len=min_len)
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_table_path = OUTPUT_DIR / f"{file_base}.tsv"
    output_phage_table_path = OUTPUT_DIR / f"{file_base}_phages.tsv"

    if output_table_path.exists() and not kwargs.get("overwrite"):
        logger.error(
            "output file exists. enable --overwrite option to overwrite the output file."
        )
        sys.exit(1)

    if not model_info["graph"].exists():
        logger.error(f"could not find model graph. please check {model_paths}")
        sys.exit(1)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    except RuntimeError:
        # Thread counts can only be set before the TF runtime initializes;
        # keep the existing configuration in already-initialized processes.
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

        # Set mixed precision policy if requested
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
    logger.info(f"dustmask: {kwargs.get('dustmask', True)}")
    logger.info(f"batch size: {kwargs.get('batch')}")
    logger.info(f"mode: {mode}")
    logger.info(f"model: {model_id}")
    logger.info(f"avail mem: {psutil.virtual_memory().available / (GB_BYTES):.2f}GB")
    logger.info(
        f"intra threads: {tf.config.threading.get_intra_op_parallelism_threads()}"
    )
    logger.info(
        f"inter threads: {tf.config.threading.get_inter_op_parallelism_threads()}"
    )
    logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
    logger.info(
        f"memory usage : {current_process.memory_full_info().rss / GB_BYTES:.2f}GB ({current_process.memory_percent():.2f}%)"
    )

    device = tf.config.list_logical_devices(mode)
    device_names = [get_device_name(i) for i in device]
    logger.debug(f"{device}, {device_names}")
    if len(device) > 1:
        logger.info(f"Using MirroredStrategy {device_names}")
        strategy = tf.distribute.MirroredStrategy(device_names)
    else:
        logger.info(f"Using OneDeviceStrategy {device_names}")
        strategy = tf.distribute.OneDeviceStrategy(device_names[0])
    # # 1. Split the input FASTA
    # chunk_files = split_fasta_with_pyfastx(input_file_path, CHUNKS_DIR, chunks=16)
    # # Or: chunk_files = split_fasta_with_pyfastx(input_fasta, chunks_dir, chunks=10)

    # # 2. Run TRF on chunks in parallel
    # masked_files = run_batch(chunk_files, out_dir=MAKSED_DIR, n_threads=16)

    # # 3. Merge all masked files into one
    # merge_masked_files(masked_files, INPUT_FILE_MASKED)

    # # 4. Cleanup temporary directories
    # shutil.rmtree(CHUNKS_DIR)
    # shutil.rmtree(MAKSED_DIR)

    term_repeats = scan_for_terminal_repeats(
        # file_path=str(INPUT_FILE_MASKED),
        file_path=str(input_file_path),
        num=num,
        workers=THREADS,
        fsize=kwargs.get("fsize"),
    )

    # Select inference engine based on options
    quantized_mode = kwargs.get("quantized")
    use_xla = kwargs.get("xla", False)
    use_onnx = kwargs.get("onnx", False)
    use_onnx_int8 = kwargs.get("int8", False)

    if quantized_mode:
        # Look for quantized model in the same directory as the original model
        graph_dir = Path(model_info["graph"])
        quantized_dir = graph_dir.parent / f"{model_name}_{quantized_mode}"
        tflite_path = quantized_dir / f"{model_name}_{quantized_mode}.tflite"

        if not tflite_path.exists():
            logger.error(
                f"Quantized model not found at {tflite_path}. "
                f"Run 'jaeger utils quantize -m {model_name} -o {graph_dir.parent} --mode {quantized_mode}' first."
            )
            sys.exit(1)

        logger.info(f"Using quantized model: {tflite_path}")
        model_info_tflite = model_info.copy()
        model_info_tflite["tflite"] = tflite_path
        model = TFLiteInferModel(model_info_tflite)
    elif use_onnx:
        # Look for ONNX model (FP32 or INT8)
        graph_dir = Path(model_info["graph"])
        if use_onnx_int8:
            onnx_dir = graph_dir.parent / f"{model_name}_onnx_int8"
            onnx_path = onnx_dir / f"{model_name}_int8.onnx"
            if not onnx_path.exists():
                logger.error(
                    f"INT8 ONNX model not found at {onnx_path}. "
                    f"Run 'jaeger utils convert-graph -m {model_name} -o {graph_dir.parent} --mode onnx --int8' first."
                )
                sys.exit(1)
            logger.info(f"Using INT8 ONNX model: {onnx_path}")
        else:
            onnx_dir = graph_dir.parent / f"{model_name}_onnx"
            onnx_path = onnx_dir / f"{model_name}.onnx"

            if not onnx_path.exists():
                logger.error(
                    f"ONNX model not found at {onnx_path}. "
                    f"Run 'jaeger utils convert-graph -m {model_name} -o {graph_dir.parent} --mode onnx' first."
                )
                sys.exit(1)

            logger.info(f"Using ONNX model: {onnx_path}")

        model_info_onnx = model_info.copy()
        model_info_onnx["onnx"] = onnx_path

        model = ONNXEngine(model_info_onnx)
    else:
        if use_xla:
            logger.info(
                "Using XLA-compiled inference (first batch may be slow due to compilation)"
            )
        model = InferModel(model_info, use_xla=use_xla)

    string_processor_config = model.string_processor_config
    batch_size = kwargs.get("batch")
    fsize = kwargs.get("fsize")
    stride = kwargs.get("stride")
    dynamic_stride = kwargs.get("dynamic_stride", False)
    dustmask = kwargs.get("dustmask", True)
    dynamic_stride_threshold = kwargs.get("dynamic_stride_threshold", 10.0)

    _trained_codons = string_processor_config.get("crop_size_codons")
    _trained_nt = string_processor_config.get("crop_size_nt")
    if _trained_codons is not None:
        logger.info(
            f"model trained fragment length: {_trained_codons} codons "
            f"({_trained_nt} nt)"
        )
    elif _trained_nt is not None:
        logger.info(f"model trained fragment length: {_trained_nt} nt")
    _crop_msg = _crop_length_warning(_trained_codons, _trained_nt, fsize)
    if _crop_msg is not None:
        logger.warning(_crop_msg)

    user_min_len = kwargs.get("min_len")
    min_len = user_min_len or fsize

    if user_min_len is not None and user_min_len < fsize:
        logger.info(
            f"Two-pass prediction: long contigs (>= {fsize} bp) then short contigs "
            f"({user_min_len}-{fsize - 1} bp)"
        )
        long_dataset = _build_prediction_dataset(
            input_file_path=input_file_path,
            num=num,
            string_processor_config=string_processor_config,
            fragsize=fsize,
            stride=stride,
            batch=batch_size,
            min_len=fsize,
            max_len=None,
            dynamic_stride=dynamic_stride,
            dynamic_stride_threshold=dynamic_stride_threshold,
            dustmask=dustmask,
            use_padded_batch=False,
        )
        short_dataset = _build_prediction_dataset(
            input_file_path=input_file_path,
            num=num,
            string_processor_config=string_processor_config,
            fragsize=fsize,
            stride=stride,
            batch=batch_size,
            min_len=user_min_len,
            max_len=fsize - 1,
            dynamic_stride=dynamic_stride,
            dynamic_stride_threshold=dynamic_stride_threshold,
            dustmask=dustmask,
            use_padded_batch=True,
        )
        with strategy.scope():
            try:
                logger.info("starting model inference on long contigs")
                y_pred_long = model.predict(long_dataset, no_progress=True)
                logger.info("starting model inference on short contigs")
                y_pred_short = model.predict(short_dataset, no_progress=True)
                y_pred = _concat_predictions(y_pred_long, y_pred_short)
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.error(
                    f"an error {e} occured during inference on {'|'.join(device_names)}! check {log_file} for traceback."
                )
                sys.exit(1)
    else:
        idataset = _build_prediction_dataset(
            input_file_path=input_file_path,
            num=num,
            string_processor_config=string_processor_config,
            fragsize=fsize,
            stride=stride,
            batch=batch_size,
            min_len=min_len,
            max_len=None,
            dynamic_stride=dynamic_stride,
            dynamic_stride_threshold=dynamic_stride_threshold,
            dustmask=dustmask,
            use_padded_batch=False,
        )
        with strategy.scope():
            try:
                logger.info("starting model inference")
                y_pred = model.predict(idataset, no_progress=True)
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.error(
                    f"an error {e} occured during inference on {'|'.join(device_names)}! check {log_file} for traceback."
                )
                sys.exit(1)

    # Remove the CLI model name from kwargs to avoid shadowing the loaded
    # ``model`` object passed explicitly below.
    kwargs.pop("model", None)
    _write_prediction_outputs(
        y_pred,
        model=model,
        model_name=model_name,
        model_info=model_info,
        input_file_path=input_file_path,
        output_table_path=output_table_path,
        output_phage_table_path=output_phage_table_path,
        file_base=file_base,
        OUTPUT_DIR=OUTPUT_DIR,
        term_repeats=term_repeats,
        num=num,
        logger=logger,
        **kwargs,
    )
    # INPUT_FILE_MASKED.unlink()
