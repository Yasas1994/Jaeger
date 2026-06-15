"""PyTorch-based ``jaeger predict`` command."""

from __future__ import annotations

import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from jaeger.inference.pytorch.runner import PyTorchInferenceRunner
from jaeger.postprocess.collect import pred_to_dict, write_output
from jaeger.seqops.io import fragment_generator, validate_fasta_entries
from jaeger.seqops.maps import CODON_ID, CODONS
from jaeger.utils.logging import description, get_logger
from jaeger.utils.misc import load_model_config
from jaeger.utils.termini import scan_for_terminal_repeats

GB_BYTES = 1024**3


def _resolve_codon_value(value):
    """Resolve a codon table configuration value to a list."""
    if isinstance(value, str):
        if value in ("CODON", "CODONS"):
            return CODONS
        if value == "CODON_ID":
            return CODON_ID
        raise ValueError(f"Unsupported codon table name: {value!r}")
    return value


def _build_codon_table(codon_cfg, codon_id_cfg):
    """Build a codon -> integer ID lookup from config values."""
    codons = _resolve_codon_value(codon_cfg)
    codon_ids = _resolve_codon_value(codon_id_cfg)
    return dict(zip(codons, codon_ids))


def _translate_fragment(seq: str, codon_table: dict, crop_size: int) -> torch.Tensor:
    """Translate a DNA fragment into a ``(6, L)`` codon index tensor.

    The six reading frames are produced using the same offset/striding logic as
    the training-time raw sequence processor.
    """
    seq = seq.upper()[:crop_size]
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    rev = "".join(comp.get(base, "N") for base in reversed(seq))

    def _codon_ids(s: str):
        return [codon_table.get(s[i : i + 3], -1) for i in range(len(s) - 2)]

    forward = _codon_ids(seq)
    reverse = _codon_ids(rev)

    frames = [forward[offset::3] for offset in range(3)] + [
        reverse[offset::3] for offset in range(3)
    ]

    target_len = max(0, len(seq) // 3 - 1)
    frames = [f[:target_len] + [-1] * (target_len - len(f)) for f in frames]

    arr = np.array(frames, dtype=np.int64) + 1  # 0 is reserved for padding/masking
    return torch.from_numpy(arr)


def _aggregate_predictions(outputs: dict) -> dict:
    """Average each tensor in ``outputs`` over the batch dimension."""
    return {k: v.mean(dim=0) for k, v in outputs.items()}


def _append_outputs(container: dict, outputs: dict):
    """Append CPU numpy arrays from a runner output dict to a container."""
    for k, v in outputs.items():
        container.setdefault(k, []).append(v.cpu().numpy())


def _concat_outputs(container: dict) -> dict:
    """Concatenate lists of numpy arrays per output key."""
    return {k: np.concatenate(vs, axis=0) for k, vs in container.items()}


def _build_class_map(config: dict) -> dict:
    """Build a class_map compatible with ``jaeger.postprocess.collect``."""
    class_label_map = config.get("model", {}).get("class_label_map", [])
    classes = [entry["class"] for entry in class_label_map]
    indices = [entry["label"] for entry in class_label_map]
    return {"num_classes": len(classes), "class": classes, "index": indices}


def run_core(**kwargs):
    """Run the PyTorch Jaeger inference pipeline."""
    try:
        import psutil

        current_process = psutil.Process()
    except Exception:
        current_process = None

    model_path = kwargs.get("model_path")
    config_path = kwargs.get("config")
    checkpoint_path = kwargs.get("checkpoint")

    if model_path:
        model_path = Path(model_path)
        config_file = None
        for name in ("config.yaml", "config.yml", "config.json"):
            candidate = model_path / name
            if candidate.exists():
                config_file = candidate
                break
        if config_file is None:
            raise FileNotFoundError(f"No config file found in {model_path}")
        config = load_model_config(config_file)
        checkpoint_path = model_path / "model.pt"
        model_name = config.get("model", {}).get("name", model_path.name)
    elif config_path and checkpoint_path:
        config = load_model_config(Path(config_path))
        checkpoint_path = Path(checkpoint_path)
        model_name = config.get("model", {}).get("name", "jaeger_pytorch")
    else:
        raise NotImplementedError(
            "Default TensorFlow model loading is not supported in PyTorch mode. "
            "Please provide --model_path (with config.yaml and model.pt) or both "
            "--config and --checkpoint."
        )

    input_file_path = Path(kwargs["input"])
    file_base = input_file_path.stem
    model_id = model_name.replace(" ", "_")
    output_dir = Path(kwargs["output"]) / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(f"{file_base}_jaeger.log")
    logger = get_logger(output_dir, log_file, level=kwargs.get("verbose", 1))
    logger.info(
        description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters")
    )
    logger.debug(config)

    fsize = kwargs.get("fsize", 2000)
    stride = kwargs.get("stride", 2000)
    threads = kwargs.get("workers", 4)
    batch_size = kwargs.get("batch", 96)

    try:
        num = validate_fasta_entries(str(input_file_path), min_len=fsize)
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_table_path = output_dir / f"{file_base}.tsv"
    output_phage_table_path = output_dir / f"{file_base}_phages.tsv"

    if output_table_path.exists() and not kwargs.get("overwrite"):
        logger.error(
            "output file exists. enable --overwrite option to overwrite the output file."
        )
        sys.exit(1)

    for opt in ("quantized", "xla", "onnx", "int8"):
        if kwargs.get(opt):
            logger.warning(
                f"--{opt} is not supported by the PyTorch inference path and will be ignored."
            )
    precision = kwargs.get("precision", "fp32")
    if precision and precision != "fp32":
        logger.warning(f"--precision {precision} is not supported and will be ignored.")

    if kwargs.get("cpu") or not torch.cuda.is_available():
        device = torch.device("cpu")
        mode = "CPU"
        if not kwargs.get("cpu") and not torch.cuda.is_available():
            logger.warning(
                "could not find a GPU on the system. "
                "For optimal performance run Jaeger on a GPU."
            )
    else:
        device = torch.device("cuda")
        mode = "GPU"

    logger.info(f"pytorch: {version('torch')}")
    logger.info(f"input file: {input_file_path.name}")
    logger.info(f"log file: {log_file.name}")
    logger.info(f"outpath: {output_dir.resolve()}")
    logger.info(f"fragment size: {fsize}")
    logger.info(f"stride: {stride}")
    logger.info(f"batch size: {batch_size}")
    logger.info(f"mode: {mode}")
    logger.info(f"model: {model_id}")

    if current_process is not None:
        logger.info(f"avail mem: {psutil.virtual_memory().available / GB_BYTES:.2f}GB")
        logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
        logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
        logger.info(
            f"memory usage : {current_process.memory_full_info().rss / GB_BYTES:.2f}GB "
            f"({current_process.memory_percent():.2f}%)"
        )

    torch.set_num_threads(threads)

    runner = PyTorchInferenceRunner(
        config, checkpoint_path=checkpoint_path, device=device
    )
    class_map = _build_class_map(config)

    term_repeats = scan_for_terminal_repeats(
        file_path=str(input_file_path),
        num=num,
        workers=threads,
        fsize=fsize,
    )

    sp_config = config.get("model", {}).get("string_processor", {})
    crop_size = sp_config.get("crop_size", fsize)
    codon_table = _build_codon_table(
        sp_config.get("codon", CODONS), sp_config.get("codon_id", CODON_ID)
    )

    meta = {
        "headers": [],
        "index": [],
        "contig_end": [],
        "window_idx": [],
        "length": [],
        "c_count": [],
        "g_count": [],
        "a_count": [],
        "t_count": [],
        "gc_skew": [],
    }
    inputs: list[torch.Tensor] = []
    outputs_container: dict = {}

    for frag_str in fragment_generator(
        file_path=str(input_file_path),
        fragsize=fsize,
        stride=stride,
        num=num,
        no_progress=True,
    ):
        parts = frag_str.split(",")
        seq = parts[0]
        meta["headers"].append(parts[1])
        meta["index"].append(int(parts[2]))
        meta["contig_end"].append(int(parts[3]))
        meta["window_idx"].append(int(parts[4]))
        meta["length"].append(int(parts[5]))
        meta["g_count"].append(float(parts[6]))
        meta["c_count"].append(float(parts[7]))
        meta["a_count"].append(float(parts[8]))
        meta["t_count"].append(float(parts[9]))
        meta["gc_skew"].append(float(parts[10]))

        inputs.append(_translate_fragment(seq, codon_table, crop_size))
        if len(inputs) >= batch_size:
            batch_x = torch.stack(inputs)
            out = runner.predict(batch_x, batch_size=batch_size)
            _append_outputs(outputs_container, out)
            inputs.clear()

    if inputs:
        batch_x = torch.stack(inputs)
        out = runner.predict(batch_x, batch_size=batch_size)
        _append_outputs(outputs_container, out)
        inputs.clear()

    if not outputs_container:
        logger.error("no fragments were generated for inference")
        sys.exit(1)

    y_pred = _concat_outputs(outputs_container)
    y_pred["meta_0"] = np.array(meta["headers"], dtype=str)
    y_pred["meta_1"] = np.array(meta["index"], dtype=np.int32)
    y_pred["meta_2"] = np.array(meta["contig_end"], dtype=np.int32)
    y_pred["meta_3"] = np.array(meta["window_idx"], dtype=np.int32)
    y_pred["meta_4"] = np.array(meta["length"], dtype=np.int32)
    y_pred["meta_5"] = np.array(meta["c_count"], dtype=np.float32)
    y_pred["meta_6"] = np.array(meta["g_count"], dtype=np.float32)
    y_pred["meta_7"] = np.array(meta["a_count"], dtype=np.float32)
    y_pred["meta_8"] = np.array(meta["t_count"], dtype=np.float32)
    y_pred["meta_9"] = np.array(meta["gc_skew"], dtype=np.float32)

    try:
        data, data_full = pred_to_dict(
            y_pred,
            class_map=class_map,
            fsize=fsize,
            term_repeats=term_repeats,
        )
        num_written = write_output(
            data,
            labels=class_map["class"],
            indices=class_map["index"],
            output_table_path=output_table_path,
            output_phage_table_path=output_phage_table_path,
            reliability_cutoff=kwargs.get("rc", 0.5),
            phage_score=kwargs.get("pc", 1),
        )
        logger.info(f"processed {num_written}/{num} sequences")
    except Exception as e:
        logger.error(f"an error {e} occurred during postprocessing")
        logger.debug(traceback.format_exc())
        sys.exit(1)

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
                class_map=class_map,
                cmdline_kwargs=kwargs,
                headers=data_full["headers"],
                predictions=data_full["predictions"],
                lengths=data_full["lengths"],
                gc_skews=data_full["gc_skews"],
                gcs=data_full["gcs"],
            ):
                logger.info("identifying prophages")
                pro_dir = output_dir / f"{file_base}_prophages"
                plots_dir = pro_dir / "plots"
                for d in [pro_dir, plots_dir]:
                    d.mkdir(parents=True, exist_ok=True)

                phage_cord = segment(
                    logits_df,
                    outdir=plots_dir,
                    cutoff_length=kwargs.get("lc", 500_000),
                    sensitivity=kwargs.get("sensitivity", 1.5),
                    identifier="phage",
                )
                plot_type = kwargs.get("plot_type", "circular")
                config_labels = {i: c for i, c in enumerate(class_map.get("class", []))}
                if plot_type in ("circular", "both"):
                    plot_scores(
                        logits_df,
                        config={"all_labels": config_labels},
                        model=model_name,
                        fsize=fsize,
                        infile_base=file_base,
                        outdir=plots_dir,
                        phage_cordinates=phage_cord,
                    )
                if plot_type in ("linear", "both"):
                    plot_scores_linear(
                        logits_df,
                        config={"all_labels": config_labels},
                        model=model_name,
                        fsize=fsize,
                        infile_base=file_base,
                        outdir=plots_dir,
                        phage_cordinates=phage_cord,
                    )
                prophage_report(
                    fsize=fsize,
                    filehandle=str(input_file_path),
                    prophage_cordinates=phage_cord,
                    outdir=pro_dir,
                )
            else:
                logger.info("no prophage regions found")
        except Exception as e:
            logger.error(f"an error {e} occurred during the prophage prediction step")
            logger.debug(traceback.format_exc())

    if kwargs.get("getsequences"):
        from jaeger.postprocess.collect import write_fasta_from_results

        output_fasta_file_path = output_dir / f"{file_base}_phages_jaeger.fasta"
        if output_phage_table_path.exists():
            write_fasta_from_results(
                input_fasta=str(input_file_path),
                output_tsv=str(output_phage_table_path),
                output_fasta=str(output_fasta_file_path),
            )
            logger.info(f"{output_fasta_file_path.name} created")
        else:
            logger.info("no phage sequences to write")

    if kwargs.get("window_scores"):
        output_scores_path = output_dir / f"{file_base}_window_scores.npz"
        logger.info(f"writing window-wise scores and metadata to {output_scores_path}")
        np.savez(
            output_scores_path,
            headers=data_full["headers"],
            lengths=data_full["lengths"],
            predictions=np.array(data_full["predictions"], dtype=object),
            gc_skews=np.array(data_full["gc_skews"], dtype=object),
            gcs=np.array(data_full["gcs"], dtype=object),
        )
        logger.info(f"{output_scores_path.name} created")

    if current_process is not None:
        logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
        logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
        logger.info(
            f"memory usage : {current_process.memory_full_info().rss / GB_BYTES:.2f}GB "
            f"({current_process.memory_percent():.2f}%)"
        )

    if "embedding" in y_pred:
        np.savez(
            output_dir / f"{file_base}_embedding.npz",
            embedding=y_pred["embedding"],
            headers=y_pred["meta_0"],
        )
    if "nmd" in y_pred:
        np.savez(
            output_dir / f"{file_base}_nmd.npz",
            embedding=y_pred["nmd"],
            headers=y_pred["meta_0"],
        )
