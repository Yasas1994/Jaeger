"""Training CLI entry point for Jaeger.

All core model-building logic lives in `jaeger.nnlib.builder`.
This module is a thin Click wrapper that orchestrates:
- strategy selection (GPU/CPU, mixed precision)
- data pipeline construction
- model training loops
- saving
"""

from __future__ import annotations

import gc
import json
import os
from typing import Any

# temporary fix
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

import click
import tensorflow as tf
from pathlib import Path

from jaeger.nnlib.builder import DynamicModelBuilder, check_files
from jaeger.seqops.encode import process_string_train
from jaeger.utils.misc import load_model_config, numerize
from jaeger.utils.logging import get_logger
from jaeger.data.loaders import _load_numpy_dataset
from jaeger.dataops.reliability_generator import generate_reliability_data

try:
    from icecream import ic

    ic.configureOutput(prefix="Jaeger |")
except ImportError:

    def ic(*args, **kwargs):
        pass


logger = get_logger(log_file=None, log_path=None, level=3)


def _write_convergence_marker(
    checkpoint_dir: str | Path, branch: str, epoch: int
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    marker_path = checkpoint_dir / "converged.json"
    marker_path.write_text(
        json.dumps(
            {"is_converged": True, "branch": branch, "epoch": epoch},
            indent=2,
        )
    )
    logger.info(f"Recorded convergence for {branch} at epoch {epoch}")


def _apply_ignore_convergence(
    checkpoint: dict[str, Any], ignore: bool, branches: list[str]
) -> None:
    """Reset ``is_converged`` for listed branches when asked to ignore markers.

    This mutates ``checkpoint`` in place so that downstream training gates
    treat converged branches as runnable for the current invocation only.
    """
    if not ignore:
        return
    for branch in branches:
        branch_ckpt = checkpoint.get(branch)
        if branch_ckpt is not None:
            branch_ckpt["is_converged"] = False
            logger.info(f"Ignoring convergence marker for {branch}")


def _resolve_numpy_crop_params(
    string_processor_config: dict[str, Any], split: str
) -> tuple[Any, Any, Any]:
    """Return crop_sizes, strides, overlap for a train/validation NumPy split.

    Training uses the multi-crop config (``crop_sizes`` / ``strides`` /
    ``overlap``). Validation defaults to a single ``crop_size`` crop or no
    runtime cropping, unless ``validation_crop_sizes`` is explicitly provided.
    This keeps validation throughput high while still allowing multi-crop
    evaluation when desired.
    """
    if split == "validation":
        crop_sizes = string_processor_config.get("validation_crop_sizes")
        strides = string_processor_config.get("validation_strides")
        overlap = string_processor_config.get("validation_overlap")
    else:
        crop_sizes = string_processor_config.get("crop_sizes")
        strides = string_processor_config.get("strides")
        overlap = string_processor_config.get("overlap")

    # Fallback to a single crop_size if crop_sizes is not provided.
    if crop_sizes is None:
        crop_size = string_processor_config.get("crop_size")
        if crop_size is not None:
            crop_sizes = [crop_size]

    return crop_sizes, strides, overlap


def _replica_round(batch_size: int, num_replicas: int) -> int:
    """Return the largest replica-divisible batch size <= batch_size."""
    if num_replicas <= 1:
        return batch_size
    return (batch_size // num_replicas) * num_replicas


def _apply_grouped_batching(
    ds: tf.data.Dataset,
    batching_cfg: dict[str, Any],
    num_replicas: int,
    feature_key: str,
) -> tf.data.Dataset:
    """Batch sequences so every batch contains one exact length.

    Per-length batch sizes come from ``length_batch_sizes``; unlisted lengths
    use ``default_batch_size``. Batch sizes are rounded down to multiples of
    ``num_replicas`` when running on multiple devices.
    """
    if "default_batch_size" not in batching_cfg:
        raise ValueError(
            "grouped batching requires string_processor.batching.default_batch_size"
        )

    length_batch_sizes = {
        int(k): int(v) for k, v in batching_cfg.get("length_batch_sizes", {}).items()
    }
    default_batch_size = int(batching_cfg["default_batch_size"])

    # Build a lookup table: length -> effective batch size.
    # StaticHashTable does not accept empty key/value tensors, so we always
    # seed it with at least one dummy entry.
    default_value = _replica_round(default_batch_size, num_replicas)
    if default_value <= 0:
        raise ValueError(
            f"default_batch_size ({default_batch_size}) rounds to 0 for "
            f"{num_replicas} replicas; increase it."
        )

    if length_batch_sizes:
        keys = tf.constant(list(length_batch_sizes.keys()), dtype=tf.int64)
        vals = tf.constant(
            [
                (
                    default_value
                    if _replica_round(length_batch_sizes[k], num_replicas) == 0
                    else _replica_round(length_batch_sizes[k], num_replicas)
                )
                for k in length_batch_sizes.keys()
            ],
            dtype=tf.int64,
        )
    else:
        keys = tf.constant([-1], dtype=tf.int64)
        vals = tf.constant([default_value], dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(default_value, dtype=tf.int64),
    )

    def _key_fn(features, _label):
        feature = features[feature_key]
        return tf.cast(tf.shape(feature)[1], tf.int64)

    def _reduce_func(_key, dataset):
        batch_size = table.lookup(_key)
        return dataset.batch(batch_size, drop_remainder=True)

    def _window_size_func(key):
        return table.lookup(key)

    return ds.group_by_window(
        key_func=_key_fn,
        reduce_func=_reduce_func,
        window_size_func=_window_size_func,
    )


def _build_numpy_split(
    path: str,
    num_classes: int,
    string_processor_config: dict[str, Any],
    batching_cfg: dict[str, Any],
    batch_size: int,
    multi_gpu: bool,
    num_replicas: int,
    buffer_size: int | None,
    split: str,
) -> tf.data.Dataset:
    """Build a batched/prefetched NumPy dataset for one split."""
    _crop_sizes, _strides, _overlap = _resolve_numpy_crop_params(
        string_processor_config, split
    )
    _onehot_buffer = (
        buffer_size if buffer_size is not None and buffer_size > 0 else None
    )
    batching_strategy = batching_cfg.get("strategy", "padded")
    _data = _load_numpy_dataset(
        path,
        input_type=string_processor_config.get("input_type"),
        seq_onehot=string_processor_config.get("seq_onehot"),
        codon_depth=string_processor_config.get("codon_depth"),
        nucleotide_onehot_map=string_processor_config.get("nucleotide_onehot_map"),
        num_classes=num_classes,
        one_hot_labels=True,
        buffer_size=_onehot_buffer,
        crop_sizes=_crop_sizes,
        strides=_strides,
        overlap=_overlap,
        pad_to_max=(batching_strategy != "grouped"),
    )

    ds = _data
    if _onehot_buffer is None and not _is_ragged_dataset(_data):
        ds = ds.cache()
    ds = ds.shuffle(
        buffer_size=buffer_size if buffer_size != -1 else 100000,
    )

    if batching_strategy == "grouped":
        feature_specs = _data.element_spec[0]
        if len(feature_specs) != 1:
            raise ValueError(
                "grouped batching currently requires a single input feature "
                f"(got {sorted(feature_specs.keys())})."
            )
        feature_key = next(iter(feature_specs.keys()))
        ds = _apply_grouped_batching(
            ds,
            batching_cfg,
            num_replicas=num_replicas,
            feature_key=feature_key,
        )
    else:
        padded_shapes = (
            tf.nest.map_structure(lambda spec: spec.shape, _data.element_spec[0]),
            _data.element_spec[1].shape,
        )
        ds = ds.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=multi_gpu,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------
# CLI command
# ------------------------------------------------------------------


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--mixed_precision", is_flag=True, default=False)
@click.option("--from_last_checkpoint", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
@click.option("--only_classification_head", is_flag=True, default=False)
@click.option("--only_reliability_head", is_flag=True, default=False)
@click.option("--only_heads", is_flag=True, default=False)
@click.option("--only_save", is_flag=True, default=False)
@click.option("--save_model", is_flag=True, default=False)
@click.option("--self_supervised_pretraining", is_flag=True, default=False)
@click.option(
    "--xla",
    is_flag=True,
    default=False,
    help="Enable XLA JIT compilation for training.",
)
@click.option(
    "--ignore_convergence",
    is_flag=True,
    default=False,
    help="Ignore convergence markers and re-train from the last checkpoint",
)
@click.option("--meta", type=click.Path(), default=None)
def train_fragment(
    config,
    mixed_precision,
    from_last_checkpoint,
    force,
    only_classification_head,
    only_reliability_head,
    only_heads,
    only_save,
    save_model,
    self_supervised_pretraining,
    xla,
    ignore_convergence,
    meta,
):
    """Train a fragment-level Jaeger model."""
    train_fragment_core(
        config=config,
        mixed_precision=mixed_precision,
        from_last_checkpoint=from_last_checkpoint,
        force=force,
        only_classification_head=only_classification_head,
        only_reliability_head=only_reliability_head,
        only_heads=only_heads,
        only_save=only_save,
        save_model=save_model,
        self_supervised_pretraining=self_supervised_pretraining,
        xla=xla,
        ignore_convergence=ignore_convergence,
        meta=meta,
    )


# ------------------------------------------------------------------
# Core training orchestration (CLI glue)
# ------------------------------------------------------------------


def _is_ragged_dataset(ds: tf.data.Dataset) -> bool:
    feat_spec = ds.element_spec[0]
    for spec in tf.nest.flatten(feat_spec):
        if any(d is None for d in spec.shape.as_list()):
            return True
    return False


def train_fragment_core(**kwargs):
    """Train fragment classification and reliability models."""
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)
    logger.info(f"Physical GPUs detected: {num_gpus}")

    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(
            f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas"
        )
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
        logger.info("Using OneDeviceStrategy on /GPU:0")
    else:
        strategy = tf.distribute.get_strategy()
        logger.info("No GPU detected, using default CPU strategy")

    if kwargs.get("mixed_precision", False):
        logger.info("experimental: using mix precision floats for faster training")
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    multi_gpu = strategy.num_replicas_in_sync > 1

    with strategy.scope():
        logger.info("initializing model")
        config = load_model_config(Path(kwargs.get("config")))
        config["mix_precision"] = kwargs.get("mixed_precision", False)
        config["from_last_checkpoint"] = kwargs.get("from_last_checkpoint")
        config["force"] = kwargs.get("force")
        config["ignore_convergence"] = kwargs.get("ignore_convergence", False)
        config["use_xla"] = kwargs.get("xla", False)
        config["generate_reliability_data"] = kwargs.get(
            "generate_reliability_data", False
        )
        if config["use_xla"]:
            logger.info("Using XLA JIT compilation for training")

        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()
        for (
            _,
            m,
        ) in models.items():
            m.summary()
        model_num_params = numerize(models.get("rep_model").count_params(), decimal=1)

        # ================= train classifier ======================
        builder.compile_model(models, train_branch="classifier")

        string_processor_config = builder._get_string_processor_config()
        batching_cfg = string_processor_config.get("batching", {})
        batching_strategy = batching_cfg.get("strategy", "padded")
        if batching_strategy not in ("padded", "grouped"):
            raise ValueError(
                f"Invalid batching strategy: {batching_strategy!r}. "
                "Use 'padded' or 'grouped'."
            )
        _train_data = builder._get_fragment_paths()
        train_data = {"train": None, "validation": None}

        data_format = string_processor_config.get("data_format", "csv")
        _buffer_size = string_processor_config.get("buffer_size")

        # ============ check convergence & train classifier ===========
        checkpoint = builder._checkpoints
        _apply_ignore_convergence(
            checkpoint, kwargs.get("ignore_convergence", False), ["classifier"]
        )
        cls_converged = checkpoint and checkpoint.get("classifier", {}).get(
            "is_converged", False
        )
        if kwargs.get("only_save", False) is False:
            if cls_converged:
                logger.info(
                    "Classifier training already converged; skipping. "
                    "Use --ignore_convergence to retrain."
                )
            elif not kwargs.get("only_reliability_head", False):
                if kwargs.get("only_classification_head", False) or kwargs.get(
                    "only_heads", False
                ):
                    models.get("rep_model").trainable = False

                # load train data here
                for k, v in _train_data.items():
                    paths = check_files(v.get("paths"))
                    if not paths:
                        logger.warning(
                            "no valid files in paths=%r %r", k, v.get("paths")
                        )
                        exit(1)

                    if data_format == "csv":
                        _data = tf.data.TextLineDataset(
                            paths, num_parallel_reads=len(paths), buffer_size=200
                        )
                        if string_processor_config.get("input_type") == "translated":
                            padded_shape = {
                                "translated": [6, None]
                                if string_processor_config.get("use_embedding_layer")
                                is True
                                else [
                                    6,
                                    None,
                                    string_processor_config.get("codon_depth"),
                                ]
                            }
                        elif string_processor_config.get("input_type") == "nucleotide":
                            padded_shape = {
                                "nucleotide": [
                                    2,
                                    string_processor_config.get("crop_size"),
                                    4,
                                ]
                            }
                        train_data[k] = (
                            _data.map(
                                process_string_train(
                                    codons=string_processor_config.get("codon"),
                                    codon_num=string_processor_config.get("codon_id"),
                                    codon_depth=string_processor_config.get(
                                        "codon_depth"
                                    ),
                                    label_original=string_processor_config.get(
                                        "classifier_labels", None
                                    ),
                                    label_alternative=string_processor_config.get(
                                        "classifier_labels_map", None
                                    ),
                                    ngram_width=string_processor_config.get(
                                        "ngram_width"
                                    ),
                                    seq_onehot=string_processor_config.get(
                                        "seq_onehot"
                                    ),
                                    crop_size=string_processor_config.get("crop_size"),
                                    input_type=string_processor_config.get(
                                        "input_type"
                                    ),
                                    masking=string_processor_config.get("masking"),
                                    mutate=string_processor_config.get("mutate"),
                                    mutation_rate=string_processor_config.get(
                                        "mutation_rate"
                                    ),
                                    num_classes=builder.classifier_out_dim,
                                    class_label_onehot=False
                                    if "binary"
                                    in builder.train_cfg.get(
                                        "loss_classifier", "categorical_crossentropy"
                                    ).lower()
                                    else True,
                                    shuffle=string_processor_config.get("shuffle"),
                                    shuffle_frames=string_processor_config.get(
                                        "shuffle_frames", False
                                    ),
                                ),
                                num_parallel_calls=tf.data.AUTOTUNE,
                            )
                            .shuffle(
                                buffer_size=_data.cardinality()
                                if _buffer_size == -1
                                else _buffer_size,
                            )
                            .padded_batch(
                                batch_size=builder.train_cfg.get("batch_size"),
                                padded_shapes=(
                                    padded_shape,
                                    [builder.classifier_out_dim],
                                ),
                                drop_remainder=multi_gpu,
                            )
                            .prefetch(tf.data.AUTOTUNE)
                        )
                    elif data_format == "numpy":
                        logger.info(f"Loading {k} data from NumPy: {paths}")
                        if len(paths) > 1:
                            logger.warning(
                                "NumPy format only supports a single .npz file per split; using first: %s",
                                paths[0],
                            )
                        train_data[k] = _build_numpy_split(
                            paths[0],
                            num_classes=builder.classifier_out_dim,
                            string_processor_config=string_processor_config,
                            batching_cfg=batching_cfg,
                            batch_size=builder.train_cfg.get("batch_size"),
                            multi_gpu=multi_gpu,
                            num_replicas=strategy.num_replicas_in_sync,
                            buffer_size=_buffer_size,
                            split=k,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported data_format: {data_format}. Use 'csv' or 'numpy'."
                        )

                train_args = {
                    "validation_data": train_data.get("validation").take(
                        builder.train_cfg.get("classifier_validation_steps")
                    ),
                    "epochs": builder.train_cfg.get("classifier_epochs"),
                    "callbacks": builder.get_callbacks(branch="classifier"),
                }
                if checkpoint:
                    train_args["initial_epoch"] = checkpoint.get("classifier", {}).get(
                        "epoch", 0
                    )

                # self-supervised pre-training
                if kwargs.get("self_supervised_pretraining", False):
                    builder.compile_model(models, train_branch="pretrain")
                    # models.get("jaeger_projection").summary()
                    self_supervised_train_args = {
                        "validation_data": train_data.get("validation").take(
                            builder.train_cfg.get("classifier_validation_steps")
                        ),
                        "epochs": builder.train_cfg.get("projection_epochs"),
                        "callbacks": builder.get_callbacks(branch="projection"),
                    }
                    if checkpoint:
                        self_supervised_train_args["initial_epoch"] = checkpoint.get(
                            "projection", {}
                        ).get("epoch", 0)
                    models.get("jaeger_projection").fit(
                        train_data.get("train").take(
                            builder.train_cfg.get("classifier_train_steps")
                        ),
                        **self_supervised_train_args,
                    )

                # train classification model
                classifier_history = models.get("jaeger_classifier").fit(
                    train_data.get("train").take(
                        builder.train_cfg.get("classifier_train_steps"),
                    ),
                    class_weight=builder.train_cfg.get("classifier_class_weights"),
                    **train_args,
                )
                if (
                    classifier_history.epoch
                    and classifier_history.epoch[-1] < train_args["epochs"] - 1
                ):
                    classifier_dir = builder.train_cfg.get("classifier_dir")
                    if classifier_dir is not None:
                        _write_convergence_marker(
                            classifier_dir,
                            branch="classifier",
                            epoch=int(classifier_history.epoch[-1]),
                        )
                    else:
                        logger.warning(
                            "classifier_dir is not configured; skipping convergence marker"
                        )

                # unload classifier data before loading reliability data
                logger.info("unloading classifier training data")
                train_data = {"train": None, "validation": None}
                gc.collect()
            else:
                logger.info("Skipping training — classification model")

        # ============== generate reliability data ========================
        if kwargs.get("generate_reliability_data", False):
            generator_cfg = builder.train_cfg.get("reliability_data_generation", {})

            raw_csv_paths = generator_cfg.get("raw_csv_paths") or {}
            if data_format == "csv":
                raw_csv_path = (
                    raw_csv_paths.get("train")
                    or _train_data.get("train", {}).get("paths", [None])[0]
                )
            else:
                raw_csv_path = raw_csv_paths.get("train") or generator_cfg.get(
                    "raw_csv_path"
                )

            if not raw_csv_path:
                raise ValueError(
                    "--generate_reliability_data requires raw CSV sequences. "
                    "Set reliability_data_generation.raw_csv_paths.train in the config "
                    "or use data_format: csv for classifier training."
                )

            if builder._get_reliability_fragment_paths().get("train", {}).get("paths"):
                logger.warning(
                    "--generate_reliability_data is active; ignoring "
                    "fragment_reliability_data paths provided in the config"
                )

            output_dir = generator_cfg.get("output_dir") or str(
                Path(raw_csv_path).parent / "reliability_generated"
            )
            logger.info(f"Generating reliability data in {output_dir}")

            id_threshold = kwargs.get("id_threshold")
            if id_threshold is None:
                id_threshold = generator_cfg.get("id_threshold", 0.8)
            synthetic_ood_threshold = kwargs.get("synthetic_ood_threshold")
            if synthetic_ood_threshold is None:
                synthetic_ood_threshold = generator_cfg.get(
                    "synthetic_ood_threshold", 0.8
                )
            synthetic_ood_multiplier = kwargs.get("synthetic_ood_multiplier")
            if synthetic_ood_multiplier is None:
                synthetic_ood_multiplier = generator_cfg.get(
                    "synthetic_ood_multiplier", 1.0
                )

            _rel_train_data = generate_reliability_data(
                classifier=models["jaeger_classifier"],
                raw_csv_path=raw_csv_path,
                output_dir=output_dir,
                string_processor_config=string_processor_config,
                model_cfg=builder.model_cfg,
                classifier_out_dim=builder.classifier_out_dim,
                reliability_out_dim=builder.reliability_out_dim,
                batch_size=builder.train_cfg.get("batch_size"),
                id_threshold=id_threshold,
                synthetic_ood_threshold=synthetic_ood_threshold,
                synthetic_ood_multiplier=synthetic_ood_multiplier,
                generator_cfg=generator_cfg,
                inference_batch_size=generator_cfg.get("inference_batch_size"),
            )
            data_format = "numpy"
            logger.info(
                "Reliability data generated; switching reliability loader to numpy"
            )

            if models.get("reliability_head") is not None:
                rel_train_paths = _rel_train_data.get("train", {}).get("paths", [])
                if rel_train_paths:
                    builder._set_reliability_bias(
                        models["reliability_head"], rel_train_paths[-1]
                    )

        # ============== reliability model ========================
        builder.compile_model(models, train_branch="reliability")

        if not kwargs.get("generate_reliability_data", False):
            _rel_train_data = builder._get_reliability_fragment_paths()
        rel_train_data = {"train": None, "validation": None}

        # ============== check convergence & train reliability ========
        checkpoint = builder._checkpoints
        _apply_ignore_convergence(
            checkpoint, kwargs.get("ignore_convergence", False), ["reliability"]
        )
        rel_converged = checkpoint and checkpoint.get("reliability", {}).get(
            "is_converged", False
        )
        if kwargs.get("only_save", False) is False:
            if rel_converged:
                logger.info(
                    "Reliability training already converged; skipping. "
                    "Use --ignore_convergence to retrain."
                )
            elif (
                not kwargs.get("only_classification_head", False)
                and models.get("jaeger_reliability") is not None
            ):
                # load reliability model data
                for k, v in _rel_train_data.items():
                    paths = check_files(v.get("paths"))
                    if not paths:
                        logger.warning(
                            "no valid files in paths=%r %r", k, v.get("paths")
                        )
                        exit(1)

                    if data_format == "csv":
                        _data = tf.data.TextLineDataset(
                            v.get("paths"),
                            num_parallel_reads=len(v.get("paths")),
                            buffer_size=200,
                        )
                        if string_processor_config.get("input_type") == "translated":
                            padded_shape = {
                                "translated": [6, None]
                                if string_processor_config.get("use_embedding_layer")
                                is True
                                else [
                                    6,
                                    string_processor_config.get("crop_size") // 3 - 1,
                                    string_processor_config.get("codon_depth"),
                                ]
                            }
                        elif string_processor_config.get("input_type") == "nucleotide":
                            padded_shape = {
                                "nucleotide": [
                                    2,
                                    string_processor_config.get("crop_size"),
                                    4,
                                ]
                            }
                        rel_train_data[k] = (
                            _data.map(
                                process_string_train(
                                    codons=string_processor_config.get("codon"),
                                    codon_num=string_processor_config.get("codon_id"),
                                    codon_depth=string_processor_config.get(
                                        "codon_depth"
                                    ),
                                    ngram_width=string_processor_config.get(
                                        "ngram_width"
                                    ),
                                    seq_onehot=string_processor_config.get(
                                        "seq_onehot"
                                    ),
                                    crop_size=string_processor_config.get("crop_size"),
                                    input_type=string_processor_config.get(
                                        "input_type"
                                    ),
                                    masking=string_processor_config.get("masking"),
                                    num_classes=builder.reliability_out_dim,
                                    class_label_onehot=False,
                                    shuffle_frames=string_processor_config.get(
                                        "shuffle_frames", False
                                    ),
                                ),
                                num_parallel_calls=tf.data.AUTOTUNE,
                            )
                            .shuffle(
                                buffer_size=_data.cardinality()
                                if _buffer_size == -1
                                else _buffer_size,
                            )
                            .padded_batch(
                                batch_size=builder.train_cfg.get("batch_size"),
                                padded_shapes=(
                                    padded_shape,
                                    [builder.reliability_out_dim],
                                ),
                                drop_remainder=multi_gpu,
                            )
                            .prefetch(tf.data.AUTOTUNE)
                        )
                    elif data_format == "numpy":
                        logger.info(f"Loading reliability {k} data from NumPy: {paths}")
                        if len(paths) > 1:
                            logger.warning(
                                "NumPy format only supports a single .npz file per split; using first: %s",
                                paths[0],
                            )
                        rel_train_data[k] = _build_numpy_split(
                            paths[0],
                            num_classes=builder.reliability_out_dim,
                            string_processor_config=string_processor_config,
                            batching_cfg=batching_cfg,
                            batch_size=builder.train_cfg.get("batch_size"),
                            multi_gpu=multi_gpu,
                            num_replicas=strategy.num_replicas_in_sync,
                            buffer_size=_buffer_size,
                            split=k,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported data_format: {data_format}. Use 'csv' or 'numpy'."
                        )

                rel_train = rel_train_data.get("train")
                rel_val = rel_train_data.get("validation")
                if rel_train is None or rel_val is None:
                    logger.warning("Skipping training — reliability data not available")
                else:
                    train_args = {
                        "validation_data": rel_val.take(
                            builder.train_cfg.get("reliability_validation_steps")
                        ),
                        "epochs": builder.train_cfg.get("reliability_epochs"),
                        "callbacks": builder.get_callbacks(branch="reliability"),
                    }
                    if checkpoint:
                        train_args["initial_epoch"] = checkpoint.get(
                            "reliability", {}
                        ).get("epoch", 0)

                    reliability_history = models.get("jaeger_reliability").fit(
                        rel_train.take(
                            builder.train_cfg.get("reliability_train_steps")
                        ),
                        class_weight=builder.train_cfg.get("reliability_class_weights"),
                        **train_args,
                    )
                    if (
                        reliability_history.epoch
                        and reliability_history.epoch[-1] < train_args["epochs"] - 1
                    ):
                        reliability_dir = builder.train_cfg.get("reliability_dir")
                        if reliability_dir is not None:
                            _write_convergence_marker(
                                reliability_dir,
                                branch="reliability",
                                epoch=int(reliability_history.epoch[-1]),
                            )
                        else:
                            logger.warning(
                                "reliability_dir is not configured; skipping convergence marker"
                            )
            else:
                logger.info("Skipping training — reliability model")

        # ============= test final model =========================
        logger.info("testing the final model")
        models.get("jaeger_model").trainable = False
        # Classifier validation data is unloaded earlier; prefer reliability
        # validation data when available.
        test_val_data = rel_train_data.get("validation") or train_data.get("validation")
        if test_val_data is not None:
            models.get("jaeger_model").predict(test_val_data.take(100))
        else:
            logger.info("no validation data available for final test, skipping")
        logger.info("training completed!")

        # ============= saving ===================================
        if kwargs.get("save_model", False):
            builder.save_model(
                model=models.get("jaeger_model"),
                num_params=model_num_params,
                suffix="fragment",
                metadata=kwargs.get("meta", None),
            )
            builder.save_embedding_model(
                models=models,
                num_params=model_num_params,
                suffix="fragment",
                metadata=kwargs.get("meta", None),
            )


if __name__ == "__main__":
    train_fragment()
