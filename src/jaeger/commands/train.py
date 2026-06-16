"""Training CLI entry point for Jaeger.

All core model-building logic lives in `jaeger.nnlib.builder`.
This module is a thin Click wrapper that orchestrates:
- strategy selection (GPU/CPU, mixed precision)
- data pipeline construction
- model training loops
- saving
"""

from __future__ import annotations

import os

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

try:
    from icecream import ic

    ic.configureOutput(prefix="Jaeger |")
except ImportError:

    def ic(*args, **kwargs):
        pass


logger = get_logger(log_file=None, log_path=None, level=3)


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
        meta=meta,
    )


# ------------------------------------------------------------------
# Core training orchestration (CLI glue)
# ------------------------------------------------------------------


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
        config["use_xla"] = kwargs.get("xla", False)
        if config["use_xla"]:
            logger.info("Using XLA JIT compilation for training")

        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()
        models.get("rep_model").summary()
        model_num_params = numerize(models.get("rep_model").count_params(), decimal=1)

        # ================= train classifier ======================
        builder.compile_model(models, train_branch="classifier")

        string_processor_config = builder._get_string_processor_config()
        _train_data = builder._get_fragment_paths()
        train_data = {"train": None, "validation": None}

        data_format = string_processor_config.get("data_format", "csv")
        _buffer_size = string_processor_config.get("buffer_size")

        for k, v in _train_data.items():
            paths = check_files(v.get("paths"))
            if not paths:
                logger.warning("no valid files in paths=%r %r", k, v.get("paths"))
                exit(1)

            if data_format == "csv":
                _data = tf.data.TextLineDataset(
                    paths, num_parallel_reads=len(paths), buffer_size=200
                )
                if string_processor_config.get("input_type") == "translated":
                    padded_shape = {
                        "translated": [6, None]
                        if string_processor_config.get("use_embedding_layer") is True
                        else [
                            6,
                            None,
                            string_processor_config.get("codon_depth"),
                        ]
                    }
                elif string_processor_config.get("input_type") == "nucleotide":
                    padded_shape = {
                        "nucleotide": [2, string_processor_config.get("crop_size"), 4]
                    }
                train_data[k] = (
                    _data.map(
                        process_string_train(
                            codons=string_processor_config.get("codon"),
                            codon_num=string_processor_config.get("codon_id"),
                            codon_depth=string_processor_config.get("codon_depth"),
                            label_original=string_processor_config.get(
                                "classifier_labels", None
                            ),
                            label_alternative=string_processor_config.get(
                                "classifier_labels_map", None
                            ),
                            ngram_width=string_processor_config.get("ngram_width"),
                            seq_onehot=string_processor_config.get("seq_onehot"),
                            crop_size=string_processor_config.get("crop_size"),
                            input_type=string_processor_config.get("input_type"),
                            masking=string_processor_config.get("masking"),
                            mutate=string_processor_config.get("mutate"),
                            mutation_rate=string_processor_config.get("mutation_rate"),
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
                _data = _load_numpy_dataset(
                    paths[0],
                    input_type=string_processor_config.get("input_type"),
                    seq_onehot=string_processor_config.get("seq_onehot"),
                    codon_depth=string_processor_config.get("codon_depth"),
                    nucleotide_onehot_map=string_processor_config.get(
                        "nucleotide_onehot_map"
                    ),
                    num_classes=builder.classifier_out_dim,
                    one_hot_labels=True,
                )
                padded_shapes = (
                    tf.nest.map_structure(
                        lambda spec: spec.shape, _data.element_spec[0]
                    ),
                    _data.element_spec[1].shape,
                )
                train_data[k] = (
                    _data.cache()
                    .shuffle(
                        buffer_size=_buffer_size if _buffer_size != -1 else 100000,
                    )
                    .padded_batch(
                        batch_size=builder.train_cfg.get("batch_size"),
                        padded_shapes=padded_shapes,
                        drop_remainder=multi_gpu,
                    )
                    .prefetch(tf.data.AUTOTUNE)
                )
            else:
                raise ValueError(
                    f"Unsupported data_format: {data_format}. Use 'csv' or 'numpy'."
                )

        # ============ check convergence & train classifier ===========
        checkpoint = builder._checkpoints
        cls_converged = checkpoint and checkpoint.get("classifier", {}).get(
            "is_converged", False
        )
        if kwargs.get("only_save", False) is False:
            if not cls_converged and not kwargs.get("only_reliability_head", False):
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

                if kwargs.get("only_classification_head", False) or kwargs.get(
                    "only_heads", False
                ):
                    models.get("rep_model").trainable = False

                # self-supervised pre-training
                if kwargs.get("self_supervised_pretraining", False):
                    builder.compile_model(models, train_branch="pretrain")
                    models.get("jaeger_projection").summary()
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
                models.get("jaeger_classifier").fit(
                    train_data.get("train").take(
                        builder.train_cfg.get("classifier_train_steps"),
                    ),
                    class_weight=builder.train_cfg.get("classifier_class_weights"),
                    **train_args,
                )
            else:
                logger.info("Skipping training — classification model")

        # ============== reliability model ========================
        builder.compile_model(models, train_branch="reliability")

        _rel_train_data = builder._get_reliability_fragment_paths()
        rel_train_data = {"train": None, "validation": None}
        for k, v in _rel_train_data.items():
            paths = check_files(v.get("paths"))
            if not paths:
                logger.warning("no valid files in paths=%r", k, v.get("paths"))
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
                        if string_processor_config.get("use_embedding_layer") is True
                        else [
                            6,
                            string_processor_config.get("crop_size") // 3 - 1,
                            string_processor_config.get("codon_depth"),
                        ]
                    }
                elif string_processor_config.get("input_type") == "nucleotide":
                    padded_shape = {
                        "nucleotide": [2, string_processor_config.get("crop_size"), 4]
                    }
                rel_train_data[k] = (
                    _data.map(
                        process_string_train(
                            codons=string_processor_config.get("codon"),
                            codon_num=string_processor_config.get("codon_id"),
                            codon_depth=string_processor_config.get("codon_depth"),
                            ngram_width=string_processor_config.get("ngram_width"),
                            seq_onehot=string_processor_config.get("seq_onehot"),
                            crop_size=string_processor_config.get("crop_size"),
                            input_type=string_processor_config.get("input_type"),
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
                _data = _load_numpy_dataset(
                    paths[0],
                    input_type=string_processor_config.get("input_type"),
                    seq_onehot=string_processor_config.get("seq_onehot"),
                    codon_depth=string_processor_config.get("codon_depth"),
                    nucleotide_onehot_map=string_processor_config.get(
                        "nucleotide_onehot_map"
                    ),
                    num_classes=builder.reliability_out_dim,
                    one_hot_labels=True,
                )
                padded_shapes = (
                    tf.nest.map_structure(
                        lambda spec: spec.shape, _data.element_spec[0]
                    ),
                    _data.element_spec[1].shape,
                )
                rel_train_data[k] = (
                    _data.cache()
                    .shuffle(
                        buffer_size=_buffer_size if _buffer_size != -1 else 100000,
                    )
                    .padded_batch(
                        batch_size=builder.train_cfg.get("batch_size"),
                        padded_shapes=padded_shapes,
                        drop_remainder=multi_gpu,
                    )
                    .prefetch(tf.data.AUTOTUNE)
                )
            else:
                raise ValueError(
                    f"Unsupported data_format: {data_format}. Use 'csv' or 'numpy'."
                )

        # ============== check convergence & train reliability ========
        checkpoint = builder._checkpoints
        rel_converged = checkpoint and checkpoint.get("reliability", {}).get(
            "is_converged", False
        )
        if kwargs.get("only_save", False) is False:
            if (
                not rel_converged
                and not kwargs.get("only_classification_head", False)
                and models.get("jaeger_reliability") is not None
            ):
                train_args = {
                    "validation_data": rel_train_data.get("validation").take(
                        builder.train_cfg.get("reliability_validation_steps")
                    ),
                    "epochs": builder.train_cfg.get("reliability_epochs"),
                    "callbacks": builder.get_callbacks(branch="reliability"),
                }
                if checkpoint:
                    train_args["initial_epoch"] = checkpoint.get("reliability", {}).get(
                        "epoch", 0
                    )

                models.get("jaeger_reliability").fit(
                    rel_train_data.get("train").take(
                        builder.train_cfg.get("reliability_train_steps")
                    ),
                    class_weight=builder.train_cfg.get("reliability_class_weights"),
                    **train_args,
                )
            else:
                logger.info("Skipping training — reliability model")

        # ============= test final model =========================
        logger.info("testing the final model")
        models.get("jaeger_model").trainable = False
        models.get("jaeger_model").predict(train_data.get("validation").take(100))
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
