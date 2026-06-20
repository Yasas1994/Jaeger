"""Model builders for Jaeger neural networks.

This module contains the `DynamicModelBuilder` class which constructs
Keras models from YAML/JSON configuration dictionaries. It is used by
the training pipeline and can be imported independently of the CLI layer.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import tensorflow as tf
import yaml

from jaeger.seqops.maps import (
    AA_ID,
    CODONS,
    CODON_ID,
    DICODONS,
    DICODON_ID,
    MURPHY10_ID,
    PC5_ID,
)
from jaeger.nnlib.metrics import (
    PrecisionForClass,
    RecallForClass,
    SpecificityForClass,
)
from jaeger.nnlib.v2.layers import (
    AxialAttention,
    CrossFrameAttention,
    GatedFrameGlobalMaxPooling,
    LocalAttention,
    MaskedBatchNorm,
    MaskedDYT,
    MaskedGlobalAvgPooling,
    MaskedConv1D,
    MaskedLayerNormalization,
    MetricModel,
    MultiScaleConv1D,
    OODSignalLayer,
    ResidualBlock_wrapper,
    TransformerEncoder,
)
from jaeger.nnlib.v2.losses import ArcFaceLoss, HierarchicalLoss
from jaeger.nnlib.v2.nmd import NMDLayer, NMDMerge
from jaeger.utils.logging import get_logger
from jaeger.utils.misc import clear_directory

logger = get_logger(log_file=None, log_path=None, level=3)


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_files(files: list[str]) -> list[str]:
    """Return subset of *files* that exist on disk."""
    return [f for f in files if Path(f).is_file()]


GRAPH_RE = re.compile(
    r"^jaeger_(?P<id>[0-9a-fA-F]+)_(?P<size>\d+(?:\.\d+)?[A-Z])_fragment_graph$"
)


def find_existing_graph_id(path: Path) -> Optional[str]:
    """Search *direct children* of *path* for jaeger_<id>_1.2M_fragment_graph.

    Returns the extracted <id> or ``None`` if not found.
    """
    if not path.exists() or not path.is_dir():
        return None
    for p in path.iterdir():
        m = GRAPH_RE.match(p.name)
        if m:
            return m.group("id")
    return None


class DynamicModelBuilder:
    """Builds Keras models from a configuration dictionary.

    The builder supports:
    - Embedding layers (translated codons or nucleotide one-hot)
    - Representation learners (residual blocks, transformers, attention)
    - Projection heads (for self-supervised pre-training with ArcFace loss)
    - Classification heads
    - Reliability heads (binary/multi-class confidence estimation)
    - Combined ``jaeger_model`` that exposes all outputs

    Parameters
    ----------
    config:
        Full training configuration dict. Must contain ``model`` and
        ``training`` top-level keys.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.uid: str = uuid4().hex[:8]
        self.cfg: dict[str, Any] = config
        self.model_cfg: dict[str, Any] = config.get("model", {}) or {}
        self.train_cfg: dict[str, Any] = config.get("training", {}) or {}

        self.inputs: Optional[tf.Tensor] = None
        self.outputs: list[tf.Tensor] = []

        embedding_cfg = self.model_cfg.get("embedding", {}) or {}
        self.input_shape = embedding_cfg.get("input_shape")

        self._fragment_paths = self._get_fragment_paths()
        self._contig_paths = self._get_contig_paths()
        self._reliability_fragment_paths = self._get_reliability_fragment_paths()
        self._reliability_contig_paths = self._get_reliability_contig_paths()

        self._saving_config = self._get_model_saving_configuration()
        self._from_last_checkpoint: bool = bool(
            config.get("from_last_checkpoint", False)
        )
        self._force: bool = bool(config.get("force", False))
        self.use_xla: bool = bool(config.get("use_xla", False))
        self.generate_reliability_data: bool = bool(
            config.get("generate_reliability_data", False)
        )
        self._checkpoints: dict[str, Path] = {}

        self.classifier_out_dim: int = int(self.model_cfg.get("classifier_out_dim", 0))
        self.reliability_out_dim: int = int(
            self.model_cfg.get("reliability_out_dim", 0)
        )

        self.loss_classifier_name = self.train_cfg.get(
            "loss_classifier", "categorical_crossentropy"
        ).lower()
        self.loss_reliability_name = self.train_cfg.get(
            "loss_reliability", "binary_crossentropy"
        ).lower()
        self.loss_classifier = None
        self.loss_reliability = None
        self.metrics_classifier: list[Any] = []
        self.metrics_reliability: list[Any] = []

        self._regularizer = {
            "l2": tf.keras.regularizers.L2,
            "l1": tf.keras.regularizers.L1,
        }

        self._layers = {
            "masked_conv1d": MaskedConv1D,
            "multi_scale_conv": MultiScaleConv1D,
            "conv1d": tf.keras.layers.Conv1D,
            "masked_batchnorm": MaskedBatchNorm,
            "masked_dyt": MaskedDYT,
            "masked_layernorm": MaskedLayerNormalization,
            "layernorm": tf.keras.layers.LayerNormalization,
            "batchnorm": tf.keras.layers.BatchNormalization,
            "transformer_encoder": TransformerEncoder,
            "cross_frame_attention": CrossFrameAttention,
            "axial_attention": AxialAttention,
            "local_attention": LocalAttention,
            "residual_block": ResidualBlock_wrapper,
            "nmd": NMDLayer,
            "ood_signal_layer": OODSignalLayer,
            "dense": tf.keras.layers.Dense,
            "activation": tf.keras.layers.Activation,
            "relu": tf.keras.layers.Activation,
            "gelu": tf.keras.layers.Activation,
            "sigmoid": tf.keras.layers.Activation,
            "softmax": tf.keras.layers.Activation,
            "tanh": tf.keras.layers.Activation,
            "dropout": tf.keras.layers.Dropout,
            "crop": tf.keras.layers.Cropping2D,
        }

        self._load_training_params()
        self._prepare_checkpoint_dirs()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _prepare_checkpoint_dirs(self, config: dict | None = None) -> None:
        if not config:
            config = self.cfg
        callbacks_cfg = config.get("training", {}).get("callbacks", {})
        directories = callbacks_cfg.get("directories", []) or []

        should_exit = False
        for dir_str in directories:
            path = Path(dir_str)
            if path.exists():
                if self._from_last_checkpoint:
                    self._checkpoints[path.name] = self.get_latest_h5_with_metadata(
                        path
                    )
                elif self._force:
                    shutil.rmtree(path)
                else:
                    logger.warning(
                        "Checkpoint(s) exist at %s. "
                        "Use --force to delete the existing checkpoints and continue!",
                        path,
                    )
                    logger.info(
                        "Or set --from_last_checkpoint to continue training "
                        "from the last checkpoint."
                    )
                    should_exit = True
                    continue
            path.mkdir(parents=True, exist_ok=True)
        if should_exit:
            exit(1)

    def get_latest_h5_with_metadata(
        self,
        path: str | Path,
        check_convergence: str = "classifier",
        pattern: str = r"epoch:(\d+)-loss:(\d+\.\d+)",
    ) -> dict:
        """Scan *path* for the most recent ``.h5`` checkpoint matching *pattern*."""
        path = Path(path)
        h5_files = sorted(
            path.glob("*.h5"), key=lambda f: f.stat().st_mtime, reverse=True
        )
        for file in h5_files:
            match = re.search(pattern, file.name)
            if match:
                epoch, loss = match.groups()
                return {
                    "path": file,
                    "epoch": int(epoch),
                    "loss": float(loss),
                    "is_converged": False if path.name == check_convergence else False,
                }
        return {"path": None, "epoch": 0, "loss": None, "is_converged": False}

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_fragment_classifier(self) -> dict[str, tf.keras.Model]:
        """Build the full fragment-level model graph.

        Returns a dict with keys such as ``rep_model``, ``classification_head``,
        ``reliability_head``, ``jaeger_classifier``, ``jaeger_reliability``,
        ``jaeger_model``, etc.
        """
        models: dict[str, tf.keras.Model] = {}

        # === 1. EMBEDDING ===
        if "embedding" in self.model_cfg:
            inputs, x = self._build_embedding(self._get_string_processor_config())
            self.inputs = inputs
        else:
            raise ValueError("Missing 'embedding' section in config")

        rep_cfg = self.model_cfg.get("representation_learner", {})
        class_cfg = self.model_cfg.get("classifier", {})
        rep_is_branched = "branch" in rep_cfg
        class_is_branched = "branch" in class_cfg

        # === 2. REPRESENTATION LEARNER ===
        if "representation_learner" in self.model_cfg:
            if rep_is_branched:
                rep_out = self._build_branched_block(
                    x,
                    rep_cfg["branch"],
                    prefix="rep",
                    merge_method=None,
                )
            else:
                merge_cfg = self.model_cfg.get("reliability_model", {}).get("merge")
                rep_out = self._build_block(
                    x,
                    self.model_cfg["representation_learner"],
                    prefix="rep",
                    nmd_merge=merge_cfg,
                )
            models["rep_model"] = tf.keras.Model(
                inputs=self.inputs, outputs=rep_out, name="rep_model"
            )
            models["rep_model"].summary()

        # === 3. PRETRAINING (projection head) ===
        if "projection" in self.model_cfg:
            input_shape = (self.model_cfg["projection"].get("input_shape"),)
            inputs = tf.keras.Input(shape=input_shape, name="projection_input")
            x_projection = self._build_block(
                inputs, self.model_cfg["projection"], prefix="projection"
            )
            models["projection_head"] = tf.keras.Model(
                inputs=inputs, outputs=x_projection, name="projection_head"
            )
            projection_dim = self.model_cfg["projection"]["hidden_layers"][-2][
                "config"
            ].get("units")

            rep_output = models["rep_model"].output
            if isinstance(rep_output, (list, tuple)):
                x = rep_output[0]
            else:
                x = rep_output
            x = models["projection_head"](x)
            models["jaeger_projection"] = MetricModel(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_projection"
            )
            labels = tf.keras.Input(shape=(self.classifier_out_dim,), name="labels")
            embeddings = tf.keras.Input(shape=(projection_dim,), name="embedding")
            loss = ArcFaceLoss(
                num_classes=self.classifier_out_dim,
                embedding_dim=projection_dim,
                margin=self.model_cfg["projection"]["margin"],
                scale=self.model_cfg["projection"]["scale"],
                onehot=True,
            )(labels, embeddings)
            models["arcface_loss"] = tf.keras.Model(
                inputs=[labels, embeddings], outputs=loss, name="Arcface"
            )
            if self._checkpoints.get("projection", {}).get("path", False):
                models["jaeger_projection"].load_weights(
                    self._checkpoints.get("projection").get("path"),
                    skip_mismatch=True,
                )
                logger.info(
                    f"Loaded projection model weights from "
                    f"{self._checkpoints.get('projection').get('path')}"
                )

        # === 3. CLASSIFIER ===
        if "classifier" in self.model_cfg:
            if class_is_branched:
                branch_hidden = list(class_cfg["branch"].get("hidden_layers", []))
                if not branch_hidden or branch_hidden[-1].get("name") != "merge":
                    raise ValueError(
                        "Branched classifier must end with a 'merge' layer"
                    )
                merge_cfg = branch_hidden[-1].get("config", {})
                merge_method = merge_cfg.get("method", "average")
                head_cfg = {"hidden_layers": branch_hidden[:-1]}

                rep_out = models["rep_model"].output
                num_branches = len(rep_out) if isinstance(rep_out, list) else 1
                first_branch = rep_out[0] if isinstance(rep_out, list) else rep_out
                input_shape = int(first_branch.shape[-1])
                branch_inputs = [
                    tf.keras.Input(shape=(input_shape,), name=f"classifier_input_{i}")
                    for i in range(num_branches)
                ]
                x_classifier = self._build_branched_block(
                    branch_inputs,
                    head_cfg,
                    prefix="classifier",
                    merge_method=merge_method,
                )
                inputs = branch_inputs
            else:
                input_shape = (self.model_cfg["classifier"].get("input_shape"),)
                inputs = tf.keras.Input(shape=input_shape, name="classifier_input")
                x_classifier = self._build_block(
                    inputs, self.model_cfg["classifier"], prefix="classifier"
                )

            models["classification_head"] = tf.keras.Model(
                inputs=inputs, outputs=x_classifier, name="classification_head"
            )

            rep_out = models["rep_model"].output
            if isinstance(rep_out, list) and class_is_branched:
                x = models["classification_head"](rep_out)
            elif isinstance(rep_out, list):
                x_rep = tf.keras.layers.Concatenate(axis=-1)(rep_out)
                x = models["classification_head"](x_rep)
            elif isinstance(rep_out, tuple):
                x_rep = rep_out[0]
                x = models["classification_head"](x_rep)
            else:
                x = models["classification_head"](rep_out)

            models["jaeger_classifier"] = tf.keras.Model(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_classifier"
            )
            if self._checkpoints.get("classifier", {}).get("path", False):
                models["jaeger_classifier"].load_weights(
                    self._checkpoints.get("classifier").get("path"),
                    skip_mismatch=True,
                )
                logger.info(
                    f"Loaded classification model weights from "
                    f"{self._checkpoints.get('classifier').get('path')}"
                )

        # === 4. RELIABILITY ===
        if "reliability_model" in self.model_cfg:
            reliability_cfg = self.model_cfg["reliability_model"]
            mode = reliability_cfg.get("mode", "nmd")
            if mode not in ("nmd", "nmd_plus_signals"):
                raise ValueError(
                    f"Unsupported reliability_model.mode: {mode!r}. "
                    "Use 'nmd' or 'nmd_plus_signals'."
                )

            rep_out = models["rep_model"].output
            if not isinstance(rep_out, (list, tuple)) or len(rep_out) < 2:
                raise ValueError(
                    "reliability_model is configured but the representation learner "
                    "produced no NMD tensor. Add an `nmd` layer or set "
                    "return_nmd: true on a layer that supports it."
                )
            nmd = rep_out[1]
            nmd_dim = int(tf.keras.backend.int_shape(nmd)[-1])

            if mode == "nmd_plus_signals":
                default_signals = [
                    "max_prob",
                    "entropy",
                    "energy",
                    "margin",
                    "nmd_norm",
                ]
                signals = reliability_cfg.get("signals", default_signals)
                signal_layer = OODSignalLayer(signals=signals, name="ood_signals")
                reliability_input_dim = nmd_dim + len(signals)
            else:
                reliability_input_dim = nmd_dim

            expected_dim = reliability_cfg.get("input_shape")
            if expected_dim is None:
                reliability_cfg = dict(reliability_cfg)
                reliability_cfg["input_shape"] = reliability_input_dim
            elif expected_dim != reliability_input_dim:
                raise ValueError(
                    f"reliability_model.input_shape ({expected_dim}) does not match "
                    f"computed reliability input dimension ({reliability_input_dim}). "
                    f"Set input_shape to None or omit it when using mode={mode!r}."
                )

            # Decide whether we can compute the bias now or need to defer/skip.
            reliability_train_paths = (
                self._get_reliability_fragment_paths().get("train", {}).get("paths", [])
            )
            reliability_data_configured = bool(reliability_train_paths)
            reliability_data_exists = (
                reliability_data_configured
                and Path(reliability_train_paths[-1]).exists()
            )

            build_reliability_head = True
            if self.generate_reliability_data:
                logger.info(
                    "Deferring reliability head bias initialization until "
                    "after OOD data generation."
                )
                reliability_cfg.pop("bias_initializer", None)
                for layer in reliability_cfg.get("hidden_layers", []):
                    layer.get("config", {}).pop("bias_initializer", None)
            elif reliability_data_configured and not reliability_data_exists:
                logger.warning(
                    "Reliability training data is configured but missing and "
                    "--generate_reliability_data is not set. "
                    "The reliability model will not be constructed."
                )
                build_reliability_head = False
            elif not reliability_data_configured:
                logger.info(
                    "No reliability training data is configured; "
                    "building reliability head with default bias."
                )
                reliability_cfg.pop("bias_initializer", None)
                for layer in reliability_cfg.get("hidden_layers", []):
                    layer.get("config", {}).pop("bias_initializer", None)

            if build_reliability_head:
                inputs = tf.keras.Input(
                    shape=(reliability_input_dim,), name="reliability_input"
                )
                x_reliability = self._build_block(
                    inputs, reliability_cfg, prefix="reliability"
                )
                models["reliability_head"] = tf.keras.Model(
                    inputs=inputs, outputs=x_reliability, name="reliability_head"
                )

                def _reliability_from_rep(rep_out):
                    if mode == "nmd_plus_signals":
                        emb = rep_out[0]
                        nmd_ = rep_out[1]
                        logits = models["classification_head"](emb)
                        sig = signal_layer({"logits": logits, "nmd": nmd_})
                        return tf.keras.layers.Concatenate(axis=-1)([nmd_, sig])
                    return rep_out[1]

                x = models["reliability_head"](_reliability_from_rep(rep_out))
                models["jaeger_reliability"] = tf.keras.Model(
                    inputs=models["rep_model"].input,
                    outputs=x,
                    name="Jaeger_reliability",
                )
                try:
                    if self._checkpoints.get("reliability", {}).get("path", False):
                        models["jaeger_reliability"].load_weights(
                            self._checkpoints.get("reliability").get("path"),
                            skip_mismatch=True,
                        )
                        logger.info(
                            f"Loaded reliability model weights from "
                            f"{self._checkpoints.get('reliability').get('path')}"
                        )
                except Exception:
                    logger.warning(
                        "could not load the weights to reliability model from checkpoint. "
                        "trying to load weights partially"
                    )
                    models["jaeger_reliability"].load_weights(
                        self._checkpoints.get("reliability").get("path"),
                        skip_mismatch=True,
                    )
                    self._checkpoints["reliability"] = {
                        "path": None,
                        "epoch": 0,
                        "loss": None,
                        "is_converged": False,
                    }

        # === 5. COMBINED MODEL ===
        rep_out = models["rep_model"].output
        if rep_is_branched or class_is_branched:
            prediction = models["jaeger_classifier"].output

            if isinstance(rep_out, list):
                embedding = tf.keras.layers.Average(name="embedding_avg")(rep_out)
            elif isinstance(rep_out, tuple):
                embedding = rep_out[0]
            else:
                embedding = rep_out

            models["jaeger_model"] = tf.keras.Model(
                inputs=models["rep_model"].input,
                outputs={"prediction": prediction, "embedding": embedding},
                name="Jaeger_model",
            )
        else:
            has_reliability = "reliability_head" in models
            if isinstance(rep_out, (list, tuple)):
                if len(rep_out) == 2:
                    x1, x2 = rep_out
                    class_ = models["classification_head"](x1)
                    outputs: dict[str, Any] = {
                        "prediction": class_,
                        "embedding": x1,
                        "nmd": x2,
                    }
                    if has_reliability:
                        outputs["reliability"] = models["reliability_head"](
                            _reliability_from_rep(rep_out)
                        )
                    models["jaeger_model"] = tf.keras.Model(
                        inputs=models["rep_model"].input,
                        outputs=outputs,
                        name="Jaeger_model",
                    )
                elif len(rep_out) == 3:
                    x1, x2, g = rep_out
                    class_ = models["classification_head"](x1)
                    outputs = {
                        "prediction": class_,
                        "embedding": x1,
                        "nmd": x2,
                        "gate": g,
                    }
                    if has_reliability:
                        outputs["reliability"] = models["reliability_head"](
                            _reliability_from_rep(rep_out)
                        )
                    models["jaeger_model"] = tf.keras.Model(
                        inputs=models["rep_model"].input,
                        outputs=outputs,
                        name="Jaeger_model",
                    )
            else:
                class_ = models["classification_head"](rep_out)
                models["jaeger_model"] = tf.keras.Model(
                    inputs=models["rep_model"].input,
                    outputs={"prediction": class_, "embedding": rep_out},
                    name="Jaeger_model",
                )

        return models

    def build_contig_classifier(self) -> None:
        """Placeholder for future contig-level consensus model."""
        pass

    def _build_embedding(self, cfg: dict[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
        """Create the embedding layer from *cfg*."""
        input_shape = cfg.get("input_shape", (6, None, 64))
        embedding_size = cfg.get("embedding_size", 4)

        inputs = tf.keras.Input(shape=input_shape, name=cfg.get("input_type"))
        masked_inputs = tf.keras.layers.Masking(name="input_mask", mask_value=0.0)(
            inputs
        )

        match cfg.get("input_type"):
            case "translated":
                if embedding_size > 0:
                    if cfg.get("use_embedding_layer", False):
                        x = tf.keras.layers.Embedding(
                            input_dim=cfg.get("vocab_size"),
                            output_dim=embedding_size,
                            mask_zero=True,
                            embeddings_initializer=tf.keras.initializers.Orthogonal(),
                            embeddings_regularizer=self._regularizer.get(
                                cfg.get("embedding_regularizer"),
                            )(cfg.get("embedding_regularizer_w")),
                        )(inputs)
                    else:
                        x = tf.keras.layers.Dense(
                            embedding_size,
                            name=f"{cfg.get('input_type')}_embedding",
                            use_bias=False,
                            kernel_initializer=tf.keras.initializers.Orthogonal(),
                            kernel_regularizer=self._regularizer.get(
                                cfg.get("embedding_regularizer"),
                            )(cfg.get("embedding_regularizer_w")),
                        )(masked_inputs)
                else:
                    x = masked_inputs
            case "nucleotide":
                x = masked_inputs
            case _:
                raise ValueError(f"{cfg.get('input_type')} is invalid")

        if cfg.get("use_positional_embeddings", False):
            from jaeger.nnlib.v2.layers import SinusoidalPositionEmbedding

            positional = SinusoidalPositionEmbedding(
                max_wavelength=cfg.get("positional_embedding_length")
            )(x)
            x = tf.keras.layers.Add()([x, positional])

        return inputs, x

    def _get_bias(self, data_path: str, kind: str, label_map: list) -> np.ndarray:
        """Compute class-frequency bias for the final layer."""

        def _sigmoid(f: dict):
            n, p = f.values()
            t = p / (p + n)
            return np.log(t / (1 - t))

        def _softmax(f: dict):
            f = np.array(list(f.values()))
            return np.log(f / np.sum(f))

        def _correct_label_map(f: dict, label_map: list):
            _tmp = {i: 0 for i in range(max(label_map) + 1)}
            for k, v in f.items():
                _tmp[label_map[k]] += v
            return _tmp

        def _load_counts(path: str) -> dict[int, int]:
            if path.endswith(".npz"):
                data = np.load(path, allow_pickle=True)
                if "labels" in data:
                    labels = data["labels"]
                elif "label" in data:
                    labels = data["label"]
                else:
                    label_keys = [k for k in data.files if k.startswith("labels_")]
                    if not label_keys:
                        raise ValueError(
                            f"NPZ file {path!r} contains no 'labels', 'label', "
                            f"or sharded 'labels_*' arrays."
                        )
                    labels = np.concatenate([data[k] for k in sorted(label_keys)])
                labels = np.asarray(labels)
                if labels.ndim > 1:
                    labels = np.argmax(labels, axis=-1)
                labels = labels.ravel()
                unique, counts = np.unique(labels, return_counts=True)
                return {int(k): int(v) for k, v in zip(unique, counts)}
            else:
                import polars as pl

                df = pl.read_csv(path, columns=[0], has_header=False)
                counts = df["column_1"].value_counts().to_dict(as_series=False)
                counts = dict(zip(counts["column_1"], counts["count"]))
                return {k: counts[k] for k in sorted(list(counts.keys()))}

        counts_dict = _load_counts(data_path)
        if len(label_map) > 0:
            counts_dict = _correct_label_map(counts_dict, label_map)
        match kind:
            case "softmax":
                return _softmax(counts_dict)
            case "sigmoid":
                return _sigmoid(counts_dict)

    def _set_reliability_bias(self, model: tf.keras.Model, data_path: str) -> None:
        """Recompute and assign class-frequency bias to the reliability head.

        This is used when the reliability head was built before the training
        data existed (e.g. ``--generate_reliability_data``).
        """
        reliability_label_map = self.model_cfg.get("string_processor", {}).get(
            "reliability_labels_map", []
        )
        bias = self._get_bias(
            data_path,
            kind="sigmoid" if "binary" in self.loss_reliability_name else "softmax",
            label_map=reliability_label_map,
        )

        dense_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                dense_layer = layer
                break
        if dense_layer is None:
            raise ValueError(
                "Cannot set reliability bias: no Dense layer found in reliability head"
            )
        bias_arr = np.broadcast_to(
            np.asarray(bias, dtype=np.float32), (dense_layer.units,)
        )
        dense_layer.bias.assign(bias_arr)
        logger.info(f"Updated reliability head bias from generated data: {data_path}")

    def _build_block(
        self,
        x,
        cfg: dict[str, Any],
        prefix: str,
        nmd_merge: dict[str, Any] | None = None,
    ):
        """Build a stack of layers from *cfg* and return the output tensor(s).

        Parameters
        ----------
        x:
            Input tensor.
        cfg:
            Layer configuration dictionary.
        prefix:
            Name prefix for layers.
        nmd_merge:
            Optional configuration for merging collected NMD tensors before
            returning from the representation learner.
        """
        nmd = []
        previous_channels = None
        for i, layer_cfg in enumerate(cfg.get("hidden_layers", [])):
            layer_name = layer_cfg.get("name", "").lower()
            cfg_layer = dict(layer_cfg.get("config", {}))
            cfg_layer["name"] = f"{prefix}_{layer_name}_{i}"

            layer_class = self._layers.get(layer_name)
            if layer_class is None:
                raise ValueError(f"Unknown layer type: {layer_name}")

            if layer_name in {"relu", "gelu", "sigmoid", "softmax", "tanh"}:
                cfg_layer["activation"] = layer_name

            if "kernel_regularizer" in cfg_layer:
                reg_name = cfg_layer.pop("kernel_regularizer")
                reg_w = cfg_layer.pop("kernel_regularizer_w", None)
                cfg_layer["kernel_regularizer"] = self._regularizer[reg_name](reg_w)

            if "kernel_initializer" in cfg_layer:
                init_name = cfg_layer.pop("kernel_initializer")
                cfg_layer["kernel_initializer"] = tf.keras.initializers.get(init_name)

            if "bias_initializer" in cfg_layer:
                if "calculate_from" in cfg_layer.get("bias_initializer"):
                    classifier_label_map = self.model_cfg.get(
                        "string_processor", {}
                    ).get("classifier_labels_map", [])
                    reliability_label_map = self.model_cfg.get(
                        "string_processor", {}
                    ).get("reliability_labels_map", [])
                    if "relia" in prefix:
                        path = (
                            self._get_reliability_fragment_paths()
                            .get("train", {})
                            .get("paths", [])[-1]
                        )
                        cfg_layer["bias_initializer"] = tf.keras.initializers.Constant(
                            self._get_bias(
                                path,
                                kind="sigmoid"
                                if "binary" in self.loss_reliability_name
                                else "softmax",
                                label_map=reliability_label_map,
                            )
                        )
                    elif "classi" in prefix:
                        path = (
                            self._get_fragment_paths()
                            .get("train", {})
                            .get("paths", [])[-1]
                        )
                        cfg_layer["bias_initializer"] = tf.keras.initializers.Constant(
                            self._get_bias(
                                path,
                                kind="sigmoid"
                                if "binary" in self.loss_classifier_name
                                else "softmax",
                                label_map=classifier_label_map,
                            )
                        )

            if "block_size" in cfg_layer:
                block_size = cfg_layer.pop("block_size")
                shape = (self.input_shape[0], None, previous_channels)
                if block_size > 0:
                    if cfg_layer.get("return_nmd", False):
                        x, nmd_ = layer_class(block_size, shape, **cfg_layer)(x)
                        nmd.append(nmd_)
                    else:
                        x = layer_class(block_size, shape, **cfg_layer)(x)
                    if "filters" in cfg_layer:
                        previous_channels = cfg_layer.get("filters")
                    elif "units" in cfg_layer:
                        previous_channels = cfg_layer.get("units")
                continue

            if "return_nmd" in cfg_layer:
                if cfg_layer.get("return_nmd"):
                    x, nmd_ = layer_class(**cfg_layer)(x)
                    nmd.append(nmd_)
                else:
                    x = layer_class(**cfg_layer)(x)
                if "filters" in cfg_layer:
                    previous_channels = cfg_layer.get("filters")
                elif "units" in cfg_layer:
                    previous_channels = cfg_layer.get("units")
                continue

            # Standalone NMD layer: produce a side-output vector while leaving
            # the main feature map unchanged for downstream layers / pooling.
            if layer_name == "nmd":
                nmd_ = layer_class(**cfg_layer)(x)
                nmd.append(nmd_)
                continue

            x = layer_class(**cfg_layer)(x)

            if "filters" in cfg_layer:
                previous_channels = cfg_layer.get("filters")
            elif "units" in cfg_layer:
                previous_channels = cfg_layer.get("units")
            elif "embed_dim" in cfg_layer:
                previous_channels = cfg_layer.get("embed_dim")

        # Aggregation
        if "pooling" in cfg:
            pooling = cfg.get("pooling", "average").lower()
            pooler = self._get_pooler(pooling)
            has_nmd = len(nmd) > 0
            if has_nmd:
                if len(nmd) == 1:
                    nmd = nmd[0]
                elif nmd_merge is not None:
                    merge_kwargs = dict(nmd_merge)
                    merge_kwargs.pop("name", None)
                    nmd = NMDMerge(name=f"{prefix}_nmd_merge", **merge_kwargs)(nmd)
                else:
                    nmd = tf.keras.layers.Concatenate(
                        axis=-1, name=f"{prefix}_nmd_concat"
                    )(nmd)
            if "gated" not in pooling:
                x = pooler(name=f"global_{pooling}pool")(x)
                return (x, nmd) if has_nmd else x
            else:
                x, g = pooler(return_gate=True, name=f"global_{pooling}pool")(x)
                return (x, nmd, g) if has_nmd else (x, g)
        return x

    def _build_branched_block(
        self,
        x,
        cfg: dict[str, Any],
        prefix: str,
        merge_method: str | None = None,
    ):
        """Build a shared-weight branch stack, split input on axis 1, and merge.

        Parameters
        ----------
        x:
            Input tensor with a branch dimension in axis 1 (e.g.
            ``(batch, 2, length, channels)``), or a list of branch tensors.
        cfg:
            Layer configuration for one branch (may contain ``hidden_layers``
            and ``pooling``).
        prefix:
            Name prefix for layers.
        merge_method:
            How to merge branch outputs: ``average``, ``sum``, ``concat``, ``max``.
            If None, return a list of branch outputs.

        Returns
        -------
        Merged tensor, or list of branch tensors if ``merge_method`` is None.
        """
        if isinstance(x, list):
            branches = x
        else:
            num_branches = int(x.shape[1])

            def _split_and_squeeze(t):
                return [
                    tf.squeeze(b, axis=1)
                    for b in tf.split(t, num_or_size_splits=num_branches, axis=1)
                ]

            branches = tf.keras.layers.Lambda(
                _split_and_squeeze,
                name=f"{prefix}_split_branches",
            )(x)
        branch_shape = branches[0].shape[1:]

        branch_input = tf.keras.Input(shape=branch_shape, name=f"{prefix}_branch_input")
        branch_output = self._build_block(branch_input, cfg, prefix=f"{prefix}_branch")
        branch_model = tf.keras.Model(
            branch_input, branch_output, name=f"{prefix}_branch"
        )

        branch_outputs = []
        for b in branches:
            bo = branch_model(b)
            if isinstance(bo, (list, tuple)):
                bo = bo[0]
            branch_outputs.append(bo)

        if merge_method is None:
            return branch_outputs

        merge_method = merge_method.lower()
        if merge_method == "average":
            return tf.keras.layers.Average(name=f"{prefix}_merge_avg")(branch_outputs)
        if merge_method == "sum":
            return tf.keras.layers.Add(name=f"{prefix}_merge_sum")(branch_outputs)
        if merge_method == "max":
            return tf.keras.layers.Maximum(name=f"{prefix}_merge_max")(branch_outputs)
        if merge_method == "concat":
            return tf.keras.layers.Concatenate(axis=-1, name=f"{prefix}_merge_concat")(
                branch_outputs
            )
        raise ValueError(f"Unknown merge method: {merge_method}")

    # ------------------------------------------------------------------
    # Metrics / loss / optimizer factories
    # ------------------------------------------------------------------

    def _get_metrics(self, config: list[dict]) -> list[Any]:
        _metrics = {
            "categorical_accuracy": tf.keras.metrics.CategoricalAccuracy,
            "binary_accuracy": tf.keras.metrics.BinaryAccuracy,
            "sparse_categorical_accuracy": tf.keras.metrics.SparseCategoricalAccuracy,
            "categorical_crossentropy": tf.keras.metrics.CategoricalCrossentropy,
            "binary_crossentropy": tf.keras.metrics.BinaryCrossentropy,
            "sparse_categorical_crossentropy": tf.keras.metrics.SparseCategoricalCrossentropy,
            "auc": tf.keras.metrics.AUC,
            "precision": tf.keras.metrics.Precision,
            "recall": tf.keras.metrics.Recall,
            "per_class_precision": PrecisionForClass,
            "per_class_recall": RecallForClass,
            "per_class_specificity": SpecificityForClass,
        }
        metrics = []
        for c in config:
            if "per_class_" not in c.get("name"):
                if c.get("params") is not None:
                    metrics.append(_metrics.get(c.get("name"))(**c.get("params")))
                else:
                    metrics.append(_metrics.get(c.get("name"))())
            else:
                for cls in range(self.classifier_out_dim):
                    metrics.append(_metrics.get(c.get("name"))(class_id=cls))
        return metrics

    def _load_training_params(self) -> None:
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        self.optimizer = self._get_optimizer(opt_name, opt_params)
        loss_classifier_name = self.train_cfg.get(
            "loss_classifier", "categorical_crossentropy"
        ).lower()
        loss_classifier_params = self.train_cfg.get("loss_params_classifier", {})
        self.loss_classifier = self._get_loss(
            loss_classifier_name, loss_classifier_params
        )

        loss_reliability_name = self.train_cfg.get(
            "loss_reliability", "binary_crossentropy"
        ).lower()
        loss_reliability_params = self.train_cfg.get("loss_params_reliability", {})
        self.loss_reliability = self._get_loss(
            loss_reliability_name, loss_reliability_params
        )

        if len(self.train_cfg.get("metrics_classifier", [])) > 0:
            self.metrics_classifier = self._get_metrics(
                self.train_cfg.get("metrics_classifier")
            )
        else:
            self.metrics_classifier = self._get_default_metrics(branch="classifier")

        if len(self.train_cfg.get("metrics_reliability", [])) > 0:
            self.metrics_reliability = self._get_metrics(
                self.train_cfg.get("metrics_reliability")
            )
        else:
            self.metrics_reliability = self._get_default_metrics(branch="reliability")

    def _get_default_metrics(self, branch: str) -> list[Any]:
        match branch:
            case "classifier":
                return [tf.keras.metrics.CategoricalAccuracy(name="acc")]
            case "reliability":
                return [
                    tf.keras.metrics.AUC(name="auc", from_logits=True),
                    tf.keras.metrics.BinaryAccuracy(name="acc"),
                ]

    def compile_model(self, model: dict, train_branch: str = "classifier") -> None:
        """Compile a specific branch of the model graph."""
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        self.optimizer = self._get_optimizer(opt_name, opt_params)
        jit_compile = self.use_xla
        if train_branch == "pretrain":
            model.get("rep_model").trainable = True
            model.get("jaeger_projection").compile(
                optimizer=self.optimizer,
                loss_fn=model.get("arcface_loss"),
                run_eagerly=False,
                jit_compile=jit_compile,
            )
            logger.info(f"model compiled for {train_branch} (xla={jit_compile})")
        elif train_branch == "classifier":
            model.get("rep_model").trainable = True
            model.get("jaeger_classifier").compile(
                optimizer=self.optimizer,
                loss=self.loss_classifier,
                metrics=self.metrics_classifier,
                jit_compile=jit_compile,
            )
            logger.info(f"model compiled for {train_branch} (xla={jit_compile})")
        elif train_branch == "reliability":
            if model.get("jaeger_reliability") is None:
                logger.warning(
                    "jaeger_reliability not built — skipping reliability compilation"
                )
                return
            model.get("rep_model").trainable = False
            if model.get("classification_head") is not None:
                model.get("classification_head").trainable = False
            model.get("jaeger_reliability").compile(
                optimizer=self.optimizer,
                loss=self.loss_reliability,
                metrics=self.metrics_reliability,
                jit_compile=jit_compile,
            )
        else:
            raise ValueError(
                "train_branch must be 'pretrain', 'classifier' or 'reliability'"
            )

    # ------------------------------------------------------------------
    # Saving / callbacks
    # ------------------------------------------------------------------

    def _prepare_save_path(
        self,
        num_params: str | None = None,
        suffix: str | None = None,
        metadata: str | None = None,
        clear: bool = True,
    ) -> tuple[Path, str]:
        """Create the save directory and build the model name.

        Returns the target path and the computed model name.
        """
        path = Path(self._saving_config.get("path"))
        path.mkdir(parents=True, exist_ok=True)

        if metadata:
            metadata = Path(metadata).resolve()
            metadata.write_text(
                json.dumps(
                    {"model_path": str(path), "experiment_path": str(path.parent)},
                    indent=2,
                )
            )
        graph_id = find_existing_graph_id(path)

        if clear and any(path.iterdir()):
            logger.warning(f"{path} is not empty. deleting existing files!")
            if graph_id is not None:
                logger.warning("if a graph is found, it's id will be reused")
                logger.warning(f"found existing graph id: {graph_id}")
            clear_directory(path=path)

        model_name = self.model_cfg.get("name")
        model_name += (
            f"{'_' + graph_id if graph_id else '_' + self.uid}"
            f"{'_' + num_params if num_params else ''}"
            f"{'_' + suffix if suffix else ''}"
        )
        return path, model_name

    def save_model(
        self,
        model: tf.keras.Model,
        num_params: str | None = None,
        suffix: str | None = None,
        metadata: str | None = None,
    ) -> None:
        """Save model weights, SavedModel graph, class map and project config."""
        path, model_name = self._prepare_save_path(
            num_params=num_params, suffix=suffix, metadata=metadata, clear=True
        )

        if self._saving_config.get("save_weights"):
            model.save_weights(path / f"{model_name}.weights.h5")
            logger.info(
                f"model weights are written to {path / f'{model_name}.weights.h5'}"
            )

        if self._saving_config.get("save_exec_graph"):
            tf.saved_model.save(model, path / f"{model_name}_graph")
            logger.info(
                f"model computational graph is written to "
                f"{path / f'{model_name}_graph'}"
            )

        with open(path / f"{model_name}_classes.yaml", "w") as yaml_file:
            logger.info("writing class_labels_map")
            yaml.dump(
                dict(classes=self.model_cfg.get("class_label_map")),
                yaml_file,
                default_flow_style=False,
            )

        shutil.copy(self.cfg.get("config_path"), path / f"{model_name}_project.yaml")

    def save_embedding_model(
        self,
        models: dict[str, tf.keras.Model],
        num_params: str | None = None,
        suffix: str | None = None,
        metadata: str | None = None,
    ) -> None:
        """Save a dedicated embedding-only SavedModel graph.

        The embedding graph returns the representation vector just before the
        classifier head. This is used by downstream tasks such as the taxonomy
        workflow, and is kept separate so it can be loaded without the larger
        classification / reliability heads.
        """
        if not self._saving_config.get("save_embedding_graph", True):
            return

        rep_model = models.get("rep_model")
        if rep_model is None:
            logger.warning(
                "No 'rep_model' found; skipping embedding-only graph export."
            )
            return

        path, model_name = self._prepare_save_path(
            num_params=num_params, suffix=suffix, metadata=metadata, clear=False
        )

        rep_output = rep_model.output
        if isinstance(rep_output, (list, tuple)):
            rep_output = rep_output[0]

        # Wrap the representation in an Identity layer and export it under the
        # key "embedding" so the SavedModel signature is easy to consume.
        # Use a distinct operation name to avoid collisions with an existing
        # "embedding" layer in the representation model.
        embedding_output = tf.keras.layers.Lambda(lambda x: x, name="embedding_output")(
            rep_output
        )

        embedding_model = tf.keras.Model(
            inputs=rep_model.input,
            outputs={"embedding": embedding_output},
            name=f"{model_name}_embedding",
        )

        graph_path = path / f"{model_name}_embedding_graph"
        tf.saved_model.save(embedding_model, graph_path)
        logger.info(f"embedding-only computational graph is written to {graph_path}")

    def get_callbacks(self, branch: str = "classifier") -> list[Any]:
        """Build Keras callback list from config for a given *branch*."""
        cb_list = self.train_cfg.get("callbacks", dict())
        callbacks = []
        for cb in cb_list.get(branch, []):
            name = cb.get("name")
            params = cb.get("params", {})
            try:
                cb_class = getattr(tf.keras.callbacks, name)
                callbacks.append(cb_class(**params))
            except AttributeError:
                raise ValueError(f"Unsupported callback: {name}")
        return callbacks

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_string_processor_config(self) -> dict[str, Any]:
        _map = {
            "CODON": CODONS,
            "CODON_ID": CODON_ID,
            "AA_ID": AA_ID,
            "MURPHY10_ID": MURPHY10_ID,
            "PC5_ID": PC5_ID,
            "DICODON": DICODONS,
            "DICODON_ID": DICODON_ID,
        }
        _config = dict(self.model_cfg.get("embedding", {}))
        _config.update(self.model_cfg.get("string_processor", {}))
        _config["data_format"] = _config.get("data_format", "csv")
        _config["seq_onehot"] = _config.get("seq_onehot", False)
        if _config["input_type"] == "translated":
            _config["codon"] = _map.get(_config.get("codon"))
            _config["codon_id"] = _map.get(_config.get("codon_id"))
            _config["codon_depth"] = max(_config.get("codon_id", [])) + 1
            _config["vocab_size"] = len(_config.get("codon_id", [])) + 1
            _config["ngram_width"] = int(math.log(len(_config.get("codon", [])), 4))
            if _config["seq_onehot"] is False:
                _config["codon_depth"] = 1
        else:
            _config["codon"] = None
            _config["codon_id"] = None
            _config["codon_depth"] = None
            _config["vocab_size"] = 4
            _config["ngram_width"] = None
        return _config

    def _get_optimizer(self, name: str, kwargs: dict) -> Any:
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "adamw": tf.keras.optimizers.AdamW,
            "muon": tf.keras.optimizers.Muon,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "adagrad": tf.keras.optimizers.Adagrad,
        }
        return optimizers[name](**kwargs)

    def _get_pooler(self, name: str):
        poolers = {
            "max": tf.keras.layers.GlobalMaxPooling2D,
            "average": tf.keras.layers.GlobalAveragePooling2D,
            "max1d": tf.keras.layers.GlobalMaxPooling1D,
            "average1d": tf.keras.layers.GlobalAveragePooling1D,
            "masked_average": MaskedGlobalAvgPooling,
            "gatedframe": GatedFrameGlobalMaxPooling,
        }
        return poolers[name]

    def _get_loss(self, name: str, kwargs: dict) -> Any:
        losses = {
            "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
            "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy,
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
            "mse": tf.keras.losses.MeanSquaredError,
            "hierachical_loss": HierarchicalLoss,
        }
        return losses[name](**kwargs)

    def _get_paths(self, key: str) -> dict[str, Any]:
        fcd_dict = self.train_cfg.get(key, {})
        paths: dict[str, Any] = {}
        for fcd_k, fcd_v in fcd_dict.items():
            tmp_paths = []
            tmp_class = []
            tmp_label = []
            for i in fcd_v:
                tmp_class.extend(i.get("class", []))
                tmp_label.extend(i.get("label", []))
                tmp_paths.extend(i.get("path", []))
            paths[fcd_k] = {
                "paths": tmp_paths,
                "class": tmp_class,
                "label": tmp_label,
            }
        return paths

    def _get_last_checkpoint(self) -> None:
        """Placeholder for checkpoint resumption logic."""
        pass

    def _get_model_saving_configuration(self) -> dict:
        return self.train_cfg.get("model_saving", {})

    def _get_fragment_paths(self) -> dict[str, Any]:
        return self._get_paths("fragment_classifier_data")

    def _get_contig_paths(self) -> dict[str, Any]:
        return self._get_paths("contig_classifier_data")

    def _get_reliability_fragment_paths(self) -> dict[str, Any]:
        return self._get_paths("fragment_reliability_data")

    def _get_reliability_contig_paths(self) -> dict[str, Any]:
        return self._get_paths("contig_reliability_data")
