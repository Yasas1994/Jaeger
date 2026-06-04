from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
import yaml
import math
import tensorflow as tf
import numpy as np
from jaeger.utils.misc import track_ms as track
from jaeger.preprocess.latest.maps import (
    CODON_ID,
    CODONS,
    AA_ID,
    MURPHY10_ID,
    PC5_ID,
    DICODONS,
    DICODON_ID,
)

# Neural‐net building blocks
from jaeger.nnlib.v2.layers import (
    GeLU,
    ReLU,
    MaskedBatchNorm,
    MaskedConv1D,
    ResidualBlock,
)


def evaluate(model, x) -> dict[str, float]:
    accum = defaultdict(list)
    for j, i in track(x, description="[cyan]Crunching data..."):
        y_hat = model(j, training=False)
        y_true = i
        accum["logits"].append(y_hat["prediction"].numpy())
        accum["y_true"].append(y_true.numpy())

    # Concatenate over all batches
    logits = np.concatenate(accum["logits"], axis=0)
    y_true = np.concatenate(accum["y_true"], axis=0)

    # Compute loss (categorical cross-entropy from logits)
    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=True)
    ).numpy()

    # Compute accuracy
    pred_labels = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(pred_labels == true_labels)

    return {"loss": float(loss), "accuracy": float(accuracy)}


class JaegerModel(tf.keras.Model):
    """
    Custom model for Jaeger with training, testing, and prediction steps.

    Methods:
        compile: Compiles the model with loss function, optimizer, and metrics.
        train_step: Performs a training step with gradient calculation and
                    weight updates.
        test_step: Performs a testing step with loss calculation and metric
                   updates.
        predict_step: Performs a prediction step with inference mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function(jit_compile=False)
    def predict_step(self, data):
        # Unpack the data
        x, y = data[0], data[1:]
        # set model to inference mode
        y_logits = self(x, training=False)
        return {"y_hat": y_logits, "meta": y}


class DynamicInferenceModelBuilder:
    """
    rebuilds a model from a configuration file: probably faster than the
    tf.saved.Model graph
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.model_cfg = config.get("model")
        self.class_map = self.model_cfg.get("class_label_map")
        tf.random.set_seed(self.model_cfg.get("seed"))
        np.random.seed(self.model_cfg.get("seed"))
        self.inputs = None
        self.outputs = list()
        self.string_processor_config = self._get_string_processor_config()
        self._regularizer = {
            "l2": tf.keras.regularizers.L2,
            "l1": tf.keras.regularizers.L1,
        }

        match config.get("activation", "gelu"):
            case "gelu":
                self.Activation = GeLU
            case "relu":
                self.Activation = ReLU

        self.models = self.build_fragment_classifier()

    def build_fragment_classifier(self):
        """
        returns rep_model, classification_head and reliability head
        """
        models = {}
        # === 1. EMBEDDING ===
        if "embedding" in self.model_cfg:
            inputs, x = self._build_embedding(self.model_cfg["embedding"])
            self.inputs = inputs
        else:
            raise ValueError("Missing 'embedding' section in config")

        # === 2. REPRESENTATION LEARNER ===
        if "representation_learner" in self.model_cfg:
            x, r_lbn = self._build_representation_learner(
                self.model_cfg["representation_learner"], x
            )
        else:
            raise ValueError("Missing 'representation_learner' section in config")

        # === 3. CLASSIFIER ===
        if "classifier" in self.model_cfg:
            y = self._build_head(self.model_cfg["classifier"], x)
            models["classifier"] = tf.keras.Model(inputs=self.inputs, outputs=y)
        else:
            raise ValueError("Missing 'classifier' section in config")

        # === 4. RELIABILITY MODEL ===
        if "reliability_model" in self.model_cfg:
            r = self._build_head(self.model_cfg["reliability_model"], r_lbn)
            models["reliability"] = tf.keras.Model(inputs=self.inputs, outputs=r)
        else:
            raise ValueError("Missing 'reliability_model' section in config")

        return models

    def _build_embedding(self, cfg: Dict[str, Any]):
        """Builds the embedding layer based on config."""
        input_shape = cfg.get("input_shape")
        frames = cfg.get("frames", 6)
        embedding_size = cfg.get("embedding_size", 128)
        seq_type = cfg.get("type", "translated")

        if seq_type == "translated":
            inputs = tf.keras.Input(shape=input_shape, name="translated_input")
            x = inputs
        elif seq_type == "nucleotide":
            inputs = tf.keras.Input(shape=input_shape, name="nucleotide_input")
            x = inputs
        elif seq_type == "both":
            # Handle dual input
            inputs = tf.keras.Input(shape=input_shape, name="combined_input")
            x = inputs
        else:
            raise ValueError(f"Unknown embedding type: {seq_type}")

        return inputs, x

    def _build_representation_learner(self, cfg: Dict[str, Any], x):
        """Builds the representation learner (conv blocks)."""
        block_sizes = cfg.get("block_sizes", [2, 2, 2])
        block_filters = cfg.get("block_filters", [128, 128, 128])
        block_kernel_size = cfg.get("block_kernel_size", [5, 5, 5])
        block_kernel_dilation = cfg.get("block_kernel_dilation", [3, 3, 3])
        block_kernel_strides = cfg.get("block_kernel_strides", [1, 1, 1])
        block_regularizer = cfg.get("block_regularizer", ["l2", "l2", "l2"])
        block_regularizer_w = cfg.get("block_regularizer_w", [0, 0, 0])
        pooling = cfg.get("pooling", "max")

        # Initial conv
        x = MaskedConv1D(
            filters=cfg.get("masked_conv1d_1_filters", 128),
            kernel_size=cfg.get("masked_conv1d_1_kernel_size", 7),
            strides=cfg.get("masked_conv1d_1_strides", 1),
            dilation_rate=cfg.get("masked_conv1d_1_dilation_rate", 1),
            padding="same",
            kernel_regularizer=self._regularizer.get(
                cfg.get("masked_conv1d_1_regularizer", "l2"),
                tf.keras.regularizers.L2,
            )(cfg.get("masked_conv1d_1_regularizer_w", 0)),
            name="masked_conv1d_initial",
        )(x)
        x = self.Activation(name="activation_initial")(x)

        # Residual blocks
        for i, (size, filters, ksize, dilation, strides, reg, reg_w) in enumerate(
            zip(
                block_sizes,
                block_filters,
                block_kernel_size,
                block_kernel_dilation,
                block_kernel_strides,
                block_regularizer,
                block_regularizer_w,
            )
        ):
            for j in range(size):
                x = ResidualBlock(
                    filters=filters,
                    kernel_size=ksize,
                    dilation_rate=dilation,
                    strides=strides if j == 0 else 1,
                    kernel_regularizer=self._regularizer.get(reg, tf.keras.regularizers.L2)(
                        reg_w
                    ),
                    name=f"resblock_{i}_{j}",
                )(x)
            if pooling == "max":
                x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"pool_{i}")(x)
            elif pooling == "avg":
                x = tf.keras.layers.AveragePooling1D(pool_size=2, name=f"pool_{i}")(x)

        # Final conv
        x = MaskedConv1D(
            filters=block_filters[-1],
            kernel_size=cfg.get("masked_conv1d_final_kernel_size", 5),
            strides=cfg.get("masked_conv1d_final_strides", 1),
            dilation_rate=cfg.get("masked_conv1d_final_dilation_rate", 1),
            padding="same",
            kernel_regularizer=self._regularizer.get(
                cfg.get("masked_conv1d_final_regularizer", "l2"),
                tf.keras.regularizers.L2,
            )(cfg.get("masked_conv1d_final_regularizer_w", 0)),
            name="masked_conv1d_final",
        )(x)
        x = self.Activation(name="activation_final")(x)

        # Global pooling for reliability model
        r_lbn = tf.keras.layers.GlobalAveragePooling1D(name="gap_reliability")(x)

        return x, r_lbn

    def _build_head(self, cfg: Dict[str, Any], x):
        """Builds a classification head."""
        hidden_layers = cfg.get("hidden_layers", [])

        for i, layer_cfg in enumerate(hidden_layers):
            x = tf.keras.layers.Dense(
                units=layer_cfg.get("units", 128),
                activation=None,
                use_bias=layer_cfg.get("use_bias", False),
                kernel_regularizer=self._regularizer.get(
                    layer_cfg.get("kernel_regularizer", "l2"),
                    tf.keras.regularizers.L2,
                )(layer_cfg.get("kernel_regularizer_w", 1e-5)),
                name=f"dense_{i}",
            )(x)
            x = self.Activation(name=f"activation_dense_{i}")(x)
            if layer_cfg.get("dropout_rate", 0) > 0:
                x = tf.keras.layers.Dropout(layer_cfg["dropout_rate"], name=f"dropout_{i}")(x)

        # Output layer
        x = tf.keras.layers.Dense(
            units=cfg.get("output_units", 6),
            activation=cfg.get("output_activation", None),
            use_bias=cfg.get("output_use_bias", False),
            name="output",
        )(x)

        return x

    def _get_string_processor_config(self):
        """Extract string processor config from model config."""
        # This is a simplified version
        return {"input_type": "translated"}


class InferModel:
    """
    loads a graph given a dict with model graph location and class map
    consumnes batched iterators and returns logits per iterator element
    (works with latest generation of models)
    """

    def __init__(self, path_dict):
        self.class_map = self._load_class_map(path_dict.get("classes"))
        self.loaded_model = tf.saved_model.load(path_dict.get("graph"))
        self.inference_fn = self.loaded_model.signatures["serving_default"]
        self.string_processor_config = self._load_string_processor_config(
            path_dict.get("project", None)
        )

    @tf.function(jit_compile=False)
    def _predict_step(self, x):
        # Unpack the data
        # set model to inference mode
        y_logits = self.inference_fn(
            inputs=x.get(self.string_processor_config.get("input_type"))
        )
        return y_logits

    def predict(self, dataset, no_progress: bool = False) -> dict[str, np.ndarray]:
        """
        dataset: yields tuples (inputs_dict, meta0, meta1, …)
        Returns a dict of numpy arrays, each of shape (N, …)

        To avoid GPU OOM on large datasets, each batch is moved to host
        (CPU) memory immediately after inference.
        """
        # accumulate NumPy arrays on CPU — prevents GPU memory growth
        acc: dict[str, list[np.ndarray]] = defaultdict(list)

        # inline for speed
        inf_fn = self._predict_step

        for inputs, *meta in track(
            dataset, description="[cyan]Crunching data…", disable=no_progress
        ):
            # call the tf.function
            logits = inf_fn(inputs)

            # move logits to CPU immediately to bound GPU memory usage
            for k, t in logits.items():
                acc[k].append(t.numpy())

            # meta data is already on CPU (strings / python objects)
            for idx, m in enumerate(meta):
                acc[f"meta_{idx}"].append(m)

        # concatenate NumPy arrays on CPU
        result: dict[str, np.ndarray] = {}
        for k, arr_list in acc.items():
            result[k] = np.concatenate(arr_list, axis=0)
        return result

    def evaluate(self, dataset, no_progress: bool = False) -> dict[str, float]:
        """
        dataset: yields tuples (inputs_dict, y_true_onehot)
        Returns loss and accuracy

        Moves each batch to CPU immediately to avoid GPU OOM.
        """
        logits_acc: list[np.ndarray] = []
        true_acc: list[np.ndarray] = []

        inf_fn = self._predict_step

        for inputs, y_true in track(
            dataset, description="[cyan]Evaluating…", disable=no_progress
        ):
            logits = inf_fn(inputs)["prediction"]
            # move to CPU immediately
            logits_acc.append(logits.numpy())
            true_acc.append(y_true.numpy())

        # concatenate on CPU
        logits = np.concatenate(logits_acc, axis=0)
        y_true = np.concatenate(true_acc, axis=0)

        # loss & accuracy
        loss = tf.keras.losses.categorical_crossentropy(
            y_true, logits, from_logits=True
        )
        loss = float(tf.reduce_mean(loss).numpy())

        pred_labels = np.argmax(logits, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = float(np.mean(pred_labels == true_labels))

        return {"loss": loss, "accuracy": accuracy}

    def _load_class_map(self, path):
        with open(path) as f:
            _class_map = yaml.safe_load(f)["classes"]

            return {
                "num_classes": len(_class_map),
                "class": [i["class"] for i in _class_map],
                "index": [i["label"] for i in _class_map],
            }

    def _load_string_processor_config(self, path) -> Dict:
        _map = {
            "CODON": CODONS,
            "CODON_ID": CODON_ID,
            "AA_ID": AA_ID,
            "MURPHY10_ID": MURPHY10_ID,
            "PC5_ID": PC5_ID,
            "DICODON": DICODONS,
            "DICODON_ID": DICODON_ID,
        }

        if path is None:
            # Legacy models without project.yaml — return minimal defaults
            return {"input_type": "translated"}

        _config = yaml.safe_load(Path(path).read_text()) or {}
        _model_cfg = _config.get("model")
        _config = _model_cfg.get("embedding")
        _config.update(_model_cfg.get("string_processor"))

        # Set input_type from embedding type (e.g., "translated", "nucleotide", "both")
        _config["input_type"] = _config.get("type", "translated")

        if _config["codon"] is not None and _config["codon_id"] is not None:
            _config["codon"] = _map.get(_config.get("codon"))
            _config["codon_id"] = _map.get(_config.get("codon_id"))
            _config["codon_depth"] = max(_config.get("codon_id")) + 1  # num_codons
            _config["vocab_size"] = len(_config.get("codon_id")) + 1  # num_codons + 1
            _config["ngram_width"] = int(math.log(len(_config["codon"]), 4))
            # Infer seq_onehot from embedding input_shape if not explicitly set.
            # input_shape like [6, None, 64] means one-hot with depth=64;
            # [6, None] means embedding lookup (no one-hot).
            input_shape = _model_cfg.get("embedding", {}).get("input_shape")
            if _config.get("seq_onehot") is None and input_shape is not None:
                # input_shape is [frames, timesteps, codon_depth] for one-hot
                _config["seq_onehot"] = (
                    len(input_shape) == 3
                    and input_shape[-1] is not None
                    and input_shape[-1] > 1
                )
            _config["seq_onehot"] = _config.get("seq_onehot", False)
            if _config["seq_onehot"] is False:
                _config["codon_depth"] = 1
        return _config


class TFLiteInferModel:
    """
    TFLite inference wrapper for quantized Jaeger models.

    Provides the same interface as InferModel but runs inference via
    TensorFlow Lite interpreter. Supports dynamic input resizing for
    variable sequence lengths.
    """

    def __init__(self, path_dict):
        self.class_map = self._load_class_map(path_dict.get("classes"))
        self.string_processor_config = self._load_string_processor_config(
            path_dict.get("project", None)
        )

        # Load TFLite model
        tflite_path = path_dict.get("tflite")
        if tflite_path is None or not Path(tflite_path).exists():
            raise ValueError(
                f"TFLite model not found at {tflite_path}. "
                "Run 'jaeger utils quantize' first."
            )

        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, dataset, no_progress: bool = False) -> dict[str, np.ndarray]:
        """
        dataset: yields tuples (inputs_dict, meta0, meta1, …)
        Returns a dict of numpy arrays, each of shape (N, …)
        """
        acc: dict[str, list[np.ndarray]] = defaultdict(list)

        for inputs, *meta in track(
            dataset, description="[cyan]Crunching data…", disable=no_progress
        ):
            # Get the input tensor (e.g., "translated")
            input_type = self.string_processor_config.get("input_type", "translated")
            x = inputs.get(input_type)

            if isinstance(x, tf.Tensor):
                x = x.numpy()

            batch_size, *shape = x.shape

            # Resize interpreter if needed
            current_shape = self.input_details[0]["shape"]
            if list(current_shape) != [batch_size, *shape]:
                self.interpreter.resize_tensor_input(
                    self.input_details[0]["index"], [batch_size, *shape], strict=True
                )
                self.interpreter.allocate_tensors()
                # Refresh details after resize
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

            # Set input and invoke
            self.interpreter.set_tensor(self.input_details[0]["index"], x)
            self.interpreter.invoke()

            # Get outputs - map to same keys as SavedModel
            # Output order: prediction, reliability (based on our conversion)
            pred = self.interpreter.get_tensor(self.output_details[0]["index"])
            rel = self.interpreter.get_tensor(self.output_details[1]["index"])

            acc["prediction"].append(pred)
            acc["reliability"].append(rel)

            # meta data
            for idx, m in enumerate(meta):
                if isinstance(m, tf.Tensor):
                    m = m.numpy()
                acc[f"meta_{idx}"].append(m)

        # concatenate arrays; keep lists for non-array metadata
        result: dict[str, np.ndarray] = {}
        for k, arr_list in acc.items():
            if k.startswith("meta_"):
                # Metadata may be strings or other non-array types
                try:
                    result[k] = np.concatenate(arr_list, axis=0)
                except ValueError:
                    result[k] = arr_list  # Keep as list if concatenation fails
            else:
                result[k] = np.concatenate(arr_list, axis=0)
        return result

    def evaluate(self, dataset, no_progress: bool = False) -> dict[str, float]:
        """Evaluate on a dataset. Not yet implemented for TFLite."""
        raise NotImplementedError("TFLite evaluation not yet implemented")

    def _load_class_map(self, path):
        with open(path) as f:
            _class_map = yaml.safe_load(f)["classes"]
            return {
                "num_classes": len(_class_map),
                "class": [i["class"] for i in _class_map],
                "index": [i["label"] for i in _class_map],
            }

    def _load_string_processor_config(self, path) -> Dict:
        _map = {
            "CODON": CODONS,
            "CODON_ID": CODON_ID,
            "AA_ID": AA_ID,
            "MURPHY10_ID": MURPHY10_ID,
            "PC5_ID": PC5_ID,
            "DICODON": DICODONS,
            "DICODON_ID": DICODON_ID,
        }

        if path is None:
            return {"input_type": "translated"}

        _config = yaml.safe_load(Path(path).read_text()) or {}
        _model_cfg = _config.get("model")
        _config = _model_cfg.get("embedding")
        _config.update(_model_cfg.get("string_processor"))
        _config["input_type"] = _config.get("type", "translated")

        if _config["codon"] is not None and _config["codon_id"] is not None:
            _config["codon"] = _map.get(_config.get("codon"))
            _config["codon_id"] = _map.get(_config.get("codon_id"))
            _config["codon_depth"] = max(_config.get("codon_id")) + 1
            _config["vocab_size"] = len(_config.get("codon_id")) + 1
            _config["ngram_width"] = int(math.log(len(_config["codon"]), 4))
            input_shape = _model_cfg.get("embedding", {}).get("input_shape")
            if _config.get("seq_onehot") is None and input_shape is not None:
                _config["seq_onehot"] = (
                    len(input_shape) == 3
                    and input_shape[-1] is not None
                    and input_shape[-1] > 1
                )
            _config["seq_onehot"] = _config.get("seq_onehot", False)
            if _config["seq_onehot"] is False:
                _config["codon_depth"] = 1
        return _config
