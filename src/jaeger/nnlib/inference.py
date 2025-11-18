from collections import defaultdict
from typing import Any, Dict
import yaml
import math
import tensorflow as tf
import numpy as np
from jaeger.utils.misc import track_ms as track
from jaeger.preprocess.latest.maps import CODON_ID, CODONS, AA_ID, MURPHY10_ID, PC5_ID, DICODONS, DICODON_ID

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
            r, r_lbn = self._build_representation_learner(
                x, self.model_cfg["representation_learner"]
            )
            models["rep_model"] = tf.keras.Model(
                inputs=self.inputs, outputs=[r, r_lbn], name="rep_model"
            )

        # === 3. CLASSIFIER ===
        if "classifier" in self.model_cfg:
            input_shape = (
                self.model_cfg["representation_learner"].get("block_filters")[-1],
            )
            inputs = tf.keras.Input(shape=input_shape, name="classifier_input")
            x_classifier = self._build_perceptron(
                inputs, self.model_cfg["classifier"], prefix="classifier"
            )
            models["classification_head"] = tf.keras.Model(
                inputs=inputs, outputs=x_classifier, name="classification_head"
            )
            # combine with the representation learner
            x = models["rep_model"].output[0]
            x = models["classification_head"](x)
            models["jaeger_classifier"] = tf.keras.Model(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_classifier"
            )

        # === 4. RELIABILITY ===
        if "reliability_model" in self.model_cfg:
            input_shape = (
                self.model_cfg["representation_learner"].get("block_filters")[-1],
            )
            inputs = tf.keras.Input(shape=input_shape, name="reliability_input")
            x_reliability = self._build_perceptron(
                inputs, self.model_cfg["reliability_model"], prefix="reliability"
            )
            models["reliability_head"] = tf.keras.Model(
                inputs=inputs, outputs=x_reliability, name="reliability_head"
            )
            # combine withe representation learner
            x = models["rep_model"].output[1]
            x = models["reliability_head"](x)
            models["jaeger_reliability"] = tf.keras.Model(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_reliability"
            )

        # ==== 5. COMBINED MODEL ====
        x1, x2 = models["rep_model"].output
        reliability = models["reliability_head"](x2)
        class_ = models["classification_head"](x1)
        models["jaeger_model"] = JaegerModel(
            inputs=models["rep_model"].input,
            outputs={"prediction": class_, "reliability": reliability},
            name="Jaeger_model",
        )

        return models

    def build_contig_classifier(self):
        """
        to do: contig consensus prediction model
        currently, the final predictions per-contig is obtained by averaing the
        per-fragment logits. Instead, we can learn a function to combine information
        from all fragments.
        """
        pass

    def _build_embedding(self, cfg: Dict[str, Any]):
        """
        creates the embedding layer
        """
        input_shape = cfg.get("input_shape", (6, None, 64))
        embedding_size = cfg.get("embedding_size", 4)

        inputs = tf.keras.Input(shape=input_shape, name=cfg.get("type"))
        masked_inputs = tf.keras.layers.Masking(name="input_mask", mask_value=0.0)(
            inputs
        )

        match cfg.get("type"):
            case "translated":
                x = tf.keras.layers.Dense(
                    embedding_size,
                    name=f"{cfg.get('type')}_embedding",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.Orthogonal(),
                )(masked_inputs)
            case "nucleotide":
                x = masked_inputs
            case _:
                raise ValueError(f"{cfg.get('type')} is invalid")
        return inputs, x

    def _build_representation_learner(self, x, cfg: Dict[str, Any]):
        """
        X -> [X'-1, X']
        """
        x = MaskedConv1D(
            filters=cfg.get("masked_conv1d_1_filters"),
            kernel_size=cfg.get("masked_conv1d_1_kernel_size"),
            strides=cfg.get("masked_conv1d_1_strides"),
            dilation_rate=cfg.get("masked_conv1d_1_dilation_rate"),
            use_bias=False,
            name="masked_conv1d_1",
            activation=None,
            kernel_regularizer=self._regularizer.get(
                cfg.get("masked_conv1d_1_regularizer"),
            )(cfg.get("masked_conv1d_1_regularizer_w")),
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )(x)
        # using batchnorm here gives a big advantage. You get a model that can work well with different input size.
        # infact the accuracy increase with the increasing input size.

        x = MaskedBatchNorm(name="masked_batchnorm_1")(x)
        x = self.Activation(name="activation_1")(x)

        for block, (
            block_size,
            block_filter,
            ksize,
            kdilation,
            kstride,
            kreg,
            kregw,
        ) in enumerate(
            zip(
                cfg.get("block_sizes", [3, 3, 3]),
                cfg.get("block_filters", [128, 256, 512]),
                cfg.get("block_kernel_size", [5, 5, 5]),
                cfg.get("block_kernel_dilation", [1, 1, 1]),
                cfg.get("block_kernel_strides", [2, 2, 2]),
                cfg.get("block_regularizer", ["l2", "l2", "l2"]),
                cfg.get("block_regularizer_w", [1e-6, 1e-6, 1e-6]),
            ),
            start=1,
        ):
            # ========== blockn (compress -> res) =============
            x = MaskedConv1D(
                filters=block_filter,
                kernel_size=ksize,
                strides=kstride,
                dilation_rate=kdilation,
                use_bias=False,
                name=f"ds_masked_conv1d_{block}",
                kernel_regularizer=self._regularizer.get(kreg)(kregw),
                activation=None,
                kernel_initializer=tf.keras.initializers.HeUniform(),
            )(x)
            x = MaskedBatchNorm(name=f"ds_masked_batchnorm_{block}")(x)
            x = self.Activation(name=f"ds_activation_{block}")(x)

            for i in range(block_size):
                x = ResidualBlock(
                    block_filter,
                    kernel_size=ksize,
                    block_number=f"{block}{i}",
                    name=f"masked_resblock_{block}{i}",
                )(x)

        # ============ final block ============
        x = MaskedConv1D(
            filters=block_filter,
            kernel_size=cfg.get("masked_conv1d_final_kernel_size"),
            strides=cfg.get("masked_conv1d_final_strides"),
            dilation_rate=cfg.get("masked_conv1d_final_dilation_rate"),
            use_bias=False,
            name="masked_conv1d_final",
            kernel_regularizer=self._regularizer.get(
                cfg.get("masked_conv1d_final_regularizer"),
            )(cfg.get("masked_conv1d_final_regularizer_w")),
            activation=None,
            kernel_initializer=tf.keras.initializers.HeUniform(),
        )(x)
        # this layers mean vector is used as u[train] to calculate nmd u[example] - u[train]
        x, nmd = MaskedBatchNorm(name="masked_batchnorm_final", return_nmd=True)(x)
        x = self.Activation(name="activation_final")(x)
        # =========== Aggregation ==============
        x = self._get_pooler(cfg.get("pooling"))(
            name=f"global_{cfg.get('pooling')}pool"
        )(x)
        return x, nmd

    def _build_perceptron(self, x, cfg: Dict[str, Any], prefix: str):
        """
        Get an MLP for a given configuration
        prefixes : reliability, projection, classifier
        """

        for i, layer_cfg in enumerate(cfg.get("hidden_layers"), 1):
            x = tf.keras.layers.Dense(
                layer_cfg.get("units"),
                use_bias=layer_cfg.get("use_bias"),
                name=f"{prefix}_dense_{i}",
                # activation=layer_cfg.get("activation")
            )(x)
            x = self.Activation(name=f"{prefix}_activation_{i}")(x)
            if layer_cfg.get("dropout_rate", 0) > 0:
                x = tf.keras.layers.Dropout(
                    layer_cfg.get("dropout_rate"),
                    name=f"{prefix}_dropout_{i}",
                )(x)

        outputs = tf.keras.layers.Dense(
            cfg.get("output_units"),
            use_bias=layer_cfg.get("output_use_bias"),
            name=prefix,
            activation=cfg.get("output_activation"),
        )(x)
        return outputs

    def _get_string_processor_config(self) -> Dict:
        _map = {
            "CODON": CODONS,
            "CODON_ID": CODON_ID,
            "AA_ID": AA_ID,
            "MURPHY10_ID": MURPHY10_ID,
        }
        emb_config = self.model_cfg.get("embedding")
        sp_config = self.model_cfg.get("string_processor")

        _config = {
            "input_type": emb_config.get("input_type"),
            "codon": _map.get(sp_config.get("codon")),
            "codon_id": _map.get(sp_config.get("codon_id")),
            "codon_depth": max(_map.get(sp_config.get("codon_id"))) + 1,
            "crop_size": sp_config.get("crop_size"),
            "buffer_size": sp_config.get("buffer_size"),
        }

        return _config

    def _get_pooler(self, name):
        poolers = {
            "max": tf.keras.layers.GlobalMaxPooling2D,
            "average": tf.keras.layers.GlobalAveragePooling2D,
        }
        return poolers[name]


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
                x.get(self.string_processor_config.get("input_type"))
            )
        return y_logits

    def predict(self, dataset, no_progress: bool = False) -> dict[str, np.ndarray]:
        """
        dataset: yields tuples (inputs_dict, meta0, meta1, …)
        Returns a dict of numpy arrays, each of shape (N, …)
        """
        # accumulate TF tensors
        acc: dict[str, list[tf.Tensor]] = defaultdict(list)

        # inline for speed
        inf_fn = self._predict_step

        for inputs, *meta in track(
            dataset, description="[cyan]Crunching data…", disable=no_progress
        ):
            # call the tf.function
            logits = inf_fn(inputs)

            # collect all logits tensors
            for k, t in logits.items():
                acc[k].append(t)

            # collect meta tensors
            for idx, m in enumerate(meta):
                acc[f"meta_{idx}"].append(m)

        # now concatenate once per key, then move to NumPy
        result: dict[str, np.ndarray] = {}
        for k, tensor_list in acc.items():
            # tf.concat stays on device, one pass
            cat = tf.concat(tensor_list, axis=0)
            # then one host-device copy per key
            result[k] = cat.numpy()
        return result

    def evaluate(self, dataset, no_progress: bool = False) -> dict[str, float]:
        """
        dataset: yields tuples (inputs_dict, y_true_onehot)
        Returns loss and accuracy
        """
        logits_acc: list[tf.Tensor] = []
        true_acc: list[tf.Tensor] = []

        inf_fn = self._predict_step

        for inputs, y_true in track(
            dataset, description="[cyan]Evaluating…", disable=no_progress
        ):
            logits = inf_fn(inputs)["prediction"]
            logits_acc.append(logits)
            true_acc.append(y_true)

        # bulk-concatenate and to NumPy once
        logits = tf.concat(logits_acc, axis=0).numpy()
        y_true = tf.concat(true_acc, axis=0).numpy()

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
            "DICODON_ID": DICODON_ID
        }
        
        _config = yaml.safe_load(path.read_text()) or {}
        _model_cfg = _config.get("model")
        _config = _model_cfg.get("embedding")
        _config.update(_model_cfg.get("string_processor"))


        _config["codon"] = _map.get(_config.get("codon"))
        _config["codon_id"] = _map.get(_config.get("codon_id"))
        _config["codon_depth"] = max(_config.get("codon_id")) + 1
        _config["vocab_size"]  = max(_config.get("codon_id")) + 1
        _config["ngram_width"] = int(math.log( len(_config["codon"]) , 4))
        _config["seq_onehot"] = _config.get("seq_onehot", False)
        if _config["seq_onehot"] is False:
             _config["codon_depth"] = 1
        return _config