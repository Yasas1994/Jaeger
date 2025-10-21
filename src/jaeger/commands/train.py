"""
Copyright 2024 R. Y. Wijesekara - University Medicine Greifswald, Germany

Identifying phage genome sequences concealed in metagenomes is a
long standing problem in viral metagenomics and ecology.
The Jaeger approach uses homology-free machine learning to identify
 both phages and prophages in metagenomic assemblies.
"""

import os

# temporary fix
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"
from jaeger.preprocess.latest.convert import process_string_train
from jaeger.preprocess.latest.maps import CODONS, CODON_ID, MURPHY10_ID, AA_ID, PC5_ID, DICODONS, DICODON_ID
import yaml
import shutil
import math
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict, List, Optional
from jaeger.nnlib.v2.layers import (
    GeLU,
    ReLU,
    MaskedBatchNorm,
    MaskedConv1D,
    ResidualBlock_wrapper,
    MetricModel,
    GatedFrameGlobalMaxPooling,
    TransformerEncoder
)
from jaeger.nnlib.v2.losses import ArcFaceLoss, HierarchicalLoss
from jaeger.nnlib.inference import InferModel, evaluate
from jaeger.nnlib.metrics import PrecisionForClass, RecallForClass, SpecificityForClass
import logging
import re
from jaeger.utils.misc import numerize, load_model_config, AvailableModels

# dev
import numpy as np
from icecream import ic

# todo
# prevent over writing existing experiments
# train from last checkpoint

logger = logging.getLogger("Jaeger")
ic.configureOutput(prefix="Jaeger |")


class DynamicModelBuilder:
    """
    to do: implement feature correlation based out-of-distribution detection
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.model_cfg = config.get("model")
        self.train_cfg = config.get("training")
        tf.random.set_seed(self.model_cfg.get("seed"))
        np.random.seed(self.model_cfg.get("seed"))
        self.inputs = None
        self.outputs = list()
        self.input_shape = self.model_cfg["embedding"].get("input_shape")
        self._fragment_paths = self._get_fragment_paths()
        self._contig_paths = self._get_contig_paths()
        self._reliability_fragment_paths = self._get_reliability_fragment_paths()
        self._reliability_contig_paths = self._get_reliability_contig_paths()
        ic(self._fragment_paths)
        # ic(self._reliability_fragment_paths)
        self._saving_config = self._get_model_saving_configuration()
        ic(config.get("from_last_checkpoint"))
        self._from_last_checkpoint = config.get("from_last_checkpoint")
        self.classifier_out_dim = self.model_cfg.get("classifier_out_dim")
        self.reliability_out_dim = self.model_cfg.get("reliability_out_dim")
        self.loss_classifier = None
        self.loss_reliability = None
        self.metrics_classifier = []
        self.metrics_reliability = []
        self._checkpoints = dict()
        self._load_training_params()
        self._regularizer = {
            "l2": tf.keras.regularizers.L2,
            "l1": tf.keras.regularizers.L1,
        }
        self._layers = {
            "masked_conv1d": MaskedConv1D,
            "conv1d": tf.keras.layers.Conv1D,
            "masked_batchnorm": MaskedBatchNorm,
            "batchnorm": tf.keras.layers.BatchNormalization,
            "transformer_encoder": TransformerEncoder,
            "residual_block": ResidualBlock_wrapper,
            "dense": tf.keras.layers.Dense,
            "activation": tf.keras.layers.Activation,
            "dropout": tf.keras.layers.Dropout

        }
        # self._activations = {
        #     "gelu": GeLU,
        #     "relu": ReLU,
        # }
        cb_list = config["training"].get("callbacks", dict())
        for p in cb_list.get("directories"):
            p = Path(p)
            if p.exists() and not self._from_last_checkpoint:
                ic(f"removing the old found checkpoint at {p}")
                ic(
                    "set --from_last_checkpoint flag to continue training from the last checkpoint"
                )
                shutil.rmtree(p)
            elif p.exists() and self._from_last_checkpoint:
                # None is not available
                self._checkpoints[p.name] = self.get_latest_h5_with_metadata(p)

            p.mkdir(parents=True, exist_ok=True)
        ic(self._checkpoints)


    def get_latest_h5_with_metadata(
        self,
        path: str | Path,
        check_convergence: str = "classifier",
        pattern: str = r"epoch:(\d+)-loss:(\d+\.\d+)",
    ) -> Optional[tuple[Path, dict]]:
        path = Path(path)
        """
        to do: create a checkpoint file once the model converges
        """
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

    def build_fragment_classifier(self):
        """
        returns rep_model, classification_head and reliability head
        """
        models = {}
        # === 1. EMBEDDING ===
        if "embedding" in self.model_cfg:
            inputs, x = self._build_embedding(self._get_string_processor_config())
            self.inputs = inputs
        else:
            raise ValueError("Missing 'embedding' section in config")

        # === 2. REPRESENTATION LEARNER ===
        if "representation_learner" in self.model_cfg:
            rep_out = self._build_block(
                x, 
                self.model_cfg["representation_learner"],
                prefix="rep"
            )
            models["rep_model"] = tf.keras.Model(
                inputs=self.inputs, outputs=rep_out, name="rep_model"
            )
            models["rep_model"].summary()
        # === 3. PRETRAINING ==
        if "projection" in self.model_cfg:
            input_shape = (
                self.model_cfg["representation_learner"].get("block_filters")[-1],
            )
            inputs = tf.keras.Input(shape=input_shape, name="projection_input")
            x_projection = self._build_block(
                inputs, self.model_cfg["projection"], prefix="projection"
            )
            models["projection_head"] = tf.keras.Model(
                inputs=inputs, outputs=x_projection, name="projection_head"
            )
            num_class = self.num_class
            projection_dim = self.model_cfg["projection"]["hidden_layers"][-1]["units"]

            # combine with the representation learner
            x = models["rep_model"].output[0]
            x = models["projection_head"](x)
            models["jaeger_projection"] = MetricModel(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_projection"
            )
            # define arcface loss model
            labels = tf.keras.Input(shape=(num_class,), name="labels")
            embeddings = tf.keras.Input(
                shape=(projection_dim,), name="embedding"
            )  # output of the projection head
            loss = ArcFaceLoss(
                num_classes=num_class,
                embedding_dim=projection_dim,
                margin=self.model_cfg["projection"]["margin"],
                scale=self.model_cfg["projection"]["scale"],
                onehot=True,
            )(labels, embeddings)
            models["arcface_loss"] = tf.keras.Model(
                inputs=[labels, embeddings], outputs=loss, name="Arcface"
            )

        # === 3. CLASSIFIER ===
        if "classifier" in self.model_cfg:
            input_shape = (
                self.model_cfg["classifier"].get("input_shape"),
            )
            inputs = tf.keras.Input(shape=input_shape, name="classifier_input")
            ic(inputs)
            x_classifier = self._build_block(
                inputs, 
                self.model_cfg["classifier"], 
                prefix="classifier"
            )
            models["classification_head"] = tf.keras.Model(
                inputs=inputs, 
                outputs=x_classifier, 
                name="classification_head"
            )
            # combine with the representation learner
            x_rep = models["rep_model"].output[0]
            x = models["classification_head"](x_rep)
            models["jaeger_classifier"] = tf.keras.Model(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_classifier"
            )
            if self._checkpoints.get("classifier", {}).get("path", False):
                # loads weights from the last checkpoint
                models["jaeger_classifier"].load_weights(
                    self._checkpoints.get("classifier").get("path")
                )
                ic(
                    f"Loaded classification model weights from {self._checkpoints.get('classifier').get('path')}"
                )

        # === 4. RELIABILITY ===
        if "reliability_model" in self.model_cfg:
            input_shape = (
                self.model_cfg["reliability_model"].get("input_shape"),
            )
            inputs = tf.keras.Input(shape=input_shape, name="reliability_input")
            x_reliability = self._build_block(
                inputs, 
                self.model_cfg["reliability_model"], 
                prefix="reliability"
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

            if self._checkpoints.get("reliability", {}).get("path", False):
                # loads weights from the last checkpoint
                models["jaeger_reliability"].load_weights(
                    self._checkpoints.get("reliability").get("path")
                )
                ic(
                    f"Loaded reliability model weights from {self._checkpoints.get('reliability').get('path')}"
                )
        # ==== 5. COMBINED MODEL ====
        rep_out = models["rep_model"].output
        if isinstance(rep_out, tuple):
            if len(rep_out) == 2:
                x1, x2 = rep_out
                reliability = models["reliability_head"](x2) # NMD
                class_ = models["classification_head"](x1)
                models["jaeger_model"] = tf.keras.Model(
                    inputs=models["rep_model"].input,
                    outputs={"prediction": class_, "reliability": reliability, "embedding": x1, "nmd": x2},
                    name="Jaeger_model",
            )
            elif len(rep_out) == 3:
                x1, x2, g = rep_out
                reliability = models["reliability_head"](x2) # NMD
                class_ = models["classification_head"](x1)
                models["jaeger_model"] = tf.keras.Model(
                    inputs=models["rep_model"].input,
                    outputs={"prediction": class_, "reliability": reliability, "embedding": x1, "nmd": x2, "gate": g },
                    name="Jaeger_model",
                )
        ic(models)

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
                            embeddings_regularizer=self._regularizer.get(
                        cfg.get("embedding_regularizer"),
                    )(cfg.get("embedding_regularizer_w"))
                        )(inputs)
                    else:
                        x = tf.keras.layers.Dense(
                            embedding_size,
                            name=f"{cfg.get('input_type')}_embedding",
                            use_bias=False,
                            kernel_initializer=tf.keras.initializers.Orthogonal(),
                            kernel_regularizer=self._regularizer.get(
                        cfg.get("embedding_regularizer"),
                    )(cfg.get("embedding_regularizer_w"))
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
    
    def _get_bias(self, data_path: str, kind: str):
        """
        initialize final layer bias connsidering the
        class frequencies of the training datasets
        path to train file and kind[softmax, sigmoid]
        """
        import polars as pl
        def _sigmoid(f:Dict):
            n, p = f.values()
            t = p / (p + n)
            ic(np.log(t / (1-t)))
            return np.log(t / (1-t))
        
        def _softmax(f:Dict):
            f = np.array(list(f.values()))
            ic(np.log(f/np.sum(f)))
            return np.log(f/np.sum(f))


        df = pl.read_csv(data_path, columns=[0], has_header=False)

        counts_dict = (
            df["column_1"]
            .value_counts()
            .to_dict(as_series=False)  # gives {column_name: [...], count: [...]}
        )

        counts_dict = dict(zip(counts_dict["column_1"], counts_dict["count"]))
        counts_dict = {k:counts_dict[k] for k in  sorted(list(counts_dict.keys()))}
        match kind:
            case "softmax":
                return _softmax(counts_dict)
            case "sigmoid":
                return _sigmoid(counts_dict)

    def _build_block(self, x, cfg: Dict[str, Any], prefix:str):
        """
        Build representation learner from configuration.
        Input:  X
        Output: [X'-1, X']
        """
        nmd = None  # default in case no layer returns it
        previous_channels = None
        for i, layer_cfg in enumerate(cfg.get("hidden_layers", [])):
            layer_name = layer_cfg.get("name", "").lower()
            cfg_layer = dict(layer_cfg.get("config", {}))
            cfg_layer["name"] = f"{prefix}_{layer_name}_{i}"

            layer_class = self._layers.get(layer_name)
            if layer_class is None:
                raise ValueError(f"Unknown layer type: {layer_name}")

            # Handle kernel_regularizer
            if "kernel_regularizer" in cfg_layer:
                reg_name = cfg_layer.pop("kernel_regularizer")
                reg_w = cfg_layer.pop("kernel_regularizer_w", None)
                cfg_layer["kernel_regularizer"] = self._regularizer[reg_name](reg_w)

            # Handle kernel_initializer (expand if needed later)
            if "kernel_initializer" in cfg_layer:
                init_name = cfg_layer["kernel_initializer"]
                cfg_layer["kernel_initializer"] = tf.keras.initializers.get(init_name)

            if "bias_initializer" in cfg_layer:
                if "calculate_from" in cfg_layer.get("bias_initializer"):
                    if "relia" in prefix:
                        path = self._get_reliability_fragment_paths().get('train').get('paths')[-1]
                        cfg_layer["bias_initializer"] = tf.keras.initializers.Constant(self._get_bias(path, kind="sigmoid"))
                    elif "classi" in prefix:
                        path = self._get_fragment_paths().get('train').get('paths')[-1]
                        cfg_layer["bias_initializer"] = tf.keras.initializers.Constant(self._get_bias(path, kind="softmax"))

            # Handle residual blocks
            if "block_size" in cfg_layer:
                block_size = cfg_layer.pop("block_size")
                shape = (self.input_shape[0], None, previous_channels)
                if block_size > 0:
                    if cfg_layer.get("return_nmd", False):
                        x, nmd = layer_class(block_size, shape, **cfg_layer)(x)
                    else:
                        x = layer_class(block_size, shape, **cfg_layer)(x)
                    if 'filters' in cfg_layer:
                        previous_channels = cfg_layer.get('filters')
                    elif 'units' in cfg_layer:
                        previous_channels = cfg_layer.get('units')
                continue

            # Handle return_nmd case
            if "return_nmd" in cfg_layer:
                if cfg_layer.get("return_nmd"):
                    x, nmd = layer_class(**cfg_layer)(x)
                else:
                    x = layer_class(**cfg_layer)(x)
                if 'filters' in cfg_layer:
                    previous_channels = cfg_layer.get('filters')
                elif 'units' in cfg_layer:
                    previous_channels = cfg_layer.get('units')
                continue

            x = layer_class(**cfg_layer)(x)

            if 'filters' in cfg_layer:
                previous_channels = cfg_layer.get('filters')
            elif 'units' in cfg_layer:
                previous_channels = cfg_layer.get('units')

        # ===== Aggregation =====
        if "pooling" in cfg:
            pooling = cfg.get("pooling", "average").lower()
            pooler = self._get_pooler(pooling)

            if "gated" not in pooling:
                x = pooler(name=f"global_{pooling}pool")(x)
                return (x, nmd) if nmd is not None else x
            else:
                x, g = pooler(return_gate=True, name=f"global_{pooling}pool")(x)
                return (x, nmd, g) if nmd is not None else (x, g)
        return x
    
    def _get_metrics(self, config):
        metrics = []
        _metrics = {
            "categorical_accuracy": tf.keras.metrics.CategoricalAccuracy,
            "binary_accuracy": tf.keras.metrics.BinaryAccuracy,
            "sparse_categorical_accuracy": tf.keras.metrics.SparseCategoricalAccuracy,
            "categorical_crossentropy": tf.keras.metrics.CategoricalCrossentropy,
            "binary_crossentropy": tf.keras.metrics.BinaryCrossentropy,
            "sparse_categorical_crossentropy": tf.keras.metrics.SparseCategoricalCrossentropy ,
            "auc": tf.keras.metrics.AUC,
            "precision": tf.keras.metrics.Precision,
            "recall": tf.keras.metrics.Recall,
            "per_class_precision": PrecisionForClass,
            "per_class_recall": RecallForClass,
            "per_class_specificity": SpecificityForClass

        }
        ic(config)
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

    def _load_training_params(self):
        """
        this method class method is essential when training models with NMD based
        out-of-distribution detection model. once the classifier is trained, reliability
        model can be trained by recompiling the model to train the reliability branch
        of the model, after freezing the classifier weights.
        """
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        self.optimizer = self._get_optimizer(opt_name, opt_params)
        ic(self.optimizer)
        ic(opt_params)
        ic(opt_name)
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
        # load metrics from configuration
        if len(self.train_cfg.get("metrics_classifier", [])) > 0:
            self.metrics_classifier = self._get_metrics(self.train_cfg.get("metrics_classifier"))
        else:    
            self.metrics_classifier = self._get_default_metrics(branch="classifier")
        
        if len(self.train_cfg.get("metrics_reliability", [])) > 0:
            self.metrics_reliability = self._get_metrics(self.train_cfg.get("metrics_reliability"))
        else:    
            self.metrics_reliability = self._get_default_metrics(branch="reliability")
        
        ic(self.loss_reliability)
        ic(self.metrics_classifier)
        ic(self.metrics_reliability)

    def _get_default_metrics(self, branch):
        match branch:
            case "classifier":
                return [tf.keras.metrics.CategoricalAccuracy(name="acc")]
            case "reliability":
                return [tf.keras.metrics.AUC(name="auc", from_logits=True),
                        tf.keras.metrics.BinaryAccuracy(name="acc")]


    def compile_model(self, model, train_branch="classifier"):
        """
        compiles the reliability model or the classification model
        """
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        self.optimizer = self._get_optimizer(opt_name, opt_params)
        if train_branch == "pretrain":
            model.get("rep_model").trainable = True
            model.get("jaeger_projection").compile(
                optimizer=self.optimizer,
                loss_fn=model.get("arcface_loss"),
                run_eagerly=True,
            )
            ic(f"model compiled for {train_branch}")

        elif train_branch == "classifier":
            model.get("rep_model").trainable = True
            model.get("jaeger_classifier").compile(
                optimizer=self.optimizer,
                loss=self.loss_classifier,
                metrics=self.metrics_classifier,
            )
            ic(f"model compiled for {train_branch}")

        elif train_branch == "reliability":
            # Freeze classifier and representation learner
            model.get("rep_model").trainable = False
            model.get("jaeger_reliability").compile(
                optimizer=self.optimizer,
                loss=self.loss_reliability,
                metrics=self.metrics_reliability,
            )

        else:
            raise ValueError(
                "train_branch must be 'pretrain', 'classifier' or 'reliability'"
            )

    def save_model(self, model, suffix=None):
        """
        saves models (graph, weights or both) to the output directory
        """
        path = Path(self._saving_config.get("path"))
        path.mkdir(parents=True, exist_ok=True)
        model_name = self.model_cfg.get("name")
        if suffix:
            model_name += f"_{suffix}"

        if self._saving_config.get("save_weights"):
            model.save_weights(path / f"{model_name}.weights.h5")
            ic(f"model weights are written to {path / f'{model_name}.weights.h5'}")

        if self._saving_config.get("save_exec_graph"):
            # this way, you don't need the model configuration to rebuild the model
            tf.saved_model.save(model, path / f"{model_name}_graph")
            ic(
                f"model computational graph is written to {path / f'{model_name}_graph'}"
            )
        # save output indices -> class mapping in the same directory
        with open(path / f"{model_name}_classes.yaml", "w") as yaml_file:
            ic("writing class_labels_map")
            yaml.dump(
                dict(classes=self.model_cfg.get("class_label_map")),
                yaml_file,
                default_flow_style=False,
            )

    def save_config(self, suffix=None):
        """
        saves project config to the model output directory
        """
        path = Path(self._saving_config.get("path"))
        model_name = self.model_cfg.get("name")

        if suffix:
            model_name += f"_{suffix}"

        with open(path / f"{model_name}_project.yaml", "w+") as yaml_file:
            ic("saving project config")
            yaml.dump(self.cfg, yaml_file, default_flow_style=False)

    def get_callbacks(self, branch="classifier") -> List:
        cb_list = self.train_cfg.get("callbacks", dict())
        callbacks = []
        for cb in cb_list.get(branch):
            name = cb.get("name")
            params = cb.get("params", {})
            try:
                cb_class = getattr(tf.keras.callbacks, name)
                callbacks.append(cb_class(**params))
            except AttributeError:
                raise ValueError(f"Unsupported callback: {name}")
        return callbacks

    def _get_string_processor_config(self) -> Dict:
        _map = {
            "CODON": CODONS,
            "CODON_ID": CODON_ID,
            "AA_ID": AA_ID,
            "MURPHY10_ID": MURPHY10_ID,
            "PC5_ID": PC5_ID,
            "DICODON": DICODONS,
            "DICODON_ID": DICODON_ID
        }
        _config = self.model_cfg.get("embedding")
        _config.update(self.model_cfg.get("string_processor"))


        _config["codon"] = _map.get(_config.get("codon"))
        _config["codon_id"] = _map.get(_config.get("codon_id"))
        _config["codon_depth"] = max(_config.get("codon_id")) + 1
        _config["vocab_size"]  = max(_config.get("codon_id")) + 1
        _config["ngram_width"] = int(math.log( _config["codon_depth"] , 4))
        _config["seq_onehot"] = _config.get("seq_onehot", False)
        if _config["seq_onehot"] is False:
             _config["codon_depth"] = 1
        return _config

    def _get_optimizer(self, name, kwargs) -> Any:
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "adagrad": tf.keras.optimizers.Adagrad,
        }
        return optimizers[name](**kwargs)

    def _get_pooler(self, name):
        poolers = {
            "max": tf.keras.layers.GlobalMaxPooling2D,
            "average": tf.keras.layers.GlobalAveragePooling2D,
            "gatedframe": GatedFrameGlobalMaxPooling
        }
        return poolers[name]

    def _get_loss(self, name, kwargs) -> Any:
        losses = {
            "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
            "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy,
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
            "mse": tf.keras.losses.MeanSquaredError,
            "hierachical_loss" : HierarchicalLoss
        }
        ic(kwargs)
        return losses[name](**kwargs)

    def _get_paths(self, key: str) -> Dict:
        fcd_dict = self.train_cfg.get(key, {})
        paths = {}  # train:[], validation:[]
        for fcd_k, fcd_v in fcd_dict.items():
            tmp_paths = []
            tmp_class = []
            tmp_label = []
            for i in fcd_v:
                tmp_class.extend(i.get("class"))
                tmp_label.extend(i.get("label"))
                tmp_paths.extend(i.get("path"))
            paths[fcd_k] = {"paths": tmp_paths, "class": tmp_class, "label": tmp_label}
        return paths

    def _get_last_checkpoint(self):
        """
        initilize the model with last checkpoint's model weights
        """
        pass

    def _get_model_saving_configuration(self):
        return self.train_cfg.get("model_saving", {})

    def _get_fragment_paths(self) -> Dict:
        return self._get_paths("fragment_classifier_data")

    def _get_contig_paths(self) -> Dict:
        return self._get_paths("contig_classifier_data")

    def _get_reliability_fragment_paths(self) -> Dict:
        return self._get_paths("fragment_reliability_data")

    def _get_reliability_contig_paths(self) -> Dict:
        return self._get_paths("contig_reliability_data")


def train_fragment_core(**kwargs):
    """
    trains fragment classification model and reliability prediction model.
    """
    # strategy = tf.distribute.MirroredStrategy()
    # ic(f"Number of devices: {strategy.num_replicas_in_sync}")

    # with strategy.scope():

    ic(kwargs.get("config"))
    ic(kwargs.get("from_last_checkpoint"))
    config = load_model_config(Path(kwargs.get("config")))
    config["from_last_checkpoint"] = kwargs.get("from_last_checkpoint")
    # Initialize the model
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    models.get("rep_model").summary()
    model_num_params = numerize(models.get("rep_model").count_params(), decimal=1)
    ic(model_num_params)

    # =================train classifier ======================
    builder.compile_model(models, train_branch="classifier")

    for i in models.get("jaeger_classifier").layers:
        ic(i.name, i.trainable)

    # =================load train data ======================
    string_processor_config = builder._get_string_processor_config()
    _train_data = builder._get_fragment_paths()
    train_data = {"train": None, "validation": None}
    # classifier_epochs: 50
    # reliability_epochs: 50
    # reliability_train_steps: -1 # -1 to run till the generator exhausts

    for k, v in _train_data.items():
        ic(k, v)
        _data = tf.data.TextLineDataset(
            v.get("paths"), num_parallel_reads=len(v.get("paths")), buffer_size=200
        )
        _buffer_size = string_processor_config.get("buffer_size")
        if string_processor_config.get("input_type") == "translated":
            padded_shape = {
                "translated": [
                    6,
                    None
                ] if string_processor_config.get("use_embedding_layer") is True else
                [
                    6,
                    string_processor_config.get("crop_size") // 3 - 1,
                    string_processor_config.get("codon_depth"),
                ]
            }
        elif string_processor_config.get("input_type") == "nucleotide":
            padded_shape = {
                "nucleotide": [2, string_processor_config.get("crop_size"), 4]
            }
        ic(padded_shape)
        train_data[k] = (
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
                    mutate=string_processor_config.get("mutate"),
                    mutation_rate=string_processor_config.get("mutation_rate"),
                    num_classes=builder.classifier_out_dim,
                    class_label_onehot=True,
                    shuffle=string_processor_config.get("shuffle"),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(
                buffer_size=_data.cardinality() if _buffer_size == -1 else _buffer_size,
                # reshuffle_each_iteration=string_processor_config.get("reshuffle_each_iteration")
            )
            .padded_batch(
                batch_size=builder.train_cfg.get("batch_size"),
                padded_shapes=(
                    padded_shape,
                    [builder.classifier_out_dim],
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )
    ic(builder.train_cfg.get("classifier_train_steps"))
    ic(builder.train_cfg.get("classifier_epochs"))
    # for i in train_data["train"].take(1):
    #     ic(i)

    # ============ check if the model has converged ===========
    checkpoint = builder._checkpoints
    converged = checkpoint and checkpoint.get("classifier", {}).get(
        "is_converged", False
    )
    if  kwargs.get("only_save", False) is False:
        if (not converged and not kwargs.get("only_reliability_head", False)):
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

            # ============ self-supervised pre-training =============================
            if kwargs.get("self_supervised_pretraining", False):
                builder.compile_model(models, train_branch="pretrain")
                models.get("jaeger_projection").summary()
                self_suoervised_train_args = {
                    "validation_data": train_data.get("validation").take(
                        builder.train_cfg.get("classifier_validation_steps")
                    ),
                    "epochs": builder.train_cfg.get("projection_epochs"),
                    "callbacks": builder.get_callbacks(branch="classifier"),
                }
                models.get("jaeger_projection").fit(
                    train_data.get("train").take(
                        builder.train_cfg.get("classifier_train_steps")
                    ),
                    **self_suoervised_train_args,
                )
            # ============== train the classification model ==========================
            models.get("jaeger_classifier").fit(
                train_data.get("train").take(
                    builder.train_cfg.get("classifier_train_steps"),
                
                ),
                class_weight = builder.train_cfg.get("class_weights"),
                **train_args,
            )
        else:
            ic("Skipping training — classification model")

    # ============== reliability model ========================
    builder.compile_model(models, train_branch="reliability")
    for i in models.get("jaeger_reliability").layers:
        ic(i.name, i.trainable)

    _rel_train_data = builder._get_reliability_fragment_paths()
    rel_train_data = {"train": None, "validation": None}
    for k, v in _rel_train_data.items():
        _data = tf.data.TextLineDataset(
            v.get("paths"), num_parallel_reads=len(v.get("paths")), buffer_size=200
        )
        if string_processor_config.get("input_type") == "translated":
            padded_shape = {
                "translated": [
                    6,
                    # string_processor_config.get("crop_size") // 3 - 1,
                    # string_processor_config.get("codon_depth"),
                    None
                ] if string_processor_config.get("use_embedding_layer") is True else
                [
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
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(
                buffer_size=_data.cardinality() if _buffer_size == -1 else _buffer_size,
                # reshuffle_each_iteration=string_processor_config.get("reshuffle_each_iteration")
            )
            .padded_batch(
                batch_size=builder.train_cfg.get("batch_size"),
                padded_shapes=(
                    padded_shape,
                    [builder.reliability_out_dim],
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )
    # ============== check if the model has converged ========

    checkpoint = builder._checkpoints
    converged = checkpoint and checkpoint.get("reliability", {}).get(
        "is_converged", False
    )
    if kwargs.get("only_save", False) is False:
        ic(kwargs.get("only_save", False))
        if (not converged and not kwargs.get("only_classification_head", False)):
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
                **train_args,
            )
        else:
            ic("Skipping training — reliability model")

    # ============= test final model =========================
    ic("testing the final model")

    models.get("jaeger_model").trainable = False
    models.get("jaeger_classifier").evaluate(
        train_data.get("validation").take(
            100
        )
    )

    predictions = models.get("jaeger_model").predict(
        train_data.get("validation").take(
            100
        )
    )
    ic(predictions)
    ic(predictions.keys())
    # ============= saving ===================================
    builder.save_model(
        model=models.get("jaeger_model"), suffix=f"{model_num_params}_fragment"
    )
    builder.save_config(suffix=f"{model_num_params}_fragment")

    # ============= load saved model and infer ===============================
    # model_paths = AvailableModels(path=builder._saving_config.get("path"))
    # ic(model_paths.info)
    # mname_ = list(model_paths.info.keys())[0]
    # model = InferModel(model_paths.info.get(mname_))
    # ic(
    #     model.evaluate(
    #         train_data.get("validation").take(
    #             builder.train_cfg.get("classifier_validation_steps")
    #         )
    #     )
    # )


# def train_contig_core(**kwargs):
#     """
#     to do: contig consensus prediction model
#     currently, the final predictions per-contig is obtained by averaing the
#     per-fragment logits. Instead, we can learn a function to combine information
#     from all fragments.
#     """
#     with open(kwargs.get("config"), "r") as f:
#         config = yaml.safe_load(f)
