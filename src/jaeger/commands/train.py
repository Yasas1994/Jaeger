import os
# temporary fix
os.environ['WRAPT_DISABLE_EXTENSIONS'] = "true" 
from jaeger.preprocess.latest.convert import process_string_train
from jaeger.preprocess.latest.maps import *
import yaml
import jinja2
import shutil
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict, List, Optional
from jaeger.nnlib.v2.layers import GeLU, ReLU, MaskedAdd, MaskedBatchNorm, MaskedConv1D, ResidualBlock
import logging

# dev
import numpy as np
from icecream import ic

logger = logging.getLogger("Jaeger")
ic.configureOutput(prefix="Jaeger |")

def load_model_config(path: Path) -> Dict:
    '''
    loads the configuration file from the template
    '''
    
    with open(path) as fp:
        _data = yaml.safe_load(fp)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=path.parent))
    template = env.get_template(path.name)

    data = yaml.safe_load(template.render(_data))

    return data

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
        self.outputs = []
        self._fragment_paths = self._get_fragment_paths()
        self._contig_paths = self._get_contig_paths()
        self._reliability_fragment_paths = self._get_reliability_fragment_paths()
        self._reliability_contig_paths = self._get_reliability_contig_paths()
        ic(self._fragment_paths)
        # ic(self._reliability_fragment_paths)
        self._saving_config = self._get_model_saving_configuration()
        self.optimizer = None
        self.loss_classifier = None
        self.loss_reliability = None
        self.training_epochs = config["training"].get("classifier_epochs")
        self.reliability_epochs = config["training"].get("reliability_epochs")
        self._load_training_params()

        cb_list = config["training"].get("callbacks", dict())
        for p in cb_list.get("directories"):
            p =Path(p)
            if p.exists() and cb_list.get("clean_old"):
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

        match config.get("activation", "gelu"):
            case "gelu":
                self.Activation = GeLU
            case "relu":
                self.Activation = ReLU

    def build_fragment_classifier(self):
        # === 1. EMBEDDING ===
        if "embedding" in self.model_cfg:
            inputs, x = self._build_embedding(self.model_cfg["embedding"])
            self.inputs = inputs
        else:
            raise ValueError("Missing 'embedding' section in config")

        # === 2. REPRESENTATION LEARNER ===
        if "representation_learner" in self.model_cfg:
            r, r_lbn = self._build_representation_learner(x, self.model_cfg["representation_learner"])

        # === 3. CLASSIFIER ===
        if "classifier" in self.model_cfg:
            x_classifier = self._build_classifier(r, self.model_cfg["classifier"])

        # === 4. RELIABILITY ===
        if "reliability_model" in self.model_cfg:
            x_reliability = self._build_reliability_model(r_lbn, self.model_cfg["reliability_model"])

        self.outputs = {'classifier': x_classifier, 
                        'reliability': x_reliability
                        }

        return tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name="JaegerModel")
    
    def build_contig_classifier(self):
        '''
        to do: contig consensus prediction model
        currently, the final predictions per-contig is obtained by averaing the
        per-fragment logits. Instead, we can learn a function to combine information
        from all fragments. 
        '''
        pass

    def _build_embedding(self, cfg: Dict[str, Any]):
        '''
        creates the embedding layer
        '''
        input_shape = cfg.get("input_shape", (6, None, 64))
        embedding_size = cfg.get("embedding_size", 4)

        inputs = tf.keras.Input(shape=input_shape, name=cfg.get('type'))
        masked_inputs = tf.keras.layers.Masking(name="input_mask", mask_value=0.0)(inputs)

        match cfg.get('type'):
            case "translated":
                x = tf.keras.layers.Dense(embedding_size,
                                          name=f'{cfg.get('type')}_embedding',
                                          use_bias=False,
                                          kernel_initializer=tf.keras.initializers.Orthogonal())(masked_inputs)
            case "nucleotide":
                 x = masked_inputs
            case _:
                raise ValueError(f"{cfg.get('type')} is invalid")
        return inputs, x

    def _build_representation_learner(self, x, cfg: Dict[str, Any]):
        '''
        X -> [X'-1, X']
        '''
        x = MaskedConv1D(filters=cfg.get("masked_conv1d_1_filters"),
                         kernel_size=cfg.get("masked_conv1d_1_kernel_size"),
                         strides=cfg.get("masked_conv1d_1_strides"),
                         dilation_rate=cfg.get("masked_conv1d_1_dilation_rate"),
                         use_bias=False,
                         name="masked_conv1d_1",
                         activation = None,
                         kernel_regularizer=tf.keras.regularizers.L2(1e-6),
                         kernel_initializer=tf.keras.initializers.HeUniform())(x)
            # using batchnorm here gives a big advantage. You get a model that can work well with different input size.
            # infact the accuracy increase with the increasing input size.

        x = MaskedBatchNorm(name="masked_batchnorm_1")(x)
        x = self.Activation(name="activation_1")(x)

        for block, (block_size, block_filter, ksize, kdilation, kstride) in enumerate(zip(cfg.get("block_sizes", [3, 3, 3]),
                                                               cfg.get("block_filters", [128, 256, 512]), 
                                                               cfg.get("block_kernel_size", [5, 5, 5]),
                                                               cfg.get("block_kernel_dilation", [1, 1, 1]),
                                                               cfg.get("block_kernel_strides", [2, 2, 2]),
                                                               ), start=1):
            # ========== blockn (compress -> res) =============
            x = MaskedConv1D(filters=block_filter,
                            kernel_size=ksize,
                            strides=kstride,
                            dilation_rate=kdilation,
                            use_bias=False,
                            name=f"ds_masked_conv1d_{block}",
                            kernel_regularizer=tf.keras.regularizers.L2(1e-6),
                            activation = None,
                            kernel_initializer=tf.keras.initializers.HeUniform())(x)
            x = MaskedBatchNorm(name=f"ds_masked_batchnorm_{block}")(x)
            x = self.Activation(name=f"ds_activation_{block}")(x)

            for i in range(block_size):
                x = ResidualBlock(block_filter,
                                  kernel_size=ksize,
                                  block_number=f"{block}{i}",
                                  name=f"masked_resblock_{block}{i}")(x)

        # ============ final block ============
        x = MaskedConv1D(filters=block_filter,
                            kernel_size=cfg.get("masked_conv1d_final_kernel_size"),
                            strides=cfg.get("masked_conv1d_final_strides"),
                            dilation_rate=cfg.get("masked_conv1d_final_dilation_rate"),
                            use_bias=False,
                            name="masked_conv1d_final",
                            kernel_regularizer=tf.keras.regularizers.L2(1e-6),
                            activation = None,
                            kernel_initializer=tf.keras.initializers.HeUniform())(x)
        # this layers mean vector is used as u[train] to calculate nmd u[example] - u[train]
        x, nmd = MaskedBatchNorm(name="masked_batchnorm_final", return_nmd=True)(x) 
        x = self.Activation(name="activation_final")(x)
        # =========== Aggregation ==============
        x = self._get_pooler(cfg.get('pooling'))(name=f"global_{cfg.get('pooling')}pool")(x)
        return x, nmd

    def _build_classifier(self, x, cfg: Dict[str, Any]):
        '''
        X -> num_classes
        '''
        x = tf.keras.layers.Dense(cfg.get("dense_1_units"),
                        name=f'classifier_dense_1',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-5),
                        use_bias=False)(x)
        x = self.Activation(name=f'classifier_activation_1')(x)
        x = tf.keras.layers.Dense(cfg.get('classes'),
                        activation=None,
                        name=f'classifier',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-5),
                        use_bias=False)(x)
        return x

    def _build_reliability_model(self, x_lbn, cfg: Dict[str, Any]):
        '''
        X -> 1
        '''
        # ============reliability=================
        # reliabily model is inspired by neural mean discrepancy method
        x = tf.keras.layers.Dense(cfg.get('dense_1_units'),
                                name='reliability_dense_1',
                                kernel_regularizer=tf.keras.regularizers.L2(1e-5),
                                use_bias=True)(x_lbn)
        x = self.Activation(name='reliability_activation_1')(x)
        reliability = tf.keras.layers.Dense(1,
                                activation=None,
                                name='reliability',
                                kernel_regularizer=tf.keras.regularizers.L2(1e-5),
                                use_bias=True)(x)
        return reliability
    
    def _build_projector(self, x, cfg: Dict[str, Any]):
        '''
        X1 -> X2
        '''
        reg_map = {"l1": tf.keras.regularizers.l1, "l2": tf.keras.regularizers.l2}

        dense_1 = cfg.get("dense_1", 32)
        dense_2 = cfg.get("dense_2", 16)

        reg1 = reg_map.get(cfg.get("kernel_regularizer_1"), lambda: None)(cfg.get("kernel_regularizer_1_w"))
        reg2 = reg_map.get(cfg.get("kernel_regularizer_2"), lambda: None)(cfg.get("kernel_regularizer_2_w"))

        # ============ projection =============================
        # this projection can be used for supervised contrastive learning
        # batch-norm and dropout do not play nicely together https://doi.org/10.48550/arXiv.1801.05134
        x = tf.keras.layers.Dense(dense_1,
                                name='projection_dense_1',
                                kernel_regularizer=reg1,
                                use_bias=False)(x)
        x = self.Activation(name='projection_activation_1')(x)
        x = tf.keras.layers.Dropout(cfg.get('dropout_rate'))(x)

        representation = tf.keras.layers.Dense(dense_2,
                                    kernel_regularizer=reg2,
                                    name='projection_dense_2',
                                    use_bias=False,
                                    dtype='float32')(x)
        return representation
    
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
        ic(opt_params)
        loss_classifier_name = self.train_cfg.get("loss_classifier", "categorical_crossentropy").lower()
        loss_classifier_params = self.train_cfg.get("loss_params_classifier", {})
        self.loss_classifier = self._get_loss(loss_classifier_name, loss_classifier_params)

        loss_reliability_name = self.train_cfg.get("loss_reliability", "binary_crossentropy").lower()
        loss_reliability_params = self.train_cfg.get("loss_params_reliability", {})
        self.loss_reliability = self._get_loss(loss_reliability_name, loss_reliability_params)
        ic(self.loss_reliability)

    def switch_branch(self, model, train_branch="classifier"):
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        self.optimizer = self._get_optimizer(opt_name, opt_params)
        if train_branch == "classifier":
            # Freeze reliability module
            for layer in model.layers:
                if "reliability" in layer.name:
                    layer.trainable = False
                else:
                    layer.trainable = True

            loss_weights = {"classifier": 1.0, "reliability": 0.0}

        elif train_branch == "reliability":
            # Freeze classifier and representation learner
            for layer in model.layers:
                if "reliability" not in layer.name:
                    layer.trainable = False
                else:
                    layer.trainable = True

            loss_weights = {"classifier": 0.0, "reliability": 1.0}
        else:
            raise ValueError("train_branch must be 'classifier' or 'reliability'")

        model.compile(
            optimizer=self.optimizer,
            loss={
                "classifier": self.loss_classifier,
                "reliability": self.loss_reliability,
                },
            loss_weights=loss_weights,
            metrics={
                "classifier": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
                "reliability": [tf.keras.metrics.AUC(name="auc", from_logits=True)]
                    }
        )
        ic(f"model compiled for {train_branch}")

    def save_model(self, model, suffix=None):
        '''
        saves models (graph, weights or both) to the output directory
        '''
        path = Path(self._saving_config.get("path"))
        path.mkdir(parents=True, exist_ok=True)
        model_name = self.model_cfg.get("name")
        if suffix:
            model_name += f"_{suffix}"

        if self._saving_config.get("save_weights"):
            model.save_weights(path / f"{model_name}.weights.h5")
            ic(f"model weights are written to {path / f"{model_name}.weights.h5"}")

        if self._saving_config.get("save_exec_graph"):
            # this way, you don't need the model configuration to rebuild the model
            tf.saved_model.save(model, path / f"{model_name}_graph" )
            ic(f"model computational graph is written to {path / f"{model_name}_graph"}")
        # save output indices -> class mapping in the same directory
        with open(path / f'{model_name}_classes.yaml', 'w') as yaml_file:
            ic("writing class_labels_map")
            yaml.dump(dict(classes=self.model_cfg.get("class_labels_map")), yaml_file, default_flow_style=False)
    
    def save_config(self):
        '''
        saves project config to the model output directory
        '''
        path = Path(self._saving_config.get("path"))
        model_name = self.model_cfg.get("name")
        with open(path / f'{model_name}_project.yaml', 'w+') as yaml_file:
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
            "CODON" : CODONS,
            "CODON_ID": CODON_ID,
            "AA_ID": AA_ID,
            "MURPHY10_ID": MURPHY10
        }
        emb_config = self.model_cfg.get("embedding")
        sp_config = self.model_cfg.get("string_processor")

        _config = {"input_type": emb_config.get("type"),
                     "codon" : _map.get(sp_config.get("codon")),
                     "codon_id": _map.get(sp_config.get("codon_id")),
                     "codon_depth": max(_map.get(sp_config.get("codon_id")))+1,
                     "crop_size": sp_config.get("crop_size")
                     }
        
        return _config

    def _get_optimizer(self, name, kwargs) -> Any:
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "adagrad": tf.keras.optimizers.Adagrad
        }
        return optimizers[name](**kwargs)

    def _get_pooler(self, name):
        poolers = {
            "max": tf.keras.layers.GlobalMaxPooling2D,
            "average": tf.keras.layers.GlobalAveragePooling2D
        }
        return poolers[name]

    def _get_loss(self, name, kwargs) -> Any:
        losses = {
            "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
            "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy,
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy,
            "mse": tf.keras.losses.MeanSquaredError
        }
        return losses[name](**kwargs)
    
    def _get_paths(self, key:str) -> Dict:
        fcd_dict = self.train_cfg.get(key, {})
        paths = {} # train:[], validation:[]
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

    '''
    trains fragment classification model and reliability prediction model.
    '''
    ic(kwargs.get("config"))
    config = load_model_config(Path(kwargs.get("config")))
    # Initialize the model
    builder = DynamicModelBuilder(config)
    model = builder.build_fragment_classifier()
    model.summary()
    # =================train classifier ======================
    builder.switch_branch(model, train_branch="classifier")

    for i in model.layers:
        ic(i.name, i.trainable)

    # =================load train data ======================
    string_processor_config = builder._get_string_processor_config()
    _train_data = builder._get_fragment_paths()
    train_data = {"train":None, "validation": None}

    for k,v in _train_data.items():
        ic(k, v)
        _data = tf.data.TextLineDataset(v.get("paths"),
                                        num_parallel_reads=len(v.get("paths")),
                                        buffer_size=200)
        
        train_data[k]=_data.map(process_string_train(
                                                        codons=string_processor_config.get("codon"),
                                                        codon_num=string_processor_config.get("codon_id"),
                                                        codon_depth=string_processor_config.get("codon_depth"),
                                                        crop_size=string_processor_config.get("crop_size"),
                                                        input_type=string_processor_config.get("input_type"),
                                                        num_classes=builder.config.get("classifier").get("classes"),
                                                        class_label_onehot=True),
                        num_parallel_calls=tf.data.AUTOTUNE)\
                    .shuffle(buffer_size=300)\
                    .batch(builder.train_cfg.get("batch_size"), drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)

    model.fit(train_data.get("train").take(10_000),
              validation_data=train_data.get("validation").take(1_000),
              epochs=builder.training_epochs,
              callbacks=builder.get_callbacks(branch="classifier"))


    # ============== reliability model ========================
    builder.switch_branch(model, train_branch="reliability")

    _rel_train_data = builder._get_reliability_fragment_paths()
    rel_train_data = {"train":None, "validation": None}
    for k,v in _rel_train_data.items():
        _data = tf.data.TextLineDataset(v.get("paths"), 
                                        num_parallel_reads=len(v.get("paths")),
                                        buffer_size=200)   
        rel_train_data[k] = _data.map(process_string_train(
                                                        codons=string_processor_config.get("codon"),
                                                        codon_num=string_processor_config.get("codon_id"),
                                                        codon_depth=string_processor_config.get("codon_depth"),
                                                        crop_size=string_processor_config.get("crop_size"), 
                                                        input_type=string_processor_config.get("input_type"),
                                                        num_classes=builder.config.get("classifier").get("classes"),
                                                        label_type='reliability',
                                                        class_label_onehot=True),
                        num_parallel_calls=tf.data.AUTOTUNE)\
                    .batch(builder.train_cfg.get("batch_size"),drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)


    model.fit(rel_train_data.get("train").take(10_000),
              validation_data=rel_train_data.get("validation").take(1_0000),
              epochs=builder.reliability_epochs,
              callbacks=builder.get_callbacks(branch="reliability"))
    # ============= saving ===================================
    builder.save_model(model=model, suffix="fragment")
    builder.save_config()


def train_contig_core(**kwargs):
    '''
    to do: contig consensus prediction model
    currently, the final predictions per-contig is obtained by averaing the
    per-fragment logits. Instead, we can learn a function to combine information
    from all fragments. 
    '''
    with open(kwargs.get("config"), "r") as f:
        config = yaml.safe_load(f)

#if '__main__' == __name__:
    """
    for testing only
    """

    # config = load_model_config(Path('/Users/javis/Documents/Programming/Jaeger/src/commands/configs/nn_config.yaml'))
    # #ic(config.get("training"))
    # def get_random_example(batch=10, size=100, channels=64):
    #     x = np.eye(channels)
    #     seqs = []
    #     for i in range(batch):
    #         seq = np.stack([x[np.random.choice(x.shape[0], size=size)] for _ in range(6)])
    #         seqs.append(seq)
    #     return np.stack(seqs)
    
    # # Initialize the model
    # builder = DynamicModelBuilder(config)
    # model = builder.build_fragment_classifier()
    
    # model.summary()
    # # =================train classifier ======================
    # builder.switch_branch(model, train_branch="classifier")

    # for i in model.layers:
    #     ic(i.name, i.trainable)
    # debug_data = '/Users/javis/Documents/Programming/Jaeger/data/val_data_1000.txt'
    # traning_data = tf.data.TextLineDataset([debug_data], num_parallel_reads=1, buffer_size=200)
    # string_processor_config = builder._get_string_processor_config()
    # from jaeger.preprocess.latest.convert import process_string_train
    # from jaeger.preprocess.latest.maps import CODONS, CODON_ID, AA_ID
    # traning_data=traning_data.map(process_string_train(
    #                                                    codons=string_processor_config.get("codon"),
    #                                                    codon_num=string_processor_config.get("codon_id"),
    #                                                    codon_depth=string_processor_config.get("codon_depth"),
    #                                                    crop_size=string_processor_config.get("crop_size"),
    #                                                    input_type=string_processor_config.get("input_type"),
    #                                                    class_label_onehot=True),
    #                    num_parallel_calls=tf.data.AUTOTUNE)\
    #               .batch(16,drop_remainder=True)\
    #               .prefetch(tf.data.AUTOTUNE).repeat()
    # # for i in traning_data.take(1):
    # #     ic(i)
    # ic(model.evaluate(traning_data.take(1)))
    # model.fit(traning_data.take(64),
    #           epochs=builder.training_epochs,
    #           callbacks=builder.get_callbacks(branch="classifier"))
    # #ic(model(get_random_example(batch=10, size=213)))
    # ic("testing post training (classification model): classifier")
    # ic(model.evaluate(traning_data.take(64)))

    # for xitr,yitr in traning_data.take(1):
    #     ic("debug classifier train data")
    #     #ic(xitr['translated'].shape)
    #     ic(yitr['reliability'].shape)
    #     ic(yitr['classifier'].shape)

    # # ============== reliability model ========================
    # builder.switch_branch(model, train_branch="reliability")

    # for i in model.layers:
    #     ic(i.name, i.trainable)

    # debug_rel_data = '/Users/javis/Documents/Programming/Jaeger/data/val_data_shuf_1000.txt'
    # rel_traning_data = tf.data.TextLineDataset([debug_rel_data], num_parallel_reads=1, buffer_size=200)
    # from jaeger.preprocess.latest.convert import process_string_train
   
    # rel_traning_data=rel_traning_data.map(process_string_train(
    #                                                    codons=string_processor_config.get("codon"),
    #                                                    codon_num=string_processor_config.get("codon_id"),
    #                                                    codon_depth=string_processor_config.get("codon_depth"),
    #                                                    crop_size=string_processor_config.get("crop_size"), 
    #                                                    input_type=string_processor_config.get("input_type"),
    #                                                    label_type='reliability',
    #                                                    class_label_onehot=True),
    #                    num_parallel_calls=tf.data.AUTOTUNE)\
    #               .batch(16,drop_remainder=True)\
    #               .prefetch(tf.data.AUTOTUNE).repeat()

    # for xitr, yitr in rel_traning_data.take(1):
    #     ic("debug reliability model train data")
    #     #ic(xitr['translated'].shape)
    #     ic(yitr['reliability'].shape)
    #     ic(yitr['classifier'].shape)

    # model.fit(rel_traning_data.take(64),
    #           epochs=builder.reliability_epochs,
    #           callbacks=builder.get_callbacks(branch="reliability"))
    
    # ic("testing post training (reliability model): classifier")
    # ic(model.evaluate(traning_data.take(64)))
    # ic("testing post training: reliability model")
    # ic(model.evaluate(rel_traning_data.take(64)))
    # #ic(model(get_random_example(batch=10, size=213, channels=21)))

    # builder.save_model(model=model)

    # load and text the saved model (2x faster than the one below)
    # ic("benchmarking the saved_model")
    # x = get_random_example(batch=16)
    # loaded_model = tf.saved_model.load('/Users/javis/Documents/Programming/Jaeger/data/test_model/jaeger_1.5M_graph')
    # inference_fn = loaded_model.signatures["serving_default"]
    # ic(timeit.timeit(lambda : inference_fn(tf.constant(x, dtype=tf.float32)), number=100)/100)

    # # test the default model
    # ic("benchmarking the default model")
    # ic(timeit.timeit(lambda : model(x), number=100)/100)

    