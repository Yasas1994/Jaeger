import os
# temporary fix
os.environ['WRAPT_DISABLE_EXTENSIONS'] = "true" 
from preprocess.v2.convert import process_string
import yaml
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, regularizers
from typing import Any, Dict, List, Optional
from nnlib.v2.layers import GeLU, ReLU, MaskedAdd, MaskedBatchNorm, MaskedConv1D, ResidualBlock
import logging

# dev
import numpy as np
from icecream import ic

logger = logging.getLogger("Jaeger")
class DynamicModelBuilder:
    """
    to do: implement feature correlation based out-of-distribution detection
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config["model"]
        self.train_cfg = config["training"]
        self.inputs = None
        self.outputs = []
        self._fragment_paths = self._get_fragment_paths()
        self._contig_paths = self._get_contig_paths()
        self._saving_config = self._get_model_saving_configuration()
        self.optimizer = None
        self.loss_classifier = None
        self.loss_reliability = None
        self._load_training_params()
        match config.get("activation", "gelu"):
            case "gelu":
                self.Activation = GeLU
            case "relu":
                self.Activation = ReLU

    def build_fragment_classifier(self):
        # === 1. EMBEDDING ===
        if "embedding" in self.config:
            inputs, x = self._build_embedding(self.config["embedding"])
            self.inputs = inputs
        else:
            raise ValueError("Missing 'embedding' section in config")

        # === 2. REPRESENTATION LEARNER ===
        if "representation_learner" in self.config:
            r, r_lbn = self._build_representation_learner(x, self.config["representation_learner"])

        # === 3. CLASSIFIER ===
        if "classifier" in self.config:
            x_classifier = self._build_classifier(r, self.config["classifier"])

        # === 4. RELIABILITY ===
        if "reliability_model" in self.config:
            x_reliability = self._build_reliability_model(r_lbn, self.config["reliability_model"])

        self.outputs = {'classifier': x_classifier, 'reliability': x_reliability}

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
            # case "nucleotide":
            #     x = masked_inputs
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
                         kernel_regularizer=tf.keras.regularizers.L2(1e-4),
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
                            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
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
                            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
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
        reg_map = {"l1": regularizers.l1, "l2": regularizers.l2}

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
                "classifier": [tf.keras.metrics.CategoricalAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc", from_logits=True)],
                "reliability": [tf.keras.metrics.AUC(name="auc", from_logits=True)]
                    }
        )
        ic(f"model compiled for {train_branch}")

    def save_model(self, model):
        path = Path(self._saving_config.get("path"))
        model_name = self.config.get("name")

        if self._saving_config.get("save_weights"):
            model.save_weights(path / f"{model_name}.weights.h5")
            ic(f"model weights are written to {path / f"{model_name}.weights.h5"}")

        if self._saving_config.get("save_exec_graph"):
            # this way, you don't need the model configuration to rebuild the model
            tf.saved_model.save(model, path / f"{model_name}_graph" )
            ic(f"model computational graph is written to {path / f"{model_name}_graph"}")
        # save output indices -> class mapping in the same directory
        with open(path / f'{model_name}.yaml', 'w') as yaml_file:
            ic("writing class_labels_map")
            yaml.dump(dict(classes=self.config.get("class_labels_map")), yaml_file, default_flow_style=False)

    def get_callbacks(self) -> List:
        cb_list = self.train_cfg.get("callbacks", [])
        callbacks = []
        for cb in cb_list:
            name = cb.get("name")
            params = cb.get("params", {})
            try:
                cb_class = getattr(tf.keras.callbacks, name)
                callbacks.append(cb_class(**params))
            except AttributeError:
                raise ValueError(f"Unsupported callback: {name}")
        return callbacks

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
            tmp = []
            for i in fcd_v:
                class_ = i.get("class")
                label_ = i.get("label")
                path_ =  i.get("path")
                tmp.append(path_)
        paths[fcd_k] = tmp
        return paths       
    def _get_model_saving_configuration(self):
        return self.train_cfg.get("model_saving", {})
    
    def _get_fragment_paths(self) -> Dict:
        return self._get_paths("fragment_classifier_data")


    def _get_contig_paths(self):
        return self.train_cfg.get("contig_classifier_data")


    def _get_shuffled_fragment_paths(self):
        return self.train_cfg.get("fragment_reliability_data")


    def _get_shuffled_contig_paths(self):
        return self.train_cfg.get("contig_reliability_data")

def train_fragment_core(**kwargs):

    '''
    trains fragment classification model and reliability prediction model.
    '''
    with open(kwargs.get("config"), "r") as f:
        config = yaml.safe_load(f)

    # Initialize the model
    builder = DynamicModelBuilder(config)
    model = builder.build_fragment_classifier()
    # compile the classifier
    model = builder.compile_for_branch(model, train_branch="classifier")
    model.summary()
    # get callbacks
    callbacks = builder.get_callbacks()

    traning_data = tf.data.TextLineDataset(builder._fragment_paths.get('train'), num_parallel_reads=5, buffer_size=200)
    traning_data=traning_data.map(process_string(maxval=100,crop_size=100),
                       num_parallel_calls=tf.data.AUTOTUNE)\
                  .batch(16,drop_remainder=True)\
                  .prefetch(tf.data.AUTOTUNE)
        
    traning_data = traning_data.cache()

    #initialize validation data gennerator 
    validation_data = tf.data.TextLineDataset(builder._fragment_paths.validation, 
                                              num_parallel_reads=len(config.validation_data_paths),
                                                buffer_size=200)
    validation_data = validation_data.map(process_string(maxval=100,crop_size=100),
                         num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(10)

    # fit fragment prediction model
    model.fit(traning_data, validation_data, epochs=10, callbacks=callbacks)

    # compile reliability model
    model = builder.compile_for_branch(model, train_branch="reliability")
    model.summary()
    # train reliability model

def train_contig_core(**kwargs):
    '''
    to do: contig consensus prediction model
    currently, the final predictions per-contig is obtained by averaing the
    per-fragment logits. Instead, we can learn a function to combine information
    from all fragments. 
    '''
    with open(kwargs.get("config"), "r") as f:
        config = yaml.safe_load(f)

if '__main__' == __name__:
    import timeit
    with open('/Users/javis/Documents/Programming/Jaeger/src/commands/configs/nn_config.yaml', "r") as f:
        config = yaml.safe_load(f)

    def get_random_example(batch=10, size=100):
        x = np.eye(64)
        seqs = []
        for i in range(batch):
            seq = np.stack([x[np.random.choice(x.shape[0], size=size)] for _ in range(6)])
            seqs.append(seq)
        return np.stack(seqs)
    
    # Initialize the model
    builder = DynamicModelBuilder(config)
    model = builder.build_fragment_classifier()

    model.summary()
    # =================train classifier ======================
    builder.switch_branch(model, train_branch="classifier")

    for i in model.layers:
        ic(i.name, i.trainable)
    debug_data = '/Users/javis/Documents/Programming/Jaeger/data/val_data_1000.txt'
    traning_data = tf.data.TextLineDataset([debug_data], num_parallel_reads=1, buffer_size=200)
    from preprocess.latest.convert import process_string_train
    traning_data=traning_data.map(process_string_train(crop_size=1024, class_label_onehot=True),
                       num_parallel_calls=tf.data.AUTOTUNE)\
                  .batch(16,drop_remainder=True)\
                  .prefetch(tf.data.AUTOTUNE).repeat()

    ic(model.predict(traning_data.take(1)))
    model.fit(traning_data.take(1), epochs=10)
    ic(model(get_random_example(batch=10, size=213)))

    for xitr,yitr in traning_data.take(1):
        ic("debug classifier train data")
        ic(xitr['translated'].shape)
        ic(yitr['reliability'].shape)
        ic(yitr['classifier'].shape)

    # ============== reliability model ========================
    builder.switch_branch(model, train_branch="reliability")

    for i in model.layers:
        ic(i.name, i.trainable)

    debug_rel_data = '/Users/javis/Documents/Programming/Jaeger/data/val_data_shuf_1000.txt'
    traning_data = tf.data.TextLineDataset([debug_rel_data], num_parallel_reads=1, buffer_size=200)
    from preprocess.latest.convert import process_string_train
    traning_data=traning_data.map(process_string_train(crop_size=1024, 
                                                       label_type='reliability',
                                                       class_label_onehot=True),
                       num_parallel_calls=tf.data.AUTOTUNE)\
                  .batch(16,drop_remainder=True)\
                  .prefetch(tf.data.AUTOTUNE).repeat()

    for xitr, yitr in traning_data.take(1):
        ic("debug reliability model train data")
        ic(xitr['translated'].shape)
        ic(yitr['reliability'].shape)
        ic(yitr['classifier'].shape)

    model.fit(traning_data.take(1), epochs=10)

    ic(model.predict(traning_data.take(1)))
    ic(model(get_random_example(batch=10, size=213)))

    builder.save_model(model=model)

    # load and text the saved model (2x faster than the one below)
    ic("benchmarking the saved_model")
    x = get_random_example(batch=16)
    loaded_model = tf.saved_model.load('/Users/javis/Documents/Programming/Jaeger/data/test_model/jaeger_graph')
    inference_fn = loaded_model.signatures["serving_default"]
    ic(timeit.timeit(lambda : inference_fn(tf.constant(x, dtype=tf.float32)), number=100)/100)

    # test the default model
    ic("benchmarking the default model")
    ic(timeit.timeit(lambda : model(x), number=100)/100)

    