import tensorflow as tf

from jaeger.commands.train import DynamicModelBuilder

config = {
    "model": {
        "name": "jaeger",
        "seed": 434,
        "classifier_out_dim": 6,
        "reliability_out_dim": 1,
        "activation": "gelu",
        "embedding": {
            "use_embedding_layer": True,
            "input_type": "translated",
            "frames": 6,
            "input_shape": [6, None],
            "embedding_size": 32,
            "embedding_regularizer": "l2",
            "embedding_regularizer_w": 1e-5,
        },
        "string_processor": {
            "seq_onehot": False,
            "codon": "CODON",
            "codon_id": "CODON_ID",
            "crop_size": 1000,
            "buffer_size": 100,
            "reshuffle_each_iteration": True,
            "shuffle": False,
            "masking": False,
            "classifier_labels": [0, 1, 2, 3, 4, 5],
        },
        "representation_learner": {
            "hidden_layers": [
                {
                    "name": "masked_conv1d",
                    "config": {
                        "filters": 32,
                        "kernel_size": 7,
                        "strides": 1,
                        "dilation_rate": 1,
                        "use_bias": True,
                        "activation": None,
                        "kernel_regularizer": "l2",
                        "kernel_regularizer_w": 1e-5,
                    },
                },
                {
                    "name": "masked_batchnorm",
                    "config": {"return_nmd": False},
                },
                {
                    "name": "activation",
                    "config": {"activation": "gelu"},
                },
                {
                    "name": "residual_block",
                    "config": {
                        "block_size": 2,
                        "filters": 32,
                        "kernel_size": 5,
                        "strides": 1,
                        "dilation_rate": 1,
                        "use_bias": True,
                        "activation": "gelu",
                        "kernel_regularizer": "l2",
                        "kernel_regularizer_w": 1e-5,
                    },
                },
                {
                    "name": "residual_block",
                    "config": {
                        "block_size": 2,
                        "filters": 32,
                        "kernel_size": 9,
                        "strides": 1,
                        "dilation_rate": 1,
                        "use_bias": False,
                        "activation": "gelu",
                        "kernel_regularizer": "l2",
                        "kernel_regularizer_w": 1e-4,
                        "return_nmd": True,
                    },
                },
            ],
            "pooling": "max",
        },
        "classifier": {
            "input_shape": 32,
            "hidden_layers": [
                {
                    "name": "dense",
                    "config": {
                        "units": 32,
                        "activation": "gelu",
                        "use_bias": True,
                        "kernel_regularizer": "l2",
                        "kernel_regularizer_w": 1e-4,
                    },
                },
                {"name": "dropout", "config": {"rate": 0.1}},
                {
                    "name": "dense",
                    "config": {
                        "units": 6,
                        "activation": None,
                        "use_bias": True,
                    },
                },
            ],
        },
        "reliability_model": {
            "input_shape": 32,
            "hidden_layers": [
                {
                    "name": "dense",
                    "config": {
                        "units": 8,
                        "activation": "gelu",
                        "use_bias": True,
                        "kernel_regularizer": "l2",
                        "kernel_regularizer_w": 1e-5,
                    },
                },
                {"name": "dropout", "config": {"rate": 0.1}},
                {
                    "name": "dense",
                    "config": {
                        "units": 1,
                        "activation": None,
                        "use_bias": True,
                    },
                },
            ],
        },
    },
    "training": {
        "optimizer": "sgd",
        "optimizer_params": {
            "learning_rate": 3e-4,
            "clipnorm": 20,
            "momentum": 0.9,
        },
        "loss_classifier": "categorical_crossentropy",
        "loss_params_classifier": {"from_logits": True},
        "loss_reliability": "binary_crossentropy",
        "loss_params_reliability": {"from_logits": True},
        "batch_size": 4,
        "callbacks": {"directories": []},
        "model_saving": {
            "path": "/tmp/jaeger_test_model",
            "save_weights": False,
            "save_exec_graph": False,
        },
    },
}

builder = DynamicModelBuilder(config)
models = builder.build_fragment_classifier()

model = models["jaeger_model"]

# Shifted token IDs: 0 = padding/mask, 1..64 = codons.
x = tf.random.uniform(
    shape=[4, 6, 120],
    minval=0,
    maxval=65,
    dtype=tf.int32,
)

out = model(x, training=False)

print("Output keys:", out.keys())
print("prediction:", out["prediction"].shape)
print("reliability:", out["reliability"].shape)
print("embedding:", out["embedding"].shape)
print("nmd:", out["nmd"].shape)

assert out["prediction"].shape == (4, 6)
assert out["reliability"].shape == (4, 1)
assert out["embedding"].shape == (4, 32)
assert out["nmd"].shape == (4, 32)

builder.compile_model(models, train_branch="classifier")
builder.compile_model(models, train_branch="reliability")

print("All checks passed.")
