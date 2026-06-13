#!/usr/bin/env python3
"""Clean evaluation without checkpoint interference."""

import sys

sys.path.insert(0, "src")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

from jaeger.utils.misc import load_model_config
from jaeger.seqops.encode import process_string_train
from jaeger.seqops.maps import CODONS, CODON_ID

CLASS_NAMES = ["chromosome", "virus", "plasmid"]


def load_model(config_path, checkpoint_path):
    """Load model without touching checkpoint dirs."""
    config = load_model_config(Path(config_path))

    # Manually build model without DynamicModelBuilder's checkpoint logic
    from jaeger.nnlib.v2.layers import (
        MaskedConv1D,
        MaskedBatchNorm,
        ResidualBlock_wrapper,
    )

    # Get model config
    model_cfg = config["model"]
    embedding_cfg = model_cfg.get("embedding", {})
    rep_cfg = model_cfg.get("representation_learner", {})
    cls_cfg = model_cfg.get("classifier", {})

    # Build inputs
    input_type = embedding_cfg.get("input_type", "translated")
    use_embedding = embedding_cfg.get("use_embedding_layer", True)

    if input_type == "translated" and use_embedding:
        inputs = tf.keras.Input(shape=(6, None), dtype=tf.int32, name="translated")
        x = tf.keras.layers.Embedding(
            65,  # codon_depth + 1 (since seq+1)
            embedding_cfg.get("embedding_size", 64),
            name="embedding",
        )(inputs)
    else:
        raise NotImplementedError("Only translated + embedding supported")

    # Build representation learner
    mask = tf.keras.layers.Lambda(lambda x: tf.not_equal(x, 0), name="mask")(inputs)

    for layer_cfg in rep_cfg.get("hidden_layers", []):
        name = layer_cfg.get("name", "").lower()
        cfg = dict(layer_cfg.get("config", {}))

        if name == "masked_conv1d":
            cfg.pop("kernel_regularizer_w", None)
            cfg.pop("kernel_regularizer", None)
            x = MaskedConv1D(**cfg)(x, mask=mask)
        elif name == "masked_batchnorm":
            x = MaskedBatchNorm(**cfg)(x)
        elif name == "activation":
            x = tf.keras.layers.Activation(**cfg)(x)
        elif name == "residual_block":
            block_size = cfg.pop("block_size", 1)
            filters = cfg.get("filters", 32)
            cfg.pop("kernel_regularizer_w", None)
            cfg.pop("kernel_regularizer", None)
            shape = (6, None, filters)
            x = ResidualBlock_wrapper(block_size, shape, **cfg)(x)
        elif name == "cross_frame_attention":
            from jaeger.nnlib.v2.layers import CrossFrameAttention

            x = CrossFrameAttention(**cfg)(x)
        elif name == "axial_attention":
            from jaeger.nnlib.v2.layers import AxialAttention

            x = AxialAttention(**cfg)(x)

    # Pooling
    pooling = rep_cfg.get("pooling", "average").lower()
    if pooling == "average":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Classifier
    for layer_cfg in cls_cfg.get("hidden_layers", []):
        name = layer_cfg.get("name", "").lower()
        cfg = dict(layer_cfg.get("config", {}))

        if name == "dropout":
            x = tf.keras.layers.Dropout(**cfg)(x)
        elif name == "dense":
            cfg.pop("kernel_regularizer_w", None)
            cfg.pop("kernel_regularizer", None)
            cfg.pop("bias_initializer", None)
            x = tf.keras.layers.Dense(**cfg)(x)

    outputs = {"prediction": x, "embedding": x}
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="jaeger_model")

    # Load weights
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Loading weights from: {checkpoint_path}")
        model.load_weights(checkpoint_path)

    return model


def create_dataset(csv_path, crop_size, max_samples=5000, batch_size=256):
    df = pd.read_csv(csv_path, header=None, names=["label", "seq"])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    csv_strings = [f"{label},{seq}" for label, seq in zip(df["label"], df["seq"])]
    labels = df["label"].values.astype(np.int32)

    processor = process_string_train(
        codons=CODONS,
        codon_num=CODON_ID,
        seq_onehot=False,
        crop_size=crop_size,
        timesteps=False,
        mutate=False,
        mutation_rate=0.0,
        shuffle=False,
        masking=False,
        class_label_onehot=False,
        ngram_width=3,
    )

    def _parse(csv_str):
        out, _ = processor(csv_str)
        return out["translated"]

    dataset = tf.data.Dataset.from_tensor_slices(csv_strings)
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, labels


def evaluate(model, dataset, true_labels, length, model_name):
    print(f"\nEvaluating {model_name} on {length}bp")
    preds = []
    for batch in dataset:
        out = model(batch, training=False)
        logits = out["prediction"].numpy()
        preds.extend(np.argmax(logits, axis=-1))
    preds = np.array(preds)
    acc = accuracy_score(true_labels, preds)
    print(f"  Accuracy: {acc:.4f}")
    print(
        f"  Report:\n{classification_report(true_labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)}"
    )
    return {"model": model_name, "length": length, "accuracy": acc}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = load_model(args.config, args.checkpoint)

    results = []
    for length in args.lengths:
        dataset, labels = create_dataset(
            args.test_csv, length, args.max_samples, args.batch_size
        )
        result = evaluate(model, dataset, labels, length, args.model_name)
        results.append(result)

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"{'Model':<15} {'500bp':<10} {'1000bp':<10} {'2000bp':<10}")
    print("-" * 50)
    accs = [r["accuracy"] for r in results]
    print(f"{args.model_name:<15} {accs[0]:<10.4f} {accs[1]:<10.4f} {accs[2]:<10.4f}")


if __name__ == "__main__":
    main()
