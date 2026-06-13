#!/usr/bin/env python3
"""Simple evaluation using DynamicModelBuilder with checkpoint protection."""

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
import shutil

from jaeger.utils.misc import load_model_config
from jaeger.commands.train import DynamicModelBuilder
from jaeger.seqops.encode import process_string_train
from jaeger.seqops.maps import CODONS, CODON_ID

CLASS_NAMES = ["chromosome", "virus", "plasmid"]


def load_model(config_path, checkpoint_path):
    config = load_model_config(Path(config_path))
    config["from_last_checkpoint"] = False
    config["force"] = False

    # Temporarily move checkpoint dir to avoid the warning
    checkpoint_dir = str(Path(checkpoint_path).parent)
    backup_dir = checkpoint_dir + "_backup"

    if Path(checkpoint_dir).exists():
        if Path(backup_dir).exists():
            shutil.rmtree(backup_dir)
        shutil.move(checkpoint_dir, backup_dir)

    try:
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()
        builder.compile_model(models, train_branch="classifier")
    finally:
        # Always restore checkpoint dir
        if Path(backup_dir).exists():
            shutil.move(backup_dir, checkpoint_dir)

    print(f"  Loading weights from: {checkpoint_path}")
    models["jaeger_model"].load_weights(checkpoint_path)
    return models["jaeger_model"]


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
        logits = out["prediction"].numpy() if isinstance(out, dict) else out.numpy()
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
