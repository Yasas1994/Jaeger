#!/usr/bin/env python3
"""
Fast evaluation of length generalization using tf.data pipeline.
"""

import sys
sys.path.insert(0, "src")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

from jaeger.utils.misc import load_model_config
from jaeger.commands.train import DynamicModelBuilder
from jaeger.preprocess.latest.convert import process_string_train
from jaeger.preprocess.latest.maps import CODONS, CODON_ID

CLASS_NAMES = ["chromosome", "virus", "plasmid"]


def load_model(config_path, checkpoint_path=None):
    import shutil
    config = load_model_config(Path(config_path))
    config["from_last_checkpoint"] = False
    config["force"] = False
    
    # Temporarily move checkpoints to avoid the warning
    checkpoint_dir = None
    if "training" in config:
        checkpoint_dir = config["training"].get("classifier_dir", "")
        checkpoint_dir = checkpoint_dir.replace("{{ model.base_dir }}", config["model"]["base_dir"])
        checkpoint_dir = checkpoint_dir.replace("{{ training.experiment_root }}", f"experiments/experiment_{config['model']['experiment']}_{config['model']['seed']}")
    
    if checkpoint_dir and Path(checkpoint_dir).exists():
        backup_dir = str(checkpoint_dir) + "_backup"
        if Path(backup_dir).exists():
            shutil.rmtree(backup_dir)
        shutil.move(checkpoint_dir, backup_dir)
    else:
        backup_dir = None
    
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    builder.compile_model(models, train_branch="classifier")
    
    # Restore checkpoints and load weights
    if backup_dir:
        shutil.move(backup_dir, checkpoint_dir)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Loading weights from: {checkpoint_path}")
        models["jaeger_model"].load_weights(checkpoint_path)
    return models["jaeger_model"]


def create_dataset(csv_path, crop_size, max_samples=5000, batch_size=256):
    """Create tf.data dataset with proper preprocessing."""
    df = pd.read_csv(csv_path, header=None, names=["label", "seq"])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Create CSV strings
    csv_strings = [f"{label},{seq}" for label, seq in zip(df["label"], df["seq"])]
    labels = df["label"].values.astype(np.int32)
    
    # Create processor
    processor = process_string_train(
        codons=CODONS, codon_num=CODON_ID, seq_onehot=False,
        crop_size=crop_size, timesteps=False, mutate=False,
        mutation_rate=0.0, shuffle=False, masking=False,
        class_label_onehot=False, ngram_width=3,
    )
    
    def _parse(csv_str):
        out, _ = processor(csv_str)
        return out["translated"]
    
    dataset = tf.data.Dataset.from_tensor_slices(csv_strings)
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, labels


def evaluate_model(model, dataset, true_labels, length, model_name):
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {length}bp sequences")
    print(f"{'='*60}")
    
    preds = []
    for batch in dataset:
        out = model(batch, training=False)
        logits = out["prediction"].numpy() if isinstance(out, dict) else out.numpy()
        preds.extend(np.argmax(logits, axis=-1))
    
    preds = np.array(preds)
    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))
    
    return {"model": model_name, "length": length, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
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
        dataset, labels = create_dataset(args.test_csv, length, args.max_samples, args.batch_size)
        result = evaluate_model(model, dataset, labels, length, args.model_name)
        results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Length':<10} {'Accuracy':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {r['length']:<10} {r['accuracy']:<10.4f}")
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    np.savez(output_dir / f"{args.model_name}_results.npz",
             lengths=np.array([r['length'] for r in results]),
             accuracies=np.array([r['accuracy'] for r in results]))


if __name__ == "__main__":
    main()
