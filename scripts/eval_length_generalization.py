#!/usr/bin/env python3
"""
Evaluate length generalization for models trained on 500bp fragments.
Tests on 500bp, 1000bp, and 2000bp sequences.
"""

import sys

sys.path.insert(0, "src")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import argparse

from jaeger.utils.misc import load_model_config
from jaeger.commands.train import DynamicModelBuilder
from jaeger.seqops.encode import process_string_train
from jaeger.seqops.maps import CODONS, CODON_ID

CLASS_NAMES = ["chromosome", "virus", "plasmid"]


def load_model(config_path, checkpoint_path=None):
    """Load model from config and optional checkpoint."""
    config = load_model_config(Path(config_path))
    config["from_last_checkpoint"] = False
    config["force"] = True

    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    builder.compile_model(models, train_branch="classifier")

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Loading weights from: {checkpoint_path}")
        models["jaeger_model"].load_weights(checkpoint_path)

    return models["jaeger_model"]


def evaluate(model, sequences, labels, crop_size, batch_size=128):
    """Evaluate model on sequences with specific crop_size."""
    # Create processor
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

    # Process sequences - format as "label,seq" strings
    processed = []
    for seq in tqdm(sequences, desc=f"Processing {crop_size}bp", leave=False):
        # Create a CSV-like string: "0,seq" (label doesn't matter for inference)
        csv_str = f"0,{seq}"
        out, _ = processor(csv_str)
        processed.append(out["translated"].numpy())

    processed = np.array(processed)

    # Predict in batches
    preds = []
    for i in tqdm(range(0, len(processed), batch_size), desc="Predicting", leave=False):
        batch = processed[i : i + batch_size]
        out = model(batch, training=False)
        if isinstance(out, dict):
            logits = out["prediction"].numpy()
        else:
            logits = out.numpy()
        preds.extend(np.argmax(logits, axis=-1))

    preds = np.array(preds)
    acc = accuracy_score(labels, preds)
    return acc, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = load_model(args.config, args.checkpoint)

    # Load test data
    print(f"Loading test data from: {args.test_csv}")
    df = pd.read_csv(args.test_csv, header=None, names=["label", "seq"])
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)

    sequences = df["seq"].values
    labels = df["label"].values.astype(np.int32)
    print(f"Test samples: {len(sequences)}")

    results = []
    for length in args.lengths:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {args.model_name} on {length}bp sequences")
        print(f"{'=' * 60}")

        acc, preds = evaluate(model, sequences, labels, length, args.batch_size)

        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, preds))

        results.append(
            {
                "model": args.model_name,
                "length": length,
                "accuracy": acc,
            }
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'Length':<10} {'Accuracy':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {r['length']:<10} {r['accuracy']:<10.4f}")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{args.model_name}_results.npz"
    np.savez(
        output_file,
        lengths=np.array([r["length"] for r in results]),
        accuracies=np.array([r["accuracy"] for r in results]),
    )
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
