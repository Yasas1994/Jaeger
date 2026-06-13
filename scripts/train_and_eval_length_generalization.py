#!/usr/bin/env python3
"""
Train models on 500bp and evaluate length generalization.
Trains baseline, cross-frame, and axial models, then evaluates on 500/1000/2000bp.
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
from sklearn.metrics import accuracy_score

from jaeger.utils.misc import load_model_config
from jaeger.commands.train import DynamicModelBuilder
from jaeger.seqops.encode import process_string_train
from jaeger.seqops.maps import CODONS, CODON_ID

CLASS_NAMES = ["chromosome", "virus", "plasmid"]


def train_model(config_path):
    """Train model and return the trained model + config."""
    config = load_model_config(Path(config_path))
    config["from_last_checkpoint"] = False
    config["force"] = True

    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    builder.compile_model(models, train_branch="classifier")

    # Get data
    string_processor_config = builder._get_string_processor_config()
    _train_data = builder._get_fragment_paths()

    train_data = {"train": None, "validation": None}
    for k, v in _train_data.items():
        from jaeger.commands.train import check_files

        paths = check_files(v.get("paths"))
        _data = tf.data.TextLineDataset(
            paths, num_parallel_reads=len(paths), buffer_size=200
        )
        _buffer_size = string_processor_config.get("buffer_size")

        padded_shape = {"translated": [6, None]}
        train_data[k] = (
            _data.map(
                process_string_train(
                    codons=string_processor_config.get("codon"),
                    codon_num=string_processor_config.get("codon_id"),
                    codon_depth=string_processor_config.get("codon_depth"),
                    label_original=string_processor_config.get(
                        "classifier_labels", None
                    ),
                    label_alternative=string_processor_config.get(
                        "classifier_labels_map", None
                    ),
                    ngram_width=string_processor_config.get("ngram_width"),
                    seq_onehot=string_processor_config.get("seq_onehot"),
                    crop_size=string_processor_config.get("crop_size"),
                    input_type=string_processor_config.get("input_type"),
                    masking=string_processor_config.get("masking"),
                    mutate=string_processor_config.get("mutate"),
                    mutation_rate=string_processor_config.get("mutation_rate"),
                    shuffle=string_processor_config.get("shuffle"),
                    shuffle_frames=string_processor_config.get("shuffle_frames"),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(_buffer_size, reshuffle_each_iteration=True)
            .padded_batch(
                builder.train_cfg.get("batch_size", 32),
                padded_shapes=(
                    padded_shape,
                    {"classifier": builder.classifier_out_dim},
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    # Train
    train_args = {
        "validation_data": train_data.get("validation").take(
            builder.train_cfg.get("classifier_validation_steps")
        ),
        "epochs": builder.train_cfg.get("classifier_epochs"),
        "callbacks": builder.get_callbacks(branch="classifier"),
    }

    models.get("jaeger_classifier").fit(
        train_data.get("train").take(builder.train_cfg.get("classifier_train_steps")),
        class_weight=builder.train_cfg.get("classifier_class_weights"),
        **train_args,
    )

    return models["jaeger_model"], builder


def create_eval_dataset(csv_path, crop_size, max_samples=5000, batch_size=256):
    """Create eval dataset."""
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
    return {"model": model_name, "length": length, "accuracy": acc}


def main():
    configs = [
        ("train_config/nn_config_500bp_baseline.yaml", "baseline"),
        ("train_config/nn_config_500bp_crossframe.yaml", "crossframe"),
        ("train_config/nn_config_500bp_axial.yaml", "axial"),
    ]

    test_csv = "/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/training/genomad_train_data/jaeger_format/val.csv"
    lengths = [500, 1000, 2000]
    max_samples = 5000

    all_results = []

    for config_path, model_name in configs:
        print(f"\n{'#' * 60}")
        print(f"# Training {model_name}")
        print(f"{'#' * 60}")

        model, builder = train_model(config_path)

        for length in lengths:
            dataset, labels = create_eval_dataset(test_csv, length, max_samples)
            result = evaluate(model, dataset, labels, length, model_name)
            all_results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'500bp':<10} {'1000bp':<10} {'2000bp':<10}")
    print("-" * 60)

    for model_name in ["baseline", "crossframe", "axial"]:
        accs = [r["accuracy"] for r in all_results if r["model"] == model_name]
        print(f"{model_name:<15} {accs[0]:<10.4f} {accs[1]:<10.4f} {accs[2]:<10.4f}")

    # Save
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    np.savez(output_dir / "length_generalization_results.npz", results=all_results)


if __name__ == "__main__":
    main()
