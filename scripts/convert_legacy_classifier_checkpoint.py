#!/usr/bin/env python3
"""Convert a legacy Jaeger classifier weights file to the current architecture.

The legacy checkpoint (e.g. jaeger_d1754a4e_3.4M_fragment.weights.h5) stores
weights for a model built with Keras Functional residual-block submodels. The
current code uses custom ResidualBlockStack layers with different weight names,
so a plain load_weights(skip_mismatch=True) leaves most layers uninitialized.

This script reads the legacy file, maps every weight to the equivalent layer in
a freshly built classifier, and writes a new .weights.h5 file that can be
loaded directly by DynamicModelBuilder.
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py
import numpy as np
import tensorflow as tf
import yaml

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.nnlib.v2.layers import MaskedConv1D, MaskedBatchNorm, ResidualBlockStack


def _read_vars(group: h5py.Group) -> list[np.ndarray]:
    """Read vars/0, vars/1, ... from an h5 weight group in order."""
    names = sorted(group["vars"].keys(), key=lambda x: int(x))
    return [group["vars"][name][:] for name in names]


def _set_layer_weights(layer: tf.keras.layers.Layer, weights: list[np.ndarray]) -> None:
    """Set layer weights, verifying the count matches."""
    expected = len(layer.weights)
    got = len(weights)
    if expected != got:
        raise ValueError(
            f"Layer {layer.name} expects {expected} weights but got {got}"
        )
    if expected == 0:
        return
    layer.set_weights(weights)


def convert_legacy_classifier_checkpoint(
    config_path: str,
    legacy_weights_path: str,
    output_path: str,
) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build only the classifier graph. Drop parts that are not in the legacy
    # classifier checkpoint and disable masking so the architecture matches the
    # SavedModel / legacy model behaviour.
    config["model"]["use_masking"] = False
    config["training"] = {}
    config["model"].pop("reliability_model", None)
    config["model"].pop("projection", None)

    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    classifier = models["jaeger_classifier"]
    rep_model = models["rep_model"]
    head = models["classification_head"]

    # Build the graph by calling it once so all sublayers are created.
    dummy_input = {
        "translated": tf.zeros((1, 6, 500), dtype=tf.float32),
    }
    _ = classifier(dummy_input, training=False)

    with h5py.File(legacy_weights_path, "r") as f:
        layers_group = f["layers"]

        # --- top-level stem layers ---------------------------------------
        stem_layers = [layer for layer in rep_model.layers if isinstance(layer, (
            tf.keras.layers.Embedding,
            MaskedConv1D,
            MaskedBatchNorm,
        ))]
        # Map by type/order rather than by fixed name because Keras assigns
        # incremental IDs that can shift across builds.
        embedding_layer = next(
            (l for l in stem_layers if isinstance(l, tf.keras.layers.Embedding)), None
        )
        conv_layer = next(
            (l for l in stem_layers if isinstance(l, MaskedConv1D)), None
        )
        bn_layer = next(
            (l for l in stem_layers if isinstance(l, MaskedBatchNorm)), None
        )

        if embedding_layer is not None and "embedding" in layers_group:
            _set_layer_weights(embedding_layer, _read_vars(layers_group["embedding"]))
            print(f"Mapped embedding -> {embedding_layer.name}")
        if conv_layer is not None and "masked_conv1d" in layers_group:
            _set_layer_weights(conv_layer, _read_vars(layers_group["masked_conv1d"]))
            print(f"Mapped masked_conv1d -> {conv_layer.name}")
        if bn_layer is not None and "masked_batch_norm" in layers_group:
            _set_layer_weights(bn_layer, _read_vars(layers_group["masked_batch_norm"]))
            print(f"Mapped masked_batch_norm -> {bn_layer.name}")

        # --- residual block stacks ----------------------------------------
        stacks = [
            layer for layer in rep_model.layers
            if isinstance(layer, ResidualBlockStack)
        ]
        functional_keys = [
            k for k in layers_group.keys()
            if k == "functional" or (
                k.startswith("functional_")
                and k.split("_", 1)[1].isdigit()
                and int(k.split("_", 1)[1]) < 8
            )
        ]
        # numeric sort: functional, functional_1, ..., functional_7
        def _functional_sort_key(k: str) -> int:
            return 0 if k == "functional" else int(k.split("_", 1)[1])
        functional_keys.sort(key=_functional_sort_key)

        if len(stacks) != len(functional_keys):
            raise ValueError(
                f"Number of residual stacks ({len(stacks)}) does not match number of "
                f"legacy functional groups ({len(functional_keys)})."
            )

        for stack, func_key in zip(stacks, functional_keys):
            func_group = layers_group[func_key]["layers"]
            block_keys = [k for k in func_group.keys() if k.startswith("residual_block")]
            # sort: residual_block, residual_block_1, residual_block_2, ...
            def _block_sort_key(k: str) -> int:
                return 0 if k == "residual_block" else int(k.rsplit("_", 1)[1])
            block_keys.sort(key=_block_sort_key)

            if len(stack.blocks) != len(block_keys):
                raise ValueError(
                    f"Stack {stack.name}: {len(stack.blocks)} blocks but legacy group "
                    f"{func_key} has {len(block_keys)} blocks."
                )

            for block, blk_key in zip(stack.blocks, block_keys):
                blk_group = func_group[blk_key]
                # conv1, conv2
                _set_layer_weights(block.conv1, _read_vars(blk_group["conv1"]))
                _set_layer_weights(block.conv2, _read_vars(blk_group["conv2"]))
                _set_layer_weights(block.bn1, _read_vars(blk_group["bn1"]))
                _set_layer_weights(block.bn2, _read_vars(blk_group["bn2"]))
                if block.conv3 is not None:
                    _set_layer_weights(block.conv3, _read_vars(blk_group["conv3"]))
                    _set_layer_weights(block.bn3, _read_vars(blk_group["bn3"]))
            print(f"Mapped {func_key} -> {stack.name} ({len(block_keys)} blocks)")

        # --- classification head ------------------------------------------
        dense_layers = [
            layer for layer in head.layers
            if isinstance(layer, tf.keras.layers.Dense)
        ]
        if "functional_8" in layers_group:
            head_group = layers_group["functional_8"]["layers"]
            dense_keys = [k for k in head_group.keys() if k.startswith("dense")]
            dense_keys.sort(key=lambda k: 0 if k == "dense" else int(k.rsplit("_", 1)[1]))

            if len(dense_layers) != len(dense_keys):
                raise ValueError(
                    f"Classification head has {len(dense_layers)} Dense layers but "
                    f"legacy functional_8 has {len(dense_keys)}."
                )

            for dense_layer, dkey in zip(dense_layers, dense_keys):
                _set_layer_weights(dense_layer, _read_vars(head_group[dkey]))
                print(f"Mapped functional_8/{dkey} -> {dense_layer.name}")

    # Save in the current Keras weights format.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_weights(str(output_path))
    print(f"\nSaved converted classifier weights to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a legacy Jaeger classifier checkpoint to the current architecture."
    )
    parser.add_argument("--config", required=True, help="Path to the Jaeger YAML config.")
    parser.add_argument("--legacy-weights", required=True, help="Path to the legacy .weights.h5 file.")
    parser.add_argument("--output", required=True, help="Path for the converted .weights.h5 file.")
    args = parser.parse_args()

    convert_legacy_classifier_checkpoint(args.config, args.legacy_weights, args.output)
