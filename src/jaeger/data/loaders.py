"""Dataset loaders for the unified NumPy NPZ training format.

Provides a single ``tf.data.Dataset`` builder for the NPZ schema produced by
``jaeger.dataops.convert._convert_to_npz``. The schema supports ``nucleotide``,
``translated``, or ``both`` input representations, integer or one-hot encodings,
and configurable nucleotide one-hot mappings.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from jaeger.dataops.convert import _get_codon_map


def _one_hot_integer_np(indices: np.ndarray, depth: int) -> np.ndarray:
    """Convert integer indices to float32 one-hot along a new last axis.

    Index 0 and any invalid index produce an all-zero vector. Valid indices
    are in [1, depth-1]. Output shape is indices.shape + (depth,).
    """
    out = np.zeros(indices.shape + (depth,), dtype=np.float32)
    flat = indices.ravel()
    valid = flat > 0
    if np.any(valid):
        out.reshape(-1, depth)[np.arange(flat.size)[valid], flat[valid]] = 1.0
    return out


def _build_nucleotide_onehot_lookup(
    nucleotide_map: dict[str, int],
    nucleotide_onehot_map: dict[str, list[float]] | None,
) -> np.ndarray:
    """Build a token -> one-hot vector lookup table.

    The NPZ ``nucleotide_map`` maps base letters to integer tokens. The config
    ``nucleotide_onehot_map`` maps base letters to one-hot vectors. This helper
    combines them so that token ``nucleotide_map[base]`` maps to
    ``nucleotide_onehot_map[base]``.

    If ``nucleotide_onehot_map`` is None, the default 4-D one-hot order
    A=[1,0,0,0], G=[0,1,0,0], T=[0,0,1,0], C=[0,0,0,1], N=[0,0,0,0] is used.
    Unknown tokens map to the zero vector.
    """
    if nucleotide_onehot_map is None:
        nucleotide_onehot_map = {
            "A": [1.0, 0.0, 0.0, 0.0],
            "G": [0.0, 1.0, 0.0, 0.0],
            "T": [0.0, 0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 0.0, 0.0],
        }
    depth = len(next(iter(nucleotide_onehot_map.values())))
    max_token = max(nucleotide_map.values())
    lookup = np.zeros((max_token + 1, depth), dtype=np.float32)
    for base, token in nucleotide_map.items():
        vec = nucleotide_onehot_map.get(base)
        if vec is not None:
            lookup[token] = np.asarray(vec, dtype=np.float32)
    return lookup


def _is_ragged(data: np.lib.npyio.NpzFile, input_type: str) -> bool:
    for key in ("nucleotide", "translated"):
        if input_type in (key, "both") and key in data:
            arr = data[key]
            if arr.ndim == 1 and arr.dtype == object:
                return True
    return False


def _load_ragged_numpy_dataset(
    data: np.lib.npyio.NpzFile,
    input_type: str,
    seq_onehot: bool,
    codon_depth: int | None,
    nucleotide_onehot_map: dict[str, list[float]] | None,
    num_classes: int | None,
    one_hot_labels: bool,
) -> tf.data.Dataset:
    """Load an NPZ where feature arrays are 1-D object arrays of crops."""
    if input_type in ("translated", "both") and codon_depth is None:
        codon_map_name = str(data["codon_map"])
        codon_map = _get_codon_map(codon_map_name)
        codon_depth = len(codon_map) + 1

    lookup = None
    if input_type in ("nucleotide", "both") and seq_onehot:
        nucleotide_map = json.loads(str(data["nucleotide_map"]))
        lookup = _build_nucleotide_onehot_lookup(nucleotide_map, nucleotide_onehot_map)

    n = len(data["labels"])

    def _sample_features(i: int) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {}
        if input_type in ("nucleotide", "both"):
            nuc = data["nucleotide"][i]
            if seq_onehot and nuc.ndim == 2 and lookup is not None:
                nuc = lookup[nuc]
            features["nucleotide"] = nuc
        if input_type in ("translated", "both"):
            trans = data["translated"][i]
            if seq_onehot and trans.ndim == 2 and codon_depth is not None:
                t = tf.cast(trans, tf.int32)
                mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
                oh = tf.one_hot(t, depth=codon_depth, dtype=tf.float32)
                trans = (oh * mask).numpy()
            features["translated"] = trans
        return features

    sample = _sample_features(0)
    output_signature = (
        {
            key: tf.TensorSpec(shape=[None] * arr.ndim, dtype=tf.as_dtype(arr.dtype))
            for key, arr in sample.items()
        },
        tf.TensorSpec(
            shape=(num_classes,)
            if one_hot_labels and num_classes and num_classes > 1
            else (1,)
            if num_classes == 1
            else (),
            dtype=tf.float32,
        ),
    )

    def generator():
        for i in range(n):
            features = _sample_features(i)
            label = int(data["labels"][i])
            if one_hot_labels and num_classes is not None and num_classes > 1:
                label_arr = np.eye(num_classes, dtype=np.float32)[label]
            elif num_classes == 1:
                label_arr = np.array([float(label)], dtype=np.float32)
            else:
                label_arr = np.float32(label)
            yield features, label_arr

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )
    return ds


def _load_sharded_numpy_dataset(
    data: np.lib.npyio.NpzFile,
    input_type: str,
    seq_onehot: bool,
    codon_depth: int | None,
    nucleotide_onehot_map: dict[str, list[float]] | None,
    num_classes: int | None,
    one_hot_labels: bool,
) -> tf.data.Dataset:
    """Load a sharded NPZ produced by the streaming converter."""
    manifest = json.loads(str(data["_jaeger_manifest"].item()))
    num_shards = int(manifest["num_shards"])
    feature_keys = [k for k in manifest["keys"] if k in ("nucleotide", "translated")]

    if input_type in ("translated", "both") and codon_depth is None:
        codon_map_name = str(manifest.get("codon_map"))
        codon_map = _get_codon_map(codon_map_name)
        codon_depth = len(codon_map) + 1

    lookup = None
    if input_type in ("nucleotide", "both") and seq_onehot:
        nucleotide_map = json.loads(str(manifest["nucleotide_map"]))
        lookup = _build_nucleotide_onehot_lookup(nucleotide_map, nucleotide_onehot_map)

    first_features = {k: data[f"{k}_00000"] for k in feature_keys}

    def _convert_feature(arr: np.ndarray, key: str) -> np.ndarray:
        if not seq_onehot:
            return arr
        if key == "nucleotide" and arr.ndim == 2 and lookup is not None:
            return lookup[arr]
        if key == "translated" and arr.ndim == 2 and codon_depth is not None:
            t = tf.cast(arr, tf.int32)
            mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
            oh = tf.one_hot(t, depth=codon_depth, dtype=tf.float32)
            return (oh * mask).numpy()
        return arr

    def _sample_features(i: int) -> dict[str, np.ndarray]:
        return {
            key: _convert_feature(first_features[key][i], key) for key in feature_keys
        }

    def _feature_spec(arr: np.ndarray) -> tf.TensorSpec:
        # Keep all axes static except the sequence-length axis (second from last).
        shape = list(arr.shape)
        if len(shape) >= 2:
            shape[-2] = None
        return tf.TensorSpec(shape=shape, dtype=tf.as_dtype(arr.dtype))

    sample = _sample_features(0)
    output_signature = (
        {key: _feature_spec(arr) for key, arr in sample.items()},
        tf.TensorSpec(
            shape=(num_classes,)
            if one_hot_labels and num_classes and num_classes > 1
            else (1,)
            if num_classes == 1
            else (),
            dtype=tf.float32,
        ),
    )

    def generator():
        for shard_idx in range(num_shards):
            shard_arrays = {key: data[f"{key}_{shard_idx:05d}"] for key in feature_keys}
            shard_labels = data[f"labels_{shard_idx:05d}"]
            # Pre-convert dense (non-object) shards in one shot to avoid
            # per-sample TensorFlow overhead.
            for key, arr in shard_arrays.items():
                if not (arr.ndim == 1 and arr.dtype == object):
                    shard_arrays[key] = _convert_feature(arr, key)
            for i in range(len(shard_labels)):
                features = {}
                for key, arr in shard_arrays.items():
                    feat = arr[i]
                    if arr.ndim == 1 and arr.dtype == object:
                        feat = _convert_feature(feat, key)
                    features[key] = feat
                label = int(shard_labels[i])
                if one_hot_labels and num_classes is not None and num_classes > 1:
                    label_arr = np.eye(num_classes, dtype=np.float32)[label]
                elif num_classes == 1:
                    label_arr = np.array([float(label)], dtype=np.float32)
                else:
                    label_arr = np.float32(label)
                yield features, label_arr

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )
    return ds


def _load_numpy_dataset(
    path: str,
    input_type: str = "translated",
    seq_onehot: bool = False,
    codon_depth: int | None = None,
    nucleotide_onehot_map: dict[str, list[float]] | None = None,
    num_classes: int | None = None,
    one_hot_labels: bool = True,
    buffer_size: int | None = None,
) -> tf.data.Dataset:
    """Load a unified NPZ file and return a ``tf.data.Dataset``.

    Parameters
    ----------
    path:
        Path to the ``.npz`` archive.
    input_type:
        One of ``"translated"``, ``"nucleotide"``, or ``"both"``.
    seq_onehot:
        If True and the stored sequence array is integer-encoded, convert it to
        one-hot floats.
    codon_depth:
        Depth to use for translated one-hot conversion. If None, the depth is
        inferred from the ``codon_map`` entry stored in the NPZ.
    nucleotide_onehot_map:
        Optional base -> one-hot mapping for nucleotide inputs. Defaults to
        A/G/T/C/N order when omitted.
    num_classes:
        Number of classes; used to one-hot integer labels when
        ``one_hot_labels`` is True.
    one_hot_labels:
        Convert integer labels to one-hot float vectors.
    buffer_size:
        If provided and positive, integer-to-one-hot conversion is performed
        batch-wise in the ``tf.data`` pipeline instead of materialising the
        whole one-hot array in memory. This mirrors the CSV loader's
        sample-by-sample preprocessing and avoids OOM for large NPZs.

    Returns
    -------
    ``tf.data.Dataset`` yielding ``(features, labels)`` tuples.
    """
    data = np.load(path, allow_pickle=True)

    if "_jaeger_manifest" in data.files:
        return _load_sharded_numpy_dataset(
            data,
            input_type=input_type,
            seq_onehot=seq_onehot,
            codon_depth=codon_depth,
            nucleotide_onehot_map=nucleotide_onehot_map,
            num_classes=num_classes,
            one_hot_labels=one_hot_labels,
        )

    if _is_ragged(data, input_type):
        return _load_ragged_numpy_dataset(
            data,
            input_type=input_type,
            seq_onehot=seq_onehot,
            codon_depth=codon_depth,
            nucleotide_onehot_map=nucleotide_onehot_map,
            num_classes=num_classes,
            one_hot_labels=one_hot_labels,
        )

    valid_input_types = {"translated", "nucleotide", "both"}
    if input_type not in valid_input_types:
        raise ValueError(
            f"Unsupported input_type: {input_type}. "
            f"Use one of {sorted(valid_input_types)}."
        )

    features: dict[str, np.ndarray] = {}
    onehot_keys: list[str] = []
    use_batchwise_onehot = buffer_size is not None and buffer_size > 0

    if input_type in ("translated", "both"):
        translated = data["translated"]
        if translated.ndim == 3:
            if seq_onehot:
                if codon_depth is None:
                    codon_map_name = str(data["codon_map"])
                    codon_map = _get_codon_map(codon_map_name)
                    codon_depth = len(codon_map) + 1
                if use_batchwise_onehot:
                    features["translated"] = translated
                    onehot_keys.append("translated")
                else:
                    translated = _one_hot_integer_np(translated, codon_depth)
                    features["translated"] = translated
            else:
                features["translated"] = translated
        elif translated.ndim != 4:
            raise ValueError(
                f"Unexpected translated array ndim: {translated.ndim} "
                "(expected 3 for integer or 4 for one-hot)"
            )
        else:
            features["translated"] = translated

    if input_type in ("nucleotide", "both"):
        nucleotide = data["nucleotide"]
        if nucleotide.ndim == 3:
            if seq_onehot:
                nucleotide_map = json.loads(str(data["nucleotide_map"]))
                lookup = _build_nucleotide_onehot_lookup(
                    nucleotide_map, nucleotide_onehot_map
                )
                if use_batchwise_onehot:
                    features["nucleotide"] = nucleotide
                    onehot_keys.append("nucleotide")
                else:
                    nucleotide = lookup[nucleotide]
                    features["nucleotide"] = nucleotide
            else:
                features["nucleotide"] = nucleotide
        elif nucleotide.ndim != 4:
            raise ValueError(
                f"Unexpected nucleotide array ndim: {nucleotide.ndim} "
                "(expected 3 for integer or 4 for one-hot)"
            )
        else:
            features["nucleotide"] = nucleotide

    if "labels" in data:
        labels = data["labels"]
    elif "label" in data:
        labels = data["label"]
    else:
        raise KeyError("NPZ must contain 'labels' or 'label' array")

    if (
        one_hot_labels
        and num_classes is not None
        and num_classes > 1
        and (labels.ndim == 1 or np.issubdtype(labels.dtype, np.integer))
    ):
        labels = np.eye(num_classes, dtype=np.float32)[labels]
    else:
        labels = labels.astype(np.float32)

    # Binary heads expect shape (N, 1), multi-class heads expect (N, num_classes).
    if labels.ndim == 1 and num_classes == 1:
        labels = labels[:, np.newaxis]

    # Keep the source arrays on CPU; the GPU only sees training batches.
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices((features, labels))

    if onehot_keys and use_batchwise_onehot:
        convert_fns: dict[str, tf.types.experimental.PolymorphicFunction] = {}

        if "translated" in onehot_keys:
            depth = int(codon_depth)  # type: ignore[arg-type]

            def _convert_translated(t: tf.Tensor, depth: int = depth) -> tf.Tensor:
                t = tf.cast(t, tf.int32)
                t = tf.where(t < depth, t, 0)
                mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
                oh = tf.one_hot(t, depth=depth, dtype=tf.float32)
                return oh * mask

            convert_fns["translated"] = _convert_translated

        if "nucleotide" in onehot_keys:
            lookup_t = tf.constant(lookup, dtype=tf.float32)  # type: ignore[possibly-undefined]

            def _convert_nucleotide(
                n: tf.Tensor, lut: tf.Tensor = lookup_t
            ) -> tf.Tensor:
                n = tf.cast(n, tf.int32)
                return tf.gather(lut, n)

            convert_fns["nucleotide"] = _convert_nucleotide

        def _map_fn(
            feats: dict[str, tf.Tensor], lbl: tf.Tensor
        ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
            out: dict[str, tf.Tensor] = {}
            for key, value in feats.items():
                fn = convert_fns.get(key)
                out[key] = fn(value) if fn is not None else value
            return out, lbl

        ds = (
            ds.batch(buffer_size)
            .map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch()
        )

    return ds
