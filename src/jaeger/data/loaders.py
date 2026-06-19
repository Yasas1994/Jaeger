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

from jaeger.dataops.convert import _crop_starts, _get_codon_map


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
            elif nuc.ndim == 2:
                nuc = nuc.astype(np.int32)
            features["nucleotide"] = nuc
        if input_type in ("translated", "both"):
            trans = data["translated"][i]
            if seq_onehot and trans.ndim == 2 and codon_depth is not None:
                t = tf.cast(trans, tf.int32)
                mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
                oh = tf.one_hot(t, depth=codon_depth, dtype=tf.float32)
                trans = (oh * mask).numpy()
            elif trans.ndim == 2:
                trans = trans.astype(np.int32)
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
        ds = ds.prefetch(tf.data.AUTOTUNE)
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
            # Keras Embedding layers expect int32 inputs.
            if arr.ndim == 2:
                return arr.astype(np.int32)
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
        # Feature arrays are (frames, length) or (frames, length, depth).
        # The sequence-length axis is always axis 1.
        shape = list(arr.shape)
        if len(shape) >= 2:
            shape[1] = None
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
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _resolve_strides(
    crop_sizes: list[int],
    strides: list[int] | None,
    overlap: float | None,
) -> list[int]:
    """Return per-crop strides from explicit strides or overlap."""
    if strides is not None:
        if len(strides) != len(crop_sizes):
            raise ValueError(
                f"strides ({len(strides)}) must match crop_sizes ({len(crop_sizes)})"
            )
        return [int(s) for s in strides]
    if overlap is not None:
        return [max(1, int(cs * (1 - overlap))) for cs in crop_sizes]
    return [int(cs) for cs in crop_sizes]


def _densify_object_array(arr: np.ndarray, pad_value: int = 0) -> np.ndarray | None:
    """Convert a 1-D object array of ndarrays into a dense padded ndarray.

    Returns ``None`` if the array is not an object array or if the inner arrays
    have incompatible shapes. Fixed-length inputs are stacked directly;
    variable-length inputs are padded to the maximum length along each axis.
    """
    if arr.ndim != 1 or arr.dtype != object:
        return arr

    try:
        # Fast path: all inner arrays have the same shape.
        return np.stack(arr)
    except ValueError:
        pass

    # Variable-length path: pad each inner array to the max shape.
    shapes = [a.shape for a in arr]
    if not shapes:
        return arr
    max_shape = tuple(max(s) for s in zip(*shapes))
    try:
        padded = []
        for a in arr:
            pad_width = [(0, m - s) for m, s in zip(max_shape, a.shape)]
            if any(p > 0 for w in pad_width for p in w):
                a = np.pad(a, pad_width, mode="constant", constant_values=pad_value)
            padded.append(a)
        return np.stack(padded)
    except Exception:
        return None


def _load_cropped_numpy_dataset(
    data: np.lib.npyio.NpzFile,
    crop_sizes: list[int],
    strides: list[int],
    input_type: str,
    seq_onehot: bool,
    codon_depth: int | None,
    nucleotide_onehot_map: dict[str, list[float]] | None,
    num_classes: int | None,
    one_hot_labels: bool,
    pad_to_max: bool = True,
) -> tf.data.Dataset:
    """Load a NumPy NPZ and slice runtime crops from full-length arrays."""
    if input_type == "both":
        raise NotImplementedError(
            "Runtime crop generation for input_type='both' is not supported yet."
        )

    if input_type in ("translated", "both") and codon_depth is None:
        codon_map_name = str(data["codon_map"])
        codon_map = _get_codon_map(codon_map_name)
        codon_depth = len(codon_map) + 1

    lookup = None
    if input_type in ("nucleotide", "both") and seq_onehot:
        nucleotide_map = json.loads(str(data["nucleotide_map"]))
        lookup = _build_nucleotide_onehot_lookup(nucleotide_map, nucleotide_onehot_map)

    if "labels" in data:
        labels = data["labels"]
    elif "label" in data:
        labels = data["label"]
    else:
        raise KeyError("NPZ must contain 'labels' or 'label' array")
    labels = np.asarray(labels, dtype=np.int32)
    n = len(labels)

    feature_keys: list[str] = []
    np_arrays: dict[str, np.ndarray] = {}
    if input_type in ("translated", "both"):
        feature_keys.append("translated")
        np_arrays["translated"] = data["translated"]
        if "translated_lengths" in data:
            lengths = np.asarray(data["translated_lengths"], dtype=np.int32)
        else:
            lengths = np.full(n, np_arrays["translated"].shape[-2], dtype=np.int32)
    if input_type in ("nucleotide", "both"):
        feature_keys.append("nucleotide")
        np_arrays["nucleotide"] = data["nucleotide"]
        if "lengths" in data:
            lengths = np.asarray(data["lengths"], dtype=np.int32)
        else:
            lengths = np.full(n, np_arrays["nucleotide"].shape[-2], dtype=np.int32)

    is_object = bool(
        np_arrays[feature_keys[0]].ndim == 1
        and np_arrays[feature_keys[0]].dtype == object
    )

    # If the NPZ stored full sequences as object arrays, try to densify them.
    # Fixed-length data (e.g. all 1800 bp) becomes a dense tensor and can use
    # the fast from_tensor_slices + parallel map path instead of a Python
    # generator. Variable-length data falls back to the generator path.
    if is_object:
        densified: dict[str, np.ndarray] = {}
        can_densify = True
        for key in feature_keys:
            dense = _densify_object_array(np_arrays[key])
            if dense is None:
                can_densify = False
                break
            densified[key] = dense
        if can_densify:
            np_arrays.update(densified)
            is_object = False

    max_crop_size = int(max(crop_sizes))

    def _encode_label(label: int) -> np.ndarray:
        if one_hot_labels and num_classes is not None and num_classes > 1:
            return np.eye(int(num_classes), dtype=np.float32)[label]
        if num_classes == 1:
            return np.array([float(label)], dtype=np.float32)
        return np.float32(label)

    def _convert_numpy_crop(crop: np.ndarray, key: str) -> np.ndarray:
        if seq_onehot and crop.ndim == 2 and key == "translated":
            t = tf.cast(crop, tf.int32)
            mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
            oh = tf.one_hot(t, depth=int(codon_depth), dtype=tf.float32)
            return (oh * mask).numpy()
        if seq_onehot and crop.ndim == 2 and key == "nucleotide" and lookup is not None:
            return lookup[crop]
        if crop.ndim == 2:
            return crop.astype(np.int32)
        return crop

    def _feature_spec(arr: np.ndarray) -> tf.TensorSpec:
        # Feature arrays are (frames, length) or (frames, length, depth).
        # The sequence-length axis is always axis 1.
        shape = list(arr.shape)
        if len(shape) >= 2:
            shape[1] = max_crop_size if pad_to_max else None
        return tf.TensorSpec(shape=shape, dtype=tf.as_dtype(arr.dtype))

    if is_object:
        # Variable-length full sequences stored as object arrays.
        # Keep the generator as light as possible (numpy slicing only) and do
        # one-hot conversion in a parallel map. This follows TensorFlow's data
        # performance guideline of keeping the generator Python-only and moving
        # heavy TF ops into vectorized/parallel map transforms.
        def _sample_features(i: int, start: int, length: int) -> dict[str, np.ndarray]:
            raw: dict[str, np.ndarray] = {}
            for key in feature_keys:
                arr = np_arrays[key][i]
                if arr.ndim == 2:
                    crop = arr[:, start : start + length]
                else:
                    crop = arr[:, start : start + length, :]
                if pad_to_max and length < max_crop_size:
                    if arr.ndim == 2:
                        pad_width = [(0, 0), (0, max_crop_size - length)]
                    else:
                        pad_width = [(0, 0), (0, max_crop_size - length), (0, 0)]
                    crop = np.pad(crop, pad_width, mode="constant", constant_values=0)
                raw[key] = crop.astype(np.int32)
            return raw

        # Build the first crop to infer output signatures.
        first_start, first_length = next(
            (
                (start, min(crop_sizes[0], int(lengths[0]) - start))
                for start in _crop_starts(int(lengths[0]), crop_sizes[0], strides[0])
            ),
            None,
        )
        if first_start is None:
            raise ValueError("No crops could be generated from the input sequences")
        sample = _sample_features(0, first_start, first_length)
        output_signature = (
            {key: _feature_spec(arr) for key, arr in sample.items()},
            tf.TensorSpec(
                shape=(int(num_classes),)
                if one_hot_labels and num_classes and num_classes > 1
                else (1,)
                if num_classes == 1
                else (),
                dtype=tf.float32,
            ),
        )

        def generator():
            for i in range(n):
                actual_len = int(lengths[i])
                label = _encode_label(int(labels[i]))
                for cs, stride in zip(crop_sizes, strides):
                    for start in _crop_starts(actual_len, cs, stride):
                        length = min(cs, actual_len - start)
                        yield _sample_features(i, start, length), label

        # After one-hot conversion the tensor shapes become dynamic; pin them
        # back to fixed values so the dataset has a known element spec.
        object_expected_shapes: dict[str, tuple[int | None, ...]] = {}
        for key, arr in sample.items():
            frames = int(arr.shape[0])
            length_dim = max_crop_size if pad_to_max else None
            if seq_onehot and arr.ndim == 2 and key == "translated":
                object_expected_shapes[key] = (frames, length_dim, int(codon_depth))
            elif (
                seq_onehot
                and arr.ndim == 2
                and key == "nucleotide"
                and lookup is not None
            ):
                object_expected_shapes[key] = (
                    frames,
                    length_dim,
                    int(lookup.shape[1]),
                )
            elif arr.ndim == 3:
                object_expected_shapes[key] = (
                    frames,
                    length_dim,
                    int(arr.shape[2]),
                )
            else:
                object_expected_shapes[key] = (frames, length_dim)

        def _convert_crops(
            features: dict[str, tf.Tensor], label: tf.Tensor
        ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
            out: dict[str, tf.Tensor] = {}
            for key in feature_keys:
                crop = features[key]
                if seq_onehot and crop.shape.rank == 2 and key == "translated":
                    t = tf.cast(crop, tf.int32)
                    mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
                    oh = tf.one_hot(t, depth=int(codon_depth), dtype=tf.float32)
                    out[key] = oh * mask
                elif (
                    seq_onehot
                    and crop.shape.rank == 2
                    and key == "nucleotide"
                    and lookup is not None
                ):
                    n = tf.cast(crop, tf.int32)
                    out[key] = tf.gather(tf.constant(lookup, dtype=tf.float32), n)
                else:
                    out[key] = crop
                out[key] = tf.ensure_shape(out[key], object_expected_shapes[key])
            return out, label

        with tf.device("/CPU:0"):
            ds = tf.data.Dataset.from_generator(
                generator, output_signature=output_signature
            )
            ds = ds.map(_convert_crops, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # Dense padded full sequences.
    arrays: dict[str, tf.Tensor] = {
        key: tf.constant(np_arrays[key]) for key in feature_keys
    }
    labels_t = tf.constant(labels)

    sample_indices: list[int] = []
    starts_list: list[int] = []
    lengths_list: list[int] = []
    for i in range(n):
        actual_len = int(lengths[i])
        for cs, stride in zip(crop_sizes, strides):
            for start in _crop_starts(actual_len, cs, stride):
                length = min(cs, actual_len - start)
                sample_indices.append(i)
                starts_list.append(start)
                lengths_list.append(length)

    if not sample_indices:
        raise ValueError("No crops could be generated from the input sequences")

    sample_indices_t = tf.constant(sample_indices, dtype=tf.int32)
    starts_t = tf.constant(starts_list, dtype=tf.int32)
    crop_lengths_t = tf.constant(lengths_list, dtype=tf.int32)

    # One-hot / padding ops erase static shapes; precompute the expected shape
    # for each feature so we can pin them back in the map function.
    length_dim = max_crop_size if pad_to_max else None
    dense_expected_shapes: dict[str, tuple[int | None, ...]] = {}
    for key in feature_keys:
        arr = arrays[key]
        frames = int(arr.shape[1])
        if seq_onehot and arr.shape.rank == 3 and key == "translated":
            dense_expected_shapes[key] = (frames, length_dim, int(codon_depth))
        elif (
            seq_onehot
            and arr.shape.rank == 3
            and key == "nucleotide"
            and lookup is not None
        ):
            dense_expected_shapes[key] = (
                frames,
                length_dim,
                int(lookup.shape[1]),
            )
        elif arr.shape.rank == 4:
            dense_expected_shapes[key] = (
                frames,
                length_dim,
                int(arr.shape[3]),
            )
        else:
            dense_expected_shapes[key] = (frames, length_dim)

    def _slice_crop(
        arr: tf.Tensor,
        idx: tf.Tensor,
        start: tf.Tensor,
        length: tf.Tensor,
        pad: bool,
    ) -> tf.Tensor:
        rank = arr.shape.rank
        begin = [idx, 0, start] + [0] * (rank - 3)
        size = [1, tf.shape(arr)[1], length] + [
            tf.shape(arr)[i] for i in range(3, rank)
        ]
        crop = tf.slice(arr, begin, size)
        crop = tf.squeeze(crop, axis=0)
        if not pad:
            return crop
        # Pad every crop to the same maximum length so batch shapes are fixed
        # and dataset cardinality is known. Token 0 is the padding value and is
        # masked during one-hot conversion.
        pad_len = tf.maximum(max_crop_size - length, 0)
        n_dims = rank - 1
        paddings = tf.concat(
            [
                tf.zeros((n_dims, 1), dtype=tf.int32),
                tf.tensor_scatter_nd_update(
                    tf.zeros((n_dims, 1), dtype=tf.int32),
                    indices=[[1]],
                    updates=[[pad_len]],
                ),
            ],
            axis=1,
        )
        return tf.pad(crop, paddings)

    def _map_fn(
        idx: tf.Tensor, start: tf.Tensor, length: tf.Tensor
    ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        features: dict[str, tf.Tensor] = {}
        if "translated" in feature_keys:
            trans = _slice_crop(
                arrays["translated"], idx, start, length, pad=pad_to_max
            )
            if seq_onehot and trans.shape.rank == 2:
                t = tf.cast(trans, tf.int32)
                mask = tf.expand_dims(tf.cast(t > 0, tf.float32), -1)
                oh = tf.one_hot(t, depth=int(codon_depth), dtype=tf.float32)
                trans = oh * mask
            elif trans.dtype != tf.float32:
                trans = tf.cast(trans, tf.int32)
            features["translated"] = tf.ensure_shape(
                trans, dense_expected_shapes["translated"]
            )
        if "nucleotide" in feature_keys:
            nuc = _slice_crop(arrays["nucleotide"], idx, start, length, pad=pad_to_max)
            if seq_onehot and nuc.shape.rank == 2:
                n = tf.cast(nuc, tf.int32)
                nuc = tf.gather(tf.constant(lookup, dtype=tf.float32), n)
            elif nuc.dtype != tf.float32:
                nuc = tf.cast(nuc, tf.int32)
            features["nucleotide"] = tf.ensure_shape(
                nuc, dense_expected_shapes["nucleotide"]
            )

        label = labels_t[idx]
        if one_hot_labels and num_classes is not None and num_classes > 1:
            label = tf.one_hot(label, depth=int(num_classes), dtype=tf.float32)
        elif num_classes == 1:
            label = tf.expand_dims(tf.cast(label, tf.float32), 0)
        else:
            label = tf.cast(label, tf.float32)
        return features, label

    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (sample_indices_t, starts_t, crop_lengths_t)
        ).map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
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
    crop_sizes: list[int] | None = None,
    strides: list[int] | None = None,
    overlap: float | None = None,
    pad_to_max: bool = True,
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
    crop_sizes:
        Optional list of crop lengths. When provided, full-length sequences in
        the NPZ are sliced into multiple crops at runtime instead of loading
        pre-generated crops.
    strides:
        Optional per-crop strides. Must match ``crop_sizes`` in length.
    overlap:
        Optional overlap fraction between 0 and 1. If ``strides`` is not given,
        strides are computed as ``int(crop_size * (1 - overlap))``.
    pad_to_max:
        When cropping, pad each crop to ``max(crop_sizes)``.

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

    if crop_sizes is not None:
        resolved_strides = _resolve_strides(crop_sizes, strides, overlap)
        return _load_cropped_numpy_dataset(
            data,
            crop_sizes=crop_sizes,
            strides=resolved_strides,
            input_type=input_type,
            seq_onehot=seq_onehot,
            codon_depth=codon_depth,
            nucleotide_onehot_map=nucleotide_onehot_map,
            num_classes=num_classes,
            one_hot_labels=one_hot_labels,
            pad_to_max=pad_to_max,
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

    # Keras Embedding layers expect int32 inputs, so upcast any smaller
    # integer feature arrays before they reach the model.
    for key in features:
        if np.issubdtype(features[key].dtype, np.integer):
            features[key] = features[key].astype(np.int32)

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

    return ds.prefetch(tf.data.AUTOTUNE)
