import tensorflow as tf
from jaeger.preprocess.latest.convert import process_string_train

ds = tf.data.Dataset.from_tensor_slices([
    "0,ATGCGCACGTAGACTACGTACGAC",
    "1,CGTACGTAGCTAGCTAGCTAGCTA",
    "0,ATGCGTACGTAGCTAGCTAGCTAGC",
])

ds = ds.map(
    process_string_train(
        input_type="both",
        num_classes=2,
        codon_depth=64,
        ngram_width=3,
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

ds = ds.shuffle(6).padded_batch(
    3,
    padded_shapes=(
        {
            "nucleotide": [2, None, 4],
            "translated": [6, None, 64],
        },
        [2],
    ),
).prefetch(tf.data.AUTOTUNE)

for outputs, label in ds:
    print(label)
    print(outputs["nucleotide"].shape)
    print(outputs["translated"].shape)

assert outputs["nucleotide"].shape[0] == 3
assert outputs["translated"].shape[0] == 3
assert outputs["translated"].shape[1] == 6
assert outputs["translated"].shape[3] == 64
print("convert_test passed")
