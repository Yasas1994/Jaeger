import tensorflow as tf

# Import these from your updated module
from jaeger.nnlib.v2.layers import (
    MaskedConv1D,
    MaskedMaxPooling1D,
    MaskedBatchNorm,
    MaskedGlobalAvgPooling,
    GatedFrameGlobalMaxPooling,
    ResidualBlock,
    SinusoidalPositionEmbedding,
    TransformerEncoder,
    CrossFrameAttention,
    AxialAttention,
)

tf.random.set_seed(7)

batch = 2
frames = 6
length = 32
channels = 4

x = tf.random.normal([batch, frames, length, channels])

# True = valid position, False = masked/invalid position
mask = tf.random.uniform([batch, frames, length]) > 0.2

print("Input:", x.shape)
print("Mask:", mask.shape)

conv = MaskedConv1D(
    filters=8,
    kernel_size=3,
    padding="same",
    activation="gelu",
)

y = conv(x, mask=mask)
y_mask = conv.compute_mask(x, mask)

print("MaskedConv1D output:", y.shape)
print("MaskedConv1D mask:", y_mask.shape)

pool = MaskedMaxPooling1D(
    pool_size=2,
    strides=2,
    padding="same",
)

yp = pool(y, mask=y_mask)
yp_mask = pool.compute_mask(y, y_mask)

print("MaskedMaxPooling1D output:", yp.shape)
print("MaskedMaxPooling1D mask:", yp_mask.shape)

bn = MaskedBatchNorm()
yb = bn(y, mask=y_mask, training=True)

print("MaskedBatchNorm output:", yb.shape)

gap = MaskedGlobalAvgPooling()
ygap = gap(y, mask=y_mask)

print("MaskedGlobalAvgPooling output:", ygap.shape)

gated_pool = GatedFrameGlobalMaxPooling(return_gate=True)
ygated, gates = gated_pool(y, mask=y_mask)

print("GatedFrameGlobalMaxPooling output:", ygated.shape)
print("Gates:", gates.shape)
print("Gate sums:", tf.reduce_sum(gates, axis=1).numpy())

resblock = ResidualBlock(
    filters=8,
    kernel_size=3,
    padding="same",
    use_1x1conv=True,
)

yr = resblock(x, mask=mask, training=True)
yr_mask = resblock.compute_mask(x, mask)

print("ResidualBlock output:", yr.shape)
print("ResidualBlock mask:", yr_mask.shape)

pos = SinusoidalPositionEmbedding()
pos_encoding = pos(y)

print("Position encoding:", pos_encoding.shape)

transformer = TransformerEncoder(
    embed_dim=8,
    num_heads=2,
    feed_forward_dim=32,
    dropout_rate=0.1,
)

yt = transformer(y, mask=y_mask, training=True)

print("TransformerEncoder output:", yt.shape)

# --- CrossFrameAttention ---
cfa = CrossFrameAttention(
    embed_dim=8,
    num_heads=2,
    feed_forward_dim=32,
    dropout_rate=0.1,
)
yc = cfa(y, training=True)
print("CrossFrameAttention output:", yc.shape)

# --- AxialAttention ---
aa = AxialAttention(
    embed_dim=8,
    num_heads=2,
    feed_forward_dim=32,
    dropout_rate=0.1,
    num_blocks=1,
)
ya = aa(y, training=True)
print("AxialAttention output:", ya.shape)

# Verify shapes
assert yc.shape == y.shape, f"CrossFrameAttention shape mismatch: {yc.shape} vs {y.shape}"
assert ya.shape == y.shape, f"AxialAttention shape mismatch: {ya.shape} vs {y.shape}"
print("\nAll checks passed.")