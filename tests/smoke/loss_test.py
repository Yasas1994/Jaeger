import tensorflow as tf
from jaeger.nnlib.v2.losses import SupervisedContrastiveLoss, ArcFaceLoss, HierarchicalLoss  
tf.random.set_seed(7)

batch_size = 8
embedding_dim = 16
num_classes = 4

labels_sparse = tf.constant([0, 0, 1, 1, 2, 2, 3, 3], dtype=tf.int32)
labels_onehot = tf.one_hot(labels_sparse, depth=num_classes)

embeddings = tf.random.normal([batch_size, embedding_dim])
fine_logits = tf.random.normal([batch_size, 6])

# 1. Supervised contrastive loss
supcon = SupervisedContrastiveLoss(temperature=0.1)

loss_sparse = supcon(labels_sparse, embeddings)
loss_onehot = supcon(labels_onehot, embeddings)

print("SupCon sparse labels:", float(loss_sparse))
print("SupCon one-hot labels:", float(loss_onehot))

# 2. ArcFace trainable loss layer
arcface = ArcFaceLoss(
    num_classes=num_classes,
    embedding_dim=embedding_dim,
    margin=0.5,
    scale=30.0,
)

with tf.GradientTape() as tape:
    tape.watch(embeddings)
    arc_loss = arcface(labels_onehot, embeddings)

grads = tape.gradient(arc_loss, [embeddings, arcface.class_weights])

print("ArcFace loss:", float(arc_loss))
print("ArcFace embedding grad is not None:", grads[0] is not None)
print("ArcFace class weight grad is not None:", grads[1] is not None)

# 3. Hierarchical loss
parent_of = [0, 0, 1, 1, 2, 2]
groups = [[0, 1], [2, 3], [4, 5]]

fine_labels = tf.constant([0, 1, 2, 3, 4, 5, 0, 2], dtype=tf.int32)

hier_loss = HierarchicalLoss(
    parent_of=parent_of,
    groups=groups,
    l_fine=1.0,
    l_coarse=1.5,
)

loss_value = hier_loss(fine_labels, fine_logits)
per_example_loss = hier_loss.call(fine_labels, fine_logits)

print("Hierarchical reduced loss:", float(loss_value))
print("Hierarchical per-example shape:", per_example_loss.shape)
print("Hierarchical per-example loss:", per_example_loss.numpy())