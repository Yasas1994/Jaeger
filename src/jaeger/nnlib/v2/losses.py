import tensorflow as tf


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        labels = tf.math.argmax(labels, axis=-1)
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return npairs_loss(tf.squeeze(labels), logits)


def npairs_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Expand to [batch_size, 1]
    y_true = tf.expand_dims(y_true, -1)
    y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    return tf.math.reduce_mean(loss)


class ArcFaceLoss(tf.keras.layers.Layer):
    def __init__(
        self, num_classes, embedding_dim, margin=0.5, scale=30.0, onehot=True, **kwargs
    ):
        super(ArcFaceLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.embedding_dim = embedding_dim
        self.onehot = onehot

        # Class weights are usually kept in float32 even in mixed precision
        self.class_weights = self.add_weight(
            name="class_weights",
            shape=(self.num_classes, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, labels, embeddings):
        """
        :param embeddings: (batch_size, embedding_dim)
        :param labels: either:
            - one-hot labels (batch_size, num_classes) if self.onehot=True
            - integer labels (batch_size,) or (batch_size, 1) if self.onehot=False
        """
        # Work in the compute dtype (usually float16 in mixed precision)
        compute_dtype = embeddings.dtype

        # eps depends on dtype
        eps = tf.constant(
            6.55e-4 if compute_dtype == tf.float16 else 1.0e-9,
            dtype=compute_dtype,
        )

        # Normalize embeddings and class weights in compute dtype
        embeddings = tf.cast(embeddings, compute_dtype)
        class_weights = tf.cast(self.class_weights, compute_dtype)

        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        class_weights = tf.nn.l2_normalize(class_weights, axis=1)

        # Cosine similarity: (batch_size, num_classes)
        cosine = tf.matmul(embeddings, class_weights, transpose_b=True)

        # Labels -> one-hot
        if self.onehot:
            # Assume labels already one-hot, just cast
            labels_one_hot = tf.cast(labels, compute_dtype)
        else:
            labels = tf.reshape(labels, [-1])             # (batch_size,)
            labels = tf.cast(labels, tf.int32)
            labels_one_hot = tf.one_hot(
                labels, depth=self.num_classes, dtype=compute_dtype
            )

        # Angle and margin
        theta = tf.acos(tf.clip_by_value(cosine, -1.0 + eps, 1.0 - eps))
        target_logits = tf.cos(theta + self.margin)

        # Construct logits
        logits = cosine * (1.0 - labels_one_hot) + target_logits * labels_one_hot

        # Apply scaling
        logits = logits * self.scale

        # ---- IMPORTANT PART FOR MIXED PRECISION ----
        # Do the actual loss calculation in float32
        logits_fp32 = tf.cast(logits, tf.float32)
        labels_fp32 = tf.cast(labels_one_hot, tf.float32)

        loss_vec = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_fp32, logits=logits_fp32
        )
        loss = tf.reduce_mean(loss_vec)

        # Always return float32 loss
        return loss

class HierarchicalLoss(tf.keras.losses.Loss):
    def __init__(self, parent_of, groups, l_fine=1.0, l_coarse=1.5, name="hier_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.parent_of = tf.constant(parent_of, tf.int32)          # (6,)
        self.groups = [tf.constant(g, tf.int32) for g in groups]   # list of tensors
        self.l_fine = float(l_fine)
        self.l_coarse = float(l_coarse)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def call(self, y_true, fine_logits):
        # accept sparse ids (N,) or one-hot (N,6)
        y_true = tf.convert_to_tensor(y_true)
        if y_true.shape.rank == 2:
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        else:
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Fine-level CE (per-sample)
        loss_fine = self.ce(y_true, fine_logits)  # (N,)

        # Coarse logits via logsumexp over each group's fine logits
        coarse_logits = tf.stack(
            [tf.reduce_logsumexp(tf.gather(fine_logits, idxs, axis=1), axis=1)
             for idxs in self.groups],
            axis=1
        )  # (N, 3)

        y_coarse = tf.gather(self.parent_of, y_true)  # (N,)
        loss_coarse = self.ce(y_coarse, coarse_logits)  # (N,)

        per_ex = self.l_fine * loss_fine + self.l_coarse * loss_coarse  # (N,)
        return tf.reduce_mean(per_ex)  # scalar
