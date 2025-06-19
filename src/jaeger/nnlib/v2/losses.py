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
        """
        Initialize the ArcFaceLoss layer for supervised metric learning.

        :param num_classes: Number of classes.
        :param margin: Angular margin to add.
        :param scale: Scaling factor for the logits.
        :param kwargs: Additional keyword arguments.
        """
        super(ArcFaceLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.embedding_dim = embedding_dim
        # self.eps = 1e-7  # Small value to avoid division by zero
        self.onehot = onehot
        # Initialize class weights for the final fully connected layer
        self.class_weights = self.add_weight(
            name="class_weights",
            shape=(self.num_classes, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, labels, embeddings):
        """
        Compute the ArcFace loss.

        :param embeddings: Embeddings from the model (batch_size, embedding_dim).
        :param labels: True labels (batch_size,).
        :return: Computed ArcFace loss.
        """
        eps = 6.55e-4 if embeddings.dtype == "float16" else 1.0e-9
        # Normalize embeddings and class weights
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        class_weights = tf.nn.l2_normalize(self.class_weights, axis=1)
        class_weights = tf.cast(class_weights, dtype=embeddings.dtype)
        # Compute cosine similarity between embeddings and class weights
        cosine = tf.matmul(embeddings, class_weights, transpose_b=True)

        # Convert labels to one-hot encoding
        if self.onehot:
            labels_one_hot = tf.cast(labels, dtype=cosine.dtype)

        else:
            labels_one_hot = tf.squeeze(
                tf.one_hot(labels, depth=self.num_classes, axis=-1), axis=1
            )

        # Compute the angle (theta) and add margin
        theta = tf.acos(tf.clip_by_value(cosine, -1.0 + eps, 1.0 - eps))
        target_logits = tf.cos(theta + self.margin)

        # Construct logits
        logits = cosine * (1 - labels_one_hot) + target_logits * labels_one_hot

        # Apply scaling
        logits *= self.scale

        # Compute softmax cross-entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot, logits=logits
        )

        return tf.reduce_mean(loss)
