import numpy as np
import tensorflow as tf


def test_tf():
    try:
        # Set the matrix size
        matrix_size = 100

        # Create two random matrices
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)

        with tf.device("CPU:0"):
            result = tf.matmul(matrix_a, matrix_b)

    except Exception as e:
        return e
    else:
        return result
