import tensorflow as tf


def pairwise_distances(x, y):
    """
    Computes the pairwise squared Euclidean distances between tensors x and y.
    """
    x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)  # (N, 1)
    y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True) if y is not None else x_norm
    y_t = tf.transpose(y) if y is not None else tf.transpose(x)
    dist = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y_t)
    
    return tf.maximum(dist, 0.0)

def compute_softdtw(D, gamma):
    """
    Vectorized implementation of Soft-DTW for a batch of distance matrices.
    
    Args:
        D: A batch of distance matrices of shape (batch_size, N, N).
        gamma: Smoothing parameter.

    Returns:
        A scalar representing the average Soft-DTW loss over the batch.
    """
    batch_size, N, M = tf.shape(D)[0], tf.shape(D)[1], tf.shape(D)[2]

    # Initialize R matrices for the batch with float32 type
    R = tf.constant(float('inf'), shape=[batch_size, N + 2, M + 2], dtype=tf.float32)
    R = tf.tensor_scatter_nd_update(R, [[b, 0, 0] for b in range(batch_size)], tf.zeros(batch_size, dtype=tf.float32))

    # Iterative computation over i and j
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[:, i - 1, j - 1] / gamma
            r1 = -R[:, i - 1, j] / gamma
            r2 = -R[:, i, j - 1] / gamma
            rmax = tf.reduce_max(tf.stack([r0, r1, r2], axis=-1), axis=-1)
            rsum = tf.reduce_sum(tf.exp(tf.stack([r0 - rmax, r1 - rmax, r2 - rmax], axis=-1)), axis=-1)
            softmin = -gamma * (tf.math.log(rsum) + rmax)
            R = tf.tensor_scatter_nd_update(R, [[b, i, j] for b in range(batch_size)], D[:, i - 1, j - 1] + softmin)

    # Extract final loss values from R matrices
    losses = R[:, -2, -2]
    return tf.reduce_mean(losses)