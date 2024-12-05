import tensorflow as tf
from code.loss.soft_dtw_tf import pairwise_distances, compute_softdtw
from code.loss.path_soft_dtw_tf import PathDTWBatchTF


def dilate_loss(outputs, targets, alpha=0.5, gamma=0.01):
    """
    Compute DILATE loss.

    Args:
        outputs: Tensor of shape (batch_size, N_output, 1)
        targets: Tensor of shape (batch_size, N_output, 1)
        alpha: Weighting factor for shape vs temporal losses
        gamma: SoftDTW gamma parameter

    Returns:
        loss: Combined DILATE loss
        loss_shape: Shape loss component
        loss_temporal: Temporal loss component
    """
    batch_size = tf.shape(outputs)[0]
    N_output = tf.cast(tf.shape(outputs)[1], tf.float32)  # Ensure N_output is float32

    # Ensure N_output is an int32 for TensorFlow operations that require shape
    N_output_int = tf.cast(N_output, tf.int32)

    # Compute pairwise distances for the batch
    D = tf.zeros((batch_size, N_output_int, N_output_int), dtype=tf.float32)
    for k in range(batch_size):
        Dk = pairwise_distances(targets[k, :, :], outputs[k, :, :])
        D = tf.tensor_scatter_nd_update(D, [[k]], [Dk])

    # Shape loss using SoftDTW
    loss_shape = compute_softdtw(D, gamma)

    # Temporal loss
    path_dtw = PathDTWBatchTF(gamma)
    path = path_dtw(D)

    Omega = pairwise_distances(tf.range(1, N_output + 1, dtype=tf.float32)[:, tf.newaxis],
                               tf.range(1, N_output + 1, dtype=tf.float32)[:, tf.newaxis])
    loss_temporal = tf.reduce_sum(path * Omega) / tf.cast((N_output * N_output), tf.float32)

    # Combined loss
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal

    return loss, loss_shape, loss_temporal
