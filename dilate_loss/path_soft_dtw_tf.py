import numpy as np
import tensorflow as tf


'''
# Provided functions
def my_max_tf(x, gamma):
    # Use the log-sum-exp trick
    max_x = tf.reduce_max(x)
    exp_x = tf.exp((x - max_x) / gamma)
    Z = tf.reduce_sum(exp_x)
    return gamma * tf.math.log(Z) + max_x, exp_x / Z

def my_min_tf(x, gamma):
    min_x, argmax_x = my_max_tf(-x, gamma)
    return -min_x, argmax_x
'''

# Provided functions
def my_max(x, gamma):
    # Use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z

def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return -min_x, argmax_x


def dtw_grad_tf(theta, gamma):
    m = tf.shape(theta)[0]
    n = tf.shape(theta)[1]

    # Initialize V matrix
    V = tf.fill((m + 1, n + 1), 1e10)
    V = tf.tensor_scatter_nd_update(V, [[0, 0]], [0.0])

    # Initialize Q matrix
    Q = tf.zeros((m + 2, n + 2, 3), dtype=tf.float32)

    # Forward pass
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            values = tf.stack([
                V[i, j - 1],  # left
                V[i - 1, j - 1],  # diagonal
                V[i - 1, j]  # up
            ])
            min_val, softmin_weights = my_min(values, gamma)
            V = tf.tensor_scatter_nd_update(V, [[i, j]], [theta[i - 1, j - 1] + min_val])
            Q = tf.tensor_scatter_nd_update(Q, [[i, j]], [softmin_weights])

    # Backward pass
    E = tf.zeros((m + 2, n + 2), dtype=tf.float32)
    E = tf.tensor_scatter_nd_update(E, [[m + 1, n + 1]], [1.0])

    # Fix the Q update for [m + 1, n + 1]
    Q = tf.tensor_scatter_nd_update(Q, [[m + 1, n + 1]], [[1.0, 1.0, 1.0]])

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E = tf.tensor_scatter_nd_update(E, [[i, j]], [
                Q[i, j + 1, 0] * E[i, j + 1] +
                Q[i + 1, j + 1, 1] * E[i + 1, j + 1] +
                Q[i + 1, j, 2] * E[i + 1, j]
            ])

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


class PathDTWBatchTF(tf.keras.layers.Layer):
    def __init__(self, gamma):
        super(PathDTWBatchTF, self).__init__()
        self.gamma = gamma

    def call(self, D):
        batch_size = tf.shape(D)[0]
        N = tf.shape(D)[1]

        grad_list = []

        for k in range(batch_size):  # Loop over batch
            _, grad_k, Q_k, E_k = dtw_grad_tf(D[k, :, :], self.gamma)
            grad_list.append(grad_k)

        # Convert to tensors
        grad_tensor = tf.stack(grad_list, axis=0)

        # Return the mean gradient
        return tf.reduce_mean(grad_tensor, axis=0)    