import numpy as np

def w_hat(w):
    omega_hat = np.array([0., -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0.]).reshape(3, 3)
    return omega_hat


def matrix_hat_inv(w):
    # w: skew-symmetric matrix
    return np.array([w[2][1], w[0][2], w[1][0]])