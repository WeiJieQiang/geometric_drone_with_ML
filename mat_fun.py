import numpy as np
from numpy.linalg import norm as no

def w_hat(w):
    omega_hat = np.array([0., -w[2], w[1], w[2], 0., -w[0], -w[1], w[0], 0.]).reshape(3, 3)
    return omega_hat


def matrix_hat_inv(w):
    # w: skew-symmetric matrix
    return np.array([w[2][1], w[0][2], w[1][0]])


def _derive_normed_vector(v, dv):
    return no(v) ** (-3) * (no(v) ** 2 * dv - np.dot(dv,v) * v)

def _derive_derived_normed_vector(v, dv, ddv):
    return (no(v)**(-3) * (np.dot(2*dv,v) * dv + no(v)**2 * ddv
                           - ((np.dot(ddv,v) + np.dot(dv,dv))*v + np.dot(dv,v) * dv))
            - 3 * np.dot(dv,v) * no(v)**(-5) * (no(v)**2 * dv - np.dot(dv,v) * v))

