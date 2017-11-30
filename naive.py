"""
Routing naive implementation

Created on Sat Nov 23 17:15:01 2017

@author: - Ruslan Grimov
"""

import numpy as np


def do_routing(x, w, r_num=3):
    '''
    Functon that calculate output using r_num loops of routing

    bt - batch size
    ch_i - number of capsules channels in layer I
    n_i - number of neurons in a capsule of layer I
    ch_j - number of capsules channels in layer J
    n_j - number of neurons in a capsule of layer J

    Parameters
    ----------
    x : ndarray
        Input activations. x must have shape [bt, ch_i, n_i]
    w : array_like
        Weights matrix of shape [ch_i, n_i, ch_j, n_j].

    Returns
    -------
    v : ndarray
        Activations of output neurons. `v` will have shape [bt, ch_j, n_j]
    '''

    eps = np.finfo(float).eps

    def nsoftmax(x):
        nonlocal eps
        ex = np.exp(x * 8)
        return ex / (ex.sum(axis=-1, keepdims=True) + eps)

    def nsquash(x):
        nonlocal eps
        xq = np.sum(np.square(x), axis=-1, keepdims=True)
        return xq / (1 + xq) * x / (np.sqrt(xq) + eps)

    bt, ch_i, n_i = x.shape
    _, _, ch_j, n_j = w.shape

    # transpose and reshape for convenience
    xr = x.reshape((bt, ch_i, 1, 1, n_i))
    wr = w.transpose([0, 2, 3, 1]).reshape((1, ch_i, ch_j * n_j, n_i, 1))

    # calculate predictions u[bt][I, J][n_j]
    u = np.matmul(xr, wr).reshape((bt, ch_i, ch_j, n_j))

    # initialize b with zeroes
    b = np.zeros((bt, ch_i, ch_j))
    for r in range(r_num):
        # calculate routing coefficients
        c = nsoftmax(b) * ch_j

        # multiply predictions by their coefficients and summarize
        s = np.sum(np.expand_dims(c, axis=-1) * u, axis=-3)

        # squash vectors s
        v = nsquash(s)
        if r == r_num - 1:
            break

        a = np.matmul(v.reshape((bt, 1, ch_j, 1, n_j)),
                      u.reshape((bt, ch_i, ch_j, n_j, 1))).reshape((bt, ch_i, ch_j))

        # update b
        b = b + a

    return v
