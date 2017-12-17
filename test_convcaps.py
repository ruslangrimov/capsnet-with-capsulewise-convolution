"""
Created on Sat Nov 25 16:38:24 2017

@author: - Ruslan Grimov
"""

import pytest
from numpy.testing import assert_allclose
import convcaps.capslayers
from convcaps.capslayers import DenseCaps, Conv2DCaps
from naive import do_routing
import numpy as np
import keras.backend as K
from keras import initializers


@pytest.mark.parametrize("r_num", [1, 3])
def test_dense(r_num):
    bt = 8
    ch_i = 3
    n_i = 4
    ch_j = 2
    n_j = 5

    x = np.arange(bt * ch_i * n_i).reshape((bt, ch_i, n_i))
    x = x / x.max() / 100

    w = np.arange(ch_i * n_i * ch_j * n_j).reshape((ch_i, n_i, ch_j, n_j))
    w = w / w.max() / 100

    nv = do_routing(x, w, r_num)

    l = DenseCaps(ch_j, n_j, r_num=r_num,
                  kernel_initializer=initializers.Constant(w))
    inp = K.variable(x)
    out = l(inp)
    fn = K.function([inp], [out])

    rv = fn([x])[0]

    assert_allclose(rv, nv, rtol=1e-04, atol=1e-06)


@pytest.mark.parametrize("r_num", [1, 3])
@pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
@pytest.mark.parametrize("useGPU", [False, True])
def test_conv(r_num, strides, useGPU):
    bt = 8
    h_i, w_i = 7, 7
    ch_i = 3
    n_i = 4

    kh, kw = 3, 3

    ch_j = 2
    n_j = 5

    h_j = (h_i - kh) // strides[0] + 1
    w_j = (w_i - kw) // strides[1] + 1

    x = np.arange(bt * h_i * w_i * ch_i * n_i).reshape((bt, h_i, w_i, ch_i, n_i))
    x = x / x.max() / 100

    w = np.arange(kh * kw * ch_i * n_i * ch_j * n_j).reshape((kh, kw, ch_i, n_i, ch_j, n_j))
    w = w / w.max() / 100

    ew = np.zeros((h_i, w_i, ch_i, n_i, h_j, w_j, ch_j, n_j))

    for r in range(0, h_i - kh + 1, strides[0]):
        for c in range(0, w_i - kw + 1, strides[1]):
            ew[r:r+kh, c:c+kw, :, :, r // strides[0], c // strides[1]] = w

    ew = ew.reshape((h_i * w_i * ch_i, n_i, h_j * w_j * ch_j, n_j))
    rx = x.reshape((bt, h_i * w_i * ch_i, n_i))

    nv = do_routing(rx, ew, r_num)

    convcaps.capslayers.useGPU = useGPU
    l = Conv2DCaps(ch_j, n_j, kernel_size=(kh, kw), strides=strides, r_num=r_num,
                   kernel_initializer=initializers.Constant(w))

    inp = K.variable(x)
    out = l(inp)
    fn = K.function([inp], [out])

    rv = fn([x])[0]
    nv = nv.reshape(rv.shape)

    assert_allclose(rv, nv, rtol=1e-04, atol=1e-06)
