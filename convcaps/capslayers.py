"""
Created on Sat Nov 24 17:22:14 2017

@author: - Ruslan Grimov
"""

import keras.backend as K

from keras.layers import Layer
from keras.layers import InputSpec

from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils

import numpy as np

import tensorflow as tf

cf = K.image_data_format() == 'channels_first'


def squeeze(s):
    sq = K.sum(K.square(s), axis=-1, keepdims=True)
    return (sq / (1 + sq)) * (s / K.sqrt(sq + K.epsilon()))


class ConvertToCaps(Layer):
    def __init__(self, **kwargs):
        super(ConvertToCaps, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(1 if cf else len(output_shape), 1)
        return tuple(output_shape)

    def call(self, inputs):
        return K.expand_dims(inputs, 1 if cf else -1)


class FlattenCaps(Layer):
    def __init__(self, **kwargs):
        super(FlattenCaps, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=4)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:-1]), input_shape[-1])

    def call(self, inputs):
        shape = K.int_shape(inputs)
        return K.reshape(inputs, (-1, np.prod(shape[1:-1]), shape[-1]))


class CapsToScalars(Layer):
    def __init__(self, **kwargs):
        super(CapsToScalars, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def call(self, inputs):
        return K.sqrt(K.sum(K.square(inputs + K.epsilon()), axis=-1))


class Conv2DCaps(Layer):
    def __init__(self, ch_j, n_j,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 r_num=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(Conv2DCaps, self).__init__(**kwargs)
        rank = 2
        self.ch_j = ch_j  # Number of capsules in layer J
        self.n_j = n_j  # Number of neurons in a capsule in J
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.r_num = r_num
        self.padding = conv_utils.normalize_padding('valid')
        self.data_format = conv_utils.normalize_data_format('channels_last')
        self.dilation_rate = (1, 1)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=rank + 3)

    def build(self, input_shape):
        # throw exeption here of non implementation for others than tf backends

        # size of layer I
        # ch_i - number of capsules (channels) is layer I
        # n_i - number of neuronts in capsule in I
        self.h_i, self.w_i, self.ch_i, self.n_i = input_shape[1:5]

        # calculate the size of output image
        self.h_j, self.w_j = [conv_utils.conv_output_length(input_shape[i + 1],
                              self.kernel_size[i],
                              padding=self.padding,
                              stride=self.strides[i],
                              dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.ah_j, self.aw_j = [conv_utils.conv_output_length(input_shape[i + 1],
                              self.kernel_size[i],
                              padding=self.padding,
                              stride=1,
                              dilation=self.dilation_rate[i]) for i in (0, 1)]

        # self.ah_j = (self.h_j - 1) * self.strides[0] + 1
        # self.aw_j = (self.w_j - 1) * self.strides[1] + 1

        #print((self.h_i, self.w_i), (self.h_j, self.w_j),
        #      (self.ah_j, self.aw_j))

        self.w_shape = self.kernel_size + (self.ch_i, self.n_i,
                                           self.ch_j, self.n_j)

        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def call(self, inputs):
        if self.r_num == 1:
            # if there is no routing (and this is so when r_num is 1 and all c are equal)
            # then this is a common convolution
            outputs = K.conv2d(K.reshape(inputs, (-1, self.h_i, self.w_i,
                                                  self.ch_i*self.n_i)),
                               K.reshape(self.w, self.kernel_size +
                                         (self.ch_i*self.n_i, self.ch_j*self.n_j)),
                               data_format='channels_last',
                               strides=self.strides,
                               padding=self.padding,
                               dilation_rate=self.dilation_rate)

            outputs = squeeze(K.reshape(outputs, ((-1, self.h_j, self.w_j,
                                          self.ch_j, self.n_j))))

        else:
            inputsr = K.reshape(inputs, (-1, self.h_i * self.w_i * self.ch_i * self.n_i))
            bt = K.int_shape(inputsr)[0]

            wi = np.zeros((self.kernel_size[0], self.kernel_size[1], self.ch_i,
                           self.kernel_size[0], self.kernel_size[1], self.ch_j,
                           self.n_j, self.ch_j, self.n_j))
            for r in range(self.kernel_size[0]):
                for c in range(self.kernel_size[1]):
                    for chj in range(self.ch_j):
                        for nj in range(self.n_j):
                            wi[r, c, :, r, c, chj, nj, chj, nj] = 1.0

            wi = K.constant(wi)
            wi = K.reshape(wi, (self.kernel_size[0], self.kernel_size[1],
                                self.ch_i*self.kernel_size[0]*self.kernel_size[1]*
                                self.ch_j*self.n_j, self.ch_j*self.n_j))

            wt = K.permute_dimensions(self.w, [2, 3, 0, 1, 4, 5])
            wt = K.reshape(wt, (self.ch_i, 1, 1, self.n_i, -1))

            bp = bt
            def fn(binputs):
                binputs = K.reshape(binputs, (-1, self.h_i, self.w_i, self.ch_i, self.n_i))
                ul = []
                for i in range(self.ch_i):
                    ul.append(K.conv2d(binputs[:, :, :, i], wt[i],
                                       data_format='channels_last'))
                u = K.stack(ul, axis=3)

                u_wo_g = K.stop_gradient(u)

                j_all = self.h_j*self.w_j*self.ch_j
                j_add = j_all - self.kernel_size[0]*self.kernel_size[1]*self.ch_j

                b = tf.constant_initializer(0.)((bp, self.h_i*self.w_i*self.ch_i,
                             self.kernel_size[0]*self.kernel_size[1]*self.ch_j))

                for r in range(self.r_num):
                    #c = ((K.softmax(b) - 0.9) * 10) * K.int_shape(b)[-1] * 10
                    ex = K.exp(b * 8)
                    c = ex / ((K.sum(ex, axis=-1, keepdims=True) + K.epsilon()) +
                              j_add) * j_all

                    c = K.expand_dims(c)
                    c = K.stop_gradient(c)
                    if r == self.r_num - 1:
                        cu = c * K.reshape(u, (-1, self.h_i*self.w_i*self.ch_i,
                                               self.kernel_size[0]*
                                               self.kernel_size[1]*
                                               self.ch_j, self.n_j))
                    else:
                        cu = c * K.reshape(u_wo_g, (-1, self.h_i*self.w_i*self.ch_i,
                                                    self.kernel_size[0]*
                                                    self.kernel_size[1]*self.ch_j,
                                                    self.n_j))

                    cu = K.reshape(cu, (-1, self.h_i, self.w_i,
                                        self.ch_i*self.kernel_size[0]*
                                        self.kernel_size[1]*self.ch_j*self.n_j))
                    s = K.conv2d(cu, wi, data_format='channels_last', strides=self.strides)
                    v = squeeze(K.reshape(s, (-1, self.h_j*self.w_j*self.ch_j,
                                              self.n_j)))
                    v = K.reshape(v, (-1, self.h_j, self.w_j, self.ch_j*self.n_j))

                    if r == self.r_num - 1:
                        break

                    v = K.stop_gradient(v)
                    ph, pw = self.kernel_size[0] - 1, self.kernel_size[1] - 1

                    if self.strides == (1, 1):
                        va = v
                    else:
                        zr = K.zeros((bt, self.w_j, self.ch_j * self.n_j))
                        zc = K.zeros((bt, self.ah_j, self.ch_j * self.n_j))
                        rl = []

                        for r in range(self.ah_j):
                            rl.append(zr if r % ph else v[:, r // ph])
                        rs = K.stack(rl, axis=1)
                        cl = []
                        for c in range(self.aw_j):
                            cl.append(zc if c % pw else rs[:, :, c // pw])
                        va = K.stack(cl, axis=-2)

                    va = K.spatial_2d_padding(va, ((ph, ph), (pw, pw)),
                                              data_format='channels_last')

                    vp = tf.extract_image_patches(va, (1, self.kernel_size[0],
                                                       self.kernel_size[1], 1),
                                                  (1, 1, 1, 1),
                                                  (1, 1, 1, 1), 'VALID')
                    vp = K.reshape(vp, (-1, self.h_i*self.w_i, self.kernel_size[0]*
                                        self.kernel_size[1],
                                        self.ch_j, 1, self.n_j))
                    vp = K.reverse(vp, axes=[2])

                    vp = K.reshape(vp, (-1, self.h_i*self.w_i,
                                        self.kernel_size[0]*
                                        self.kernel_size[1]*
                                        self.ch_j, 1, self.n_j))

                    u_wo_g = K.reshape(u_wo_g, (-1, self.h_i*self.w_i, self.ch_i,
                                                self.kernel_size[0]*
                                                self.kernel_size[1]*
                                                self.ch_j, self.n_j, 1))
                    al = []
                    for i in range(self.ch_i):
                        al.append(K.batch_dot(vp, u_wo_g[:, :, i]))
                    a = K.stack(al, axis=2)
                    a = K.reshape(a, (-1, self.h_i*self.w_i*self.ch_i,
                                      self.kernel_size[0]*
                                      self.kernel_size[1]*self.ch_j))

                    #b = K.update_add(b, a)  # WTF? It yields a result different from b = b + a
                    b = b + a

                return v


            finputs = K.reshape(inputs, (bt // bp, bp, self.h_i, self.w_i, self.ch_i, self.n_i))
            v = fn(finputs)
            #v = tf.map_fn(fn, finputs,
            #              parallel_iterations=100, back_prop=True,
            #              infer_shape=True)

            outputs = v
            outputs = K.reshape(outputs, (-1, self.h_j, self.w_j,
                                          self.ch_j, self.n_j))
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.h_j, self.w_j, self.ch_j, self.n_j)

    def get_config(self):
        config = {
            'ch_j': self.ch_j,
            'n_j': self.n_j,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            #'padding': self.padding,
            #'data_format': self.data_format,
            #'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(Conv2DCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseCaps(Layer):
    def __init__(self, ch_j, n_j,
                 r_num=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseCaps, self).__init__(**kwargs)
        self.ch_j = ch_j  # Number of capsules in layer J
        self.n_j = n_j  # Number of neurons in a capsule in J
        self.r_num = r_num
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        self.ch_i, self.n_i = input_shape[1:]

        self.w_shape = (self.ch_i, self.n_i, self.ch_j, self.n_j)

        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def call(self, inputs):
        if self.r_num == 1:
            outputs = K.dot(K.reshape(inputs, (-1, self.ch_i*self.n_i)),
                           K.reshape(self.w, (self.ch_i*self.n_i,
                                              self.ch_j*self.n_j)))
            outputs = squeeze(K.reshape(outputs, (-1, self.ch_j, self.n_j)))
        else:
            wr = K.reshape(self.w, (self.ch_i, self.n_i, self.ch_j * self.n_j))

            u = tf.transpose(tf.matmul(tf.transpose(inputs, [1, 0, 2]), wr), [1, 0, 2])

            u = K.reshape(u, (-1, self.ch_i, self.ch_j, self.n_j))

            def rt(ub):
                # b = K.zeros((self.ch_i, self.ch_j))
                b = tf.constant_initializer(0.)((self.ch_i, self.ch_j))
                ub = K.reshape(ub, (-1, self.ch_i, self.ch_j, self.n_j))
                ub_wo_g = K.stop_gradient(ub)
                for r in range(self.r_num):
                    c = K.expand_dims(K.softmax(b * 8)) * self.ch_j # distribution of weighs of capsules in I across capsules in J
                    c = K.stop_gradient(c)
                    if r == self.r_num - 1:
                        cub = c * ub
                    else:
                        cub = c * ub_wo_g
                    s = K.sum(cub, axis=-3) # vectors of capsules in J
                    v = squeeze(s)  # squeezed vectors of capsules in J
                    if r == self.r_num - 1:
                        break

                    v = K.stop_gradient(v)

                    a = tf.einsum('bjk,bijk->bij', v, ub)  # a = v dot u
                    #a = K.matmul(K.reshape(v, (-1, 1, J, 1, n_j)),
                    #             K.reshape(u, (-1, I, J, n_j, 1))).reshape((-1, I, J))

                    b = b + a  # increase those b[i,j] where v[j] dot b[i,j] is larger
                return v

            u = K.reshape(u, (-1, self.ch_i* self.ch_j * self.n_j))
            outputs = tf.map_fn(rt, u,
                          parallel_iterations=1000, back_prop=True,
                          infer_shape=False)

            outputs = K.reshape(outputs, (-1, self.ch_j, self.n_j))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ch_j, self.n_j)

    def get_config(self):
        config = {
            'ch_j': self.ch_j,
            'n_j': self.n_j,
            'r_num': self.r_num,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(DenseCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
