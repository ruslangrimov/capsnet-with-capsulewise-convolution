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

useGPU = False

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
                 b_alphas=[8, 8, 8],
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
        self.b_alphas = b_alphas
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
            bt = K.shape(inputs)[0]
            ksz = self.kernel_size[0]*self.kernel_size[1]

            xr = K.reshape(inputs, (-1, self.h_i, self.w_i,
                                    self.ch_i * self.n_i))

            pt = tf.extract_image_patches(xr, (1,)+self.kernel_size+(1,),
                                          (1,)+self.strides+(1,),
                                          (1, 1, 1, 1,),
                                          'VALID')

            pt = K.reshape(pt, (-1, ksz * self.ch_i, self.n_i))

            wr = K.reshape(self.w, (ksz * self.ch_i, self.n_i,
                                    self.ch_j * self.n_j))

            global useGPU

            # it sometimes works faster on GPU when batch is devided into two parts
            # bp = K.expand_dims(bt // 2, axis=0) if useGPU else K.constant([2], dtype=tf.int32)
            bp = K.expand_dims(bt // 1, axis=0) if useGPU else K.constant([2], dtype=tf.int32)

            if self.strides != (1, 1):
                zr_shape = K.concatenate([bp, K.constant([self.w_j,
                                       ksz * self.ch_i * self.ch_j],
                                      dtype=tf.int32)])
                zr = tf.zeros(shape=zr_shape)
                zc_shape = K.concatenate([bp, K.constant([self.ah_j,
                                       ksz * self.ch_i * self.ch_j],
                                      dtype=tf.int32)])
                zc = tf.zeros(shape=zc_shape)

            def rt(ptb):
                ptb = K.reshape(ptb, (-1, ksz * self.ch_i, self.n_i))

                if useGPU:
                    ub = tf.einsum('bin,inj->bij', ptb, wr)
                else:
                    ul = []
                    for i in range(ksz * self.ch_i):
                        ul.append(K.dot(ptb[:, i], wr[i]))
                    ub = K.stack(ul, axis=1)

                #b = tf.constant_initializer(0.)((bp, self.h_i*self.w_i*self.ch_i,
                #                           ksz * self.ch_j))
                b = 0.0

                j_all = self.h_j*self.w_j*self.ch_j
                j_add = j_all - ksz * self.ch_j

                for r in range(self.r_num):
                    ex = K.exp(b * self.b_alphas[r])
                    if r > 0:
                        c = ex / ((K.sum(ex, axis=-1, keepdims=True) + K.epsilon()) +
                                  j_add) * j_all
                        c = K.reshape(c, (-1, self.h_i, self.w_i, self.ch_i *
                                          ksz * self.ch_j))
                        c = K.stop_gradient(c)

                        pc = tf.extract_image_patches(c, (1,)+self.kernel_size+(1,),
                                                          (1,)+self.strides+(1,),
                                                          (1, 1, 1, 1,),
                                                          'VALID')
                        pc = K.reshape(pc, (-1, self.h_j, self.w_j, ksz,
                                            self.ch_i, self.kernel_size[0]*
                                            self.kernel_size[1], self.ch_j))
                        pcl = []
                        for n in range(ksz):
                            pcl.append(pc[:, :, :, n, :, self.kernel_size[0]*
                                          self.kernel_size[1]-1 - n])
                        pcc = K.stack(pcl, axis=3)

                        if useGPU:
                            pcc = K.reshape(pcc, (-1, self.h_j * self.w_j* ksz *
                                                  self.ch_i * self.ch_j, 1))
                            ub = K.reshape(ub, (-1, self.h_j * self.w_j *
                                                ksz * self.ch_i * self.ch_j,
                                                self.n_j))
                            cu = pcc * ub
                        else:
                            pcc = K.reshape(pcc, (-1, 1))
                            ub = K.reshape(ub, (-1, self.n_j, 1))
                            cul = []
                            for n in range(self.n_j):
                                cul.append(ub[:, n] * pcc)
                            cu = K.stack(cul, axis=-2)

                    else:
                        cu = ub

                    cu = K.reshape(cu, (-1, self.h_j*self.w_j,
                                        ksz*self.ch_i, self.ch_j, self.n_j))

                    s = K.sum(cu, axis=-3)

                    v = squeeze(s)
                    if r == self.r_num - 1:
                        break

                    v = K.stop_gradient(v)

                    ubr = K.reshape(K.stop_gradient(ub), (-1, self.h_j*self.w_j,
                                         ksz*self.ch_i, self.ch_j, self.n_j))

                    if True:
                    #if useGPU:
                        a = tf.einsum('bjck,bjick->bjic', v, ubr)
                    else:
                        al = []
                        for i in range(ksz * self.ch_i):
                            al.append(K.batch_dot(K.reshape(ubr[:, :, i], (-1, self.h_j*self.w_j*self.ch_j, 1, self.n_j)),
                                                  K.reshape(v, (-1, self.h_j*self.w_j*self.ch_j, self.n_j, 1))))
                        a = K.stack(al, axis=1)
                        a = K.reshape(a, (-1, ksz*self.ch_i, self.h_j*self.w_j,
                                          self.ch_j))
                        a = K.permute_dimensions(a, [0, 2, 1, 3])

                    ph, pw = 2, 2
                    a = K.reshape(a, (-1, self.h_j, self.w_j,
                                      ksz * self.ch_i * self.ch_j))

                    if self.strides == (1, 1):
                        aa = a
                    else:
                        rl = []

                        for r in range(self.ah_j):
                            rl.append(zr if r % ph else a[:, r // ph])
                        rs = K.stack(rl, axis=1)
                        cl = []
                        for c in range(self.aw_j):
                            cl.append(zc if c % pw else rs[:, :, c // pw])
                        aa = K.stack(cl, axis=-2)


                    aa = K.spatial_2d_padding(aa, ((ph, ph), (pw, pw)),
                                                          data_format='channels_last')
                    pa = tf.extract_image_patches(aa, (1,)+self.kernel_size+(1,),
                                                      (1, 1, 1, 1,), #(1,)+strides+(1,),
                                                      (1, 1, 1, 1,),
                                                      'VALID')
                    pa = K.reshape(pa, (-1, self.h_i*self.w_i,
                                        ksz, ksz, self.ch_i, self.ch_j))

                    pal = []
                    for n in range(ksz):
                        pal.append(pa[:, :, n, ksz-1 - n])
                    paa = K.stack(pal, axis=3)

                    paa = K.reshape(paa, (-1, self.h_i*self.w_i*self.ch_i,
                                          ksz*self.ch_j))
                    b = b + paa

                return v

            v = tf.map_fn(rt, K.reshape(pt, (-1, bp[0], self.h_j, self.w_j,
                                             ksz * self.ch_i, self.n_i)),
                          parallel_iterations=100, back_prop=True,
                          infer_shape=False)



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
            'b_alphas': self.b_alphas,
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
                 b_alphas=[8, 8, 8],
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseCaps, self).__init__(**kwargs)
        self.ch_j = ch_j  # number of capsules in layer J
        self.n_j = n_j  # number of neurons in a capsule in J
        self.r_num = r_num
        self.b_alphas = b_alphas
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
                ub = K.reshape(ub, (-1, self.ch_i, self.ch_j, self.n_j))
                ub_wo_g = K.stop_gradient(ub)
                b = 0.0
                for r in range(self.r_num):
                    if r > 0:
                        c = K.expand_dims(K.softmax(b * self.b_alphas[r])) * self.ch_j # distribution of weighs of capsules in I across capsules in J
                        c = K.stop_gradient(c)
                    else:
                        c = 1.0

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

            global useGPU

            if useGPU:
                outputs = rt(u)
            else:
                outputs = tf.map_fn(rt, u,
                                    parallel_iterations=100, back_prop=True,
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
            'b_alphas': self.b_alphas,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(DenseCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
