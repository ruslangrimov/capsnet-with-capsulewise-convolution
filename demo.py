"""
Demo of network with 5x5 convolutional layer, two 3x3 caps layers with
capsule-wise convolution and no routing and a capslayer with routing

Created on Sat Nov 24 16:35:22 2017

@author: - Ruslan Grimov
"""

from keras import backend as K
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D, Activation, Dense, Dropout
from keras.layers import BatchNormalization, MaxPooling2D, Flatten
from convcaps.capslayers import ConvertToCaps, Conv2DCaps, FlattenCaps
from convcaps.capslayers import DenseCaps, CapsToScalars
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras import regularizers
import numpy as np
import tensorflow as tf

img_rows, img_cols = 28, 28
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# since we use only tf the channel is last
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 32
num_epochs = 75

inp = Input(batch_shape=(batch_size,)+input_shape)
l = inp

l = Conv2D(8, (5, 5), strides=(2, 2), activation='relu')(l)  # common conv layer
l = BatchNormalization()(l)
l = ConvertToCaps()(l)
l = Conv2DCaps(4, 4, (3, 3), (2, 2), r_num=1)(l)
l = Conv2DCaps(2, 6, (3, 3), (2, 2), r_num=1)(l)
l = FlattenCaps()(l)  # transform to a dense caps layer
l = DenseCaps(10, 8, r_num=3)(l)
l = CapsToScalars()(l)

model = Model(inputs=inp, outputs=l)

model.summary()

def custom_objective(y_true, y_pred):
    return K.sum(y_true * K.clip(0.8 - y_pred, 0, 1) ** 2 + 0.5 * (1 - y_true) * K.clip(y_pred - 0.3, 0, 1) ** 2)

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss=custom_objective, #'categorical_crossentropy',  # 'mean_squared_error',
              optimizer=optimizer,
              metrics=['accuracy'])

x_train = x_train[:57600]
y_train = y_train[:57600]

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=num_epochs, initial_epoch=0,
          verbose=1, validation_split=0.09,
          callbacks=[
                     ModelCheckpoint('cc_weights.{epoch:02d}-{loss:.4f}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5',
                                     monitor='loss,acc,val_loss,val_acc', verbose=0),
                     #TensorBoard(log_dir='/opt/notebooks/logs/tensorflow/',
                     #            histogram_freq=1,
                     #            write_grads=True,
                     #            batch_size=batch_size,
                     #            write_images=True)
                     ])

sz = x_test.shape[0] // batch_size * batch_size
score = model.evaluate(x_test[:sz], y_test[:sz], verbose=1)
print('\n')
print('Test score: ', score[0])
print('Test accuracy: ', score[1])
