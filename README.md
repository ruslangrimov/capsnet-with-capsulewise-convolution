# CapsNet with capsule-wise convolution

## Train on padded and translated MNIST and then test on affNIST

### Generate train data based on standard MNIST dataset

Create dataset in which each example is an MNIST digit placed randomly on a black background of 40×40 pixels

Below is content of generate_datasets.py
'''python
import numpy as np
from keras.datasets import mnist

(t_x_train, t_y_train), _ = mnist.load_data()

t_x_train = np.repeat(t_x_train, 8, axis=0)
x_train = np.zeros((t_x_train.shape[0], 40, 40))

for i in range(0, x_train.shape[0]):
    x, y = np.random.randint(0, 12, 2)
    x_train[i, y:y+28, x:x+28] = t_x_train[i]

y_train = np.repeat(t_y_train, 8, axis=0)

np.save('generateddatasets/x_train_only_translation.npy',
        x_train.astype(np.uint8))
np.save('generateddatasets/y_train_only_translation.npy',
        y_train.astype(np.uint8))
'''

### Train CapsNet with 1 conv layer, 4 convcaps layers and 1 dense caps layer with routing only on the last layer

'''python
l2 = regularizers.l2(l=0.001)

inp = Input(shape=input_shape)
l = inp

l = Conv2D(16, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2)(l)  # common conv layer
l = BatchNormalization()(l)
l = ConvertToCaps()(l)
l = Conv2DCaps(6, 4, (3, 3), (2, 2), r_num=1, b_alphas=[1, 1, 1], kernel_regularizer=l2)(l)
l = Conv2DCaps(5, 5, (3, 3), (1, 1), r_num=1, b_alphas=[1, 1, 1], kernel_regularizer=l2)(l)
l = Conv2DCaps(4, 6, (3, 3), (1, 1), r_num=1, b_alphas=[1, 1, 1], kernel_regularizer=l2)(l)
l = Conv2DCaps(3, 7, (3, 3), (1, 1), r_num=1, b_alphas=[1, 1, 1], kernel_regularizer=l2)(l)

l = FlattenCaps()(l)  # transform to a dense caps layer
l = DenseCaps(10, 8, r_num=3, b_alphas=[1, 8, 8], kernel_regularizer=l2)(l)
l = CapsToScalars()(l)

model = Model(inputs=inp, outputs=l, name='40x40_input_capsnet')
model.summary()
'''

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 40, 40, 1)         0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 18, 18, 16)        416       
_________________________________________________________________
batch_normalization_14 (Batc (None, 18, 18, 16)        64        
_________________________________________________________________
convert_to_caps_2 (ConvertTo (None, 18, 18, 16, 1)     0         
_________________________________________________________________
conv2d_caps_5 (Conv2DCaps)   (None, 8, 8, 6, 4)        3456      
_________________________________________________________________
conv2d_caps_6 (Conv2DCaps)   (None, 6, 6, 5, 5)        5400      
_________________________________________________________________
conv2d_caps_7 (Conv2DCaps)   (None, 4, 4, 4, 6)        5400      
_________________________________________________________________
conv2d_caps_8 (Conv2DCaps)   (None, 2, 2, 3, 7)        4536      
_________________________________________________________________
flatten_caps_2 (FlattenCaps) (None, 12, 7)             0         
_________________________________________________________________
dense_caps_2 (DenseCaps)     (None, 10, 8)             6720      
_________________________________________________________________
caps_to_scalars_2 (CapsToSca (None, 10)                0         
=================================================================
Total params: 25,992
Trainable params: 25,960
Non-trainable params: 32
'''

See affNIST_test_capsnet.ipynb fore more information

This model achieved 0.9772 accuracy on train set and 0.9796 on validation set

### Results on affNIST
'''
Test score:  3.78991742519
Test accuracy:  **0.704396875**
'''



