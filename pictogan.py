# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 03:27:13 2018

@author: Shriya - sbp2148
"""

import numpy as np
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization 
from keras.layers import Reshape, UpSampling2D, Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Embedding, merge
from keras.models import Model
from keras.layers import concatenate as kerascat
from keras import activations
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import h5py as h5py
from keras import initializers
from keras import optimizers
import cv2
import matplotlib.pyplot as plt


outcount = 1000
img_w, img_h = 28, 28
padding_color = [0, 0, 0]
input_shape = (img_w * 2, img_h * 2, 1)

#1. accident

input1 = np.load("./inputs/car.npy")
data1 = np.empty((outcount, img_w * 2, img_h * 2))
for i in range(outcount): 
    img = np.concatenate((input1[i].reshape(28, 28), input1[outcount - i - 1].reshape(28, 28)), axis=1)
    img = cv2.copyMakeBorder(img, 14, 14, 0, 0, cv2.BORDER_CONSTANT, value=padding_color)
    data1[i] = img/255
data1 = np.reshape(data1, [data1.shape[0], img_w * 2, img_h * 2, 1])

#2. Night and day

input1 = np.load("./inputs/moon.npy")
input2 = np.load("./inputs/sun.npy")
data2 = np.empty((outcount, img_w * 2, img_h * 2))
for i in range(outcount):
    img = np.concatenate((input1[i].reshape(28, 28), input2[i].reshape(28, 28)), axis=1)
    img = cv2.copyMakeBorder(img, 14, 14, 0, 0, cv2.BORDER_CONSTANT, value=padding_color)
    data2[i] = img/255
data2 = np.reshape(data2, [data2.shape[0], img_w * 2, img_h * 2, 1])

#3. spider on a tree

input1 = np.load("./inputs/spider.npy")
input2 = np.load("./inputs/tree.npy")
data3 = np.empty((outcount, img_w * 2, img_h * 2))
for i in range(outcount):   
    img = np.concatenate((input1[i].reshape(28, 28), input2[i].reshape(28, 28)), axis=0)
    img = cv2.copyMakeBorder(img, 0, 0, 14, 14, cv2.BORDER_CONSTANT, value=padding_color)
    data3[i] = img/255    
data3 = np.reshape(data3, [data3.shape[0], img_w * 2, img_h * 2, 1])

#3. fruits

input1 = np.load("./inputs/apple.npy")
input2 = np.load("./inputs/banana.npy")
input3 = np.load("./inputs/grapes.npy")
input4 = np.load("./inputs/strawberry.npy")
data4 = np.empty((outcount, img_w * 2, img_h * 2))
for i in range(outcount):    
    img1 = np.concatenate((input1[i].reshape(28, 28), input2[i].reshape(28, 28)), axis=1)
    img2 = np.concatenate((input3[i].reshape(28, 28), input4[i].reshape(28, 28)), axis=1)
    img = np.concatenate((img1, img2), axis=0)
    data4[i] = img/255
data4 = np.reshape(data4, [data4.shape[0], img_w * 2, img_h * 2, 1])

#Discriminator 

depth = 64
dropout = 0.4

myinput = Input(shape=input_shape)
x = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(myinput)
x = Dropout(dropout)(x)

x = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(x)
x = Dropout(dropout)(x)

x = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(x)
x = Dropout(dropout)(x)

x = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(x)
x = Dropout(dropout)(x)

x = Flatten()(x)

out1 = Dense(1, activation='sigmoid')(x)
out2 = Dense(4, activation='softplus')(x)

Dis = Model(inputs=[myinput], outputs=[out1, out2])

#Generator 

dropout = 0.4
depth = (64+64+64+64)*2
dim = 7

num_classes = 4
input1_dims = 1
input2_dims = 3500

input1 = Input(shape = (input1_dims,))
newinput1 = Flatten()(Embedding(num_classes, input2_dims, init='glorot_normal')(input1))

input2 = Input(shape = (input2_dims,))
myinput = merge([newinput1, input2], mode='dot')

x = myinput
#x = kerascat([input1, input2])
x = Dense(dim*dim*depth)(x)
x = Activation('relu')(x)
x = Reshape((dim, dim, depth))(x)
x = Dropout(dropout)(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(int(depth/2), 5, padding='same')(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(int(depth/4), 5, padding='same')(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(int(depth/8), 5, padding='same')(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = Conv2DTranspose(int(depth/16), 5, padding='same')(x)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)

x = Conv2DTranspose(1, 5, padding='same')(x)
x = Activation('sigmoid')(x)

Gen = Model(inputs = [input1, input2], outputs = [x])

#Discriminator Model

opt = optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
DM = Dis
DM.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

#Adversarial Model

opt = optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
AM = Model(inputs = Gen.input, outputs = Dis(Gen.output))
AM.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

#Training

epochs = 1000
batch = 500

out_count = 20
out_noise = np.random.uniform(-1.0, 1.0, size=[out_count, 3500])

for iters in range(epochs):
    
    real_imgs1 = np.reshape(data1[np.random.choice(data1.shape[0],batch,replace=True)],(batch, img_w * 2, img_h * 2, 1))
    real_imgs2 = np.reshape(data2[np.random.choice(data2.shape[0],batch,replace=True)],(batch, img_w * 2, img_h * 2, 1))
    real_imgs3 = np.reshape(data3[np.random.choice(data3.shape[0],batch,replace=True)],(batch, img_w * 2, img_h * 2, 1))
    real_imgs4 = np.reshape(data4[np.random.choice(data4.shape[0],batch,replace=True)],(batch, img_w * 2, img_h * 2, 1))
    
    noise = np.random.uniform(-1.0, 1.0, size=[4 * batch, 3500])
        
    labels = np.zeros([4 * batch, 1])
    labels[batch : 2 * batch, 0] = 1
    labels[2 * batch : 3 * batch, 0] = 2
    labels[3 * batch : , 0] = 3
    
    fake_imgs = Gen.predict([labels, noise])
    
    x = np.concatenate((real_imgs1, real_imgs2, real_imgs3, real_imgs4, fake_imgs))
    
    y1 = np.ones([8 * batch, 1])
    y1[4 * batch : , 0] = 0
    
    y2 = np.zeros([8 * batch, 1])
    y2[batch : 2 * batch , 0] = 1
    y2[2 * batch : 3 * batch , 0] = 2
    y2[3 * batch : 4 * batch , 0] = 3
    y2[5 * batch : 6 * batch , 0] = 1
    y2[6 * batch : 7 * batch , 0] = 2
    y2[7 * batch : , 0] = 3
    
    y2 = np_utils.to_categorical(y2, num_classes = 4)
    
    for layer in DM.layers:
        layer.trainable = True
        
    DM.train_on_batch(x, [y1, y2])
    
    for layer in DM.layers:
        layer.trainable = False
    
    labels = np.zeros([4 * batch, 1])
    labels[batch : 2 * batch, 0] = 1
    labels[2 * batch : 3 * batch, 0] = 2
    labels[3 * batch : , 0] = 3
    noise = np.random.uniform(-1.0, 1.0, size=[4 * batch, 3500])
    
    y1 = np.ones([4 * batch, 1])
    
    y2 = labels
    y2 = np_utils.to_categorical(y2, num_classes = 4)
    
    AM.train_on_batch([labels, noise], [y1, y2])
    
    if iters % 50 == 0:

        y = np.zeros([out_count, 1])
        y[ 5 : 10, 0] = 1
        y[ 10 : 15, 0] = 2
        y[ 15 : , 0] = 3

        gen_imgs = Gen.predict([y, out_noise])
    
        for i in range(out_count):
        
            img = (gen_imgs[i]*255).reshape(56, 56)
            plt.imshow(img, cmap='gray')
            plt.imsave(str('./outputs/plswork')+str(iters)+str(i), img, cmap = 'gray')
            plt.show()

#save weights
    
DM.save_weights('./models/multi_DM.h5')
AM.save_weights('./models/multi_AM.h5')