import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv3D, MaxPooling3D, Dense, BatchNormalization, Dropout, Flatten, Activation
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.core import Lambda
from keras import initializers
from keras.utils import get_file
import tensorflow as tf
import numpy as np
import os
from keras import backend as K
from skimage.io import imread
import numpy as np
import collections


path = '/work/04381/ymarathe/maverick/yearbook/'
data_path = '/work/04381/ymarathe/maverick/yearbook/'

f = open(data_path + 'yearbook_train.txt', 'r')

freq = {};
normal_const = 0;

for line in f:
    line = line.rstrip()
    image, year = line.split("\t")
    if year in freq:
        freq[year] += 1
    else:
        freq[year] = 1

normal_const = np.sum(freq.values())
for key in freq:
    freq[key] = freq[key]/float(normal_const);
    
sorted_freq = collections.OrderedDict(sorted(freq.items()))

idx = 0;
class_weights_train = {}
idx2year = {}

for key in sorted_freq:
    class_weights_train[idx] = sorted_freq[key]
    idx2year[idx] = key
    idx += 1

def gen_batches(path, gen = ImageDataGenerator(), shuffle=True, class_mode="categorical", batch_size=32, 
                target_size=(171, 186)):
    return gen.flow_from_directory(path, shuffle=shuffle, batch_size=batch_size, target_size=target_size, 
                                   class_mode=class_mode)

def gen_batches_flow(path, gen = ImageDataGenerator(), shuffle=True, batch_size=32):
    return gen.flow(path, shuffle=shuffle, batch_size=batch_size)

female_train = gen_batches(path + 'train/F')
female_valid = gen_batches(path + 'valid/F')


alex_model = Sequential()
alex_model.add(Conv3D(96, (11,11, 3), strides=(4,4,4), input_shape=(171,186,3)))
alex_model.add(Activation('relu'))
alex_model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding='valid'))
alex_model.add(Dropout(0.25))

alex_model.add(Conv3D(256, (5,5,48), strides=(4,4,4), padding='same'))
alex_model.add(Activation('relu'))
alex_model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding='valid'))
alex_model.add(Dropout(0.5))
alex_model.add(Conv3D(384, (3,3,256)))
alex_model.add(Conv3D(384, (3,3,192)))
alex_model.add(Conv3D(256, (3,3,192)))
alex_model.add(Flatten())
alex_model.add(Dense(4096))
alex_model.add(Dense(4096))
alex_model.add(Dense(1000))

alex_wts = get_file("alexnet_weights.h5")
alex_model.load_weights(alex_wts)
alex_model.add(Dense(109, activation='softmax'))

alex_model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

alex_model.summary()

with tf.device('/gpu:0'):
	alex_model.fit_generator(female_train, steps_per_epoch = female_train.samples/female_train.batch_size, epochs=1,
                    validation_data=female_valid, validation_steps = female_valid.samples/female_valid.batch_size)
	model.save_weights(path + 'alexweights_exp1.h5')


