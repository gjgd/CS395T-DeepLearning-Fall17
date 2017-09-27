
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, Activation
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


path = '/work/04381/ymarathe/maverick/yearbook/keras_yearbook/'
#path = '/Users/anikeshkamath/Documents/CS395T-DeepLearning-Fall17/data/yearbook/' #UPDATE THIS!!

def gen_batches(path, gen = ImageDataGenerator(), shuffle=True, class_mode="categorical", batch_size=32, 
                target_size=(171, 186)):
    return gen.flow_from_directory(path, shuffle=shuffle, batch_size=batch_size, target_size=target_size, 
                                   class_mode=class_mode)

def gen_batches_flow(path, gen = ImageDataGenerator(), shuffle=True, batch_size=32):
    return gen.flow(path, shuffle=shuffle, batch_size=batch_size)

female_train = gen_batches(path + 'train/F') #UPDATE THIS!!!
male_train = gen_batches(path + 'train/M')	#UPDATE THIS!!!

female_valid = gen_batches(path + 'valid/F')
male_valid = gen_batches(path + 'valid/M')

num_classes = 104

#print x_train.shape

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(171, 186, 3)))
model.add(Conv2D(48, (11,11), strides=(4,4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5,5), strides=(4,4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(192, (1,1)))
model.add(Conv2D(192, (1,1)))
model.add(Conv2D(128, (1,1)))
model.add(Dense(2048))
model.add(Dense(2048))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#ad_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit_generator(female_train, steps_per_epoch = female_train.samples/female_train.batch_size,
                    validation_data=female_valid, validation_steps = female_valid.samples/female_valid.batch_size)
#score = model.evaluate(x_test, y_test, batch_size = 16)



