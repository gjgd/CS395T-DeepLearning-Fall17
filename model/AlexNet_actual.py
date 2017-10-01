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

def listYearbook(train=True, valid=True):
    yearbook_path = '/work/04381/ymarathe/maverick/yearbook/yearbook'
    r = []
    if train: r = r + [n.strip().split('\t') for n in open(yearbook_path+'_train.txt','r')]
    if valid: r = r + [n.strip().split('\t') for n in open(yearbook_path+'_valid.txt','r')]
    return r

def loadData():
    # Parameter to limit the size of the dataset when working locally
    num_images = 1000
    img_paths_train = listYearbook(train=True, valid=False)
    x_train = np.array([ imread('/work/04381/ymarathe/maverick/yearbook/train/' + img_path)[:,:,0] for (img_path, _) in img_paths_train[:num_images]])
    x1, x2, x3 = x_train.shape
    x_train = np.reshape(x_train, (x1, x2, x3, 1))
    y_train = np.array([ int(year) - 1905 for (_, year) in img_paths_train[:num_images] ])

    img_paths_valid = listYearbook(train=False, valid=True)
    x_valid = np.array([ imread('/work/04381/ymarathe/maverick/yearbook/valid/' + img_path)[:,:,0] for (img_path, _) in img_paths_valid[:num_images] ])
    x_valid = np.reshape(x_valid, (x1, x2, x3, 1))
    y_valid = np.array([ int(year) - 1905 for (_, year) in img_paths_valid[:num_images] ])

    return (x_train, y_train, x_valid, y_valid)

(x_train, y_train, x_valid, y_valid) = loadData()

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_valid /= 255
print('x_train shape:', x_train.shape)
print('x_ev', x_valid.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 109 #from 1905 to 2013
y_train = keras.utils.to_categorical(y_train, num_classes)

y_valid = keras.utils.to_categorical(y_valid, num_classes)


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
alex_model.add(Dense(num_classes, activation='softmax'))

alex_model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['categorical_accuracy'])

alex_model.summary()

with tf.device('/gpu:0'):
	#alex_model.fit_generator(female_train, steps_per_epoch = female_train.samples/female_train.batch_size, epochs=1,
                    #validation_data=female_valid, validation_steps = female_valid.samples/female_valid.batch_size)
	
	model.fit(x_train, y_train, batch_size=128,
          epochs=10,
          verbose=2)

	model.save_weights(path + 'alexweights_exp1.h5')
	score = model.evaluate(x_valid, y_valid, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


