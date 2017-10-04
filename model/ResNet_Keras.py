
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, Activation, Lambda, Input
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
from VGG_Pretrain import RegressDataGen
import collections

data_path = '/work/04381/ymarathe/maverick/yearbook/'
path = '/home/05145/anikeshk/CS395T-DeepLearning-Fall17/model/'

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

train = RegressDataGen(data_path + 'train',
                       data_path + 'yearbook_train.txt', 
                       class_weights_train = sorted_freq,
                       do_augmentation = True,
                      )
valid = RegressDataGen(data_path + 'valid',
                       data_path + 'yearbook_valid.txt',
                       class_weights_train = sorted_freq, 
                       do_augmentation = False,
                      )

train = train.flow_from_directory()
valid = valid.flow_from_directory(shuffle=False)

mean_value = 0
for key in freq:
    mean_value += freq[key] * float(key)

resnet_model = ResNet50(include_top=True, weights='imagenet', 
    input_tensor=None, input_shape=None, pooling=None, classes=104)

lr = 1e-4

resnet_model.compile(Adam(lr=lr), loss=['mse'], 
              metrics=['mae'],
              loss_weights = [0.5]
            )

def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

resnet_model.fit_generator(train, steps_per_epoch = train.steps, epochs = 5,
                       validation_data = valid,
                       validation_steps = valid.steps,
                       callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint(path + 'exp43.h5', save_best_only=True)])

resnet_model.save_weights("/home/05145/anikeshk/CS395T-DeepLearning-Fall17/model/" + 'resnetweights_exp1.h5')

pred = resnet_model.predict_generator(valid, shuffle=False)

score = alex_model.evaluate_generator(value, verbose=0)
print('Valid loss:', score[0])
print('Valid accuracy:', score[1])





