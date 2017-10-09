## Adapted from Chollet's representation as built in model for Keras
##https://github.com/fchollet/deep-learning-models

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import math
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
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
from keras import layers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.layers import merge

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input
from keras.models import Model

import math
import os
import numpy as np
from keras.preprocessing.image import Iterator

from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import keras.backend as K
from keras.preprocessing.image import random_rotation, random_shear, random_shift, random_zoom
from skimage import exposure

import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array

import keras
from keras.preprocessing.image import ImageDataGenerator
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
import collections



data_path = '/work/04381/ymarathe/maverick/yearbook/'
path = '/home/05145/anikeshk/CS395T-DeepLearning-Fall17/model/'

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


class RegressDataGen:
    def __init__(self, directory, map_file, target_size = (171, 186, 3), 
                 class_weights_train = None, multi_output=False, do_augmentation=True, 
                 samplewise_center = True,
                 samplewise_std_deviation = True,
                 multi_input=False
                ):
        self.directory = directory
        self.map_file = map_file
        self.filenames = []
        self.map = {}
        self.fnameToGender = {}
        self.target_size = target_size
        self.populate_filenames()
        self.populate_mapping()
        self.regressIter = None
        self.steps = 0
        self.samplewise_center = samplewise_center
        self.samplewise_std_deviation = samplewise_std_deviation
        self.height_shift_range = 0.2
        self.width_shift_range = 0.2
        self.max_rotation = 45
        self.shear = 0.785398
        self.zoom_range = (0.5, 0.5)
        self.do_augmentation = do_augmentation
        self.class_weights_train = class_weights_train
        self.equalizehist = False
        self.multi_output = multi_output
        self.multi_input = multi_input
        self.lastN = []
        
    def _recursive_list(self, subpath):
        return sorted(
            os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])
    
    def populate_mapping(self):
        f = open(self.map_file, 'r')

        for line in f:
            line = line.rstrip()
            image, year = line.split("\t")
            gender, imfilename = image.split("/")
            if gender is 'M':
                encodeGender = 1
            elif gender is 'F':
                encodeGender = 0
            self.fnameToGender[image] = encodeGender
            self.map[image] = year
            
    def populate_filenames(self):
        base_dir = self.directory
        for root, _, files in self._recursive_list(base_dir):
            for fname in files:
                if fname.lower().endswith('.' + 'png'):
                    self.filenames.append(os.path.relpath(os.path.join(root, fname), base_dir))
                    
    def preprocess(self, x):
        if self.equalizehist:
            x = exposure.equalize_hist(x)
            
        return x
            
    def augment_data(self, x):
        
        x = random_shift(x, self.width_shift_range, self.height_shift_range, 
                         row_axis=0, col_axis = 1, channel_axis = 2)
        x = random_rotation(x, self.max_rotation, 
                            row_axis = 0, col_axis = 1, channel_axis = 2)
        x = random_shear(x, self.shear, row_axis = 0, col_axis = 1, channel_axis = 2)
        x = random_zoom(x, self.zoom_range, row_axis = 0, col_axis = 1, channel_axis = 2)
        
        return x
            
    def flow_from_directory(self, batch_size = 32, shuffle = True, seed = 42):
        
        self.regressIter = Iterator(len(self.filenames), batch_size = batch_size, shuffle = shuffle, seed = seed)
        
        if self.do_augmentation:
            factor = 3
        else:
            factor = 1
        
        self.steps = math.ceil(len(self.filenames)/batch_size) * factor
        
        return self
    
    def next(self, *args, **kwargs):
           
        self.lastN = []
        
        idx_array, cur_idx, bs = next(self.regressIter.index_generator)
        
        batch_x = np.zeros(tuple([len(idx_array)] + list(self.target_size)), dtype=K.floatx())
        
        batch_y = np.zeros(tuple([len(idx_array)]), dtype=K.floatx())
        
        if self.multi_output:
            batch_y_gender = np.zeros(tuple([len(idx_array)]), dtype=K.floatx())
    
        if self.multi_input:
            batch_x_gender = np.zeros(tuple([len(idx_array)]), dtype=K.floatx())
        
        if self.class_weights_train is not None:
            sample_weights = np.ones(tuple([len(idx_array)]), dtype=K.floatx())
        
        for i, j in enumerate(idx_array):
            fname = self.filenames[j]
            self.lastN.append(fname)
            img = load_img(
                  os.path.join(self.directory, fname),
                  grayscale = True,
                  target_size= self.target_size)
            x = np.array(img_to_array(img, data_format='channels_last'))
            x = self.preprocess(x)
            batch_x[i] = x
            batch_y[i] = self.map[fname]
            
            if self.multi_output:
                batch_y_gender[i] = self.fnameToGender[fname]
            
            if self.multi_input:
                batch_x_gender[i] = self.fnameToGender[fname]
            
            if self.class_weights_train is not None:
                if self.multi_output:
                    sample_weights[i] = self.class_weights_train[batch_y[i].astype('int').astype('str')]
                else:
                    sample_weights[i] = self.class_weights_train[batch_y[i].astype('int').astype('str')]
        
        if self.samplewise_center:
            for x in batch_x:
                x -= np.mean(x)
        
        if self.samplewise_std_deviation:
            for x in batch_x:
                x /= np.std(x)
        
        if self.do_augmentation:
            for x in batch_x:
                x = self.augment_data(x)
        
        if self.multi_output:
            if self.class_weights_train is not None:
                return batch_x, {'out_year' : batch_y, 'out_gender': batch_y_gender}, {'out_year' : sample_weights, 'out_gender' : sample_weights} 
            else:
                return batch_x, {'out_year' : batch_y, 'out_gender': batch_y_gender}
            
        elif self.multi_input:
            if self.class_weights_train is not None:
                return {'input_1' : batch_x, 'input_2': batch_x_gender}, batch_y, sample_weights
            else:
                return {'input_1' : batch_x, 'input_2': batch_x_gender}, batch_y
        else:    
            if self.class_weights_train is not None:
                return (batch_x, batch_y, sample_weights)
            else:
                return (batch_x, batch_y)


f = open(data_path + 'yearbook_train.txt', 'r')

freq = {}
normal_const = 0


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

mean_value = 0
for key in freq:
    mean_value += freq[key] * float(key)

idx2year = {}
year2idx = {}
class_weights_train = {}

for idx, key in enumerate(sorted_freq):
    class_weights_train[idx] = sorted_freq[key]
    idx2year[idx] = key
    year2idx[key] = idx

train = RegressDataGen(data_path + 'train',
                       data_path + 'yearbook_train.txt', 
                       class_weights_train = sorted_freq,
                       do_augmentation = True
                       #year2idx = year2idx
                      )
valid = RegressDataGen(data_path + 'valid',
                       data_path + 'yearbook_valid.txt',
                       class_weights_train = sorted_freq, 
                       do_augmentation = False
                       #year2idx = year2idx
                      )

train = train.flow_from_directory()
valid = valid.flow_from_directory(shuffle=False)



def identity_block(input_tensor, kernel_size, filters, stage, block):
#     """The identity block is the block that has no conv layer at shortcut.
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     """
     filters1, filters2, filters3 = filters
     bn_axis = 3
     if K.image_data_format() == 'channels_last':
         bn_axis = 3
     else:
         bn_axis = 1
     
     conv_name_base = 'res' + str(stage) + block + '_branch'
     bn_name_base = 'bn' + str(stage) + block + '_branch'

     x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
     x = Activation('relu')(x)

     x = Conv2D(filters2, kernel_size,
                padding='same', name=conv_name_base + '2b')(x)
     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
     x = Activation('relu')(x)

     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

     x = layers.add([x, input_tensor])
     x = Activation('relu')(x)
     return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#     """conv_block is the block that has a conv layer at shortcut
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     Note that from stage 3, the first conv layer at main path is with strides=(2,2)
#     And the shortcut should have strides=(2,2) as well
#     """
     filters1, filters2, filters3 = filters
     if K.image_data_format() == 'channels_last':
         bn_axis = 3
     else:
         bn_axis = 1
     conv_name_base = 'res' + str(stage) + block + '_branch'
     bn_name_base = 'bn' + str(stage) + block + '_branch'

     x = Conv2D(filters1, (1, 1), strides=strides,
                name=conv_name_base + '2a')(input_tensor)
     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
     x = Activation('relu')(x)

     x = Conv2D(filters2, kernel_size, padding='same',
                name=conv_name_base + '2b')(x)
     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
     x = Activation('relu')(x)

     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

     shortcut = Conv2D(filters3, (1, 1), strides=strides,
                       name=conv_name_base + '1')(input_tensor)
     shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

     x = layers.add([x, shortcut])
     x = Activation('relu')(x)
     return x



def resnet_conv(img_input):
    bn_axis = 3
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    #x = AveragePooling2D((6, 6), name='avg_pool')(x)
    
    return x


img_input = Input(shape=(171, 186, 3))
x = resnet_conv(img_input)
resnet_model1 = Model([img_input],[x])

#resnet_model1 = ResNet50(include_top=False, weights='imagenet', 
#    input_tensor=None, input_shape=(171, 186, 3), pooling=None, classes=104)

weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        WEIGHTS_PATH_NO_TOP,
                        cache_subdir='models',
                        md5_hash='a268eb855778b3df3c7506639542a6af')

resnet_model1.load_weights(weights_path)

for layer in resnet_model1.layers:
    layer.trainable = False

y = resnet_model1.output

x = Conv2D(8, (1, 1), activation='relu')(x)
x = Flatten()(x)
x = Dense(16, activation='relu', bias_initializer=keras.initializers.Ones())(x)
x = Dense(1, bias_initializer = keras.initializers.Constant(mean_value))(x)
resnet_model = Model([img_input],[x])


lr = 1e-4

#initially weights taken as 0.5

resnet_model.compile(Adam(lr=lr), loss=['mse'], 
              metrics=['mae']#,
              #loss_weights = [0.5]
            )

def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

resnet_model.summary()

with tf.device('/gpu:0'):
    resnet_model.fit_generator(train, steps_per_epoch = train.steps, epochs = 1,
                           validation_data = valid,
                           validation_steps = valid.steps,
                           callbacks=[LearningRateScheduler(lr_schedule),
                        ModelCheckpoint(path + 'exp471.h5', save_best_only=True)])

resnet_model.save_weights("/home/05145/anikeshk/CS395T-DeepLearning-Fall17/model/" + 'resnetweights_exp231.h5')

#pred = resnet_model.predict_generator(valid, shuffle=False)
score = resnet_model.evaluate_generator(valid, valid.steps)

print('Valid loss:', score[0])
print('Valid accuracy:', score[1])





