
# coding: utf-8
import matplotlib.cm as cm
# In[1]:
import skimage.io as sio
import pylab as pl
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


# In[2]:

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

idx = 0;
class_weights_train = {}
idx2year = {}

for key in sorted_freq:
    class_weights_train[idx] = sorted_freq[key]
    idx2year[idx] = key
    idx += 1


# In[3]:

import math
import os
import numpy as np
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import keras.backend as K
from keras.preprocessing.image import random_rotation, random_shear, random_shift, random_zoom
from skimage import exposure

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
            print(x)
            x = exposure.equalize_hist(x)
            print(x)
            exit()
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


# In[4]:

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.applications.xception import Xception

train = RegressDataGen(data_path + 'train',
                       data_path + 'yearbook_train.txt', 
                       class_weights_train = sorted_freq,
                       do_augmentation = False,
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


pretrained_model = Xception(include_top=False, weights='imagenet', input_shape=(171, 186, 3))
x = pretrained_model.output
x = Conv2D(8, (1, 1), activation='relu')(x)
x = Flatten()(x)
x = Dense(16, activation='relu', bias_initializer=keras.initializers.Ones())(x)
#x = BatchNormalization()(x)
predicted_year = Dense(1, bias_initializer = keras.initializers.Constant(mean_value))(x)

model = Model(inputs=pretrained_model.input, outputs=predicted_year)

lr = 1e-3
def lr_schedule(epoch):
    return lr * (0.1 ** float(epoch / 10.0))

model.compile(Adam(lr=lr), loss='mse', metrics=['mae'])

for layer in pretrained_model.layers:
    layer.trainable = False


# In[ ]:




# In[ ]:

training = False
num_experiment = 15
import numpy as np
from scipy.misc import imsave
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

if training:
    with tf.device('/gpu:0'):
        filename = "will_{}.h5".format(num_experiment)
        print ("Starting experiment " + str(num_experiment))
        model.fit_generator(train, steps_per_epoch = train.steps, epochs = 20,                                
                                   validation_data = valid, 
                                   validation_steps = valid.steps,
                                   callbacks=[LearningRateScheduler(lr_schedule),
                                ModelCheckpoint(path + filename, save_best_only=True)]
                           )
        print("Saved " + filename)
        
else:
    img_input = Input(shape=(171, 186, 3))
    model.load_weights('/work/05148/picsou/maverick/' + 'will_{}.h5'.format(13))
    input_img = model.input
    K.set_learning_phase(0)
    
    print(len(model.layers))
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = 'block14_sepconv2'
    
    filter_idx = 3
    lay = layer_dict[layer_name]
    layer_output = layer_dict[layer_name].output
    model_input = layer_dict['input_1']
    loss = K.mean(layer_output[:,:,:,filter_idx])
    
    #lay_f = K.function([img_input, K.learning_phase()], lay.output(train=False))
    
    

    def nice_imshow( data, f_i, l_n, vmin=None, vmax=None, cmap=None):
    ###"""Wrapper around pl.imshow"""
        if cmap is None:
            cmap = cm.jet
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
        imsave('%s_filter_%d.png' % (l_n, f_i), data)
	#pl.colorbar(im, cax=cax)
        
    batch_x, batch_y, sample_weight = next(valid)
    X = batch_x[1]
    
    #pl.figure()
    #pl.title('input')
  
    nice_imshow(np.squeeze(X), filter_idx, layer_name, vmin=0, vmax=1, cmap=cm.binary)
        

    
    
    
    #grads = K.gradients(loss, input_img)[0]
    #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    #iterate = K.function([input_img], [loss, grads])
    
    '''
    batch_x, batch_y, sample_weight = next(valid)
# we start from a gray image with some noise
    input_img_data = batch_x
# run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
   '''     
    

# util function to convert a tensor into a valid image
    def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

    # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

    # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    #img = input_img_data[0]
    #img = deprocess_image(img)
    #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        
    
    
    print(model.metrics_names)
    #print("Model loaded")
    #print(model.evaluate_generator(valid, valid.steps))
    
    


# ## Xception experiments
# 
# ### Warning: Experiment 6, 7, 8 had bad learning rates
# 
# will_6.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 1e-1 with lr decay
# - conv 1x1 = 16 filers
# - dense = 16
# 
# Epoch 8/20
# 713/713 [==============================] - 640s - loss: 6.6840 - mean_absolute_error: 20.0524 - val_loss: 3.9468 - val_mean_absolute_error: 16.9953
# 
# will_7.h5:
# - loss = mae
# - epochs = 20 without data augmentation
# - lr = 1e-1 with lr decay
# - conv 1x1 = 16 filers
# - dense = 16
# 
# Epoch 3/20
# 713/713 [==============================] - 642s - loss: 0.2597 - mean_absolute_error: 19.9808 - val_loss: 0.1756 - val_mean_absolute_error: 17.1463
# 
# 
# will_8.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 1e-1 with lr decay
# - conv 1x1 = 32 filers
# - dense = 64
# 
# Epoch 5/20
# 713/713 [==============================] - 640s - loss: 6.6749 - mean_absolute_error: 20.0370 - val_loss: 3.9561 - val_mean_absolute_error: 16.9929
# 
# 
# will_9.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 1e-3 with lr decay
# - conv 1x1 = 32 filers
# - dense = 64
# 
# 713/713 [==============================] - 640s - loss: 0.0052 - mean_absolute_error: 0.5249 - val_loss: 0.2931 - val_mean_absolute_error: 3.6388
# 
# will_10.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 1e-3 with lr decay
# - conv 1x1 = 16 filers
# - dense = 32
# 
# 713/713 [==============================] - 648s - loss: 0.0275 - mean_absolute_error: 1.1997 - val_loss: 0.2914 - val_mean_absolute_error: 3.5716
# 
# will_11.h5:
# - loss = mse
# - epochs = 6 wita data augmentation * 3
# - lr = 1e-3 with lr decay
# - conv 1x1 = 16 filers
# - dense = 32
# 
# 2139/2139 [==============================] - 1847s - loss: 0.0788 - mean_absolute_error: 1.8316 - val_loss: 0.3522 - val_mean_absolute_error: 3.9830
# 
# will_12.h5:
# - loss = mae
# - epochs = 20 without data augmentation
# - lr = 1e-3 with lr decay
# - conv 1x1 = 16 filers
# - dense = 32
# 
# 713/713 [==============================] - 635s - loss: 0.0057 - mean_absolute_error: 0.6751 - val_loss: 0.0363 - val_mean_absolute_error: 3.4857
# 
# will_13.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 1e-3 with lr decay
# - conv 1x1 = 8 filers
# - dense = 16
# 
# 713/713 [==============================] - 640s - loss: 0.0076 - mean_absolute_error: 0.6360 - val_loss: 0.2842 - val_mean_absolute_error: 3.5779
# 

# ## VGG19 experiments
# 
# will_1.h5:
# - loss = mse
# - epochs = 3
# - lr = 1e-2
# - conv 1x1 = 64 filers
# - dense = 128
# 
# Epoch 1/3
# 2139/2139 [==============================] - 1777s - loss: 1.7294 - acc: 0.0433 - mean_absolute_error: 8.9744 - val_loss: 1.2350 - val_acc: 0.0403 - val_mean_absolute_error: 8.3663
# 
# Epoch 2/3
# 2139/2139 [==============================] - 1706s - loss: 1.0568 - acc: 0.0573 - mean_absolute_error: 6.9857 - val_loss: 1.3283 - val_acc: 0.0404 - val_mean_absolute_error: 8.6860
# 
# Epoch 3/3
# 2139/2139 [==============================] - 1704s - loss: 0.8709 - acc: 0.0644 - mean_absolute_error: 6.3560 - val_loss: 0.9722 - val_acc: 0.0547 - **val_mean_absolute_error: 7.2192**
# 
# will_2.h5:
# - loss = mse
# - epochs = 1
# - lr = 3e-2
# - conv 1x1 = 32 filers
# - dense = 64
# 
# Epoch 1/1
# 2139/2139 [==============================] - 1718s - loss: 1.9574 - acc: 0.0403 - mean_absolute_error: 9.6086 - val_loss: 1.6574 - val_acc: 0.0375 - **val_mean_absolute_error: 9.5908**
# 
# will_3.h5:
# - loss = mse
# - epochs = 3 without data augmentation
# - lr = 3e-2
# - conv 1x1 = 32 filers
# - dense = 64
# 
# Epoch 1/3
# 713/713 [==============================] - 389s - loss: 2.4174 - acc: 0.0371 - mean_absolute_error: 10.8405 - val_loss: 1.3939 - val_acc: 0.0381 - val_mean_absolute_error: 9.0965
# 
# Epoch 2/3
# 713/713 [==============================] - 388s - loss: 1.8113 - acc: 0.0436 - mean_absolute_error: 9.3031 - val_loss: 1.4613 - val_acc: 0.0378 - **val_mean_absolute_error: 9.0060**
# 
# Epoch 3/3
# 713/713 [==============================] - 385s - loss: 1.6857 - acc: 0.0430 - mean_absolute_error: 8.9575 - val_loss: 1.8654 - val_acc: 0.0237 - val_mean_absolute_error: 10.2079
# 
# will_4.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 3e-2
# - conv 1x1 = 50 filers
# - dense = 100
# 
# Epoch 1/20
# 713/713 [==============================] - 394s - loss: 2.4342 - acc: 0.0356 - mean_absolute_error: 10.8500 - val_loss: 2.6718 - val_acc: 0.0294 - val_mean_absolute_error: 12.2755
# 
# Epoch 2/20
# 713/713 [==============================] - 393s - loss: 1.7797 - acc: 0.0438 - mean_absolute_error: 9.2208 - val_loss: 1.3407 - val_acc: 0.0317 - val_mean_absolute_error: 8.7575
# 
# Epoch 3/20
# 713/713 [==============================] - 389s - loss: 1.6081 - acc: 0.0499 - mean_absolute_error: 8.7490 - val_loss: 1.2787 - val_acc: 0.0364 - val_mean_absolute_error: 8.6355
# 
# Epoch 4/20
# 713/713 [==============================] - 389s - loss: 1.4956 - acc: 0.0526 - mean_absolute_error: 8.3522 - val_loss: 1.2091 - val_acc: 0.0358 - val_mean_absolute_error: 8.1282
# 
# Epoch 5/20
# 713/713 [==============================] - 389s - loss: 1.3725 - acc: 0.0514 - mean_absolute_error: 8.0273 - val_loss: 1.1532 - val_acc: 0.0484 - val_mean_absolute_error: 7.7425
# 
# Epoch 6/20
# 713/713 [==============================] - 389s - loss: 1.2938 - acc: 0.0488 - mean_absolute_error: 7.8258 - val_loss: 1.0986 - val_acc: 0.0400 - val_mean_absolute_error: 7.7927
# 
# Epoch 7/20
# 713/713 [==============================] - 388s - loss: 1.2713 - acc: 0.0480 - mean_absolute_error: 7.7210 - val_loss: 1.1828 - val_acc: 0.0410 - val_mean_absolute_error: 8.0945
# 
# Epoch 8/20
# 713/713 [==============================] - 388s - loss: 1.2350 - acc: 0.0521 - mean_absolute_error: 7.6468 - val_loss: 1.5489 - val_acc: 0.0342 - val_mean_absolute_error: 9.0588
# 
# Epoch 9/20
# 713/713 [==============================] - 387s - loss: 1.2230 - acc: 0.0492 - mean_absolute_error: 7.5878 - val_loss: 1.1071 - val_acc: 0.0430 - val_mean_absolute_error: 7.7185
# Epoch 10/20
# 
# 713/713 [==============================] - 388s - loss: 1.1237 - acc: 0.0517 - mean_absolute_error: 7.2646 - val_loss: 1.3911 - val_acc: 0.0362 - val_mean_absolute_error: 8.8584
# 
# Epoch 11/20
# 713/713 [==============================] - 389s - loss: 0.9102 - acc: 0.0594 - mean_absolute_error: 6.4532 - val_loss: 0.9581 - val_acc: 0.0486 - val_mean_absolute_error: 7.0721
# 
# Epoch 12/20
# 713/713 [==============================] - 389s - loss: 0.8178 - acc: 0.0612 - mean_absolute_error: 6.1367 - val_loss: 0.9202 - val_acc: 0.0442 - val_mean_absolute_error: 7.1384
# 
# Epoch 13/20
# 713/713 [==============================] - 387s - loss: 0.8100 - acc: 0.0622 - mean_absolute_error: 6.1223 - val_loss: 0.9951 - val_acc: 0.0444 - val_mean_absolute_error: 7.2625
# 
# Epoch 14/20
# 713/713 [==============================] - 387s - loss: 0.8052 - acc: 0.0627 - mean_absolute_error: 6.1083 - val_loss: 0.9653 - val_acc: 0.0524 - val_mean_absolute_error: 7.1744
# 
# Epoch 15/20
# 713/713 [==============================] - 389s - loss: 0.7664 - acc: 0.0652 - mean_absolute_error: 5.9908 - val_loss: 0.9191 - val_acc: 0.0478 - val_mean_absolute_error: 7.0468
# 
# Epoch 16/20
# 713/713 [==============================] - 387s - loss: 0.7592 - acc: 0.0688 - mean_absolute_error: 5.8906 - val_loss: 0.9926 - val_acc: 0.0464 - val_mean_absolute_error: 7.1721
# 
# Epoch 17/20
# 713/713 [==============================] - 387s - loss: 0.7411 - acc: 0.0644 - mean_absolute_error: 5.8423 - val_loss: 0.9241 - val_acc: 0.0438 - val_mean_absolute_error: 7.1005
# 
# Epoch 18/20
# 713/713 [==============================] - 388s - loss: 0.7309 - acc: 0.0586 - mean_absolute_error: 5.8033 - val_loss: 0.9584 - val_acc: 0.0661 - val_mean_absolute_error: 7.0660
# 
# Epoch 19/20
# 713/713 [==============================] - 387s - loss: 0.7039 - acc: 0.0581 - mean_absolute_error: 5.7438 - val_loss: 0.9351 - val_acc: 0.0663 - **val_mean_absolute_error: 6.9641**
# 
# Epoch 20/20
# 713/713 [==============================] - 388s - loss: 0.6950 - acc: 0.0572 - mean_absolute_error: 5.6644 - val_loss: 0.9598 - val_acc: 0.0607 - val_mean_absolute_error: 7.1322
# 
# will_5.h5:
# - loss = mse
# - epochs = 20 without data augmentation
# - lr = 3e-2
# - conv 1x1 = 50 filers
# - dense = 100
# - Dropout = 0.5
# 
# Epoch 1/20
# 713/713 [==============================] - 392s - loss: 3.2925 - acc: 0.0291 - mean_absolute_error: 12.9005 - val_loss: 1.6876 - val_acc: 0.0294 - val_mean_absolute_error: 10.0642
# 
# Epoch 2/20
# 713/713 [==============================] - 391s - loss: 2.6793 - acc: 0.0331 - mean_absolute_error: 11.5035 - val_loss: 1.3901 - val_acc: 0.0388 - val_mean_absolute_error: 8.8441
# 
# Epoch 3/20
# 713/713 [==============================] - 385s - loss: 2.5866 - acc: 0.0285 - mean_absolute_error: 11.2949 - val_loss: 3.0742 - val_acc: 0.0189 - val_mean_absolute_error: 13.7363
# 
# Epoch 4/20
# 713/713 [==============================] - 386s - loss: 2.5862 - acc: 0.0292 - mean_absolute_error: 11.3336 - val_loss: 1.9153 - val_acc: 0.0334 - val_mean_absolute_error: 9.7895
# 
# Epoch 5/20
# 713/713 [==============================] - 386s - loss: 2.6258 - acc: 0.0278 - mean_absolute_error: 11.3908 - val_loss: 2.3101 - val_acc: 0.0185 - val_mean_absolute_error: 11.5918
# 
# Epoch 6/20
# 713/713 [==============================] - 386s - loss: 2.5469 - acc: 0.0292 - mean_absolute_error: 11.2433 - val_loss: 7.3051 - val_acc: 0.0384 - val_mean_absolute_error: 10.0754
# 
# Epoch 7/20
# 713/713 [==============================] - 385s - loss: 2.5583 - acc: 0.0292 - mean_absolute_error: 11.2443 - val_loss: 6.9639 - val_acc: 0.0416 - val_mean_absolute_error: 10.8122
# 
# Epoch 8/20
# 713/713 [==============================] - 386s - loss: 2.5845 - acc: 0.0273 - mean_absolute_error: 11.3074 - val_loss: 7.7042 - val_acc: 0.0330 - val_mean_absolute_error: 11.1933
# 
# Epoch 9/20
# 713/713 [==============================] - 385s - loss: 2.5135 - acc: 0.0310 - mean_absolute_error: 11.1403 - val_loss: 6.8166 - val_acc: 0.0384 - val_mean_absolute_error: 11.4688
# 
# Epoch 10/20
# 713/713 [==============================] - 386s - loss: 2.3537 - acc: 0.0281 - mean_absolute_error: 10.8007 - val_loss: 7.8546 - val_acc: 0.0358 - val_mean_absolute_error: 11.1792
# 
# Epoch 11/20
# 713/713 [==============================] - 385s - loss: 2.2141 - acc: 0.0310 - mean_absolute_error: 10.4251 - val_loss: 6.0965 - val_acc: 0.0382 - val_mean_absolute_error: 9.4952
# 
# Epoch 12/20
# 713/713 [==============================] - 385s - loss: 2.1059 - acc: 0.0326 - mean_absolute_error: 10.1197 - val_loss: 11.5091 - val_acc: 0.0446 - val_mean_absolute_error: 10.5134
# 
# Epoch 13/20
# 713/713 [==============================] - 386s - loss: 2.0857 - acc: 0.0334 - mean_absolute_error: 10.0760 - val_loss: 12.5309 - val_acc: 0.0436 - val_mean_absolute_error: 11.5219
# 
# Epoch 14/20
# 713/713 [==============================] - 386s - loss: 2.0577 - acc: 0.0334 - mean_absolute_error: 10.0021 - val_loss: 15.5565 - val_acc: 0.0400 - val_mean_absolute_error: 11.1698
# 
# Epoch 15/20
# 713/713 [==============================] - 386s - loss: 2.0236 - acc: 0.0343 - mean_absolute_error: 9.8573 - val_loss: 5.3131 - val_acc: 0.0388 - val_mean_absolute_error: 9.0751
# 
# Epoch 16/20
# 713/713 [==============================] - 386s - loss: 1.9644 - acc: 0.0344 - mean_absolute_error: 9.7395 - val_loss: 8.4828 - val_acc: 0.0334 - val_mean_absolute_error: 10.6123
# 
# Epoch 17/20
# 713/713 [==============================] - 386s - loss: 1.9953 - acc: 0.0345 - mean_absolute_error: 9.7768 - val_loss: 6.1084 - val_acc: 0.0378 - val_mean_absolute_error: 9.5485
# 
# Epoch 18/20
# 713/713 [==============================] - 385s - loss: 1.9610 - acc: 0.0351 - mean_absolute_error: 9.6949 - val_loss: 7.6045 - val_acc: 0.0382 - val_mean_absolute_error: 9.7756
# 
# Epoch 19/20
# 713/713 [==============================] - 386s - loss: 1.9285 - acc: 0.0355 - mean_absolute_error: 9.6024 - val_loss: 4.7563 - val_acc: 0.0372 - val_mean_absolute_error: 9.2489
# 
# Epoch 20/20
# 713/713 [==============================] - 385s - loss: 1.9519 - acc: 0.0358 - mean_absolute_error: 9.6799 - val_loss: 4.6270 - val_acc: 0.0444 - **val_mean_absolute_error: 9.0526**
# 
# 
