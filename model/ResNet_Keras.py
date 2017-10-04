
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
from VGG_Pretrain import RegressDataGen
import collections

data_path = '/work/04381/ymarathe/maverick/yearbook/'
path = '/home/05145/anikeshk/CS395T-DeepLearning-Fall17/model/'

f = open(data_path + 'yearbook_train.txt', 'r')

freq = {};
normal_const = 0;

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





