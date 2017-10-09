from skimage.io import imread
import util
from util import *
import sys
import numpy as np
import keras
from keras.models import Model
from keras.applications.xception import Xception
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, Activation, Lambda, Input
from keras.preprocessing.image import img_to_array, load_img

class Predictor:
	input_shape = (171, 186, 3)

	def get_model(self):
            pretrained_model = Xception(include_top=False, weights='imagenet', input_shape=self.input_shape)
	    x = pretrained_model.output
	    x = Conv2D(16, (1, 1), activation='relu')(x)
	    x = Flatten()(x)
	    x = Dense(32, activation='relu')(x)
	    predicted_year = Dense(1)(x)
	    model = Model(inputs=pretrained_model.input, outputs=predicted_year)
            
            return model

        def __init__(self):
            self.model = self.get_model()
	    self.model.load_weights('../data/yearbook/will_12.h5')

	def load(self, image_path):
                img = load_img(
                  image_path, 
                  grayscale = False,
                  target_size= self.input_shape)
                img = np.array(img_to_array(img, data_format='channels_last'))
                img -= np.mean(img)
                img /= np.std(img)
                return img
        
	def predict(self, image_path):
		img = self.load(image_path).reshape((1,) + self.input_shape)
                year = np.round(self.model.predict(img))                
                return year[0]
