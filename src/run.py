from skimage.io import imread
import util
from util import *
import sys
sys.path.append('/home/05148/picsou/project1/model')
import models
	
class Predictor:
	path = '/home/05148/picsou/project1/data/yearbook/keras_yearbook'
	input_shape = models.input_shape
	model = models.get_model2()
	model.load_weights(path + '/weights2.h5')
	
	def load(self, image_path):
		img = imread(image_path)[:,:,0] * 1./255
		img = img.reshape((1, self.input_shape[0], self.input_shape[1], 1))
		print(sum(sum(sum(img))))
		return img

	def predict(self, image_path):
		img = self.load(image_path)
		img_class = list(self.model.predict(img)[0]).index(1) + 1905
		print(img_class)
		return [img_class]
