from skimage.io import imread
import numpy as np
 
def listYearbook(train=True, valid=True):
    yearbook_path = "data/yearbook/yearbook" 
    r = []
    if train: r = r + [n.strip().split('\t') for n in open(yearbook_path+'_train.txt','r')]
    if valid: r = r + [n.strip().split('\t') for n in open(yearbook_path+'_valid.txt','r')]
    return r

def loadData():
    # Parameter to limit the size of the dataset when working locally
    limit = 1000
    img_paths_train = listYearbook(train=True, valid=False)
    x_train = np.array([ imread('data/yearbook/train/' + img_path)[:,:,0] for (img_path, _) in img_paths_train[:limit] ])
    x1, x2, x3 = x_train.shape 
    x_train = np.reshape(x_train, (x1, x2, x3, 1)) 
    y_train = np.array([ int(year) - 1905 for (_, year) in img_paths_train[:limit] ])
    
    img_paths_valid = listYearbook(train=False, valid=True)
    x_valid = np.array([ imread('data/yearbook/valid/' + img_path)[:,:,0] for (img_path, _) in img_paths_valid[:limit] ])
    x_valid = np.reshape(x_valid, (x1, x2, x3, 1))
    y_valid = np.array([ int(year) - 1905 for (_, year) in img_paths_valid[:limit] ])

    return (x_train, y_train, x_valid, y_valid)

(x_train, y_train, x_valid, y_valid) = loadData()

num_classes = 104
input_shape = (171, 186, 1)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_valid /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'test samples')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# convert class vectors to binary class matrices
num_classes = 109 #from 1905 to 2013
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['categorical_accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=2,
          validation_data=(x_valid, y_valid))
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
