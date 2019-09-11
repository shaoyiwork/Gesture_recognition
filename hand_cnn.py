'''
This is a Gesture three classifier.
We use CNN and other net to train the model.
'''

import os
import numpy as np
import keras
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D,MaxPooling2D,Activation
from keras.optimizers import Adam, SGD
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

nbatch_size = 32
nepochs = 5

def load_data():
    images = []
    labels = []

    path = './hand_data/train/zero/'
    files = os.listdir(path)
    for f in files:
        img_path = path + f
        img = image.load_img(img_path, target_size=(234,234,3))
        img_array = image.img_to_array(img)
        images.append(img_array)
        if 'zero' in f:
            labels.append(0)
    path = './hand_data/train/two/'
    files = os.listdir(path)
    for f in files:
        img_path = path + f
        img = image.load_img(img_path, target_size=(234,234,3))
        img_array = image.img_to_array(img)
        images.append(img_array)
        if 'two' in f:
            labels.append(1)
    path = './hand_data/train/five/'
    files = os.listdir(path)
    for f in files:
        img_path = path + f
        img = image.load_img(img_path, target_size=(234,234,3))
        img_array = image.img_to_array(img)
        images.append(img_array)
        if 'five' in f:
            labels.append(2)
    data = np.asarray(images,dtype='float32')
    labels = np.asarray(labels)
    labels = np_utils.to_categorical(labels, 3)
    return data, labels

def Mode_net_cnn():
    model = Sequential()
    model.add(Conv2D(4, (5, 5),input_shape=(234, 234,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #model.add(Conv2D(8, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1))
    #model.add(Activation('sigmoid'))
    model.add(Dense(3, activation='softmax'))
    #sgd = Adam(lr=0.0001)
    sgd = RMSprop(lr=0.0001)
    #model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                 optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    return model

def Net_model_simple():
    model = Sequential()
    model.add(Conv2D(5, (3, 3),padding="valid",input_shape=(234, 234, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(10, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
 
    model.add(Flatten())
    model.add(Dense(128)) #Full connection
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
    return model


if __name__ == '__main__':
    images, lables = load_data()
    images /= 255
    print (images.shape)
    x_train, x_test, y_train, y_test = train_test_split(images, lables, test_size=0.2)
    model = Net_model_simple()
    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs, verbose=1, validation_data=(x_test, y_test))
    model.save("hand_model.h5")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


