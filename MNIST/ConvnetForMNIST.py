# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:22:50 2016

@author: wanggd
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import adadelta
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils.np_utils import to_categorical


batch_size = 128
nb_epoch = 12

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(1,28,28))) 
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3,border_mode='same'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32*196))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=adadelta(),
              metrics=['accuracy'])

#read minst dataset from file
from six.moves import cPickle
f=open('mnist.pkl')
(X_train, y_train), (X_test, y_test) = cPickle.load(f)
f.close()

#pre-process dataset
X_train=X_train.reshape(X_train.shape[0],1,28,28)
X_test=X_test.reshape(X_test.shape[0],1,28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test,verbose=0)




