# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:22:50 2016

@author: wanggd
"""

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD,adadelta
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
import numpy as np

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")


class AccHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('acc'))

histroy=AccHistory()


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(500, input_dim=784,W_regularizer=l2(0.01),init='uniform',bias=True))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
'''
model.add(normalization.BatchNormalization(
            epsilon=1e-6,mode=0,axis=1,momentum=0.9))
'''
model.add(Dense(300, init='uniform',W_regularizer=l2(0.01)))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))

model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta(),
              metrics=['accuracy'])


#feed the datasets into memory
from six.moves import cPickle
f=open('mnist.pkl','r')
(X_train, y_train), (X_test, y_test) = cPickle.load(f)
f.close()
xx=np.copy(X_train)
'''
import pylab as pl
import matplotlib.cm as cm
pl.imshow(X_train[0], interpolation='nearest', cmap=cm.binary)
'''

#pre-process dataset
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=100,
          callbacks=[histroy])
          
score = model.evaluate(X_test, y_test, batch_size=100)

