# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:26:04 2016

@author: wanggd
"""

from keras.models import Sequential, load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.layers import Activation, SimpleRNN
from keras.optimizers import  Adadelta
from keras.regularizers import l2
from keras.callbacks import Callback

import numpy as np


class AccCollector(Callback):
    '''
        This callback class is designd to early stop the training period 
        once the accuracy has reached 100%
    '''
    def __init__(self, model_name=None):
        if model_name is None:
            self.model_name = 'model.h5'
        else:
            self.model_name = model_name
    
    def on_train_begin(self, logs=None):
        self.acc = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        if logs.get('val_acc') == 1:
            self.model.stop_training=True
            self.stopped_epoch = epoch
    
    def on_train_end(self, logs=None):
        self.model.save(self.model_name, overwrite=True)

# binary_dim represents the lenght of bits
binary_dim=16
largest_number=pow(2,binary_dim)
dataset_size=20000

def num2binary(num):
    high_bits=np.unpackbits(np.array(num/256,dtype=np.uint8))
    low_bits=np.unpackbits(np.array(num%256,dtype=np.uint8))
    return np.append(high_bits,low_bits)

def binary2num(binary):
    ans=0
    for i in range(binary_dim):
        ans+=binary[i]*pow(2,binary_dim-i-1)
    return ans

#store all numbers in the form of binary for efficient access
binary = map(num2binary,np.array([range(largest_number)]).T)

                     
#generate samples
def load_data():
    '''
    we assume that a sample has the shape of (16, 2) denoted by S for convenience
    The first addend is represented by S[:, 0][::-1]
    The second addend is represented by S[:, 1][::-1]
    The sum is also reversed i.e. the top digit is at the end of the binary sequence 
    '''
    X_train=np.array([],dtype=np.uint8)   
    Y_train=np.array([],dtype=np.uint8)   
          
    for i in range(dataset_size):
        
        a=np.random.randint(largest_number/2)
        b=np.random.randint(largest_number/2)
        ans=a+b
        
        a=binary[a]
        b=binary[b]
        ans=binary[ans]
        
        for position in range(binary_dim):
            X_train=np.append(X_train,[a[binary_dim-1-position],b[binary_dim-1-position]])
            Y_train=np.append(Y_train,ans[binary_dim-1-position])
    
    return X_train, Y_train


class Addition:
    
    def __init__(self, model_path=None):
        self.X_train, self.Y_train = load_data()
        #reshape the dataset into regular format
        self.X_train=self.X_train.reshape((dataset_size,binary_dim,2))
        self.Y_train=self.Y_train.reshape((dataset_size,binary_dim*1,1))
        if model_path is None:
            self.model = self.get_model()
        else:
            self.model = load_model(model_path)
        self.acc_collector = AccCollector()
        

    def get_model(self):    
        '''
        construct NN model
        We feed two binary digits into the model each timestep, i.e, we'll
        get 16 timesteps for each sample, from ones, tens. hundreds to the
        top digit.
        '''
        model=Sequential()
        model.add(SimpleRNN(64, 
                            input_shape=(binary_dim, 2),
                            kernel_regularizer=l2(0.01),
                            return_sequences=False))
                            
        model.add(Activation('relu'))    
        model.add(Dense(128, activation='relu'))
        model.add(RepeatVector(binary_dim))
        model.add(SimpleRNN(64, return_sequences=True))
        model.add(Activation('relu'))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adadelta(lr=2, decay=1e-5),
                      metrics=['accuracy'])
        return model
    
    def train(self):
        self.model.fit(self.X_train, self.Y_train,
                  epochs=100,
                  batch_size=128,
                  validation_data=(self.X_train, self.Y_train),
                  callbacks=[self.acc_collector])


    def add(self, a, b):
        data = np.zeros((1, 16, 2), dtype=np.uint8)
        data[0, :, 0] = binary[a][::-1]
        data[0, :, 1] = binary[b][::-1]
        res=np.array(self.model.predict_classes(data, verbose=0), dtype=np.uint8).flatten() 
        return binary2num(res[::-1])
    
    def test(self, n=10000):
        '''
        note that we have very limited training data(default 20000 in the settings)
        compared with 16-bit addition.
        So here we provide a function that can be used to test how well it fits the 
        training data. In general, we'll get several hard examples confusing the model.
        '''
        def prepare_data(n):
            '''
            generate random test data
            '''
            data = np.zeros((n, 16, 2), dtype=np.uint8)
            res = []
            for i in xrange(n):
                a = np.random.randint(0, largest_number/2)
                b = np.random.randint(0, largest_number/2)
                res.append(a+b)
                data[i, :, 0] = binary[a][::-1]
                data[i, :, 1] = binary[b][::-1]
            return data, res
        
        data, res = prepare_data(n)
        pred = np.array(self.model.predict_classes(data, verbose=0), dtype=np.uint8)
        hard_examples = []
        for i, _pred in enumerate(pred):
            _pred = _pred.flatten()
            _sum = binary2num(_pred[::-1])
            if _sum != res[i]:
                hard_examples.append([data[i]])
        return np.asarray(hard_examples)
        
 
if __name__=='__main__':
    model = Addition()
    model.train()
