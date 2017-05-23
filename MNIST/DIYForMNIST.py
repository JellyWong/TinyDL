# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:52:23 2016

@author: wanggd
"""


import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import timeit
from six.moves import cPickle
import matplotlib.pyplot as plt


class HiddenLayer(object):
    
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation=T.tanh):
        
        self.input=input
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        self.params=[self.W,self.b]        
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )


class LogisticRegression(object):
    
    def __init__(self,input,n_in,n_out):
        
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        
        self.input=input
        
    def negative_log_likehood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
    
        
class dA(object):
    
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
                
        if not W:
            
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            
            W=theano.shared(value=initial_W,name='W',borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    
    def get_hidden_values(self,input):
        return T.nnet.sigmoid(T.dot(input,self.W)+self.b)
        
        
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        
    
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
        
    def get_cost_updates(self, corruption_level, learning_rate):
        
        tilde_x=self.get_corrupted_input(self.x,corruption_level)
        y=self.get_hidden_values(tilde_x)
        z=self.get_reconstructed_input(y)
        
        L=-T.sum(self.x*T.log(z)+(1-self.x)*T.log(1-z),axis=1)

        cost=T.mean(L)
        
        gparams=T.grad(cost,self.params)
        
        updates=[(param,param-learning_rate*gparam) for param,gparam in zip(self.params,gparams)]

        return (cost,updates)



class SdA(object):
    
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 n_ins=784,
                 hidden_layers_sizes=[500,500],
                 n_outs=10,
                 corruption_levels=[0.1,0.1]):
        self.sigmoid_layers=[]
        self.dA_layers=[]
        self.params=[]
        self.n_layers=len(hidden_layers_sizes)
        
        assert self.n_layers>0
        
        if not theano_rng:
            theano_rng=RandomStreams(numpy_rng.randint(2 ** 30))
        
        self.x=T.matrix('x')
        self.y=T.ivector('y')
        
        for i in range(self.n_layers):
            
            if i==0:
                input_size=n_ins
            else:
                input_size=hidden_layers_sizes[-1]
                
            if i==0:
                layer_input=self.x
            else:
                layer_input=self.sigmoid_layers[-1].output
            
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            
            self.sigmoid_layers.append(sigmoid_layer)
            
            self.params.extend(sigmoid_layer.params)
            
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
                          
            self.dA_layers.append(dA_layer)
            
        self.logLayer=LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )
        
        self.params.extend(self.logLayer.params)
        
        self.finetune_cost=self.logLayer.negative_log_likehood(self.y)
        
        self.errors=self.logLayer.errors(self.y)
        
        
    def pretraining_functions(self, train_set_x, batch_size):
        
        index=T.iscalar('index')
        
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        
        pretrain_fns=[]
        
        for dA in self.dA_layers:
            
            cost,updates=dA.get_cost_updates(corruption_level,learning_rate)
            
            fn=theano.function(inputs=[index,
                                       theano.In(corruption_level,value=0.2),
                                       theano.In(learning_rate,value=0.1)],
                               outputs=cost,
                               updates=updates,
                               givens={self.x:train_set_x[batch_begin:batch_end]})

            pretrain_fns.append(fn)
            
        return pretrain_fns
    
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
        


def load_dataset():
    
    #feed the datasets into memory
    f=open('mnist.pkl','r')
    (X_train, Y_train), (X_test, Y_test) = cPickle.load(f)
    f.close()
    #pre-process dataset
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_train = X_train.astype('float64')
    X_train /= 255
    
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    X_test = X_test.astype('float64')
    X_test /= 255
    
    X_valid=X_train[:10000] 
    X_train=X_train[10000:]
    
    X_train=theano.shared(value=X_train,name='X_train')
    X_valid=theano.shared(value=X_valid,name='X_valid')
    X_test=theano.shared(value=X_test,name='X_test')
    
    Y_valid=Y_train[:10000].astype('int32')
    Y_train=Y_train[10000:].astype('int32')
    Y_test=Y_test.astype('int32')
    
    Y_train=theano.shared(value=Y_train,name='Y_train')
    Y_valid=theano.shared(value=Y_valid,name='Y_valid')
    Y_test=theano.shared(value=Y_test,name='Y_test')
    
    return [(X_train,Y_train),(X_valid,Y_valid),(X_test,Y_test)]


datasets=load_dataset()

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
batch_size=100

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

pretraining_lr=0.1

pretraining_epochs=10

numpy_rng = np.random.RandomState(89677)
print('... building the model')

# construct the stacked denoising autoencoder class
sda = SdA(
    numpy_rng=numpy_rng,
    n_ins=28 * 28,
    hidden_layers_sizes=[1000, 1000, 1000],
    n_outs=10
)

print('... getting the pretraining functions')
pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                            batch_size=batch_size)

print('... pre-training the model')
start_time = timeit.default_timer()
## Pre-train layer-wise
corruption_levels = [.1, .2, .3]       

for i in range(sda.n_layers):
    
    for epoch in range(pretraining_epochs):
        
        c=[]
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index,
                                        corruption=corruption_levels[i],
                                        lr=pretraining_lr))
        print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))

end_time = timeit.default_timer()

print((' ran for %.2fm' % ((end_time - start_time) / 60.)))

print('... training the model')

n_epochs=20
learning_rate=0.1
train_fn, valid_score, test_score=sda.build_finetune_functions(
                                  datasets,100,learning_rate)

acc=[]
for epoch in range(n_epochs):
    cost=[]
    for index in range(n_train_batches):
        cost.append(train_fn(index))
    acc.append(1-np.mean(test_score()))
    print('mean acc :%.5f'%acc[-1])
        
plt.scatter(acc)      
        
        
        
        
        