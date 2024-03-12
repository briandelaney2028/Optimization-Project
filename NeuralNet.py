import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

class NeuralNetwork(object):
    _estimator_type = 'regressor'
    def __init__(self, layers=(30, 20), epochs=20,  learning_rate=0.001, batch_size=64):
        self.layers = layers
        self.epochs = epochs
        self.eta = learning_rate
        self.batch_size = batch_size

        self.model = Sequential([InputLayer(input_shape=(5,))])
        for layer in layers:
            self.model.add(Dense(layer, activation='relu'))
        self.model.add(Dense(1))
    
    def get_params(self, deep=False):
        return {'learning_rate':self.eta,
                'epochs':self.epochs,
                'batch_size':self.batch_size}
    
    def fit(self, X, y, **kwargs):
        X_test, y_test = None, None
        if 'validation_data' in kwargs.keys():
            X_test, y_test = kwargs['validation_data']
        if 'learning_rate' in kwargs.keys():
            self.eta = kwargs['learning_rate']
        if 'epochs' in kwargs.keys():
            self.epochs = kwargs['epochs']
        if 'batch_size' in kwargs.keys():
            self.batch_size = kwargs['batch_size']

        # Define optimizer and loss function
        opt = Adam(self.eta)
        mse = MeanSquaredError()
        self.model.compile(optimizer=opt, loss=mse)

        if X_test is not None:
            history = self.model.fit(X, y,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                verbose=0,
                                validation_data=(X_test, y_test))
        else:
            history = self.model.fit(X, y,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                verbose=0)

        if X_test is not None:
            return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model = tf.keras.models.load_model(fname)


    
