import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class NeuralNetwork(object):
    def __init__(self, layers=(30, 20), epochs=100,  learning_rate=0.001, batch_size=64):
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
        self.eta = kwargs['learning_rate']
        self.epochs = kwargs['epochs']
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


if __name__=="__main__":
    # Data
    data = pd.read_csv("prepped_data.csv")
    X = data.drop('Vx', axis=1).values
    y = data['Vx'].values

    # Break up into test and training data
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=0.1,
    #                                                     shuffle=True)

    # NN = NeuralNetwork((30, 20)) # compare (40, 20)
    # history = NN.fit(X_train, y_train, 
    #                  learning_rate = 0.001,
    #                  epochs = 100,
    #                  batch_size = 64,
    #                  validation_data=(X_test, y_test), 
    #                  return_history=True)
    
    # preds = NN.predict(X_test)

    # print("R2:", r2_score(y_test, preds))
    # print("MAE:", mean_absolute_error(y_test, preds))
    # print("RMSE:", np.power(mean_squared_error(y_test, preds), 0.5))

    # plt.plot(history.history['loss'], label="Training Loss")
    # plt.plot(history.history['val_loss'], label="Validation Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend(loc="upper right")
    # plt.show()

    # test loading
    NN = NeuralNetwork()
    NN.load("NN.keras")
    # print(X)
    # preds = NN.predict(X)
    # print(r2_score(y, preds))
    
    v_wall = 1.2
    func_NN = lambda x: np.power(NN.predict(x) - v_wall, 2)
    print(func_NN(np.array([4.32, 0.8, 0.3, 1.0, 0.75]).reshape(1,-1)))
