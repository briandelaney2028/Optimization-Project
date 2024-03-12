import SymReg
import NeuralNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load LAMMPS solution data
data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

# fit Symbolic Regressor 
symReg = SymReg.SymReg()
symReg.fit(
    X,
    y,
    epochs=10,
    complexity=10
)
# save model
symReg.save_model('symReg.json')

# neural network does not need dataframe for training
X = data.drop('Vx', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.05,
                                                    shuffle=True)
# train ANN
NN = NeuralNet.NeuralNetwork((25, 10, 10))
history = NN.fit(X_train, y_train,
                 learning_rate=0.001,
                 batch_size=64,
                 epochs=20,
                 validation_data=(X_test, y_test))
# save ANN
NN.save("NN.keras")

# plot ANN loss metrics
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.legend(loc='upper right')  
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
fig.savefig('NN_loss.png', dpi=1000)