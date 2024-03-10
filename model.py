import SymReg
import NeuralNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

symReg = SymReg.SymReg()
symReg.fit(
    X,
    y,
    epochs=10,
    complexity=10
)
symReg.save_model('symReg.json')
x_plot = np.linspace(0, 300, 301)
y_plot = np.zeros_like(x_plot)
for i in range(x_plot.size):
    df = pd.DataFrame(np.array([[x_plot[i], 0.8, 0.196, 1, 1]]), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    y_plot[i] = np.sum(symReg.predict(df))

fig, ax = plt.subplots()
ax.plot(x_plot, y_plot[::-1], label='Symbolic Regression')


X = data.drop('Vx', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.05,
                                                    shuffle=True)
NN = NeuralNet.NeuralNetwork((25, 10, 10))
history = NN.fit(X_train, y_train,
                 learning_rate=0.001,
                 batch_size=64,
                 epochs=20,
                 validation_data=(X_test, y_test))
for i in range(x_plot.size):
    y_plot[i] = np.sum(NN.predict(np.array([[x_plot[i], 0.8, 0.196, 1, 1]])))
ax.plot(x_plot,  y_plot[::-1], label='Neural Network')
ax.legend(loc='lower right')
plt.show()

NN.save("NN.keras")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.legend(loc='upper right')  
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
fig.savefig('NN_loss.png', dpi=1000)