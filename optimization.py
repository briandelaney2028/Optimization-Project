import NeuralNet
import feyn
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

# lead Symbolic Regressor
symReg = feyn.Model.load('symReg.json')

# load Neural Network
NN = NeuralNet.NeuralNetwork()
NN.load("NN.keras")

# intial guess
# [X, Density, Incline Factor, Fluid-Wall Interaction, Wall Speed]
x0 = np.array([300.0, 0.8, 0.3, 0.95, 0.75])


# define bounds
bounds = ((0.0, 400.0), # X
          (0.75, 0.85), # Density
          (0.133, 0.5), # Incline Factor
          (0.9, 1.0),   # Fluid-Wall Interaction
          (0.5, 1.0))   # Wall Speed

# optimization options
options = {
    'maxiter':100,
    'disp':True
}

# Target wall speed
v_wall = 0.9
# Target delta x
delta_x = 300
# tolerance
epsilon = 0.01

# Linear constraints
A = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]
])
lb = np.array([delta_x - epsilon])
ub = np.array([delta_x + epsilon])
c = LinearConstraint(A, lb, ub)

# optimization function for Symbolic Regression
def func_symReg(x):
    x = np.array(x).reshape(1,-1)
    df = pd.DataFrame(x, columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    return np.power(symReg.predict(df)-v_wall, 2)

# nonlinear constraints for Symbolic Regression
def noncon_sym(x):
    x_prime = np.zeros(5)
    for i in range(1, 5):
        x_prime[i] = x[i]
    df = pd.DataFrame(x_prime.reshape(1,-1), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    return symReg.predict(df)
nc_sym = NonlinearConstraint(noncon_sym, -0.1, 0.1)

# Symbolic Regression optimizaiton
results_symReg = minimize(
    func_symReg,
    x0,
    method='SLSQP',
    bounds=bounds,
    options=options,
    constraints=(c, nc_sym)
)

# optimization function for Neural Network 
func_NN = lambda x: np.power(NN.predict(x.reshape(1, -1)) - v_wall, 2)

# nonlinear constraints for Neural Network
def noncon_NN(x):
    x_prime = np.zeros(5)
    for i in range(1, 5):
        x_prime[i] = x[i]
    return np.sum(NN.predict(x_prime.reshape(1, -1)))
nc_NN = NonlinearConstraint(noncon_NN, -0.1, 0.1)

# Neural Network optimization
results_NN = minimize(
    func_NN,
    x0,
    method='SLSQP',
    bounds=bounds,
    options=options,
    constraints=(c, nc_NN)
)

with open('optimization_results.txt', 'w') as f:
    for key in results_NN.keys():
        print("-------------- {} --------------".format(key))
        f.write("-------------- {} --------------\n".format(key))
        print("Symbolic Regression: {}".format(results_symReg[key]))
        f.write("Symbolic Regression: {}\n".format(results_symReg[key]))
        print("Neural Network     : {}".format(results_NN[key]))
        f.write("Neural Network     : {}\n".format(results_NN[key]))


# plotting
n = 100
x_sym = np.linspace(0, results_symReg['x'][0], n)
x_NN = np.linspace(0, results_NN['x'][0], n)
y_sym = np.zeros(n)
y_NN = np.zeros(n)
_, rho_sym, ig_sym, fwi_sym, ws_sym = results_symReg['x']
_, rho_NN, ig_NN, fwi_NN, ws_NN = results_NN['x']
for i in range(n):
    df = pd.DataFrame(np.array([x_sym[i], rho_sym, ig_sym, fwi_sym, ws_sym]).reshape(1, -1), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    y_sym[i] = np.sum(symReg.predict(df))
    df = pd.DataFrame(np.array([x_NN[i], rho_NN, ig_NN, fwi_NN, ws_NN]).reshape(1, -1), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    y_NN[i] = np.sum(NN.predict(df))

fig, ax = plt.subplots()
ax.plot(x_sym, y_sym[::-1], 'b-o', label="Symbolic Regression")
ax.plot(x_NN, y_NN[::-1], 'r-o', label="Neural Network")
ax.legend(loc='lower left')  
# ax.set_xlim([0, np.max((np.max(x_sym), np.max(x_NN)))])
# ax.set_ylim([0, np.max((np.max(y_sym), np.max(y_NN)))])
ax.set_xlabel('X position [Unit cell lengths]')
ax.set_ylabel('Velcoity [lattice lengths / timestep]')

fig.savefig('Optimization_Comparison.png', dpi=1000)

