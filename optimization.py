import pickle
import feyn
from scipy.optimize import minimize, LinearConstraint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

# lead Symbolic Regressor
symReg = feyn.Model.load('symReg.json')

# load guassian process regressor
with open('gp.pkl', 'rb') as f:
    gp = pickle.load(f)

# intial guess
# [X, Density, Incline Factor, Fluid-Wall Interaction, Wall Speed]
x0 = [0.0, 0.8, 0.3, 0.95, 0.75]


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
v_wall = 7
# Target delta x
delta_x = 4
# tolerance
epsilon = 0.01

# constraints
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
    df = pd.DataFrame(x, columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPPED'])
    return np.power(symReg.predict(df)-v_wall, 2)


# Symbolic Regression optimizaiton
results_symReg = minimize(
    func_symReg,
    x0,
    method='SLSQP',
    bounds=bounds,
    options=options,
    constraints=c
)

# optimization function for gaussian process regression
func_gp = lambda x: np.power(gp.predict(x.reshape(1, -1)) - v_wall, 2)

# Gaussian Process Regression optimization
results_gp = minimize(
    func_gp,
    x0,
    method='SLSQP',
    bounds=bounds,
    options=options,
    constraints=c
)

for key in results_gp.keys():
    print("-------------- {} --------------".format(key))
    print("Symbolic Regression:         {}".format(results_symReg[key]))
    print("Gaussian Process Regression: {}".format(results_gp[key]))


# plotting
n = 10
x_sym = np.linspace(0, results_symReg['x'][0], n)
x_gp = np.linspace(0, results_gp['x'][0], n)
y_sym = np.zeros(n)
y_gp = np.zeros(n)
_, rho_sym, ig_sym, fwi_sym, ws_sym = results_symReg['x']
_, rho_gp, ig_gp, fwi_gp, ws_gp = results_gp['x']
for i in range(n):
    df = pd.DataFrame(np.array([x_sym[i], rho_sym, ig_sym, fwi_sym, ws_sym]).reshape(1, -1), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    y_sym[i] = np.sum(symReg.predict(df))
    df = pd.DataFrame(np.array([x_gp[i], rho_gp, ig_gp, fwi_gp, ws_gp]).reshape(1, -1), columns=['X', 'DENSITY', 'INCLINE FACTOR', 'FLUID-WALL INTERACTION', 'WALL SPEED'])
    y_gp[i] = np.sum(gp.predict(df))

fig, ax = plt.subplots()
ax.plot(x_sym, y_sym[::-1], 'b-o', label="Symbolic Regression")
ax.plot(x_gp, y_gp[::-1], 'r-o', label="Gaussian Process Regression")
ax.legend(loc='upper right')  
ax.set_xlim([0, np.max((np.max(x_sym), np.max(x_gp)))])
ax.set_ylim([0, np.max((np.max(y_sym), np.max(y_gp)))])
ax.set_xlabel('X position [Unit cell lengths]')
ax.set_ylabel('Velcoity [lattice lengths / timestep]')

fig.savefig('Optimization_Comparison.png', dpi=1000)

