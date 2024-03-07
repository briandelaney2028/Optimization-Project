import SymReg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import pickle

data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

# symReg = SymReg.SymReg()
# symReg.fit(
#     X,
#     y,
#     epochs=10,
#     complexity=10
# )
# symReg.save_model('symReg.json')

noise_est = 0.1
kernel = 1* RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e10)) + WhiteKernel(noise_est**2, noise_level_bounds=(1e-10, 1e10))
gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
)
gp.fit(X, y)

with open('gp.pkl', 'wb') as f:
    pickle.dump(gp, f)
