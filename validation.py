import SymReg
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

# Leave one out cross validation Symbolic Regression
symReg = SymReg.SymReg()

cv_symReg = cross_validate(
    symReg,
    X,
    y,
    params={'epochs':5,
            'complexity':10},
    scoring=('neg_root_mean_squared_error',
             'neg_mean_absolute_error'),
    # cv=y.size-1,
    cv=2,
    return_train_score=True
)

print(cv_symReg)

# Leave one out cross validation Gaussian Process Regression
noise_est = 0.1
kernel = 1* RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e10)) + WhiteKernel(noise_est**2, noise_level_bounds=(1e-10, 1e10))
gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
)

cv_gp = cross_validate(
    gp,
    X,
    y,
    scoring=('neg_root_mean_squared_error',
             'neg_mean_absolute_error'),
    cv=y.size-1,
    return_train_score=True
)

print(cv_gp)



