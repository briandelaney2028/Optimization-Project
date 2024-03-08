import SymReg
import NeuralNet
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

# cross validation Symbolic Regression
symReg = SymReg.SymReg()

# cv_symReg = cross_validate(
#     symReg,
#     X,
#     y,
#     fit_params={'epochs':10,
#             'complexity':10},
#     scoring=('neg_root_mean_squared_error',
#              'neg_mean_absolute_error'),
#     cv=10,
#     return_train_score=True
# )

# print(cv_symReg)
# print("Avg Symbolic Regression Test RMSE: {}".format(np.mean(cv_symReg['test_neg_root_mean_squared_error'])))
# print("Avg Symbolic Regression Test MAE: {}".format(np.mean(cv_symReg['test_neg_mean_absolute_error'])))

X = data.drop('Vx', axis=1).values
# cross validation Neural Network
NN = NeuralNet.NeuralNetwork((30, 20))

cv_NN = cross_validate(
    NN,
    X,
    y,
    fit_params={'epochs':20,
                'batch_size':64,
                'learning_rate':0.001},
    scoring=('root_mean_squared_error',
             'mean_absolute_error'),
    cv=5,
    return_train_score=True
)

print(cv_NN)
print("Avg Neural Network Test RMSE: {}".format(np.mean(cv_NN['test_root_mean_squared_error'])))
print("Avg Neural Network Test MAE: {}".format(np.mean(cv_NN['test_mean_absolute_error'])))


