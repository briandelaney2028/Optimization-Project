import SymReg
import NeuralNet
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from mlxtend.evaluate import paired_ttest_5x2cv, paired_ttest_kfold_cv

# load LAMMPS data
data = pd.read_csv("prepped_data.csv")
X = data.drop('Vx', axis=1)
y = data['Vx'].values

sym1 = SymReg.SymReg()
sym2 = SymReg.SymReg(epochs=5, complexity=5)

NN1 = NeuralNet.NeuralNetwork(layers=(25, 10, 10))
NN2 = NeuralNet.NeuralNetwork(layers=(25, 15, 10))

# 10-fold paired t-test for SR
t, p = paired_ttest_kfold_cv(
    estimator1=sym1,
    estimator2=sym2,
    X=X,
    y=y,
    cv=10,
    random_seed=20
)

print("t statistic:", t)
print("p-value:", p)

# 10-fold paired t-test for ANN
t, p = paired_ttest_kfold_cv(
    estimator1=NN1,
    estimator2=NN2,
    X=X,
    y=y,
    cv=10,
    random_seed=20
)

print("t statistic:", t)
print("p-value:", p)

# cross validation Symbolic Regression
symReg = SymReg.SymReg()

cv_symReg = cross_validate(
    symReg,
    X,
    y,
    fit_params={'epochs':10,
            'complexity':10},
    scoring=('neg_root_mean_squared_error',
             'neg_mean_absolute_error',
             'r2'),
    cv=10,
    return_train_score=True
)

print(cv_symReg)
print("Avg Symbolic Regression Test RMSE: {}".format(-1*np.mean(cv_symReg['test_neg_root_mean_squared_error'])))
print("Avg Symbolic Regression Test MAE: {}".format(-1*np.mean(cv_symReg['test_neg_mean_absolute_error'])))
print("Avg Symbolic Regression Test R2: {}".format(np.mean(cv_symReg['test_r2'])))
with open('10_fold_cv.txt', 'w') as f:
    f.write("Avg Symbolic Regression Test RMSE: {}\n".format(-1*np.mean(cv_symReg['test_neg_root_mean_squared_error'])))
    f.write("Avg Symbolic Regression Test MAE: {}\n".format(-1*np.mean(cv_symReg['test_neg_mean_absolute_error'])))
    f.write("Avg Symbolic Regression Test R2: {}\n".format(np.mean(cv_symReg['test_r2'])))
    f.write("------------------------------------------\n")


X = data.drop('Vx', axis=1).values
# cross validation Neural Network
NN = NeuralNet.NeuralNetwork((25, 10, 10))

cv_NN = cross_validate(
    NN,
    X,
    y,
    fit_params={'epochs':20,
                'batch_size':64,
                'learning_rate':0.001},
    scoring=('neg_root_mean_squared_error',
             'neg_mean_absolute_error',
             'r2'),
    cv=10,
    return_train_score=True
)

print(cv_NN)
print("Avg Neural Network Test RMSE: {}".format(-1*np.mean(cv_NN['test_neg_root_mean_squared_error'])))
print("Avg Neural Network Test MAE: {}".format(-1*np.mean(cv_NN['test_neg_mean_absolute_error'])))
print("Avg Neural Network Test R2: {}".format(np.mean(cv_NN['test_r2'])))
with open('10_fold_cv.txt', 'a') as f:
    f.write("Avg Neural Network Test RMSE: {}\n".format(-1*np.mean(cv_NN['test_neg_root_mean_squared_error'])))
    f.write("Avg Neural Network Test MAE: {}\n".format(-1*np.mean(cv_NN['test_neg_mean_absolute_error'])))
    f.write("Avg Neural Network Test R2: {}\n".format(np.mean(cv_NN['test_r2'])))


# 10-fold paired t-test for SR against ANN
t, p = paired_ttest_kfold_cv(
    estimator1=NN,
    estimator2=symReg,
    X=X,
    y=y,
    cv=10,
    random_seed=20
)

print("t statistic:", t)
print("p-value:", p)