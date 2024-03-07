from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
# y = np.squeeze(np.sin(X) + np.sin(2*X)/2 + np.sin(3*X)/3)
y = np.squeeze(10*np.ones_like(X) -0.1 * np.power(X, 2) + np.sin(X) - np.sin(2*X) + np.sin(3*X)/3 - np.sin(4*X))

rng = np.random.RandomState(102)
training_indices = rng.choice(np.arange(y.size), size=55, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) 
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
print(gaussian_process.kernel_)

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
# plt.scatter(X_train, y_train, label="Observations")
# plt.plot(X, mean_prediction, label="Mean prediction")
# plt.fill_between(
#     X.ravel(),
#     mean_prediction - 1.96 * std_prediction,
#     mean_prediction + 1.96 * std_prediction,
#     alpha=0.5,
#     label=r"95% confidence interval",
# )
# plt.legend()
# plt.xlabel("$x$")
# plt.ylabel("$f(x)$")
# _ = plt.title("Gaussian process regression on noise-free dataset")

# add noise

noise_std = 0.50
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

# include noise parameter in GPR
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_std**2)
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)


# import feyn
# import pandas as pd
# ql = feyn.QLattice(random_seed=1000)

# train = pd.DataFrame(X_train, columns=['X'])
# train['y'] = y_train
# test = pd.DataFrame(X, columns=['X'])
# test['y'] = y

# models = ql.auto_run(
#     data=train,
#     output_name='y',
#     kind='regression'
# )

# best = models[0]
# best.plot(
#     data=train,
#     compare_data=test,
#     filename='test'
# )

# plt.plot(X, best.predict(test), label='SymReg')

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on a noisy dataset")

plt.show()