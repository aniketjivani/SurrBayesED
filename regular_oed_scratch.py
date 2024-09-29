# %%

import numpy as np
import matplotlib.pyplot as plt
import emcee
from oed_wanggang import *
# logpdf of independent normal distribution.
# x is of size (n_sample, n_param).
# loc and scale are int or numpy.ndarray of size n_param.
# output is of size n_sample.

# %% Source: https://github.com/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression_util.py

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def plot_data(x, t):
    plt.scatter(x, t, marker='o', c="k", s=20)


def plot_truth(x, y, label='Truth'):
    plt.plot(x, y, 'k--', label=label)


def plot_predictive(x, y, std, y_label='Prediction', std_label='Uncertainty', plot_xy_labels=True):
    y = y.ravel()
    std = std.ravel()

    plt.plot(x, y, label=y_label)
    plt.fill_between(x.ravel(), y + std, y - std, alpha = 0.5, label=std_label)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior_samples(x, ys, plot_xy_labels=True):
    plt.plot(x, ys[:, 0], 'r-', alpha=0.5, label='Post. samples')
    for i in range(1, ys.shape[1]):
        plt.plot(x, ys[:, i], 'r-', alpha=0.5)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior(mean, cov, w0, w1):
    resolution = 100

    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)
    plt.imshow(densities, origin='lower', extent=(-1, 1, -1, 1))
    plt.scatter(w0, w1, marker='x', c="r", s=20, label='Truth')

    plt.xlabel('w0')
    plt.ylabel('w1')


def print_comparison(title, a, b, a_prefix='np', b_prefix='br'):
    print(title)
    print('-' * len(title))
    print(f'{a_prefix}:', a)
    print(f'{b_prefix}:', b)
    print()


# %%

# nonlinear test case
def model_1(theta, d):
    """
    Nonlinear model.

    Parameters
    ----------
    theta : np.ndarray of size (n_sample or 1, n_param)
        The value of unknown linear model parameters.
    d : np.ndarray of size (n_sample or 1, n_design)
        The design variable.

    Returns
    -------
    numpy.ndarray of size (n_sample, n_obs)
        The output of the linear model.
    """
    return theta ** 3 * d ** 2 + theta * np.exp(-np.abs(0.2 - d))


def model_2(theta, d):
    """
    Nonlinear model.

    Parameters
    ----------
    theta : np.ndarray of size (n_sample or 1, n_param)
        The value of unknown linear model parameters.
    d : np.ndarray of size (n_sample or 1, n_design)
        The design variable.

    Returns
    -------
    numpy.ndarray of size (n_sample, n_obs)
        The output of the linear model.
    """
    return theta ** 1.5 * d ** 2 + theta * np.exp(-((0.2 - d) / 0.3)**2)


def model_3(theta, d, use_only_d=False):
    """
    Nonlinear model.

    Parameters
    ----------
    theta : np.ndarray of size (n_sample or 1, n_param)
        The value of unknown linear model parameters.
    d : np.ndarray of size (n_sample or 1, n_design)
        The design variable.

    Returns
    -------
    numpy.ndarray of size (n_sample, n_obs)
        The output of the linear model.
    """
    if use_only_d:
        return 0.5 * d + 2 * np.sin(0.5 * d) + 3 * np.exp(-0.1 * (d - 5)**2) + 5 * np.exp(-0.5 * (d - 15)**2)
    else:
        return theta * d + 2 * np.sin(theta * d) + 3 * np.exp(-theta**2 * (d - 5)**2) + 5 * np.exp(-theta**2 * (d - 15)**2)


def model_4(theta, d):
    """
    Nonlinear model.

    Parameters
    ----------
    theta : np.ndarray of size (n_sample or 1, n_param)
        The value of unknown linear model parameters.
    d : np.ndarray of size (n_sample or 1, n_design)
        The design variable.

    Returns
    -------
    numpy.ndarray of size (n_sample, n_obs)
        The output of the linear model.
    """
    return theta * d + 2 * np.sin(theta * d) + 3 * np.exp(-(theta**2) * (d - 5)**2) * np.sin(2 * d) + 5 * np.exp(-(theta**2) * (d - 15)**2)

n_param = 1 # Number of parameters.
n_design = 1 # Number of design variables.
n_obs = 1 # Number of observations.

low = 0
high = 1
prior_rvs = lambda n_sample: np.random.uniform(low=low,
                                               high=high,
                                               size=(n_sample, n_param))
prior_logpdf = lambda theta: uniform_logpdf(theta,
                                            low=low,
                                            high=high)

design_bounds = [(0, 1),] # lower and upper bounds of design variables.

# Noise if following N(noise_loc, noise_base_scale + noise_ratio_scale * abs(G))
noise_loc = 0
noise_base_scale = 0.01
noise_ratio_scale = 0
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale),]

# Random state could be eith an integer or None.
random_state = 2021


# %%
oed_1 = OED(model_fun=model_1,
            n_param=n_param,
            n_design=n_design,
            n_obs=n_obs,
            prior_rvs=prior_rvs,
            design_bounds=design_bounds,
            noise_info=noise_info,
            prior_logpdf=prior_logpdf,
            reward_fun=None,
            random_state=random_state)


oed_2 = OED(model_fun=model_3,
            n_param=n_param,
            n_design=n_design,
            n_obs=n_obs,
            prior_rvs=prior_rvs,
            design_bounds=design_bounds,
            noise_info=noise_info,
            prior_logpdf=prior_logpdf,
            reward_fun=None,
            random_state=random_state)


oed_4 = OED(model_fun=model_4,
            n_param=n_param,
            n_design=n_design,
            n_obs=n_obs,
            prior_rvs=prior_rvs,
            design_bounds=design_bounds,
            noise_info=noise_info,
            prior_logpdf=prior_logpdf,
            reward_fun=None,
            random_state=random_state)
# %% Plot of utility on grid:
# ds = np.linspace(design_bounds[0][0], design_bounds[0][1], 21)
# Us = []
# thetas = prior_rvs(1000)
# noises = np.random.normal(size=(1000, n_obs))

# for d in ds:
#     Us.append(oed_1.exp_utility(d, thetas, noises))

# plt.figure(figsize=(6, 4))
# plt.plot(ds, Us)
# plt.xlabel('d', fontsize=20)
# plt.ylabel('U(d)', fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(ls='--')
# plt.show()

# %%

# ds2 = np.linspace(-4, 4, 51)
# Us2 = []
# thetas2 = prior_rvs(1000)
# noises2 = np.random.normal(size=(1000, n_obs))

# for d in ds2:
#     Us2.append(oed_2.exp_utility(d, thetas2, noises2))

# plt.figure(figsize=(6, 4))
# plt.plot(ds2, Us2)
# plt.xlabel(r'$d$', fontsize=20)
# plt.ylabel(r'$U(d)$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(ls='--')
# plt.show()


# %%

# ds4 = np.linspace(-4, 4, 51)
# Us4 = []
# thetas4 = prior_rvs(1000)
# noises4 = np.random.normal(size=(1000, n_obs))

# for d in ds4:
#     Us4.append(oed_4.exp_utility(d, thetas4, noises4))

# plt.figure(figsize=(6, 4))
# plt.plot(ds4, Us4)
# plt.xlabel(r'$d$', fontsize=20)
# plt.ylabel(r'$U(d)$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(ls='--')
# plt.show()

# %% Plotting model outputs

# yy_reg = np.zeros((1000, 51))
# for i, d in enumerate(ds2):
#     yy_reg[:, i] = model_3(thetas2, d).flatten()

# yy_reg_mean = np.mean(yy_reg, axis=0)
# yy_reg_std = np.std(yy_reg, axis=0)

# # plt.plot(ds2, yy_reg_mean)
# # plt.plot(ds2, yy_reg_mean - yy_reg_std)
# # plt.plot(ds2, yy_reg_mean + yy_reg_std)

# # make filled plot
# plt.plot(ds2, yy_reg_mean, label='Mean')
# plt.fill_between(ds2, yy_reg_mean - 2 * yy_reg_std, yy_reg_mean + 2 * yy_reg_std, alpha=0.3, label='')
# plt.xlabel(r'$d$', fontsize=20)
# plt.ylabel(r'$G(\theta, d)$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(ls='--')
# plt.show()

yy_reg = np.zeros((50, 100))
# get 2D grid of theta and d
thetas_reg = np.linspace(0, 1, 50)
ds_reg = np.linspace(-4, 4, 100)
THETAS, DS = np.meshgrid(thetas_reg, ds_reg)

# evaluate for each d in ds_reg over all thetas_reg:
for i, d in enumerate(ds_reg):
    yy_reg[:, i] = model_3(thetas_reg, d).flatten()

# Contour Plot:
plt.figure(figsize=(8, 6))
plt.contourf(thetas_reg, ds_reg, yy_reg.T, levels=20)
# overlay line with calculated y for theta = 0.5.
# vline for theta
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$d$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.colorbar()

# %%


# %% REGRESSION:

# Modified from:
# https://krasserm.github.io/2019/02/23/bayesian-linear-regression/

import numpy as np


def posterior(Phi, t, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N


def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    
    return y, y_var


#%% Generate train inputs (each N x 2 arrays, one column for theta and one for d)
np.random.seed(2025)

Ntrain = 50
sigma_eta = 0.1
beta = 1.0 / sigma_eta**2 # precision
alpha = 2.0 # prior precision (prior for weights is N(0, alpha^(-1)I))

# generate random theta and d simultaneously
X_train = np.random.uniform([-4.0, 0], [4.0, 1], size=(Ntrain, 2)) # note that first column is d and not theta!



# generate observations with noise.
Y_train = np.zeros(Ntrain)
for i in range(Ntrain):
    Y_train[i] = model_3(X_train[i, 1], X_train[i, 0]) + sigma_eta*np.random.randn()



plt.figure(figsize=(8, 6))
contour_plt = plt.contourf(thetas_reg, ds_reg, yy_reg.T, levels=20)
# overlay line with training scatter:
plt.scatter(X_train[:, 1], X_train[:, 0], c='r', marker='x', s=50)
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$d$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.colorbar(contour_plt)


def tabulate_coeffs():
    """list coefficients from two term expansion of taylor series."""
    return [8, 0, 0, -1200, 0, 30, 180, 0, -8, -1/4]

def model_taylor_approx(theta_vals, d_vals):
    """return the taylor approximation of the model. raw_data is some Nx2 matrix where first column is made up of d and second column of theta"""
    coeffs = tabulate_coeffs()

    taylor_approx = np.zeros((len(theta_vals), len(d_vals)))
    for i, d in enumerate(d_vals):
        taylor_approx[:, i] = coeffs[0] + coeffs[1]*theta_vals + coeffs[2]*d + coeffs[3]*theta_vals**2 + coeffs[4]*d**2 + coeffs[5]*theta_vals*d + coeffs[6]*theta_vals**2*d + coeffs[7]*theta_vals*d**2 + coeffs[8]*theta_vals**2*d**2 + coeffs[9]*theta_vals**3*d**3

    # theta_vals = raw_data[:, 1]
    # d_vals = raw_data[:, 0]

    return taylor_approx

ytaylor = model_taylor_approx(thetas_reg, ds_reg)
plt.figure(figsize=(8, 6))
contour_plt = plt.contourf(thetas_reg, ds_reg, ytaylor.T, levels=20)
# overlay line with training scatter:
plt.scatter(X_train[:, 1], X_train[:, 0], c='r', marker='x', s=50)
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$d$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.colorbar(contour_plt)
plt.title("Linearized")

# Least squares check: How good of a fit can we get?
Ntest = 100
X_test = np.random.uniform([-4.0, 0], [4.0, 1], size=(Ntest, 2))
Y_test = np.zeros(Ntest)
for i in range(Ntest):
    Y_test[i] = model_3(X_test[i, 1], X_test[i, 0]) + sigma_eta*np.random.randn()

theta_train = X_train[:, 1]
d_train = X_train[:, 0]

theta_test = X_test[:, 1]
d_test = X_test[:, 0]

A = np.vstack([np.ones(Ntrain), theta_train, d_train, theta_train**2, d_train**2, theta_train*d_train, theta_train**2*d_train, theta_train*d_train**2, theta_train**2*d_train**2, theta_train**3*d_train**3]).T

A_test = np.vstack([np.ones(Ntest), theta_test, d_test, theta_test**2, d_test**2, theta_test*d_test, theta_test**2*d_test, theta_test*d_test**2, theta_test**2*d_test**2, theta_test**3*d_test**3]).T

beta_fit = np.linalg.lstsq(A, Y_train)[0:1][0]
Y_fit = np.dot(A, beta_fit)

Y_pred = np.dot(A_test, beta_fit)

# plot scatter of training data and fit
plt.figure(figsize=(8, 6))
plt.scatter(Y_train, Y_fit, c='r', marker='x', label='Training Data')
plt.scatter(Y_test, Y_pred, c='b', marker='o', label='Test Data')
# plot diagonal line
plt.plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], 'k--', label='')
plt.xlabel('True Y', fontsize=20)
plt.ylabel('Fitted Y', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()

# plot pred over domain:
# y_pred_domain = np.zeros((len(thetas_reg) * len(ds_reg)))

# meshgrid and flatten:
THETAS, DS = np.meshgrid(thetas_reg, ds_reg)
theta_flat = THETAS.flatten()
d_flat = DS.flatten()

A_domain = np.vstack([np.ones(len(theta_flat)), theta_flat, d_flat, theta_flat**2, d_flat**2, theta_flat*d_flat, theta_flat**2*d_flat, theta_flat*d_flat**2, theta_flat**2*d_flat**2, theta_flat**3*d_flat**3]).T

y_pred_domain_flat = np.dot(A_domain, beta_fit)

# reshape and plot:
y_pred_domain = y_pred_domain_flat.reshape(len(ds_reg), len(thetas_reg))
plt.figure(figsize=(8, 6))
contour_plt = plt.contourf(thetas_reg, ds_reg, y_pred_domain, levels=20)
# overlay line with training scatter:
plt.scatter(X_train[:, 1], X_train[:, 0], c='r', marker='x', s=50)
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$d$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.colorbar(contour_plt)
plt.title("Least Squares")



# %%

# Generate training data as G = Phi * lambda + noise

beta = 25.0
alpha = 2.0

def oracle_lambda():
    return np.array([3.946, -11.825, 0.012, 8.087, 0.165, 4.397, -2.121, -0.655, 0.61, -0.152])

def getPhiMat(theta_vals, d_vals):

    assert len(theta_vals) == len(d_vals), "theta_vals and d_vals should have the same length."
    N_vals = len(theta_vals)

    PhiMat = np.vstack([np.ones(N_vals), theta_vals, d_vals, theta_vals**2, d_vals**2, theta_vals*d_vals, theta_vals**2*d_vals, theta_vals*d_vals**2, theta_vals**2*d_vals**2, theta_vals**3*d_vals**3]).T
    return PhiMat


# Replace by nonlinear model from test_problems.py if we want to introduce model misspecification.
def getGNoisy(PhiMat, beta_val):
    """
    return training data G = PhiMat * lambda + noise where noise variance is 1/beta_val and lambda is the oracle lambda.
    """
    N, p = PhiMat.shape

    lambda_oracle = oracle_lambda()
    noise = np.random.normal(0, np.sqrt(1/beta_val), size=N)

    GNoisy = np.dot(PhiMat, lambda_oracle) + noise
    return GNoisy




# %% Plots of posterior density and samples in the predictive
# this is for illustration purposes, we will repeat this plot when we acquire new samples through OED. For now we will just show the result of using subset of the training data in each stage.



# %% Plots of analytic vs estimated utility (we will probably use lstsq from numpy with the correct feature matrix that has 1/alpha I included for regularization.)




# %% Repeat plots of posterior density and predictions but this time first figure will contain all of initial training data and rest will include batches / single batch of new samples.

