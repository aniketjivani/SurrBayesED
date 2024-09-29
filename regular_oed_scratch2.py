"""
Created on Thu Aug 15
This script continues scratchwork to set up OED for the linear Gaussian case.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import utility_estimation_lg as utils
import viz_surr_oed as viz
from sklearn.preprocessing import StandardScaler
import copy

from IPython import get_ipython
ipython = get_ipython()

ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

parula_cmap = ListedColormap(viz._parula_data, name='parula')

# %%
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

# %%
beta = 36.0
alpha = 2.0

def oracle_lambda():
    return np.array([3.946, -11.825, 0.012, 8.087, 0.165, 4.397, -2.121, -0.655, 0.61, -0.152])

def getPhiMat(theta_vals, d_vals, standardize=False):

    assert len(theta_vals) == len(d_vals), "theta_vals and d_vals should have the same length."
    N_vals = len(theta_vals)

    PhiMatRaw = np.vstack([theta_vals, d_vals, theta_vals**2, d_vals**2, theta_vals*d_vals, theta_vals**2*d_vals, theta_vals*d_vals**2, theta_vals**2*d_vals**2, theta_vals**3*d_vals**3]).T

    if standardize:
        scaler = StandardScaler()
        PhiMatScaled = scaler.fit_transform(PhiMatRaw)
    else:
        PhiMatScaled = copy.deepcopy(PhiMatRaw)

    # add column of ones for intercept
    PhiMat = np.hstack([np.ones((N_vals, 1)), PhiMatScaled])

    return PhiMat


def getGNoisy(PhiMat, beta_val):
    """
    return training data G = PhiMat * lambda + noise where noise variance is 1/beta_val and lambda is the oracle lambda.
    """
    N, p = PhiMat.shape

    lambda_oracle = oracle_lambda()
    noise = np.random.normal(0, np.sqrt(1/beta_val), size=N)

    GNoisy = np.dot(PhiMat, lambda_oracle) + noise
    # GVals = np.dot(PhiMat, lambda_oracle)
    return GNoisy

# %% Create train and test data
# first batch of training data is pilot set before we conduct OED.
# test data is always a grid on the full domain (for toy problems)

N_train = 20
N_test = 400
rng = np.random.default_rng(20241996)

X_train = rng.uniform([-4.0, 0], [4.0, 1], size=(N_train, 2)) # note that first column is d and not theta!

# X_train = np.random.uniform([-4.0, 0], [4.0, 1], size=(N_train, 2)) # note that first column is d and not theta!


d_train = X_train[:, 0]
theta_train = X_train[:, 1]

PhiMat_train = getPhiMat(theta_train, d_train, standardize=False)

d_test_lin = np.linspace(-4, 4, 20)
theta_test_lin = np.linspace(0, 1, 20)

D_TEST, THETA_TEST = np.meshgrid(d_test_lin, theta_test_lin)

d_test = D_TEST.flatten()
theta_test = THETA_TEST.flatten()

PhiMat_test = getPhiMat(theta_test, d_test, standardize=False)

G_train = getGNoisy(PhiMat_train, beta)
G_test = getGNoisy(PhiMat_test, beta)

# %%

# Redo posterior after adding points to the dataset. Plot some contours of weight PDFs, contours of posterior predictive. (in next section redo same plot with addition of utility contours)

oed = utils.OEDLG(PhiMat_train, G_train, beta, alpha, random_state=2021)
utility_analytic_oracle = oed.utility_analytical(PhiMat_test)

# %% plotting

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.2f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

fig, ax = plt.subplots()



CF = ax.contourf(D_TEST, THETA_TEST, 
                 utility_analytic_oracle.reshape(D_TEST.shape),
                 cmap=parula_cmap)
CS = ax.contour(CF, colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
ax.set_title("Closed form utility contours", fontsize=20)
ax.set_xlabel(r"$d$", fontsize=20)
ax.set_ylabel(r"$\theta$", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
# set grid params
ax.grid(True, which='both', linestyle='--')
cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)

# %% 
# plot hifidelity model evaluations at all points
fig, ax = plt.subplots()
CF = ax.contourf(D_TEST, THETA_TEST,
                 G_test.reshape(D_TEST.shape),
                 cmap=parula_cmap)
CS = ax.contour(CF, colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
ax.set_title("Noisy response surface", fontsize=20)
ax.set_xlabel(r"$d$", fontsize=20)
ax.set_ylabel(r"$\theta$", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
# set grid params
ax.grid(True, which='both', linestyle='--')

# scatter plot of training data
ax.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', s=50)

cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)



# %% Select optimal point for next experiment.
opt_x_idx = np.unravel_index(utility_arr.argmax(), utility_arr.shape)
theta_opt = THETA_TEST[opt_x_idx]
d_opt = D_TEST[opt_x_idx]

print(f"Optimal theta: {theta_opt}, optimal d: {d_opt}")
# %% Plot posterior predictive response surface.
cov_post = np.linalg.inv(beta * (PhiMat_train.T @ PhiMat_train) + alpha * np.eye(PhiMat_train.shape[1]))

mean_post = (beta * (cov_post @ PhiMat_train.T @ G_train)).reshape(-1, 1)

G_mean_post = PhiMat_test @ mean_post
G_var_post = (1/beta) + np.sum(PhiMat_test.dot(cov_post) * PhiMat_test, axis=1)

fig, ax = plt.subplots()
CF = ax.contourf(D_TEST, THETA_TEST,
                 G_mean_post.reshape(D_TEST.shape),
                 cmap=parula_cmap)
CS = ax.contour(CF, colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
ax.set_title("Posterior Predictive Mean", fontsize=20)
ax.set_xlabel(r"$d$", fontsize=20)
ax.set_ylabel(r"$\theta$", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
# set grid params
ax.grid(True, which='both', linestyle='--')

# scatter plot of training data
ax.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', s=50)

cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)

# %%
fig, ax = plt.subplots()
CF = ax.contourf(D_TEST, THETA_TEST,
                 np.sqrt(G_var_post).reshape(D_TEST.shape),
                 cmap=parula_cmap)
CS = ax.contour(CF, colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
ax.set_title("Posterior Predictive Std.", fontsize=20)
ax.set_xlabel(r"$d$", fontsize=20)
ax.set_ylabel(r"$\theta$", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
# set grid params
ax.grid(True, which='both', linestyle='--')

# scatter plot of training data
ax.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', s=50)

cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)

# %% Bayesian optimization for design of experiments.

# Calculate analytic utility at training points to use as n_init before building GP and starting optimization.
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

utility_analytic_train = oed.utility_analytical(PhiMat_train)

train_X = torch.tensor(X_train, dtype=torch.double)
train_Y = torch.tensor(utility_analytic_train.reshape(-1, 1), dtype=torch.double)

gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
bounds = torch.stack([torch.tensor([-4.0, 0.0]), torch.tensor([4.0, 1.0])]).to(torch.double)
candidate, acq_value = optimize_acqf(
    logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)

# returns -4, 1

# %% add new point to training set, update posterior and posterior predictive plots.

X_train_new = np.vstack([X_train, candidate.detach().numpy()])

d_train_new = X_train_new[:, 0]
theta_train_new = X_train_new[:, 1]

PhiMat_train_new = getPhiMat(theta_train_new, d_train_new, standardize=False)
G_train_new = getGNoisy(PhiMat_train_new, beta)

# redo posterior
cov_post_new = np.linalg.inv(beta * (PhiMat_train_new.T @ PhiMat_train_new) 
                            # + cov_post)
                             + alpha * np.eye(PhiMat_train_new.shape[1]))

# mean_post_new = beta * cov_post_new @ ((PhiMat_train_new.T @ G_train_new).reshape(-1, 1) + np.linalg.inv(cov_post) @ mean_post) @ (PhiMat_)

mean_post_new = (beta * (cov_post_new @ PhiMat_train_new.T @ G_train_new)).reshape(-1, 1)

G_mean_post_new = PhiMat_test @ mean_post_new
G_var_post_new = (1/beta) + np.sum(PhiMat_test.dot(cov_post_new) * PhiMat_test, axis=1)

# %%
fig, ax = plt.subplots()
CF = ax.contourf(D_TEST, THETA_TEST,
                 G_mean_post_new.reshape(D_TEST.shape),
                 cmap=parula_cmap)
CS = ax.contour(CF, colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
ax.set_title("Posterior Predictive Mean", fontsize=20)
ax.set_xlabel(r"$d$", fontsize=20)
ax.set_ylabel(r"$\theta$", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
# set grid params
ax.grid(True, which='both', linestyle='--')

# scatter plot of training data
ax.scatter(X_train_new[:, 0], X_train_new[:, 1], c='r', marker='x', s=50)

cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)

# %%
fig, ax = plt.subplots()
CF = ax.contourf(D_TEST, THETA_TEST,
                 np.sqrt(G_var_post_new).reshape(D_TEST.shape),
                 cmap=parula_cmap)
CS = ax.contour(CF, colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
ax.set_title("Posterior Predictive Std.", fontsize=20)
ax.set_xlabel(r"$d$", fontsize=20)
ax.set_ylabel(r"$\theta$", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
# set grid params
ax.grid(True, which='both', linestyle='--')

# scatter plot of training data
ax.scatter(X_train_new[:, 0], X_train_new[:, 1], c='r', marker='x', s=50)

cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)

# %%
utility_analytic_train = oed.utility_analytical(PhiMat_train_new)

train_X = torch.tensor(X_train_new, dtype=torch.double)
train_Y = torch.tensor(utility_analytic_train.reshape(-1, 1), dtype=torch.double)

gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
bounds = torch.stack([torch.tensor([-4.0, 0.0]), torch.tensor([4.0, 1.0])]).to(torch.double)
candidate, acq_value = optimize_acqf(
    logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
print(candidate)

# %%

# # %% make a plot of G test values, with overlaid scatter points of training data.

# fig, ax = plt.subplots()
# contour_plt = ax.contourf(D_TEST, THETA_TEST, 
#                           G_test.reshape(D_TEST.shape), 
#                           cmap=parula_cmap)
# ax.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', s=50)
# ax.set_xlabel(r'$d$', fontsize=20)
# ax.set_ylabel(r'$\theta$', fontsize=20)
# ax.set_title("Train inputs and contours of G", fontsize=20)
# ax.xaxis.set_tick_params(labelsize=15)
# ax.yaxis.set_tick_params(labelsize=15)
# cbar = fig.colorbar(contour_plt, ax=ax, fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=15)
# fig.tight_layout()
# # %%
# nOut = 100
# nIn = 100

# noises = np.random.normal(0, np.sqrt(1/beta), size=(nOut, 1))
# lambdas = oed.sample_prior(nOut)

# assert lambdas.shape[0] == nOut, "lambdas should have nOut rows"

# utility_nmc = oed.utility_dnmc_fast(lambdas, PhiMat_test, noises, nIn=nIn)
# # %%
# fig, ax = plt.subplots()



# CF = ax.contourf(D_TEST, THETA_TEST, 
#                  utility_nmc.reshape(D_TEST.shape),
#                  cmap=parula_cmap, vmin=1.2, vmax=6.0)
# CS = ax.contour(CF, colors='k')
# ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
# ax.set_title("NMC utility contours", fontsize=20)
# ax.set_xlabel(r"$d$", fontsize=20)
# ax.set_ylabel(r"$\theta$", fontsize=20)
# ax.xaxis.set_tick_params(labelsize=15)
# ax.yaxis.set_tick_params(labelsize=15)
# # set grid params
# ax.grid(True, which='both', linestyle='--')
# cbar = fig.colorbar(CF, ax=ax, fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=15)
# # %%

# %%
