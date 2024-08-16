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
beta = 25.0
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
N_test = 900

X_train = np.random.uniform([-4.0, 0], [4.0, 1], size=(N_train, 2)) # note that first column is d and not theta!


d_train = X_train[:, 0]
theta_train = X_train[:, 1]

PhiMat_train = getPhiMat(theta_train, d_train, standardize=False)

d_test_lin = np.linspace(-4, 4, 30)
theta_test_lin = np.linspace(0, 1, 30)

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



# %% make a plot of G test values, with overlaid scatter points of training data.

fig, ax = plt.subplots()
contour_plt = ax.contourf(D_TEST, THETA_TEST, 
                          G_test.reshape(D_TEST.shape), 
                          cmap=parula_cmap)
ax.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', s=50)
ax.set_xlabel(r'$d$', fontsize=20)
ax.set_ylabel(r'$\theta$', fontsize=20)
ax.set_title("Train inputs and contours of G", fontsize=20)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
cbar = fig.colorbar(contour_plt, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
# %%
