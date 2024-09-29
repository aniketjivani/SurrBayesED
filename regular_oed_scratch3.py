# %%
# Linear surrogate with GP Correction.
# Fixed GP Lengthscale.
# Plots of posterior params (analytical and MCMC)
# Utility (analytic and DNMC)
# Utility over sequential acquisition (with MCMC)
# Plots of posterior predictive params
# (Optional) - Goal oriented.

# %% Imports
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

# %% Model Setup and Pilot Samples

# Non-linear model, no noise
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

def oracle_lambda():
    return np.array([3.946, -11.825, 0.012, 8.087, 0.165, 4.397, -2.121, -0.655, 0.61, -0.152])


runNonLin = True
beta = 2.0
alpha = 2.0

def getPhiMat(theta_vals, d_vals):

    # assert len(theta_vals) == len(d_vals), "theta_vals and d_vals should have the same length."
    # N_vals = len(theta_vals)

    # check if only 1 theta_vals is supplied
    if len(theta_vals.shape) == 0:
        N_vals = 1
    else:
        N_vals = len(theta_vals)

    PhiMatRaw = np.vstack([theta_vals, d_vals, theta_vals**2, d_vals**2, theta_vals*d_vals, theta_vals**2*d_vals, theta_vals*d_vals**2, theta_vals**2*d_vals**2, theta_vals**3*d_vals**3]).T

    # add column of ones for intercept
    PhiMat = np.hstack([np.ones((N_vals, 1)), PhiMatRaw])

    return PhiMat

def oracle_lambda():
    return np.array([3.946, -11.825, 0.012, 8.087, 0.165, 4.397, -2.121, -0.655, 0.61, -0.152])

N_train = 20
N_test = 400
rng = np.random.default_rng(20241996)

lbd = -4.0
ubd = 4.0

lb_theta = 0.0
ub_theta = 1.0

X_train = rng.uniform([lbd, lb_theta], [ubd, ub_theta], size=(N_train, 2)) # note that first column is d and not theta!

# X_train = np.random.uniform([-4.0, 0], [4.0, 1], size=(N_train, 2)) # note that first column is d and not theta!


d_train = X_train[:, 0]
theta_train = X_train[:, 1]

PhiMat_train = getPhiMat(theta_train, d_train)

d_test_lin = np.linspace(lbd, ubd, 20)
theta_test_lin = np.linspace(lb_theta, ub_theta, 20)

D_TEST, THETA_TEST = np.meshgrid(d_test_lin, theta_test_lin)

d_test = D_TEST.flatten()
theta_test = THETA_TEST.flatten()

X_test = np.vstack([d_test, theta_test]).T

PhiMat_test = getPhiMat(theta_test, d_test)

# For nonlinear model
if runNonLin:
    G_train = model_4(theta_train, d_train)
    G_test = model_4(theta_test, d_test)
else:
    # For linear model
    G_train = PhiMat_train @ oracle_lambda()
    G_test = PhiMat_test @ oracle_lambda()

# %% Plot contours of G_test, scatter training points.
viz.plotGTestTrain(G_test, D_TEST, THETA_TEST, X_train, cmap=parula_cmap)

# %% Scale input and output data so regression is easier.

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

Phi_train_scaled = scaler_X.fit_transform(PhiMat_train)
G_train_scaled = scaler_Y.fit_transform(G_train.reshape(-1, 1))

Phi_test_scaled = scaler_X.transform(PhiMat_test)
G_test_scaled = scaler_Y.transform(G_test.reshape(-1, 1))

# Remove intercept column from Phi
Phi_train_scaled_reg = Phi_train_scaled[:, 1:]
Phi_test_scaled_reg = Phi_test_scaled[:, 1:]

# %% Posteriors (closed form)
mu_lambda = np.zeros((Phi_train_scaled_reg.shape[1], 1))
Tau_lambda = (1 / alpha) * np.eye(Phi_train_scaled_reg.shape[1])
Tau_lambda_inv = alpha * np.eye(Phi_train_scaled_reg.shape[1])

# GP Noise covariance matrix between training points K_tr:
# Use RBF Kernel with fixed lengthscale.

sigma_ls = 0.3

from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
from scipy.linalg import cho_factor, cho_solve

# compute pairwise dist between training points
pairwise_dists = pairwise_distances(X_train, metric='euclidean')

# K_tr = (1/beta) * np.exp(-pairwise_dists**2 / (2 * sigma_ls**2))

K_tr = (1/beta) * np.exp(-pairwise_dists**2 / (2 * sigma_ls**2))

# K_tr = np.exp(-pairwise_dists**2 / (2 * sigma_ls**2))\
    # + (1/beta) * np.eye(N_train)


L = cho_factor(K_tr)

# Tau_lambda_post_y_inv = (Tau_lambda_inv + (Phi_train_scaled_reg.T @ np.linalg.inv(K_tr) @ Phi_train_scaled_reg))

Tau_lambda_post_y_inv = (Tau_lambda_inv + (Phi_train_scaled_reg.T @ cho_solve(L, Phi_train_scaled_reg)))


# get parts of mu_lambda_post_y
# mu_lambda_post_y_sol = 
                        # (Tau_lambda_inv @ mu_lambda 
# mu_lambda_post_y_sol = Phi_train_scaled_reg.T @ np.linalg.inv(K_tr) @ G_train_scaled

mu_lambda_post_y_sol = Tau_lambda_inv @ mu_lambda + Phi_train_scaled_reg.T @ cho_solve(L, G_train_scaled)

mu_lambda_post_y = np.linalg.lstsq(Tau_lambda_post_y_inv, mu_lambda_post_y_sol)[0]

# Posterior predictive:

K_tr_star = (1/beta) * np.exp(-pairwise_distances(X_train, X_test, metric='euclidean')**2 / (2 * sigma_ls**2))

K_star_star = (1/beta) * np.exp(-pairwise_distances(X_test, metric='euclidean')**2 / (2 * sigma_ls**2))

# Calculate mean, std, and use scaler_Y to get back to original scale.
# G_test_post_mean = (Phi_test_scaled_reg @ mu_lambda_post_y + K_tr_star.T @ np.linalg.inv(K_tr) @ (G_train_scaled - Phi_train_scaled_reg @ mu_lambda_post_y))

# G_test_post_var = (Phi_test_scaled_reg @ (np.linalg.lstsq(Tau_lambda_post_y_inv, Phi_test_scaled_reg.T)[0])) + (K_star_star - K_tr_star.T @ np.linalg.inv(K_tr) @ K_tr_star)

G_test_post_mean = (Phi_test_scaled_reg @ mu_lambda_post_y + K_tr_star.T @ cho_solve(L, (G_train_scaled - Phi_train_scaled_reg @ mu_lambda_post_y)))

G_test_post_var = (Phi_test_scaled_reg @ (np.linalg.lstsq(Tau_lambda_post_y_inv, Phi_test_scaled_reg.T)[0])) + (K_star_star - K_tr_star.T @ cho_solve(L, K_tr_star))

G_test_post_var_sym = (G_test_post_var + G_test_post_var.T) / 2

eval_cov, evec_cov = np.linalg.eigh(G_test_post_var_sym)
eval_cov = np.maximum(eval_cov, 1e-10)

G_test_post_var_lr =  evec_cov @ np.diag(eval_cov) @ evec_cov.T
G_test_post_var_diag = np.diag(G_test_post_var_lr)

G_test_post_std = np.sqrt(G_test_post_var_diag).reshape(-1, 1)

G_test_post_mean_unscaled = scaler_Y.inverse_transform(G_test_post_mean)
G_test_post_std_unscaled = scaler_Y.inverse_transform(G_test_post_std)

# %% Plot posterior of parameters and predictions

# Draw samples from prior of parameters
n_samples = 1000
lambda_prior_samps = np.random.multivariate_normal(mu_lambda.flatten(), Tau_lambda, size=n_samples)

lambda_post_samps = np.random.multivariate_normal(mu_lambda_post_y.flatten(), np.linalg.inv(Tau_lambda_post_y_inv), size=n_samples)

# Posterior of parameters
fig, axs = plt.subplots(5, 2, figsize=(5, 10))
for i, ax in enumerate(axs.flatten()):
    if i != (Phi_train_scaled_reg.shape[1]):
        ax.hist(lambda_prior_samps[:, i], bins=20, color='red', alpha=0.5)
        ax.hist(lambda_post_samps[:, i], bins=20, color='blue', alpha=0.5)
        ax.set_title(f"$\lambda_{i + 1}$")

plt.tight_layout()

# %% Posterior (MCMC using emcee)

# n_param = Phi_test_scaled_reg.shape[1]

# # log_prob = lambda x : self.post_logpdf(x.reshape(-1, self.n_param), 
# #                                         d=d,
# #                                         y=y)
# # def log_prob(x, mu, cov):
# #     diff = x - mu
# #     return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

# prior_rvs = np.random.multivariate_normal(mu_lambda.flatten(), Tau_lambda, size=n_samples)





# n_dim, n_walkers = n_param, 2 * n_param
# theta0 = prior_rvs(n_walkers)
# sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
# sampler.run_mcmc(theta0, 
#                     int(n_sample // n_walkers * 1.2), 
#                     progress=False)
# return sampler.get_chain().reshape(-1, self.n_param)[-n_sample:]

# %% Predictions and separate GP Plots

# Plot Prior Predictive
# G_test_prior_mean = scaler_Y.inverse_transform(Phi_test_scaled_reg @ mu_lambda)
# G_test_prior_var = scaler_Y.inverse_transform(np.sqrt(np.diag((1/beta) + Phi_test_scaled_reg @ (np.linalg.lstsq(Tau_lambda, Phi_test_scaled_reg.T)[0]))).reshape(-1, 1))

# viz.plotPostPred(G_test_prior_mean, D_TEST, THETA_TEST, X_train, cmap=parula_cmap, plot_title="Prior predictive mean")

# viz.plotPostPred(G_test_prior_var, D_TEST, THETA_TEST, X_train, cmap=parula_cmap, plot_title="Prior predictive std")

# Plot posterior predictive mean and standard deviation
viz.plotPostPred(G_test_post_mean_unscaled, D_TEST, THETA_TEST, X_train, cmap=parula_cmap, plot_title="Posterior predictive mean")

viz.plotPostPred(G_test_post_std_unscaled, D_TEST, THETA_TEST, X_train, cmap=parula_cmap, plot_title="Posterior predictive std")

# %% Utilities (analytic)

eig_analytic = utils.utility_analytical(Phi_test_scaled_reg, Tau_lambda, beta)

viz.plotUtils(eig_analytic, D_TEST, THETA_TEST, X_train, cmap=parula_cmap)


# %% Sequential Design

opt_x_idx = np.unravel_index(eig_analytic.argmax(), D_TEST.shape)
theta_opt = THETA_TEST[opt_x_idx]
d_opt = D_TEST[opt_x_idx]

# theta_opt, d_opt

# X_train_new = np.vstack([X_train, candidate.detach().numpy()])



# scaler_X = StandardScaler()
# scaler_Y = StandardScaler()

X_train_new = np.vstack([X_train, [d_opt, theta_opt]])
X_new_added = np.array([d_opt, theta_opt]).reshape(1, -1)

Phi_new = getPhiMat(theta_opt, d_opt)
if runNonLin:
    G_new = model_4(theta_opt, d_opt).reshape(-1, 1)
else:
    G_new = (Phi_new @ oracle_lambda()).reshape(-1, 1)

Phi_new_scaled = scaler_X.transform(Phi_new)
G_new_scaled = scaler_Y.transform(G_new)

Phi_test_scaled = scaler_X.transform(PhiMat_test)
G_test_scaled = scaler_Y.transform(G_test.reshape(-1, 1))

# Remove intercept column from Phi
Phi_new_scaled_reg = Phi_new_scaled[:, 1:]
Phi_test_scaled_reg = Phi_test_scaled[:, 1:]



# likelihood update will be with a single data point.
# posterior predictive samples are obtained for all test points.

mu_lambda_post_y_new, Tau_lambda_post_y_inv_new, G_test_post_mean_new, G_test_post_std_new = utils.compute_posterior_from_priors(mu_lambda_post_y, np.linalg.inv(Tau_lambda_post_y_inv), X_new_added, Phi_new_scaled_reg, G_new_scaled, X_test, Phi_test_scaled_reg, scaler_Y, beta)

# %%
n_samples = 1000
lambda_prior_samps = np.random.multivariate_normal(mu_lambda.flatten(), Tau_lambda, size=n_samples)

lambda_post_samps = np.random.multivariate_normal(mu_lambda_post_y_new.flatten(), np.linalg.inv(Tau_lambda_post_y_inv_new), size=n_samples)

# Posterior of parameters
fig, axs = plt.subplots(5, 2, figsize=(5, 10))
for i, ax in enumerate(axs.flatten()):
    if i != (Phi_train_scaled_reg.shape[1]):
        ax.hist(lambda_prior_samps[:, i], bins=20, color='red', alpha=0.5)
        ax.hist(lambda_post_samps[:, i], bins=20, color='blue', alpha=0.5)
        ax.set_title(f"$\lambda_{i + 1}$")

plt.tight_layout()


# %% Next Steps?

viz.plotPostPred(G_test_post_mean_new, D_TEST, THETA_TEST, X_train_new, cmap=parula_cmap, plot_title="Posterior predictive mean")

viz.plotPostPred(G_test_post_std_new, D_TEST, THETA_TEST, X_train_new, cmap=parula_cmap, plot_title="Posterior predictive std")

# %%
# %% Utilities (analytic)

eig_analytic_new = utils.utility_analytical(Phi_test_scaled_reg, np.linalg.inv(Tau_lambda_post_y_inv_new), beta)

viz.plotUtils(eig_analytic_new, D_TEST, THETA_TEST, X_train_new, cmap=parula_cmap)

opt_x_idx = np.unravel_index(eig_analytic_new.argmax(), D_TEST.shape)
theta_opt = THETA_TEST[opt_x_idx]
d_opt = D_TEST[opt_x_idx]

X_train_new2 = np.vstack([X_train_new, [d_opt, theta_opt]])
X_new_added2 = np.array([d_opt, theta_opt]).reshape(1, -1)

Phi_new2 = getPhiMat(theta_opt, d_opt)

if runNonLin:
    G_new2 = model_4(theta_opt, d_opt).reshape(-1, 1)
else:
    G_new2 = (Phi_new2 @ oracle_lambda()).reshape(-1, 1)

Phi_new_scaled2 = scaler_X.transform(Phi_new2)
G_new_scaled2 = scaler_Y.transform(G_new2)

# Remove intercept column from Phi
Phi_new_scaled_reg2 = Phi_new_scaled[:, 1:]

mu_lambda_post_y_new2, Tau_lambda_post_y_inv_new2, G_test_post_mean_new2, G_test_post_std_new2 = utils.compute_posterior_from_priors(mu_lambda_post_y_new, np.linalg.inv(Tau_lambda_post_y_inv_new), X_new_added2, Phi_new_scaled_reg2, G_new_scaled2, X_test, Phi_test_scaled_reg, scaler_Y, beta)
# %%
n_samples = 1000
lambda_prior_samps = np.random.multivariate_normal(mu_lambda.flatten(), Tau_lambda, size=n_samples)

lambda_post_samps = np.random.multivariate_normal(mu_lambda_post_y_new2.flatten(), np.linalg.inv(Tau_lambda_post_y_inv_new2), size=n_samples)

# Posterior of parameters
fig, axs = plt.subplots(5, 2, figsize=(5, 10))
for i, ax in enumerate(axs.flatten()):
    if i != (Phi_train_scaled_reg.shape[1]):
        ax.hist(lambda_prior_samps[:, i], bins=20, color='red', alpha=0.5)
        ax.hist(lambda_post_samps[:, i], bins=20, color='blue', alpha=0.5)
        ax.set_title(f"$\lambda_{i + 1}$")

plt.tight_layout()


# %%
viz.plotPostPred(G_test_post_mean_new2, D_TEST, THETA_TEST, X_train_new2, cmap=parula_cmap, plot_title="Posterior predictive mean")

viz.plotPostPred(G_test_post_std_new2, D_TEST, THETA_TEST, X_train_new2, cmap=parula_cmap, plot_title="Posterior predictive std")


# %%

eig_analytic_new2 = utils.utility_analytical(Phi_test_scaled_reg, np.linalg.inv(Tau_lambda_post_y_inv_new2), beta)

viz.plotUtils(eig_analytic_new2, D_TEST, THETA_TEST, X_train_new2, cmap=parula_cmap)

# %% Lengthscale tuning? Posterior correction?