# %%

import numpy as np
import matplotlib.pyplot as plt
import emcee
from oed_wanggang import *
# logpdf of independent normal distribution.
# x is of size (n_sample, n_param).
# loc and scale are int or numpy.ndarray of size n_param.
# output is of size n_sample.


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
        return theta * d + 2 * np.sin(theta * d) + 3 * np.exp(-theta * (d - 5)**2) + 5 * np.exp(-theta * (d - 15)**2)


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
    return theta * d + 2 * np.sin(theta * d) + 3 * np.exp(-theta * (d - 5)**2) * np.sin(2 * d) + 5 * np.exp(-theta * (d - 15)**2)

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
ds = np.linspace(design_bounds[0][0], design_bounds[0][1], 21)
Us = []
thetas = prior_rvs(1000)
noises = np.random.normal(size=(1000, n_obs))

for d in ds:
    Us.append(oed_1.exp_utility(d, thetas, noises))

plt.figure(figsize=(6, 4))
plt.plot(ds, Us)
plt.xlabel('d', fontsize=20)
plt.ylabel('U(d)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
plt.show()

# %%

ds2 = np.linspace(-4, 4, 51)
Us2 = []
thetas2 = prior_rvs(1000)
noises2 = np.random.normal(size=(1000, n_obs))

for d in ds2:
    Us2.append(oed_2.exp_utility(d, thetas2, noises2))

plt.figure(figsize=(6, 4))
plt.plot(ds2, Us2)
plt.xlabel('d', fontsize=20)
plt.ylabel('U(d)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
plt.show()


# %%

ds4 = np.linspace(-4, 4, 51)
Us4 = []
thetas4 = prior_rvs(1000)
noises4 = np.random.normal(size=(1000, n_obs))

for d in ds4:
    Us4.append(oed_4.exp_utility(d, thetas4, noises4))

plt.figure(figsize=(6, 4))
plt.plot(ds4, Us4)
plt.xlabel('d', fontsize=20)
plt.ylabel('U(d)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
plt.show()

# %% Plotting model outputs

yy_reg = np.zeros((1000, 51))
for i, d in enumerate(ds2):
    yy_reg[:, i] = model_3(thetas2, d).flatten()

yy_reg_mean = np.mean(yy_reg, axis=0)
yy_reg_std = np.std(yy_reg, axis=0)

plt.plot(ds2, yy_reg_mean)
plt.plot(ds2, yy_reg_mean - yy_reg_std)
plt.plot(ds2, yy_reg_mean + yy_reg_std)


# %% Plotting model outputs

yy_reg = np.zeros((1000, 51))
for i, d in enumerate(ds4):
    yy_reg[:, i] = model_4(thetas4, d).flatten()

yy_reg_mean = np.mean(yy_reg, axis=0)
yy_reg_std = np.std(yy_reg, axis=0)

plt.plot(ds4, yy_reg_mean)
plt.plot(ds4, yy_reg_mean - yy_reg_std)
plt.plot(ds4, yy_reg_mean + yy_reg_std)

# %% we will use model 3 OR model 4 as our test example for surrogate OED .

