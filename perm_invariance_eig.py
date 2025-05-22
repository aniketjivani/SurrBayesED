#!/usr/bin/env python
# coding: utf-8

# For OED in the batch design setting, in cases where the likelihood models remain unchanged for each design, computing EIG on a grid becomes expensive as the batch size grows. In such cases, if we resort to methods like Bayesian Optimization for interpolating between limited EIG evaluations, it may be a good idea to enforce permutation invariance for the Gaussian Process, in its kernel function. One simple way to do this is via Haar scattering [https://arxiv.org/pdf/1406.2390]:
# 
# $$(\alpha, \beta) \to (\alpha + \beta, |\alpha - \beta| )$$




get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
import emcee


def norm_logpdf(x, loc=0, scale=1):
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))
    
def uniform_logpdf(x, low=0, high=1):
    return np.log(uniform_pdf(x, low, high))


def uniform_pdf(x, low=0, high=1):
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(axis=1)


class OED(object):
    """
    A Bayesian optimal experimental design class.
    This class provides design of simulation-based experiments from a Bayesian 
    perspective, accommodates both batch and greedy design, and continuous 
    design space.
    This version only uses Bayesian optimization, we will provide more 
    optimization choices in the future (probably use gradient-free-optimizers
    library).

    Parameters
    ----------
    model_fun : function
        Forward model function G(theta, d). It will be abbreviated as m_f 
        inside this class.
        The forward model function should take following inputs:
            * theta, numpy.ndarray of size (n_sample or 1, n_param)
                Parameter samples.
            * d, numpy.ndarray of size (n_sample or 1, n_design)
                Designs.
        and the output is 
            * numpy.ndarray of size (n_sample, n_obs).
        When the first dimension of theta or d is 1, it should be augmented
        to align with the first dimension of other inputs (i.e., we reuse it for
        all samples).
        The model function should accommodate multiple stages or batches.
    n_param : int
        Dimension of parameter space.
    n_design : int
        Dimension of design space. n_design should be equal
        #design variables * #stages * #batches(repeats), and the model function
        should accommodate multiple stages or batches.
    n_obs : int
        Dimension of observation space.
    prior_rvs : function
        It takes number of samples as input, and output a numpy.ndarray of size
        (n_sample, n_param) which are prior samples.
    design_bounds : list, tuple or numpy.ndarray of size (n_design, 2)
        It includes the constraints of the design variable. In this version, we
        only allow to set hard limit constraints. In the future version, we may
        allow users to provide their own constraint function.
        The length of design_bounds should be n_design.
        k-th entry of design_bounds is a list, tuple or numpy.ndarray like 
        (lower_bound, upper_bound) for the limits of k-th design variable.
    noise_info : list, tuple or numpy.ndarray of size (n_obs, 3)
        It includes the statistics of additive Gaussian noise.
        The length of noise_info should be n_obs. k-th entry of noise_info is a 
        list, tuple or numpy.ndarray including
            * noise_loc : float or int
            * noise_base_scale : float or int
                It will be abbreviated as noise_b_s in this class.
            * noise_ratio_scale : float or int
                It will be abbreviated as noise_r_s in this class.
        The corresponding noise will follow a gaussian distribution with mean
        noise_loc, std (noise_base_scale + noise_ratio_scale * abs(G)).
    prior_logpdf : function, optional(default=None)
        This function is required if users want to use post_logpdf, post_pdf
        and post_rvs function. It provides logpdf of samples in the prior
        distribution. The input is
            * theta, numpy.ndarray of size (n_sample or 1, n_param)
        and the output is
            * numpy.ndarray of size(n_sample).
    reward_fun : function, optional(default=None)
        User-provided non-KL-divergence based reward function. 
        It will be abbreviated as nlkd_rw_f inside this 
        class.
        The reward function should take following inputs:
            * d : np.ndarray of size (n_design)
                The design variable.
            * y : np.ndarray of size (n_obs)
                The observation.
        and the output is 
            * A float which is the reward.
        Note that the information gain is computed within this class, and does
        not needed to be included in reward_fun.
        When reward_fun is None, we only consider information gain.
    random_state : int, optional(default=None)
        It is used as the random seed.   
    
    Methods
    -------
    post_logpdf(), post_pdf()
        Evaluate the posterior logpdf (pdf) of parameter samples.
    post_rvs()
        Generate samples from the posterior.
    exp_utility()
        Estimate the expected utility of a given design.
    oed()
        Find the optimal design that maximizes the expected utility using
        Bayesian optimization.


    Notes
    -----
    Please refer to https://arxiv.org/pdf/1108.4146.pdf for more details.
    """
    def __init__(self, model_fun, n_param, n_design, n_obs, 
                 prior_rvs, design_bounds, noise_info, 
                 prior_logpdf=None, reward_fun=None, random_state=None):
        NoneType = type(None)
        assert isinstance(random_state, (int, NoneType)), (
               "random_state should be an integer or None.")
        np.random.seed(random_state)
        self.random_state = random_state
        
        assert callable(model_fun), (
               "model_fun should be a function.")
        self.m_f = model_fun
        self.model_fun = self.m_f

        assert isinstance(n_param, int) and n_param > 0, (
               "n_param should be an integer greater than 0.")
        self.n_param = n_param
        assert isinstance(n_design, int) and n_design > 0, (
               "n_design should be an integer greater than 0.")
        self.n_design = n_design
        assert isinstance(n_obs, int) and n_obs > 0, (
               "n_obs should be an integer greater than 0.")
        self.n_obs = n_obs

        assert callable(prior_rvs), (
               "prior_rvs should be a function to generate prior samples.")
        self.prior_rvs = prior_rvs

        assert isinstance(design_bounds, (list, tuple, np.ndarray)), (
               "design_bounds should be a list, tuple or numpy.ndarray of " 
               "size (n_design, 2).")
        assert len(design_bounds) == n_design, (
               "Length of design_bounds should equal n_design.")
        for i in range(n_design):
            assert len(design_bounds[i]) == 2, (
                   "Each entry of prior_info is of size 2, including "
                   "lower bound and upper bound.")
            l_b, u_b = design_bounds[i]
            assert isinstance(l_b, (int, float)), (
                   "{}-th lower bound should be a number.".format(i))
            assert isinstance(u_b, (int, float)), (
                   "{}-th upper_bound should be a number.".format(i))
        # size (n_design, 2)
        self.design_bounds = np.array(design_bounds)

        assert isinstance(noise_info, (list, tuple, np.ndarray)), (
               "noise_info should be a list, tuple or numpy.ndarray of " 
               "size (n_obs, 3).")
        assert len(noise_info) == n_obs, (
               "Length of noise_info should equal n_obs.")
        for i in range(n_obs):
            assert len(noise_info[i]) == 3, (
                   "Each entry of noise_info is of size 3, including "
                   "noise_loc, noise_base_scale and noise_ratio_scale.")
            noise_loc, noise_b_s, noise_r_s = noise_info[i]
            assert isinstance(noise_loc, (int, float)), (
                   "{}-th noise_loc should be a number.".format(i))
            assert isinstance(noise_b_s, (int, float)), (
                   "{}-th noise_base_scale should be a number.".format(i))
            assert isinstance(noise_r_s, (int, float)), (
                   "{}-th noise_ratio_scale should be a number.".format(i))
            assert noise_b_s ** 2 + noise_r_s ** 2 > 0, (
                   "Either {}-th noise_base_scale or noise_ratio_scale "
                   "should be greater than 0.".format(i))
        # size (n_obs, 3)
        self.noise_info = np.array(noise_info)
        self.noise_loc = self.noise_info[:, 0]
        self.noise_b_s = self.noise_info[:, 1]
        self.noise_r_s = self.noise_info[:, 2]

        if prior_logpdf is not None:
            assert callable(prior_logpdf), (
                   "prior_logpdf should be a function.")
        self.prior_logpdf = prior_logpdf

        # Non-KL-divergence based reward function
        if reward_fun is None:
            self.nkld_rw_f = lambda *args, **kws: 0
        else:
            assert callable(reward_fun), (
                   "reward_fun should be a function.")
            self.nkld_rw_f = reward_fun

        self.optimizer = None
        self.thetas = None
        self.noises = None
        return

    def post_logpdf(self, thetas, 
                    d=None, y=None, include_prior=True):
        """
        A function to compute the log-probability of unnormalized posterior  
        after observing observations 'y' by conducting experiments under 
        designs 'd'.

        Parameters
        ----------
        thetas : numpy.ndarray of size (n_sample, n_param)
            The parameter samples whose log-posterior are required.
        d : numpy.ndarray of size (n_design), optional(default=None)
            Designs.
        y : numpy.ndarray of size (n_obs), optional(default=None)
            Observations.
        include_prior : bool, optional(default=True)
            Include the prior in the posterior or not. It not included, the
            posterior is just a multiplication of likelihoods.

        Returns
        -------
        A numpy.ndarray of size (n_sample) which are log-posteriors.
        """
        if include_prior:
            assert self.prior_logpdf is not None, (
                "prior_logpdf is required to estimate posterior probabilities, "
                "which is a function to evaluate prior log probabilities, "
                "with input of size (n_sample, n_param), and output of "
                "size (n_sample).")
        if d is None: d = []
        if y is None: y = []
        logpost = np.zeros(len(thetas))
        if len(d) > 0:
            G = self.m_f(thetas, d.reshape(1, -1))
            loglikeli = norm_logpdf(y.reshape(1, -1), 
                                    G + self.noise_loc,
                                    self.noise_b_s + self.noise_r_s * np.abs(G))
            logpost += loglikeli
        if include_prior:
            logpost += self.prior_logpdf(thetas)
        return logpost

    def post_pdf(self, *args, **kws):
        return np.exp(self.post_logpdf(*args, **kws))

    def post_rvs(self, n_sample, 
                 d=None, y=None):
        """
        A function to generate samples from the posterior distribution,
        after observing observations 'y' by conducting experiments under designs
        'd'.

        Parameters
        ----------
        n_sample : int
            Number of posterior samples we want to generate.
        d : numpy.ndarray of size (n_design), optional(default=None)
            Designs.
        y : numpy.ndarray of size (n_obs), optional(default=None)
            Observations.

        Returns
        -------
        A numpy.ndarray of size (n_sample, n_param).
        """
        log_prob = lambda x : self.post_logpdf(x.reshape(-1, self.n_param), 
                                               d=d,
                                               y=y)
        n_dim, n_walkers = self.n_param, 2 * self.n_param
        theta0 = self.prior_rvs(n_walkers)
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
        sampler.run_mcmc(theta0, 
                         int(n_sample // n_walkers * 1.2), 
                         progress=False)
        return sampler.get_chain().reshape(-1, self.n_param)[-n_sample:]

    def exp_utility(self, d, thetas, noises=None):
        """
        A function to estimate the expected utility 
        U(d) = int_y p(y|d) int_theta p(theta|d,y) *
                                      ln{p(theta|d,y) / p(theta)} dtheta dy
        on given design 'd' with given prior sample "thetas". 
        User provided non-KL-divergence based reward will also be considered.

        Parameters
        ----------
        d : numpy.ndarray of size (n_design)
            Designs.
        thetas : numpy.ndarray of size (n_sample, n_param)
            Prior samples.
        noises : numpy.ndarray of size (n_sample, n_obs), optional(default=None)
            Noises sampled from standard normal distribution. Using same noises
            on different designs can make the results more consistent and 
            smoothen the utility plots.

        Returns
        -------
        A float which is the expected utility.
        """
        if noises is not None:
            assert len(thetas) == len(noises) and noises.shape[1] == self.n_obs 
        n_sample = len(thetas)
        Gs = self.m_f(thetas, d.reshape(1, -1))
        if noises is None:
            noises = np.random.normal(size=Gs.shape)
        ys = (Gs +
              self.noise_loc +
              (self.noise_b_s + self.noise_r_s * np.abs(Gs)) * noises)
        loglikelis = norm_logpdf(ys, Gs + self.noise_loc,
                                 self.noise_b_s + self.noise_r_s * np.abs(Gs))
        evids = np.zeros(n_sample)
        nkld_rewards = np.zeros(n_sample)
        for i in range(n_sample):
            inner_likelis = norm_pdf(ys[i:i + 1],
                                     Gs + self.noise_loc,
                                     self.noise_b_s + 
                                     self.noise_r_s * np.abs(Gs))
            evids[i] = np.mean(inner_likelis)
            nkld_rewards[i] = self.nkld_rw_f(d, ys[i])
        return (loglikelis - np.log(evids) + nkld_rewards).mean()

    def oed(self, n_sample=1000, n_init=10, n_iter=90,
            restart=False):
        """
        A function to find the optimal design with maximum expected utility.

        Parameters
        ----------
        n_sample : int, optional(default=1000)
            Number of samples we will use to estimate the expected utility.
        n_init : int, optional(default=10)
            Number of initial exploration of Bayesian optimization.
        n_iter : int, optional(default=90)
            Number of iterations of Bayesian optimization after initial search.
        restart : bool, optional(default=False)
            If True, will reinitialize the optimizer.

        Returns
        -------
        A numpy.ndarray of size (n_design) which is the optimal design.
        A float which is the maximum expected utility.
        """
        print("Optimizing")
        pbounds = {}
        for i in range(self.n_design):
            pbounds['d' + str(i + 1)] = self.design_bounds[i, :]
        if self.thetas is None:
            self.thetas = self.prior_rvs(n_sample)
            self.noises = np.random.normal(size=(n_sample, self.n_obs))
        elif len(self.thetas) < n_sample:
            self.thetas = np.r_[self.thetas, self.prior_rvs(n_sample - 
                                                            len(self.thetas))]
            self.noises = np.r_[self.noises, 
                                np.random.normal(size=(n_sample - 
                                                       len(self.noises),
                                                       self.n_obs))]
        else:
            self.thetas = self.thetas[:n_sample]
            self.noises = self.noises[:n_sample]
        objective = lambda **kws: self.exp_utility(np.array(list(kws.values())),
                                                   self.thetas,
                                                   self.noises)
        if self.optimizer is None or restart:
            self.optimizer = BayesianOptimization(
                f=objective, pbounds=pbounds,
                random_state=self.random_state)
        self.optimizer.maximize(init_points=n_init,
                                n_iter=n_iter,
                                acq='ucb')
        self.d_opt = np.array(list(self.optimizer.max['params'].values()))
        self.U_opt = self.optimizer.max['target']
        self.opt_res = self.optimizer.res
        return self.d_opt, self.U_opt

def model_2(theta, d):
    d1 = d[:, 0:1]
    d2 = d[:, 1:2]
    return np.c_[theta ** 3 * d1 ** 2 + theta * np.exp(-np.abs(0.2 - d1)),
                 theta ** 3 * d2 ** 2 + theta * np.exp(-np.abs(0.2 - d2))]


n_param = 1 # Number of parameters.
n_design = 2 # Number of design variables.
n_obs = 2 # Number of observations.

low = 0
high = 1
prior_rvs = lambda n_sample: np.random.uniform(low=low,
                                               high=high,
                                               size=(n_sample, n_param))
prior_logpdf = lambda theta: uniform_logpdf(theta,
                                            low=low,
                                            high=high)

design_bounds = [(0, 1), (0, 1)]
noise_loc = 0
noise_base_scale = 0.01
noise_ratio_scale = 0
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale),
              (noise_loc, noise_base_scale, noise_ratio_scale)]
random_state = 2021
oed_2 = OED(model_fun=model_2,
            n_param=n_param,
            n_design=n_design,
            n_obs=n_obs,
            prior_rvs=prior_rvs,
            design_bounds=design_bounds,
            noise_info=noise_info,
            prior_logpdf=prior_logpdf,
            reward_fun=None,
            random_state=random_state)

ds = np.linspace(design_bounds[0][0], design_bounds[0][1], 11)
dX, dY = np.meshgrid(ds, ds)
Us = np.zeros_like(dX)
thetas = prior_rvs(1000)
noises = np.random.normal(size=(1000, n_obs))
for i in range(11):
    for j in range(11):
        Us[i][j] = oed_2.exp_utility(np.array([dX[i, j], dY[i, j]]), 
                                     thetas, noises)


# In[243]:


# nrand=10
# grid_random = np.linspace(design_bounds[0][0], design_bounds[0][1], nrand)
# dX_random, dY_random = np.meshgrid(grid_random, grid_random)
# ds_random = np.stack((dX_random.flatten(), dY_random.flatten())).T
# ds_random = np.random.uniform(0, 1, size=(nrand, 2))
# Us_random = np.zeros(nrand**2)
# for i in range(nrand**2):
#     Us_random[i] = oed_2.exp_utility(np.array([ds_random[i, 0], ds_random[i, 1]]), thetas,noises)

plt.figure(figsize=(6, 4))
plt.contourf(dX, dY, Us, vmin=Us.min(), vmax=Us.max())
# sc = plt.scatter(ds_random[:, 0], ds_random[:, 1], c=Us_random, 
#                  edgecolors='k',
#                  linewidths=0.8,
#                  vmin=Us.min(), vmax=Us.max())
plt.xlabel('$d_1$', fontsize=20)
plt.ylabel('$d_2$', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')


dx_dy_flat = np.stack((dX.flatten(), dY.flatten())).T


from botorch.utils import standardize
from botorch.utils.transforms import normalize, unnormalize


# In[269]:


import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
import copy


# In[270]:


torch.manual_seed(0)
n_train = 50
x1 = torch.rand(n_train)
x2 = torch.rand(n_train)
mask = x1 > x2
x1, x2 = x1[mask], x2[mask]
train_x_raw = torch.tensor(torch.stack([x1, x2], dim=-1))
n_u = train_x_raw.shape[0]

Us_random = np.zeros(n_u)
for i in range(n_u):
    Us_random[i] = oed_2.exp_utility(np.array([train_x_raw[i, 0].item(), train_x_raw[i, 1].item()]), thetas,noises)



# train_x = torch.tensor(dx_dy_flat)
# train_x = torch.tensor(ds_random)
# train_y_raw = torch.tensor(Us_flat).reshape(-1, 1)
# train_y_raw = torch.tensor(Us_random).reshape(-1, 1)
# train_y_ts = torch.log(train_y_raw)
# train_y_ts = copy.deepcopy(train_y_raw)
# train_y_std = standardize(train_y_ts)
# # train_y_std += 0.05 * torch.randn_like(train_y_std)




# # In[271]:


# train_y_ts.mean(), train_y_ts.std()
train_x = train_x_raw.double()
train_y_raw = torch.tensor(Us_random).reshape(-1, 1)
train_y_log = torch.log(train_y_raw)
train_y_std = standardize(train_y_log)

class SymmetricMaternKernel(MaternKernel):
    """
    A variant of the Matern kernel that enforces symmetry by computing:
        k_sym(x1, x2) = k_Matern(x1, x2) + k_Matern(-x1, x2)
    This modification ensures that the kernel remains invariant to sign flips in x1.
    Args:
        nu (float): The smoothness parameter of the Matern kernel (0.5, 1.5, or 2.5).
        kwargs: Additional arguments for the base kernel.
    """
    def __init__(self, nu= 2.5, **kwargs):
        super().__init__(nu=nu, **kwargs)
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the symmetric kernel function.
        """
        matern_value = super().forward(x1, x2, diag=diag, **params)
        matern_value_symmetric = super().forward(-x1, x2, diag=diag, **params)
        return matern_value + matern_value_symmetric

class SymmetricMaternKernel2(MaternKernel):
    def forward(self, x1, x2, diag=False, **params):
        k1 = super().forward(x1, x2, diag=diag, **params)
        k2 = super().forward(x1.flip(-1), x2, diag=diag, **params)
        return 0.5 * (k1 + k2)
    
import torch
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.lazy import NonLazyTensor
from torch.nn.functional import pairwise_distance
# class SetMeanKernel(Kernel):
#     """Symmetric kernel on sets using averaged pairwise RBF."""
#     def __init__(self, base_kernel=None, **kwargs):
#         super().__init__(**kwargs)
#         self.base_kernel = base_kernel or RBFKernel()

#     def forward(self, x1, x2, **params):
#         # x1, x2 shape: (batch_size, set_size, d)
#         bsz, n1, d = x1.shape
#         _, n2, _ = x2.shape

#         # Expand for broadcasting
#         x1_exp = x1.unsqueeze(2)  # (bsz, n1, 1, d)
#         x2_exp = x2.unsqueeze(1)  # (bsz, 1, n2, d)

#         # Compute pairwise base kernel
#         pairwise_k = self.base_kernel(
#             x1_exp.reshape(-1, d),
#             x2_exp.reshape(-1, d)
#         ).reshape(bsz, n1, n2)

#         # Mean over all pairwise entries to get symmetric kernel value
#         return pairwise_k.mean(dim=(-2, -1))  # shape: (bsz,)

class SetMeanKernel(Kernel):
    """
    A kernel that computes mean embeddings of input sets.
    Assumes inputs of shape (batch_size, set_size, dim)
    """
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        # x1, x2: (B1, N, D), (B2, N, D)
        # Take mean over sets
        x1_mean = x1.mean(dim=-2)  # (B1, D)
        x2_mean = x2.mean(dim=-2)  # (B2, D)

        # Compute squared Euclidean distance
        x1_ = x1_mean.unsqueeze(1)  # (B1, 1, D)
        x2_ = x2_mean.unsqueeze(0)  # (1, B2, D)
        dist_sq = ((x1_ - x2_)**2).sum(dim=-1)  # (B1, B2)

        # RBF kernel
        scaled_dist_sq = dist_sq / (self.lengthscale**2)
        K = torch.exp(-0.5 * scaled_dist_sq)  # (B1, B2)

        return K

base_kernel = MaternKernel(nu=2.5)
# set_mean_kernel = SetMeanKernel(base_kernel=base_kernel)
# set_mean_kernel = SetMeanKernel()
sym_kernel = SymmetricMaternKernel2(nu=2.5)
# gp = SingleTaskGP(train_x, train_y_std, covar_module=sym_kernel)
gp = SingleTaskGP(train_x, train_y_std, covar_module=sym_kernel)

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)


# In[272]:


# def phi(X):
#     """Transform function phi(x) = (x1 + x2/2, |x1 - x2| / 2)"""
#     x1, x2 = X[:, 0], X[:, 1]
#     return torch.stack((x1 + x2 / 2, torch.abs(x1 - x2) / 2), dim=-1)


# In[282]:


# bounds = torch.stack([torch.zeros(2), torch.ones(2)]).to(torch.double)


# In[283]:


# train_x_transformed = phi(train_x)
# phi_bounds = torch.stack([train_x_transformed.min() * torch.ones(2), 
#                           train_x_transformed.max() * torch.ones(2)]).to(torch.double)
# train_x_transformed_normalized = botorch.utils.transforms.normalize(train_x_transformed, bounds=phi_bounds)


x1 = x2 = torch.linspace(0, 1, 50)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)

# with torch.no_grad():
#     post = gp.posterior(grid)
#     mean = post.mean.reshape(50, 50)
#     std = post.variance.sqrt().reshape(50, 50)

with torch.no_grad():
    post = gp.posterior(grid)
    mean_log = post.mean.reshape(50, 50)
    std_log = post.variance.sqrt().reshape(50, 50)
    mean_y = torch.exp(mean_log)  # convert back to y-space
    std_y = torch.exp(std_log)
    
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(X1.numpy(), X2.numpy(), mean_y.numpy(), levels=20)
plt.plot([0.0, 1.0], [0.0, 1.0], "--", linewidth=2, c='w')
plt.scatter(train_x[:, 0], train_x[:, 1], c='r', edgecolors='k')
plt.title("Posterior Mean (log scale)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.contourf(X1.numpy(), X2.numpy(), std_y.numpy(), levels=20)
plt.scatter(train_x[:, 0], train_x[:, 1], c='r', edgecolors='k')
plt.title("Posterior Std Dev")
plt.colorbar()
plt.tight_layout()
plt.show()
    

gp2 = SingleTaskGP(train_x, train_y_std)
mll2 = ExactMarginalLogLikelihood(gp2.likelihood, gp2)
fit_gpytorch_mll(mll2)

with torch.no_grad():
    post2 = gp2.posterior(grid)
    mean_log2 = post2.mean.reshape(50, 50)
    std_log2 = post2.variance.sqrt().reshape(50, 50)
    mean_y2 = torch.exp(mean_log2)  # convert back to y-space
    std_y2 = torch.exp(std_log2)

    
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(X1.numpy(), X2.numpy(), mean_y2.numpy(), levels=20)
plt.plot([0.0, 1.0], [0.0, 1.0], "--", linewidth=2, c='w')
plt.scatter(train_x[:, 0], train_x[:, 1], c='r', edgecolors='k')
plt.title("Posterior Mean (log scale)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.contourf(X1.numpy(), X2.numpy(), std_y2.numpy(), levels=20)
plt.scatter(train_x[:, 0], train_x[:, 1], c='r', edgecolors='k')
plt.title("Posterior Std Dev")
plt.colorbar()
plt.tight_layout()
plt.show()
