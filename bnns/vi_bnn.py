import warnings
warnings.filterwarnings('ignore')

# Framework written in Pytorch and design based on ideas from:
# 1. Audrey Olivier's TF-implementation of BNNs
# 2. IntelLabs' Bayesian Torch - https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/layers/base_variational_layer.py - we will modify our variational layer to include flipout from the same repo.
# 3. https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb

# . We try to keep training as close as possible to training a vanilla NN in Pytorch, with all bells and whistles included. This means writing new layers for `BayesLinear` and (future) `BayesConv2D`.

import torch
import torch.nn as nn
import torch.distributions as distributions
from itertools import repeat
import collections
from scipy.special import logsumexp

# We will probably not implement a Base Regressor class.
class Regressor:
    pass

class BaseVariationalLayer(nn.Module):
    """
    prior_means: mean value of Gaussian prior, float or list (len 2 * n_uq_layers) of floats

    prior_stds: std value of Gaussian prior, float or list (len 2 * n_uq_layers) of floats

    random_seed: int or None, random seed generator for reproducibility
    """


    def __init__(self, 
                 hidden_units, 
                 input_dim=1, 
                 output_dim=1, 
                 prior_means=0., 
                 prior_stds=1., 
                 random_seed=None):
        super().__init__()
        self._dnn_to_bnn_flag=False
        self.random_seed = random_seed
        
        # Check if this is best practice.
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        self.prior_means = prior_means
        self.prior_stds = prior_stds


        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_uq_layers = len(self.hidden_units) + 1

        self.activation = nn.ReLU()

        # nn sequential
        self.layers = nn.Sequential(nn.Linear(self.input_dim, self.hidden_units),
                                    self.activation,
                                    nn.Linear(self.hidden_units, self.output_dim))
        


    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag
    
    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value


    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
                * mu_q: torch.Tensor -> mu parameter of distribution Q
                * sigma_q: torch.Tensor -> sigma parameter of distribution Q
                * mu_p: float -> mu parameter of distribution P
                * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
        
        return kl # Check.

    def kl_gauss_gauss(mean_1, std_1, mean_2, std_2, axis_sum=None):
        """
        KL divergence between two multivariation Gaussians
        """
        pass

    def kl_gauss_mixgauss(mean_1, std_1, mean_2, std_2, compmix_2):
        """
        KL divergence between a Gaussian and a mixture of Gaussians
        """
        pass

    def sample_aleatoric(self, size):
        pass


    def generate_seed_layers(self, nfigures=4, previous_seeds=()):
        pass

    # originally def compute_predictions
    def forward(self, x):
        pass

    def log_prior_pdf(self, sum_over_ns=True):
        pass

    def sample_weights_from_prior(self, ns, random_seed=None):
        pass

    # originally def predict_uq_from prior
    def prior_predict(self, x, ns):
        pass


    def predict_uq_from_samples(self, x, return_std=True, return_MC=10, return_percentiles=(2.5, 97.5), aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        pass

    def neg_log_like(y_true, y_pred, do_sum=True, axis_sum=None):
        pass

    def scaled_post_pdf(self, xn, yn):
        pass
        # return log_prior + log_likelihood

    def log_approx_post_pdf(self, sum_over_ns=True):
        """
        log(q)
        """
        pass

    def sample_weights_from_post_q(self, 
                                   ns, random_seed=None,
                                   evaluate_log_pdf=False,
                                   sum_over_ns=False,
                                   truncated=False):
        pass

    def predict_y_uq(self, x, ns, return_std=True, return_MC=10, return_percentiles=(2.5, 97.5), aleatoric_in_std_perc=True, aleatoric_in_MC=False):
        pass

    

    def predict_y_deterministic(self, x):
        """
        Use mean of posterior weights learnt to predict a deterministic output.
        """

        pass

    def post_predict(self, x, y, ns=10000):
        """
        Posterior predictive PDF
        """
        pass

    @staticmethod
    def _sigma(rho):
        """
        Calculate sigma = log(1 + exp(rho))
        """
        return torch.log1p(torch.exp(rho))
    
    @staticmethod
    def _rho(sigma):
        """
        Calculate rho = log(exp(sigma) - 1)
        """
        return torch.log(torch.exp(sigma) - 1)

    

# class VI_BNN(Regressor):
#     pass


class BayesByBackprop(VI_BNN):
    pass

class BBLowRank(VI_BNN):
    pass


class BBWithFlipout(VI_BNN):
    pass
