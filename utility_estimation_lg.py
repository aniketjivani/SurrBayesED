import numpy as np
import torch
import botorch
import emcee

class OEDLG(object):
    """
    A Bayesian optimal experimental design class. This class aims to provide numerical and analytical calculations of utility as well as batch design for linear Gaussian case. Optimization for U_KL is performed using Bayesian Optimization via BoTorch.
    """

    def __init__(self, PhiMat, G, beta, alpha, random_state=None, q_acq = 1):
        """
        PhiMat: design matrix
        G: response variable from high-fidelity model
        beta: inverse variance / precision of Gaussian distributed noise.
        alpha: prior inverse variance / precision of weights lambda.
        random_state: seed for random number generator.
        q_acq: number of points to select during optimization of acquisition function.
        """
        self.PhiMat = PhiMat
        self.G = G
        self.beta = beta
        self.alpha = alpha

        NoneType = type(None)
        assert isinstance(random_state, (int, NoneType)), (
               "random_state should be an integer or None.")
        np.random.seed(random_state)
        self.random_state = random_state

    def info(self):
        """
        Returns a dictionary containing relevant class attributes.
        Private attributes (starting with '_') are excluded.
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

    def ppinfo(self):
        """
        Pretty prints the info dictionary. Example usage: 
        oed_obj = OEDLG(PhiMat, G, beta, alpha, random_state)
        oed_obj.ppinfo()
        """
        info_dict = self.info()
        print("Person Information:")
        for key, value in info_dict.items():
            print(f"  {key}: {value}")


    def loglikelihood(self):
        """
        self.PhiMat: design matrix
        self.G: response variable
        self.beta: inverse variance / precision of Gaussian distributed noise.
        """
        pass


    def logprior(self):
        """
        self.alpha: prior inverse variance / precision of weights lambda.
        i.e. 
        """
        pass

    def post_logpdf(self, include_prior=True):
        """
        Logprobability of unnormalized posterior after observing data y.
        """
        pass


    def post_rvs(self, n_post_samples):
        """
        Generate n_post_samples from the posterior distribution.
        """
        pass

    def utility_dnmc():
        pass


    def utility_analytical():
        pass


    def optimize_acqf_and_get_observation(self, acq_func):
        """
        self.q_acq: number of points to select during optimization of acquisition function.
        """
        NUM_RESTARTS = 10
        RAW_SAMPLES = 512

        print("Optimizing the acquisition function")
        candidates, _  = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.q_acq,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        new_theta_d = candidates.detach().numpy()
        obj_analytic = self.utility_analytical(new_theta_d).unsqueeze(-1)
        obj_dnmc = self.utility_dnmc(new_theta_d).unsqueeze(-1)

        return new_theta_d, obj_analytic, obj_dnmc