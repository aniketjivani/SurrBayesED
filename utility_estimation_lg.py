import numpy as np
import torch
import botorch
import emcee
from sklearn.utils.extmath import fast_logdet

def norm_logpdf(x, loc=0, scale=1):
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))


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
        


    def logprior(self, lambdas):
        """
        self.alpha: prior inverse variance / precision of weights lambda.
        lambdas: size n_post_samples x p
        """
        return norm_logpdf(lambdas, loc=0, scale=1/np.sqrt(self.alpha))
        

    def post_logpdf_mcmc(self, include_prior=True):
        """
        Logprobability of unnormalized posterior after observing data y.
        """
        pass


    def post_rvs_mcmc(self, n_post_samples):
        """
        Generate n_post_samples from the posterior distribution.
        """
        pass

    def post_logpdf_closed_form(self, include_prior=True):
        """
        Logprobability for (Gaussian) posterior after observing data 
        """
        pass

    def post_rvs_closed_form(self, n_post_samples):
        """
        Generate n_post_samples from the posterior distribution.
        """
        pass

    def utility_dnmc():
        pass


    def utility_analytical(self):
        """
        Calculate Fisher-information matrix and maximize EIG i.e. getting a D - optimal design.
        F = PhiMat^T * \Tau_G | \lambda^-1 * PhiMat
        If PhiMat is n x p, then F is p x p.
        \Tau_G | \lambda  is cov of Gaussian RV noise. If noise distribution has precision \beta, then \Tau_G | \lambda = (1/\beta)I_n and \Tau_G | \lambda^-1 = \beta I_n.
        EIG = (1/2)(log det \Tau_G (\theta, d) - log det \Tau_G|\lambda (\theta, d))

        where \Tau_G(\theta, d) = marginal prior predictive covariance = PhiMat * \Tau_\lambda * PhiMat^T + \Tau_G | \lambda 
        and \Tau_\lambda = (1 / \alpha)I_p
        """
        
        n, p = self.PhiMat.shape
        Tau_G_cond_lambda = (1 / self.beta) * np.eye(n)
        Tau_G_cond_lambda_inv = self.beta * np.eye(n)
        Tau_lambda = (1 / self.alpha) * np.eye(p)
        Tau_G = self.PhiMat @ Tau_lambda @ self.PhiMat.T + Tau_G_cond_lambda

        eig_analytic = (1/2) * (self.log_det_mat(Tau_G) - self.log_det_mat(Tau_G_cond_lambda))

        return eig_analytic
        

    @staticmethod
    def log_det_mat(mat):
        """
        Calculate the log determinant of a matrix.
        """
        return fast_logdet(mat)
    
    def log_det_mat_numpy(mat):
        """
        Calculate the log determinant of a matrix.
        """
        return np.linalg.slogdet(mat)[1]

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