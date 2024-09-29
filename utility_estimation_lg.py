import numpy as np
import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
import emcee
from sklearn.utils.extmath import fast_logdet

# logpdf of independent normal distribution.
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

    def __init__(self, PhiMat, G, beta, alpha, 
                 random_state=None, 
                 q_acq = 1, 
                 n_param=10, 
                 n_design=2, 
                #  prior_rvs=None, 
                #  prior_logpdf=None,
                ):
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
        
        # self.prior_rvs = prior_rvs
        # self.prior_logpdf = prior_logpdf

        NoneType = type(None)
        assert isinstance(random_state, (int, NoneType)), (
               "random_state should be an integer or None.")
        np.random.seed(random_state)
        self.random_state = random_state
        self.n_param = n_param
        self.n_design = n_design

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
        print("OED object Information:")
        for key, value in info_dict.items():
            print(f"  {key}: {value}")


    def ridge_regression(self, PhiMat, G):
        """
        Perform ridge regression to estimate the weights lambda.

        lambda = \beta (\beta PhiMat^T PhiMat + \alpha I)^{-1} PhiMat^T G
        """
        beta = self.beta
        alpha = self.alpha

        # allow supplying PhiMat and G as arguments.
        # PhiMat = self.PhiMat
        # G = self.G

        N, p = PhiMat.shape

        assert N == len(G), "Number of rows in PhiMat should be equal to the length of G."

        lambda_est = np.linalg.solve((beta * (PhiMat.T @ PhiMat) + alpha * np.eye(p)), beta * (PhiMat.T @ G))

        return lambda_est

    def loglikelihood(self, PhiMat, G):
        """
        self.PhiMat: design matrix
        self.G: response variable
        self.beta: inverse variance / precision of Gaussian distributed noise.
        """
        beta = self.beta
        alpha = self.alpha
        lambda_est = self.ridge_regression(PhiMat, G)

        N, p = PhiMat.shape

        lstsq_estimate = (PhiMat @ lambda_est).reshape(-1, 1)
        yvals = G.reshape(-1, 1)

        # log_lik = (N/2) * np.log(beta) - (N/2) * np.log(2 * np.pi) - (beta/2) * np.linalg.norm(yvals - (PhiMat @ lambda_est))**2

        log_lik = norm_logpdf(yvals, loc=lstsq_estimate, scale=1/np.sqrt(beta))

        return log_lik
        

    def logprior(self, lambdas):
        """
        self.alpha: prior inverse variance / precision of weights lambda.
        lambdas: size n_post_samples x p
        """
        return norm_logpdf(lambdas, loc=0, scale=1/np.sqrt(self.alpha))
        
    def sample_prior(self, n_prior_samples):
        """
        Generate n_prior_samples from the prior distribution.
        """
        p = self.n_param
        prior_samples = np.random.normal(0, 1/np.sqrt(self.alpha), size=(n_prior_samples, p))
        return prior_samples

    def post_logpdf_mcmc(self, lambdas, PhiMat, G, include_prior=True):
        """
        Logprobability of unnormalized posterior after observing data y (following Bayes rule)
        include_prior : bool, optional(default=True)
        Include the prior in the posterior or not. It not included, the
        posterior is just a multiplication of likelihoods.

        Returns:
        A numpy.ndarray of size n_samples which are log-posteriors.
        """
        n_samples, n_params = lambdas.shape

        lambda_est = self.ridge_regression(PhiMat, G)
        lstsq_estimate = (PhiMat @ lambda_est).reshape(-1, 1)
        yvals = G.reshape(-1, 1)


        logpost = np.zeros(n_samples)

        loglikelihoods = norm_logpdf(yvals, loc=lstsq_estimate, scale=1/np.sqrt(self.beta))
        logpost += loglikelihoods

        if include_prior:
            # logprior = self.prior_logpdf(lambdas)
            logprior = self.logprior(lambdas)
            logpost += logprior

        return logpost

    def post_rvs_mcmc(self, n_post_samples, PhiMat, G):
        """
        Generate n_post_samples from the posterior distribution.
        """
        log_prob = lambda x: self.post_logpdf_mcmc(x.reshape(-1, self.n_param), PhiMat, G, include_prior=True)

        n_dim, n_walkers = self.n_param, 2 * self.n_param

        theta0 = self.prior_rvs(n_walkers)

        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
        sampler.run_mcmc(theta0,
                         int(n_post_samples // n_walkers * 1.2),
                         progress=True)
        
        return sampler.get_chain().reshape(-1, self.n_param)[-n_post_samples:]



    def post_logpdf_closed_form(self, lambdas, PhiMat, G, include_prior=True):
        """
        Logprobability for (Gaussian) posterior after observing data y (following Bayes rule, closed form for linear Gaussian model)
        """
        pass
        
        

    def post_rvs_closed_form(self, n_post_samples, PhiMat, G):
        """
        Generate n_post_samples from the posterior distribution.
        """

        beta = self.beta
        alpha = self.alpha

        lambda_est = self.ridge_regression(PhiMat, G)


        N, p = PhiMat.shape

        S_N_inv = alpha * np.eye(p) + beta * PhiMat.T @ PhiMat
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * np.dot(S_N, np.dot(PhiMat.T, G))

        post_samples = np.random.multivariate_normal(m_N, S_N, size=n_post_samples)

        return post_samples


    def posterior_predictive(self, PhiMat, G, PhiMatPost):
        """
        Generate n_post_samples from the posterior predictive distribution.
        PhiMat and G typically will be the matrices used to build the surrogate model.
        """
        beta = self.beta
        alpha = self.alpha

        # lambda_est = self.ridge_regression(PhiMat, G)

        N, p = PhiMat.shape

        n_post_samples = PhiMatPost.shape[0]

        assert p == PhiMatPost.shape[1]

        S_N_inv = alpha * np.eye(p) + beta * PhiMat.T @ PhiMat
        S_N = np.linalg.inv(S_N_inv)

        m_N = beta * np.dot(S_N, np.dot(PhiMat.T, G))

        predictive_mean = np.dot(PhiMatPost, m_N)
        predictive_var = (1/beta) + np.sum(np.dot(PhiMatPost, S_N) @ PhiMatPost, axis=1)

        return predictive_mean, predictive_var        

    def utility_dnmc(self, lambdas, PhiMat, noises, nIn=5e2):
        """
        NMC estimator.
        U(\theta, d) = (1/N_out)sum_i [log p(G_i | \theta, \lambda, d) - (1/N_in)sum_j log p(G_i | \theta, d, \lambda_i,j)]

        Can optionally supply same set of noises to be used on all designs.
        """
        N, p = PhiMat.shape

        beta = self.beta
        alpha = self.alpha

        nOut = lambdas.shape[0]
        nIn = min(int(nIn), nOut) 

        evids = np.zeros(nOut)

        nD = PhiMat.shape[0]

        utilities_dnmc = np.zeros(nD)
        if noises is None:
            noises = np.random.normal(0, 1/np.sqrt(beta), size=(nOut, 1))
        for dIdx in range(nD):
            print("Evaluating estimator at design index: ", dIdx)
            yvals = np.zeros((nOut, 1))

            loglikelis_out = np.zeros((nOut, 1))
            for i in range(nOut):
                lambda_sample = lambdas[[i], :]
                yvals[i, 0] = (PhiMat[[dIdx], :] @ lambda_sample.T)[0][0] + noises[i, 0]

                loglikelis_out[i, 0] = norm_logpdf(yvals[i:(i + 1)], loc=(PhiMat[[dIdx], :] @ lambda_sample.T)[0][0], scale=1/np.sqrt(beta))

            for i in range(nOut):
                for j in range(nIn):
                    inner_likelis = np.exp(norm_logpdf(yvals[i:(i+1)], loc=(PhiMat[[dIdx], :] @ lambdas[[j], :].T)[0][0], scale=1/np.sqrt(beta)))[0]
                    evids[i] += inner_likelis
                evids[i] = evids[i] / nIn

            utility_dnmc = (loglikelis_out - np.log(evids)).mean()
            utilities_dnmc[dIdx] = utility_dnmc

        return utilities_dnmc
    

    def utility_dnmc_fast(self, lambdas, PhiMat, noises, nIn=5e2):
        """
        NMC estimator, but faster version because we are reusing outer loop samples for computing inner evidence term.
        U(\theta, d) = (1/N_out)sum_i [log p(G_i | \theta, \lambda, d) - (1/N_in)sum_j log p(G_i | \theta, d, \lambda_i,j)]

        Can optionally supply same set of noises to be used on all designs (its supposed to give smoother utility estimates)
        """
        N, p = PhiMat.shape

        beta = self.beta
        alpha = self.alpha

        nOut = lambdas.shape[0]
        nIn = min(int(nIn), nOut) 

        evids = np.zeros(nOut)

        nD = PhiMat.shape[0]

        utilities_dnmc = np.zeros(nD)
        if noises is None:
            noises = np.random.normal(0, 1/np.sqrt(beta), size=(nOut, 1))
        for dIdx in range(nD):
            print("Evaluating estimator at design index: ", dIdx)
            yvals = np.zeros((nOut, 1))

            loglikelis_out = np.zeros((nOut, 1))
            for i in range(nOut):
                lambda_sample = lambdas[[i], :]
                yvals[i, 0] = (PhiMat[[dIdx], :] @ lambda_sample.T)[0][0] + noises[i, 0]

                loglikelis_out[i, 0] = norm_logpdf(yvals[i:(i + 1)], loc=(PhiMat[[dIdx], :] @ lambda_sample.T)[0][0], scale=1/np.sqrt(beta))

            # for i in range(nOut):
            # for j in range(nIn):
                # inner_likelis = np.exp(norm_logpdf(yvals[i:(i+1)], loc=(PhiMat[[dIdx], :] @ lambdas[[j], :].T)[0][0], scale=1/np.sqrt(beta)))[0]

            for i in range(nOut):
                inner_likelis = np.exp(loglikelis_out[:nIn, :])
                evids[i] = inner_likelis.mean()

            utility_dnmc = (loglikelis_out - np.log(evids)).mean()
            utilities_dnmc[dIdx] = utility_dnmc

        # return utility_dnmc
        return utilities_dnmc

    def utility_analytical(self, PhiMat):
        """
        Calculate Fisher-information matrix and maximize EIG i.e. getting a D - optimal design.
        F = PhiMat (\theta, d) ^T * \Tau_G | \lambda^-1 * PhiMat (\theta, d)

        When we use ith row of PhiMat(\theta, d), we take log_det over [1x1] submatrix, by passing all rows, we can aggregate results into a vector specifying eig at each sample.

        \Tau_G | \lambda  is cov of Gaussian RV noise. If noise distribution has precision \beta, then \Tau_G | \lambda = (1/\beta)I_n and \Tau_G | \lambda^-1 = \beta I_n.
        EIG = (1/2)(log det \Tau_G (\theta, d) - log det \Tau_G|\lambda (\theta, d))

        where \Tau_G(\theta, d) = marginal prior predictive covariance = PhiMat * \Tau_\lambda * PhiMat^T + \Tau_G | \lambda 
        and \Tau_\lambda = (1 / \alpha)I_p
        """
        if PhiMat is None:
            PhiMat = self.PhiMat
        n, p = PhiMat.shape

        eig_analytic = np.zeros(n)
        for i in range(n):
            PhiI = PhiMat[i, :].reshape(1, -1)

            # Tau_G_cond_lambda = (1 / self.beta) * np.eye(n)
            # Tau_G_cond_lambda_inv = self.beta * np.eye(n)

            Tau_G_cond_lambda = (1 / self.beta) * np.eye(1)
            Tau_lambda = (1 / self.alpha) * np.eye(p)
            Tau_G = PhiI @ Tau_lambda @ PhiI.T + Tau_G_cond_lambda

            eig_analytic[i] = (1/2) * (self.log_det_mat_numpy(Tau_G) - self.log_det_mat_numpy(Tau_G_cond_lambda))

        return eig_analytic
    
    def utility_analytical_alternate(self, PhiMat):
        """
        Calculate Fisher-information matrix and maximize EIG i.e. getting a D - optimal design.
        F = PhiMat (\theta, d) ^T * \Tau_G | \lambda^-1 * PhiMat (\theta, d)

        EIG = (1/2)(log det \Tau_\Lambda - log det \Tau_\Lambda | )



        \Tau_G | \lambda  is cov of Gaussian RV noise. If noise distribution has precision \beta, then \Tau_G | \lambda = (1/\beta)I_n and \Tau_G | \lambda^-1 = \beta I_n.
        EIG = (1/2)(log det \Tau_G (\theta, d) - log det \Tau_G|\lambda (\theta, d))

        where \Tau_G(\theta, d) = marginal prior predictive covariance = PhiMat * \Tau_\lambda * PhiMat^T + \Tau_G | \lambda 
        and \Tau_\lambda = (1 / \alpha)I_p
        """
        if PhiMat is None:
            PhiMat = self.PhiMat
        n, p = PhiMat.shape

        eig_analytic = np.zeros(n)
        for i in range(n):
            PhiI = PhiMat[i, :].reshape(1, -1)

            # Tau_G_cond_lambda = (1 / self.beta) * np.eye(n)
            # Tau_G_cond_lambda_inv = self.beta * np.eye(n)

            Tau_G_cond_lambda = (1 / self.beta) * np.eye(1)
            Tau_lambda = (1 / self.alpha) * np.eye(p)
            Tau_G = PhiI @ Tau_lambda @ PhiI.T + Tau_G_cond_lambda

            eig_analytic[i] = (1/2) * (self.log_det_mat_numpy(Tau_G) - self.log_det_mat_numpy(Tau_G_cond_lambda))

        return eig_analytic
    
    @staticmethod
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
    

class OEDLGGaussian(object):
    """
    A Bayesian optimal experimental design class. This class aims to provide numerical and analytical calculations of utility as well as batch design for linear Gaussian surrogate model and GP correction. Optimization for U_KL can also be performed using Bayesian Optimization via BoTorch.
    """

    def __init__(self, PhiMat, G, beta, alpha, 
                 random_state=None, 
                 q_acq = 1, 
                 n_param=10, 
                 n_design=2, 
                #  prior_rvs=None, 
                #  prior_logpdf=None,
                ):
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
        
        # self.prior_rvs = prior_rvs
        # self.prior_logpdf = prior_logpdf

        NoneType = type(None)
        assert isinstance(random_state, (int, NoneType)), (
               "random_state should be an integer or None.")
        np.random.seed(random_state)
        self.random_state = random_state
        self.n_param = n_param
        self.n_design = n_design

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
        print("OED object Information:")
        for key, value in info_dict.items():
            print(f"  {key}: {value}")


    def ridge_regression(self, PhiMat, G):
        """
        Perform ridge regression to estimate the weights lambda.

        lambda = \beta (\beta PhiMat^T PhiMat + \alpha I)^{-1} PhiMat^T G
        """
        beta = self.beta
        alpha = self.alpha

        # allow supplying PhiMat and G as arguments.
        # PhiMat = self.PhiMat
        # G = self.G

        N, p = PhiMat.shape

        assert N == len(G), "Number of rows in PhiMat should be equal to the length of G."

        lambda_est = np.linalg.solve((beta * (PhiMat.T @ PhiMat) + alpha * np.eye(p)), beta * (PhiMat.T @ G))

        return lambda_est

    def loglikelihood(self, PhiMat, G):
        """
        self.PhiMat: design matrix
        self.G: response variable
        self.beta: inverse variance / precision of Gaussian distributed noise.
        """
        beta = self.beta
        alpha = self.alpha
        lambda_est = self.ridge_regression(PhiMat, G)

        N, p = PhiMat.shape

        lstsq_estimate = (PhiMat @ lambda_est).reshape(-1, 1)
        yvals = G.reshape(-1, 1)

        # log_lik = (N/2) * np.log(beta) - (N/2) * np.log(2 * np.pi) - (beta/2) * np.linalg.norm(yvals - (PhiMat @ lambda_est))**2

        log_lik = norm_logpdf(yvals, loc=lstsq_estimate, scale=1/np.sqrt(beta))

        return log_lik
        

    def logprior(self, lambdas):
        """
        self.alpha: prior inverse variance / precision of weights lambda.
        lambdas: size n_post_samples x p
        """
        return norm_logpdf(lambdas, loc=0, scale=1/np.sqrt(self.alpha))
        
    def sample_prior(self, n_prior_samples):
        """
        Generate n_prior_samples from the prior distribution.
        """
        p = self.n_param
        prior_samples = np.random.normal(0, 1/np.sqrt(self.alpha), size=(n_prior_samples, p))
        return prior_samples

    def post_logpdf_mcmc(self, lambdas, PhiMat, G, include_prior=True):
        """
        Logprobability of unnormalized posterior after observing data y (following Bayes rule)
        include_prior : bool, optional(default=True)
        Include the prior in the posterior or not. It not included, the
        posterior is just a multiplication of likelihoods.

        Returns:
        A numpy.ndarray of size n_samples which are log-posteriors.
        """
        n_samples, n_params = lambdas.shape

        lambda_est = self.ridge_regression(PhiMat, G)
        lstsq_estimate = (PhiMat @ lambda_est).reshape(-1, 1)
        yvals = G.reshape(-1, 1)


        logpost = np.zeros(n_samples)

        loglikelihoods = norm_logpdf(yvals, loc=lstsq_estimate, scale=1/np.sqrt(self.beta))
        logpost += loglikelihoods

        if include_prior:
            # logprior = self.prior_logpdf(lambdas)
            logprior = self.logprior(lambdas)
            logpost += logprior

        return logpost

    def post_rvs_mcmc(self, n_post_samples, PhiMat, G):
        """
        Generate n_post_samples from the posterior distribution.
        """
        log_prob = lambda x: self.post_logpdf_mcmc(x.reshape(-1, self.n_param), PhiMat, G, include_prior=True)

        n_dim, n_walkers = self.n_param, 2 * self.n_param

        theta0 = self.prior_rvs(n_walkers)

        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
        sampler.run_mcmc(theta0,
                         int(n_post_samples // n_walkers * 1.2),
                         progress=True)
        
        return sampler.get_chain().reshape(-1, self.n_param)[-n_post_samples:]



    def post_logpdf_closed_form(self, lambdas, PhiMat, G, include_prior=True):
        """
        Logprobability for (Gaussian) posterior after observing data y (following Bayes rule, closed form for linear Gaussian model)
        """
        pass
        
        

    def post_rvs_closed_form(self, n_post_samples, PhiMat, G):
        """
        Generate n_post_samples from the posterior distribution.
        """

        beta = self.beta
        alpha = self.alpha

        lambda_est = self.ridge_regression(PhiMat, G)


        N, p = PhiMat.shape

        S_N_inv = alpha * np.eye(p) + beta * PhiMat.T @ PhiMat
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * np.dot(S_N, np.dot(PhiMat.T, G))

        post_samples = np.random.multivariate_normal(m_N, S_N, size=n_post_samples)

        return post_samples


    def posterior_predictive(self, PhiMat, G, PhiMatPost):
        """
        Generate n_post_samples from the posterior predictive distribution.
        PhiMat and G typically will be the matrices used to build the surrogate model.
        """
        beta = self.beta
        alpha = self.alpha

        # lambda_est = self.ridge_regression(PhiMat, G)

        N, p = PhiMat.shape

        n_post_samples = PhiMatPost.shape[0]

        assert p == PhiMatPost.shape[1]

        S_N_inv = alpha * np.eye(p) + beta * PhiMat.T @ PhiMat
        S_N = np.linalg.inv(S_N_inv)

        m_N = beta * np.dot(S_N, np.dot(PhiMat.T, G))

        predictive_mean = np.dot(PhiMatPost, m_N)
        predictive_var = (1/beta) + np.sum(np.dot(PhiMatPost, S_N) @ PhiMatPost, axis=1)

        return predictive_mean, predictive_var        

    def utility_dnmc(self, lambdas, PhiMat, noises, nIn=5e2):
        """
        NMC estimator.
        U(\theta, d) = (1/N_out)sum_i [log p(G_i | \theta, \lambda, d) - (1/N_in)sum_j log p(G_i | \theta, d, \lambda_i,j)]

        Can optionally supply same set of noises to be used on all designs.
        """
        N, p = PhiMat.shape

        beta = self.beta
        alpha = self.alpha

        nOut = lambdas.shape[0]
        nIn = min(int(nIn), nOut) 

        evids = np.zeros(nOut)

        nD = PhiMat.shape[0]

        utilities_dnmc = np.zeros(nD)
        if noises is None:
            noises = np.random.normal(0, 1/np.sqrt(beta), size=(nOut, 1))
        for dIdx in range(nD):
            print("Evaluating estimator at design index: ", dIdx)
            yvals = np.zeros((nOut, 1))

            loglikelis_out = np.zeros((nOut, 1))
            for i in range(nOut):
                lambda_sample = lambdas[[i], :]
                yvals[i, 0] = (PhiMat[[dIdx], :] @ lambda_sample.T)[0][0] + noises[i, 0]

                loglikelis_out[i, 0] = norm_logpdf(yvals[i:(i + 1)], loc=(PhiMat[[dIdx], :] @ lambda_sample.T)[0][0], scale=1/np.sqrt(beta))

            for i in range(nOut):
                for j in range(nIn):
                    inner_likelis = np.exp(norm_logpdf(yvals[i:(i+1)], loc=(PhiMat[[dIdx], :] @ lambdas[[j], :].T)[0][0], scale=1/np.sqrt(beta)))[0]
                    evids[i] += inner_likelis
                evids[i] = evids[i] / nIn

            utility_dnmc = (loglikelis_out - np.log(evids)).mean()
            utilities_dnmc[dIdx] = utility_dnmc

        return utilities_dnmc
    

    def utility_dnmc_fast(self, lambdas, PhiMat, noises, nIn=5e2):
        """
        NMC estimator, but faster version because we are reusing outer loop samples for computing inner evidence term.
        U(\theta, d) = (1/N_out)sum_i [log p(G_i | \theta, \lambda, d) - (1/N_in)sum_j log p(G_i | \theta, d, \lambda_i,j)]

        Can optionally supply same set of noises to be used on all designs (its supposed to give smoother utility estimates)
        """
        N, p = PhiMat.shape

        beta = self.beta
        alpha = self.alpha

        nOut = lambdas.shape[0]
        nIn = min(int(nIn), nOut) 

        evids = np.zeros(nOut)

        nD = PhiMat.shape[0]

        utilities_dnmc = np.zeros(nD)
        if noises is None:
            noises = np.random.normal(0, 1/np.sqrt(beta), size=(nOut, 1))
        for dIdx in range(nD):
            print("Evaluating estimator at design index: ", dIdx)
            yvals = np.zeros((nOut, 1))

            loglikelis_out = np.zeros((nOut, 1))
            for i in range(nOut):
                lambda_sample = lambdas[[i], :]
                yvals[i, 0] = (PhiMat[[dIdx], :] @ lambda_sample.T)[0][0] + noises[i, 0]

                loglikelis_out[i, 0] = norm_logpdf(yvals[i:(i + 1)], loc=(PhiMat[[dIdx], :] @ lambda_sample.T)[0][0], scale=1/np.sqrt(beta))

            # for i in range(nOut):
            # for j in range(nIn):
                # inner_likelis = np.exp(norm_logpdf(yvals[i:(i+1)], loc=(PhiMat[[dIdx], :] @ lambdas[[j], :].T)[0][0], scale=1/np.sqrt(beta)))[0]

            for i in range(nOut):
                inner_likelis = np.exp(loglikelis_out[:nIn, :])
                evids[i] = inner_likelis.mean()

            utility_dnmc = (loglikelis_out - np.log(evids)).mean()
            utilities_dnmc[dIdx] = utility_dnmc

        # return utility_dnmc
        return utilities_dnmc

    
    def utility_analytical_alternate(self, PhiMat):
        """
        Calculate Fisher-information matrix and maximize EIG i.e. getting a D - optimal design.
        F = PhiMat (\theta, d) ^T * \Tau_G | \lambda^-1 * PhiMat (\theta, d)

        EIG = (1/2)(log det \Tau_\Lambda - log det \Tau_\Lambda | )



        \Tau_G | \lambda  is cov of Gaussian RV noise. If noise distribution has precision \beta, then \Tau_G | \lambda = (1/\beta)I_n and \Tau_G | \lambda^-1 = \beta I_n.
        EIG = (1/2)(log det \Tau_G (\theta, d) - log det \Tau_G|\lambda (\theta, d))

        where \Tau_G(\theta, d) = marginal prior predictive covariance = PhiMat * \Tau_\lambda * PhiMat^T + \Tau_G | \lambda 
        and \Tau_\lambda = (1 / \alpha)I_p
        """
        if PhiMat is None:
            PhiMat = self.PhiMat
        n, p = PhiMat.shape

        eig_analytic = np.zeros(n)
        for i in range(n):
            PhiI = PhiMat[i, :].reshape(1, -1)

            # Tau_G_cond_lambda = (1 / self.beta) * np.eye(n)
            # Tau_G_cond_lambda_inv = self.beta * np.eye(n)

            Tau_G_cond_lambda = (1 / self.beta) * np.eye(1)
            Tau_lambda = (1 / self.alpha) * np.eye(p)
            Tau_G = PhiI @ Tau_lambda @ PhiI.T + Tau_G_cond_lambda

            eig_analytic[i] = (1/2) * (self.log_det_mat_numpy(Tau_G) - self.log_det_mat_numpy(Tau_G_cond_lambda))

        return eig_analytic
    
    @staticmethod
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
    

def compute_posterior_from_priors(mu_lambda, Tau_lambda, X_train, Phi_train_scaled_reg, G_train_scaled, X_test, Phi_test_scaled_reg, scaler_Y, beta):
    # mu_lambda = np.zeros((Phi_train_scaled_reg.shape[1], 1))
    # Tau_lambda = (1 / alpha) * np.eye(Phi_train_scaled_reg.shape[1])
    # Tau_lambda_inv = alpha * np.eye(Phi_train_scaled_reg.shape[1])
    Tau_lambda_inv = np.linalg.inv(Tau_lambda)



    # GP Noise covariance matrix between training points K_tr:
    # Use RBF Kernel with fixed lengthscale.

    sigma_ls = 0.6

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

    return mu_lambda_post_y, Tau_lambda_post_y_inv, G_test_post_mean_unscaled, G_test_post_std_unscaled


def compute_posterior_pointwise(mu_lambda, Tau_lambda, X_train, Phi_train_scaled_reg, G_train_scaled, X_test):
    # mu_lambda = np.zeros((Phi_train_scaled_reg.shape[1], 1))
    # Tau_lambda = (1 / alpha) * np.eye(Phi_train_scaled_reg.shape[1])
    # Tau_lambda_inv = alpha * np.eye(Phi_train_scaled_reg.shape[1])
    Tau_lambda_inv = np.linalg.inv(Tau_lambda)

    # GP Noise covariance matrix between training points K_tr:
    # Use RBF Kernel with fixed lengthscale.

    sigma_ls = 0.6

    from scipy.spatial.distance import pdist
    from sklearn.metrics import pairwise_distances
    from scipy.linalg import cho_factor, cho_solve

    # compute pairwise dist between training points
    pairwise_dists = pairwise_distances(X_train, metric='euclidean')

    # K_tr = (1/beta) * np.exp(-pairwise_dists**2 / (2 * sigma_ls**2))

    K_tr = np.exp(-pairwise_dists**2 / (2 * sigma_ls**2))

    # K_tr = np.exp(-pairwise_dists**2 / (2 * sigma_ls**2))\
    # + (1/beta) * np.eye(N_train)


    L = cho_factor(K_tr)

    Tau_lambda_post_y_inv = (Tau_lambda_inv + (Phi_train_scaled_reg.T @ cho_solve(L, Phi_train_scaled_reg)))

    # get parts of mu_lambda_post_y
    # mu_lambda_post_y_sol = 
                            # (Tau_lambda_inv @ mu_lambda 
    # mu_lambda_post_y_sol = Phi_train_scaled_reg.T @ np.linalg.inv(K_tr) @ G_train_scaled

    mu_lambda_post_y_sol = Tau_lambda_inv @ mu_lambda + Phi_train_scaled_reg.T @ cho_solve(L, G_train_scaled)

    mu_lambda_post_y = np.linalg.lstsq(Tau_lambda_post_y_inv, mu_lambda_post_y_sol)[0]

    return mu_lambda_post_y, Tau_lambda_post_y_inv

def log_det_mat_numpy(mat):
    """
    Calculate the log determinant of a matrix.
    """
    return np.linalg.slogdet(mat)[1]

def utility_analytical(PhiMat, Tau_lambda, beta):
    from scipy.spatial.distance import pdist
    from sklearn.metrics import pairwise_distances
    from scipy.linalg import cho_factor, cho_solve

    n, p = PhiMat.shape

    eig_analytic = np.zeros(n)
    for i in range(n):
        PhiI = PhiMat[i, :].reshape(1, -1)

        # Tau_G_cond_lambda = (1 / self.beta) * np.eye(n)
        # Tau_G_cond_lambda_inv = self.beta * np.eye(n)

        Tau_G_cond_lambda = (1/beta) * np.eye(n)
        Tau_G_cond_lambda_inv = beta * np.eye(n)

        
        Tau_G = PhiI @ Tau_lambda @ PhiI.T + Tau_G_cond_lambda

        eig_analytic[i] = (1/2) * (log_det_mat_numpy(Tau_G) - log_det_mat_numpy(Tau_G_cond_lambda))

    return eig_analytic