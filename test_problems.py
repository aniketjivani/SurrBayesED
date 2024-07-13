# %% Lovely test problems for building simple metamodels

import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

# sns.set_context('talk')
# sns.set_style('whitegrid')
# %% Test problem 1: Borehole function
# Source: https://www.sfu.ca/~ssurjano/borehole.html

class Borehole:
    def __init__(self):
        # create input uncertainty objects for all arguments passed to calculate the water flow rate through the borehole!
        self.rw_dist = stats.norm(loc=0.1, scale=0.0161812)
        self.r_dist = stats.lognorm(s=1.0056, scale=np.exp(7.71))
        self.Tu_dist = stats.uniform(63070, 115600)
        self.Hu_dist = stats.uniform(990, 1110)
        self.Tl_dist = stats.uniform(63.1, 116)
        self.Hl_dist = stats.uniform(700, 820)
        self.L_dist = stats.uniform(1120, 1680)
        self.Kw_dist = stats.uniform(9855, 12045)


    def input_samples(self, n_samples):
        """
        Return samples of input parameters alone.
        """
        rw = self.rw_dist.rvs(n_samples)
        r = self.r_dist.rvs(n_samples)
        Tu = self.Tu_dist.rvs(n_samples)
        Hu = self.Hu_dist.rvs(n_samples)
        Tl = self.Tl_dist.rvs(n_samples)
        Hl = self.Hl_dist.rvs(n_samples)
        Lth = self.L_dist.rvs(n_samples)
        Kw = self.Kw_dist.rvs(n_samples)

        return rw, r, Tu, Hu, Tl, Hl, Lth, Kw

    def sample(self, n_samples):
        """
        Borehole function: response is water flow through borehole in m^3 per year.

        Parameters:
        rw: radius of borehole in m
        r:  radius of influence in m
        Tu: transmissivity of upper aquifer in m^2/year
        Hu: potentiometric head of upper aquifer in m
        Tl: transmissivity of lower aquifer in m^2/year
        Hl: potentiometric head of lower aquifer in m
        Lth:  length of borehole in m
        Kw: hydraulic conductivity of borehole in m/year

        Use distribution objects defined in __init__ to sample n_samples from each parameter distribution. Then compute the water flow rate and return as a numpy array.
        """

        rw_samples = self.rw_dist.rvs(n_samples)
        r_samples = self.r_dist.rvs(n_samples)
        Tu_samples = self.Tu_dist.rvs(n_samples)
        Hu_samples = self.Hu_dist.rvs(n_samples)
        Tl_samples = self.Tl_dist.rvs(n_samples)
        Hl_samples = self.Hl_dist.rvs(n_samples)
        Lth_samples = self.L_dist.rvs(n_samples)
        Kw_samples = self.Kw_dist.rvs(n_samples)



        # let's just do it in a for loop.
        Q = np.zeros(n_samples)
        for i in range(n_samples):
            numerator = 2 * math.pi * Tu_samples[i] * (Hu_samples[i] - Hl_samples[i])

            denom1 = np.log(r_samples[i] / rw_samples[i])
            denom2 = 2 * Lth_samples[i] * Tu_samples[i] / (Kw_samples[i] * r_samples[i]**2)
            denom3 = Tu_samples[i] / Tl_samples[i]

            Q[i] = numerator / (denom1 * (1 + (denom2 / denom1) + denom3))

        return Q


# %% Test Problem 2: Wing weight function 
# Source: https://www.sfu.ca/~ssurjano/wingweight.html
# Source: Forrester et al. (2008) "Engineering design via surrogate modelling"

class WingWeight:
    """
    Models weight of a light aircraft wing as a function of 10 design parameters.

    Parameters:
    Sw: wing surface area in ft^2
    Fw: weight of wing fuel in lb
    A: aspect ratio
    sweep: quarter-chord sweep in degrees
    q: dynamic pressure at cruise in lb/ft^2
    tp: taper ratio
    tc: thickness-to-chord ratio
    Nz: ultimate load factor
    W: flight design gross weight in lb
    p: paint weight in lb/ft^2

    """

    def __init__(self):
        self.Sw_dist = stats.uniform(150, 200)
        self.Fw_dist = stats.uniform(220, 300)
        self.A_dist = stats.uniform(6, 10)
        self.sweep_dist = stats.uniform(-10, 10)
        self.q_dist = stats.uniform(16, 45)
        self.tp_dist = stats.uniform(0.5, 1.0)
        self.tc_dist = stats.uniform(0.08, 0.18)
        self.Nz_dist = stats.uniform(2.5, 6)
        self.W_dist = stats.uniform(1700, 2500)
        self.p_dist = stats.uniform(0.025, 0.08)

    def input_samples(self, n_samples):
        """
        Return samples of input parameters alone.
        """
        Sw = self.Sw_dist.rvs(n_samples)
        Fw = self.Fw_dist.rvs(n_samples)
        AspRt = self.A_dist.rvs(n_samples)
        sweep = self.sweep_dist.rvs(n_samples)
        q = self.q_dist.rvs(n_samples)
        tp = self.tp_dist.rvs(n_samples)
        tc = self.tc_dist.rvs(n_samples)
        Nz = self.Nz_dist.rvs(n_samples)
        W_gross = self.W_dist.rvs(n_samples)
        p = self.p_dist.rvs(n_samples)

        return Sw, Fw, AspRt, sweep, q, tp, tc, Nz, W_gross, p
    
    def sample(self, n_samples):
        """
        Wing weight function: response is weight of the wing in lb.
        """
        Sw_samples = self.Sw_dist.rvs(n_samples)
        Fw_samples = self.Fw_dist.rvs(n_samples)
        A_samples = self.A_dist.rvs(n_samples)
        sweep_samples = self.sweep_dist.rvs(n_samples)
        q_samples = self.q_dist.rvs(n_samples)
        tp_samples = self.tp_dist.rvs(n_samples)
        tc_samples = self.tc_dist.rvs(n_samples)
        Nz_samples = self.Nz_dist.rvs(n_samples)
        W_samples = self.W_dist.rvs(n_samples)
        p_samples = self.p_dist.rvs(n_samples)

        W_wing = np.zeros(n_samples)

        for i in range(n_samples):
            fac1 = 0.036 * (Sw_samples[i]**0.758) * (Fw_samples[i]**0.0035)
            fac2 = (A_samples[i] / (np.cos(sweep_samples[i] * np.pi / 180)) ** 2) ** 0.6
            fac3 = q_samples[i]**0.006 * (tp_samples[i]**0.04)
            fac4 = (100 * tc_samples[i] / np.cos(sweep_samples[i] * np.pi / 180))**(-0.3)
            fac5 = (Nz_samples[i] * W_samples[i])**0.49
            term1 = Sw_samples[i] * p_samples[i]

            W_wing[i] = fac1 * fac2 * fac3 * fac4 * fac5 + term1

        return W_wing
    

class WingWeightReduced:
    """
    Since only a few parameters are influential in the wing weight function, we can reduce the number of input parameters - we will use 2.
    """
    def __init__(self, n_active=2):

        self.n_active = n_active
        if n_active == 2:
            self.Sw_dist = stats.uniform(150, 200)
            self.Nz_dist = stats.uniform(2.5, 6)

            # initialize other parameters to their nominal values!
            self.Fw = 252
            self.A = 7.52
            self.sweep = 0
            self.q = 34
            self.tp = 0.672
            self.tc = 0.12
            self.W = 2000
            self.p = 0.064
        elif n_active == 1:
            self.Sw_dist = stats.uniform(150, 200)

            # initialize other parameters to their nominal values!
            self.Fw = 252
            self.A = 7.52
            self.sweep = 0
            self.q = 34
            self.tp = 0.672
            self.tc = 0.12
            self.Nz = 3.8
            self.W = 2000
            self.p = 0.064


        # self.W_dist = stats.uniform(1700, 2500)

    def input_samples(self, n_samples):
        """
        Return samples of input parameters alone.
        """
        if self.n_active == 2:
            Sw = self.Sw_dist.rvs(n_samples)
            Nz = self.Nz_dist.rvs(n_samples)
            return Sw, Nz
        elif self.n_active == 1:
            Sw = self.Sw_dist.rvs(n_samples)
            return Sw
  

    def compute_weight(self, Sw_val, Fw_val, A_val, sweep_val, q_val, tp_val, tc_val, Nz_val, W_val, p_val):
        """
        Compute wing weight as a function of all 10 variables (call from within for loops or directly on grids of parameters)
        """
        fac1 = 0.036 * (Sw_val**0.758) * (Fw_val**0.0035)
        fac2 = (A_val / (np.cos(sweep_val * np.pi / 180)) ** 2) ** 0.6
        fac3 = q_val**0.006 * (tp_val**0.04)
        fac4 = (100 * tc_val / np.cos(sweep_val * np.pi / 180))**(-0.3)
        fac5 = (Nz_val * W_val)**0.49
        term1 = Sw_val * p_val

        W_wing = fac1 * fac2 * fac3 * fac4 * fac5 + term1

        return W_wing



    def sample(self, n_samples):
        """
        Compute wing weight as a function of all 10 variables (3 varying, 7 fixed)
        """
        if self.n_active == 2:
            Sw_samples = self.Sw_dist.rvs(n_samples)
            Nz_samples = self.Nz_dist.rvs(n_samples)
            W_wing = np.zeros(n_samples)

            for i in range(n_samples):
                W_wing[i] = self.compute_weight(Sw_samples[i], self.Fw, self.A, self.sweep, self.q, self.tp, self.tc, Nz_samples[i], self.W, self.p)
        elif self.n_active == 1:
            Sw_samples = np.sort(self.Sw_dist.rvs(n_samples))
            W_wing = np.zeros(n_samples)

            for i in range(n_samples):
                W_wing[i] = self.compute_weight(Sw_samples[i], self.Fw, self.A, self.sweep, self.q, self.tp, self.tc, self.Nz, self.W, self.p)

        return W_wing
    

    def inputs_and_outputs(self, n_samples):
        """
        Return samples of input parameters and corresponding output.
        """
        if self.n_active == 2:
            Sw = self.Sw_dist.rvs(n_samples)
            Nz = self.Nz_dist.rvs(n_samples)
            W_wing = np.zeros(n_samples)

            for i in range(n_samples):
                W_wing[i] = self.compute_weight(Sw[i], self.Fw, self.A, self.sweep, self.q, self.tp, self.tc, Nz[i], self.W, self.p)

            return Sw, Nz, W_wing
        
        elif self.n_active == 1:
            Sw = np.sort(self.Sw_dist.rvs(n_samples))
            W_wing = np.zeros(n_samples)

            for i in range(n_samples):
                W_wing[i] = self.compute_weight(Sw[i], self.Fw, self.A, self.sweep, self.q, self.tp, self.tc, self.Nz, self.W, self.p)

            return Sw, W_wing
    

    def plotOnGrid(self, n_samples):
        """
        Create grid of 50 samples for each of the 3 parameters.
        Make a contour plot of wing weight on this grid.
        Overlay random samples on this plot.
        """


        if self.n_active == 2:
            Sw, Nz, W_wing_samples = self.inputs_and_outputs(n_samples)

            Sw_vals = np.linspace(self.Sw_dist.support()[0],
                                self.Sw_dist.support()[1], 50)
            Nz_vals = np.linspace(self.Nz_dist.support()[0],
                                self.Nz_dist.support()[1], 50)
            # W_vals = np.linspace(self.W_dist.support()[0],
                                #  self.W_dist.support()[1], 50)
            
            Sw_grid, Nz_grid = np.meshgrid(Sw_vals, Nz_vals, indexing='ij')

            W_wing_grid = self.compute_weight(Sw_grid, self.Fw, self.A, self.sweep, self.q, self.tp, self.tc, Nz_grid, self.W, self.p)

            # make 1 3d plot and 1 contour plot.
            fig = plt.figure(figsize=(6, 10))
            ax1 = fig.add_subplot(211, projection='3d')
            # set projection for ax[0]
            surf = ax1.plot_surface(Sw_grid, 
                            Nz_grid,
                            W_wing_grid, 
                            cmap='viridis',
                            vmin=np.min(W_wing_grid),
                            vmax=np.max(W_wing_grid),
                            alpha=0.5)

            # overlay 3d scatter plot of samples.
            ax1.scatter(Sw, Nz, W_wing_samples, c='r', s=50)
            ax1.set_xlabel('Sw')
            ax1.set_ylabel('Nz')
            ax1.set_zlabel('W_wing')

            # Colorbar
            cbar = plt.colorbar(surf)

            ax2 = fig.add_subplot(212)
            cf = ax2.contourf(Sw_grid, Nz_grid, W_wing_grid, 
                            cmap='viridis',
                            vmin=np.min(W_wing_grid),
                            vmax=np.max(W_wing_grid))
            
            ax2.set_xlabel('Sw')
            ax2.set_ylabel('Nz')

            cbar2 = plt.colorbar(cf)

            fig.tight_layout()

            plt.show()
        elif self.n_active == 1:
            # we can only do scatter plots!
            Sw_samples, W_wing_samples = self.inputs_and_outputs(n_samples)

            Sw_grid = np.linspace(self.Sw_dist.support()[0],
                                self.Sw_dist.support()[1], 200)
            

            W_wing_grid = self.compute_weight(Sw_grid, self.Fw, self.A, self.sweep, self.q, self.tp, self.tc, self.Nz, self.W, self.p)

            fig, ax = plt.subplots()
            ax.plot(Sw_grid, W_wing_grid, lw=2, c='b', label='Wing weight function')
            ax.scatter(Sw_samples, W_wing_samples, c='r', s=50, label='Samples')

            ax.set_xlabel('Sw')
            ax.set_ylabel('Wing weight')
            ax.legend()
            plt.show() 

# %%
# Commenting out, rough checks for seeing if linear model is a good fit
# when n_active = 1
# TLDR - its quite good, barring extreme ends. 

ww = WingWeightReduced(n_active=1)
Sw, W_wing = ww.inputs_and_outputs(4)

# plt.scatter(Sw, W_wing)

# Sw, W_wing = ww.inputs_and_outputs(4)

A = np.vstack([Sw, np.ones(len(Sw))]).T
m, c = np.linalg.lstsq(A, W_wing)[0]
print(m, c)

_ = plt.plot(Sw, W_wing, 'o', label='Original data', markersize=4)
_ = plt.plot(Sw, m*Sw + c, 'r', label='Fitted line')
_ = plt.legend()
_ = plt.show()

# %%
