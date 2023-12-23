import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import product
from functools import reduce
import matplotlib.cm as cm
import scipy
import time

from .qst import QST


class BWQST(QST):

    def __init__(self, L: int, psi: np.ndarray):
        super().__init__(L, psi)

    def sampling_operator(self, U: np.ndarray, Pk: np.ndarray):
        """
        Parameters:
            U: dxd density matrix
            Pk: m x d x d x d array of projection operators
        Returns:
            A(U) = [ [tr(UP_1), ...], ..., [..., tr(UP_md)] ]
        """
        return np.array([[
                np.real(np.trace(P @ U)) for P in P_setting ]
            for P_setting in Pk ])

    def simulate_measurement(self, N: int, m: int):
        """
        Parameters:
            N: number of measurement repetitions
            m: number of measurement settings
        Computes Y, the vector of sampled measurements
        """
        measures = np.random.randint(0, 3, size=(m, self.L))
        self.proj_k = self.get_projectors(measures) # Random projectors
        self.proj_flat = self.proj_k.reshape((m*self.d, self.d, self.d))
        
        probabilities = self.sampling_operator(self.rho, self.proj_k)
        probabilities[probabilities < 1e-15] = 0

        samples = np.array([ np.random.multinomial(N, basis_probs) for basis_probs in probabilities ])

        self.Y = samples.flatten() / N
    
    def grad_F(self, sigma_0):
        inv_sqrt_C_n = np.sqrt(self.d) * np.eye(self.d)
        def get_Xi(i: int):
            return self.Y[i] * inv_sqrt_C_n @ self.proj_flat[i] @ inv_sqrt_C_n
        
        exp_X_Q = 0
        for i in range(self.m*self.d):
            Xi = get_Xi(i)
            if (self.Y[i] >= 1e-10): # If measurement nonzero
                exp_X_Q += Xi / np.trace(Xi @ sigma_0)

        T = (1 / (self.m*self.d)) * exp_X_Q

        return np.eye(self.d) - T
    
    def grad_step(self, sigma, eta=1):

        factor = np.eye(self.d) - eta * self.grad_F(sigma)
        return factor @ sigma @ factor
    
    def run(self, N: int, m: int, verbose=False):

        self.simulate_measurement(N, m)

        self.m = m
        sigma = np.random.random((self.d, self.d)) + np.random.random((self.d, self.d)) * 1j
        eta = 0.1

        steps = 0

        while (steps < 1000):
            sigma = self.grad_step(sigma)
            steps += 1
        
        self.reconstruction = sigma / np.trace(sigma)


    def run_avg(self, N_avgs: int, N: int, m: int, **kwargs):
        """Runs QST multiple times and averages fidelities"""

        fidelities = []
        print(f"Running with m={m} ...")
        for i in range(N_avgs):
            self.run(N, m, **kwargs)
            fidelities.append(self.get_fidelity())
        fidelities = np.array(fidelities)

        return np.mean(fidelities), np.std(fidelities)
