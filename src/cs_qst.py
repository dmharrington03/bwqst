import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import product
from functools import reduce
import matplotlib.cm as cm
import scipy
import time

from .qst import QST


class CSQST(QST):

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
        probabilities = self.sampling_operator(self.rho, self.proj_k)
        probabilities[probabilities < 1e-15] = 0
        # print(probabilities[probabilities < 0])

        samples = np.array([ np.random.multinomial(N, basis_probs) for basis_probs in probabilities ])

        self.Y = samples.flatten() / N
    
    def run(self, N: int, m: int, epsilon=0.35, verbose=False):

        self.simulate_measurement(N, m)

        proj_flat = self.proj_k.reshape(self.proj_k.shape[0]*self.proj_k.shape[1], self.d, self.d)
        md = self.proj_k.shape[0]*self.proj_k.shape[1]

        X = cp.Variable((self.d, self.d), complex=True)

        AX = cp.reshape(cp.vstack([cp.trace(proj_flat[i] @ X) for i in range(md)]), (md))
        constraints = [X >> 0]

        constraints += [cp.norm(AX - self.Y) <= epsilon]

        prob = cp.Problem(cp.Minimize(cp.normNuc(X)),   
                        constraints)

        if (verbose):
            print(f"Solving convex problem...")
        start = time.time()
        argmin = prob.solve()
        end = time.time()
        if (verbose):
            print(f"Sovled! In {(end - start):.3f} s")
        # print(f"Minimizing trace norm: {argmin:.3f}")
        
        if ((X.value is None) and verbose):
            print("Unable to solve")

        self.reconstruction = X.value
        if (verbose):
            print(f"Fidelity: {self.get_fidelity():.3f}")

    def run_avg(self, N_avgs: int, N: int, m: int, epsilon=0.35, **kwargs):
        """Runs QST multiple times and averages fidelities"""

        fidelities = []
        print(f"Running with m={m}, eps={epsilon:.4f} ...")
        for i in range(N_avgs):
            self.run(N, m, **kwargs)
            fidelities.append(self.get_fidelity())
        fidelities = np.array(fidelities)

        return np.mean(fidelities), np.std(fidelities)
