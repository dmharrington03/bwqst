import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from itertools import product
from functools import reduce
import matplotlib.cm as cm
import scipy

class QST:

    def __init__(self, L: int, psi: np.ndarray):
        self.L = L # Num of qubits
        self.d = 2**L # Dim of Hilbert space
        self.rho = np.outer(psi, psi) # Density matrix

        # Define Pauli Matrices
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.paulis = np.array([self.X, self.Y, self.Z])
        self.reconstruction = None

    def show_fig(self, matrix: np.ndarray, title: str):
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection="3d")
        _x = np.arange(0.5, self.d + .5, 1)
        _y = np.arange(0.5, self.d + .5, 1)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        amplitudes = np.abs(matrix)
        phases = np.angle(matrix)

        top = amplitudes.ravel()
        bottom = np.zeros_like(top)
        width = depth = 1

        ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=plt.cm.hsv((3.9 + phases.flatten())/(2*np.pi)))
        ax.set_title(title)
        ax.set_xlabel("j")
        ax.set_ylabel("k")
        ax.set_zlabel(r"$|\rho_{jk}|$")
        plt.show()

    def show_target(self, title: str):
        self.show_fig(self.rho, title)


    def show_output(self, title: str):
        if self.reconstruction is not None:
            self.show_fig(self.reconstruction, title)
        else:
            print("No reconstruction available")

    def get_projectors(self, measure_settings: np.ndarray):
        """
        Parameters:
            measure_settings: m by L array of 0, 1, or 2 determining Pauli basis
                where m is number of measurement settings
        Returns:
            mxd array of random Pauli basis projectors,
            i.e. returns list of P_k for eigenspace projectors P_k
        """
        eigvecs = np.array([ np.linalg.eig(sigma).eigenvectors for sigma in self.paulis ])
        # Number of qubits
        m = measure_settings.shape[0]

        # Set of dxd projectors for each basis for each measurement setting
        P = np.zeros((m, self.d, self.d, self.d), dtype="complex128")

        for j, pauli_idx in enumerate(measure_settings):

            pauli_eigs_l = eigvecs[pauli_idx]

            # Get all combinations of 1 eigenvector from each pauli across L paulis
            eigs_sets = list(product(*pauli_eigs_l)) # 2^L total

            for k, set in enumerate(eigs_sets):
                # Tensor product all eigenvectors
                v_k = reduce(np.kron, set)
                # Form projector onto eigenspace
                proj_k = np.outer(np.conj(v_k), v_k)
                P[j, k, :, :] = proj_k

        return P
    
    def compute_fidelity(self, rho: np.ndarray, sigma: np.ndarray):
        """Returns the fidelity between the two states"""
        sqrt_rho = scipy.linalg.sqrtm(rho)
        return np.real(np.trace(scipy.linalg.sqrtm(sqrt_rho @ sigma @ sqrt_rho)))

    def get_fidelity(self):
        """Returns fidelity between reconstruction and target"""
        if self.reconstruction is not None:
            return self.compute_fidelity(self.rho, self.reconstruction)
        else:
            print("No reconstruction available")
            return 0
        