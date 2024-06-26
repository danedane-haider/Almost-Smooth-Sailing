import numpy as np
from scipy.sparse.linalg import svds

class Regularizer:
    def __init__(self, B, beta):
        self.B = B
        self.beta = beta
        self.N = np.min(B.shape)
        self.value = self._compute_value()
        self.gradient = self._compute_gradient()

    def _compute_value(self):
        fro_norm = np.linalg.norm(self.B, ord='fro')**2
        spectral_norm = np.linalg.norm(self.B, ord=2)**2
        return 0.5 * spectral_norm - 0.5 / self.N * fro_norm

    def _compute_gradient(self):
        U, Sigma, Vt = svds(self.B, k=1) #Truncated SVD as we only need the largest singular value
        sigma_max = Sigma[0]
        u1 = U[:, 0]
        v1 = Vt[0, :]

        gradient = self.beta  * (sigma_max * np.outer(u1, v1) - (1 / self.N) * self.B)

        return gradient

# Example usage:
B = np.random.randn(5, 5)
beta = 0.1
reg = Regularizer(B, beta)
print("Regularizer value:", reg.value)
print("Gradient:", reg.gradient)
print(np.linalg.norm(reg.gradient - grad_reg(B,beta)))