from .Gaussian_Process import beta_processing

import numpy as np

from .kernel_ingres import integration_funcs as ki
from .kernel_ingres import length_scale_funcs as kl
from .kernel_ingres import regularisation_funcs as kr
from .kernel_ingres import stationary_kernel_funcs as ks


class Kernel:

    def __init__(self, l_scale, stat_kern, int_kern, reg, int_2D):
        self.l_scale = l_scale
        self.stat_kern = stat_kern
        self.int_kern = int_kern
        self.reg = reg
        self.kernel = self.kern
        self.D = 1
        self.int_2D = int_2D

    def kern(self, x, y, beta):
        l_scaleX = self.l_scale(x, beta)
        l_scaleY = self.l_scale(y, beta)
        arg = np.abs(x - y) / np.sqrt(l_scaleX ** 2 + l_scaleY ** 2)
        return (np.exp(2 * beta[0])
                * np.sqrt((l_scaleX * l_scaleY)
                / (l_scaleX ** 2 + l_scaleY ** 2))
                * self.stat_kern(arg))

    def reg_nd(self, X, beta, fun):
        if self.D == 1:
            return self.reg(X, beta, fun)
        else:
            beta_d = beta_processing(beta, self.D)
            X_T = X.T
            r = self.reg(X_T[0], beta_d[0], fun)
            for i in range(self.D - 1):
                r = r * self.reg(X_T[i + 1], beta_d[i + 1], fun)
            return r

l_linear_10 = lambda x, beta: kl.l_linear(x, beta, 10)
r_lin_10 = lambda X, beta, lambd: kr.r_lin(X, beta, lambd, 10)
int_lin_10 = lambda fun, x, beta, simp=False: ki.int_lin(fun, x, beta, 10, simp)
int_2D_10 = lambda fun, beta, simp=False: ki.int_2D(fun, beta, 10, simp)

K_lin_mat2_10 = Kernel(l_linear_10, ks.matern_32, int_lin_10, r_lin_10, int_2D_10)