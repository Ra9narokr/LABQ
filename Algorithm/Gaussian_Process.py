from .Matrix import to_SPDM

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
import scipy.linalg as la

def beta_processing(beta, dimension):
    """
    当维度 d>1 时，将每个维度的参数矢量拆分成 d 个独立的参数矢量
    For dimension d>1, splits the parameter vector into d individual parameter vectors for each dimension
    """

    if dimension == 1:
        return beta
    else:
        first = beta[1:]
        n_beta = np.concatenate(
            (np.tile(np.array([[beta[0]]]), dimension).T,
             np.split(first, dimension)),
             axis=1)
        return n_beta



class GaussianProcess:

    def __init__(self, Kernel, beta, X, Y, mesh=np.arange(0, 1.01, 0.01)):
        self.Kernel = Kernel
        self.beta = beta
        self.X = X
        self.Y = Y
        self.D = len(X[0])
        self.mesh = mesh
        self.Kernel.D = self.D
        self.kernel = self.Kernel.kernel
        self.cov_matrix = to_SPDM(self.cov_matrix_(self.mesh, self.mesh, self.beta))

    def cov_matrix_(self, X1, X2, beta):
        if self.D == 1:
            return self.kernel(X1.flatten()[:, np.newaxis], X2.flatten(), beta)
        else:
            betaD = beta_processing(beta, self.D)
            X1_T, X2_T = X1.T, X2.T
            cov_mat = self.kernel(X1_T[0][:, np.newaxis], X2_T[0], betaD[0])
            for i in range(self.D - 1):
                cov_mat = cov_mat * self.kernel(X1_T[i + 1][:, np.newaxis], X2_T[i + 1], betaD[i + 1])
            return cov_mat

    def mean_(self, X):
        return np.zeros(len(X)) + np.mean(self.Y)

    def log_likelihood(self, X, Y, beta):
        return mvn.logpdf(Y, np.zeros(len(X)), to_SPDM(self.cov_matrix_(X, X, beta)), allow_singular=True)

    def fit(self, init, lambd, adapt=False, info = False):
        X, Y = self.X, self.Y

        loss = lambda beta: -self.log_likelihood(X, Y, beta) + self.Kernel.reg_nd(X, beta, lambd)
        beta_fit = minimize(loss, self.beta, method='BFGS')
        if info == True:
            print("beta: ", beta_fit.x, "\nMessage: ", beta_fit.message, "\nValue: ",
                  beta_fit.fun, "\nSuccess: ", beta_fit.success, "\nLog-Likelihood: ",
                  self.log_likelihood(self.X, self.Y, beta_fit.x))
        return beta_fit.x

    def sample(self, n_samps, X=None, Y=None, mesh=None):

        if X is None and Y is None:
            X, Y = self.X, self.Y
        if mesh is None:
            mesh = self.mesh
            K_mesh = self.cov_matrix
        else:
            K_mesh = self.cov_matrix_(mesh, mesh, self.beta)

        K_data = to_SPDM(self.cov_matrix_(X, X, self.beta))
        L_data = la.cholesky(K_data, lower=True)
        K_s = self.cov_matrix_(X, mesh, self.beta)
        L_solved = la.solve_triangular(L_data, K_s, check_finite=False, lower=True)
        post_mean_vec = self.mean_(mesh) + np.dot(L_solved.T, la.solve_triangular(L_data, Y - self.mean_(X).flatten(),
                                                                                  check_finite=False,
                                                                                  lower=True)).reshape((len(mesh),))
        L_post = la.cholesky(to_SPDM(K_mesh - np.dot(L_solved.T, L_solved)), lower=True)

        stdv = np.nan_to_num(np.sqrt(np.diag(K_mesh) - np.sum(L_solved ** 2, axis=0)))

        return (post_mean_vec.reshape(-1, 1) + np.dot(L_post, np.random.normal(size=(len(mesh), n_samps))),
                post_mean_vec.reshape(-1, 1), stdv)
