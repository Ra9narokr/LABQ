import numpy as np
from scipy.stats import norm


def r_lin(X, beta, lambd, n):  # lambd is the user input
    return (lambd[0] * 1 / (2 * n) * (np.sum(np.exp(beta[1:])) + np.sum(np.exp(beta[2:-1]))) +
            lambd[1] * (1 / n) * np.sum((beta[2:] - beta[1:-1]) / (np.exp(beta[2:]) - np.exp(beta[1:-1]))))



def r_pconst(X, beta, lambd, n):
    return lambd[0] * np.mean(np.exp(beta[1:])) + lambd[1] * np.mean(np.exp(-beta[1:]))


def r_explinear(X, beta, lambd, n):
    _y = beta[2:] - beta[1:-1]
    return (lambd[0] * np.mean((np.exp(beta[2:]) - np.exp(beta[1:-1])) / _y) -
            lambd[1] * np.mean((np.exp(-beta[2:]) - np.exp(-beta[1:-1])) / _y))


def r_const(X, beta, lambd):
    return lambd[0] * np.exp(beta[1]) + lambd[1] * np.exp(-beta[1])


def r_gauss_1D(X, beta, lambd):
    rv = norm(np.arctan(beta[2]) / np.pi + 0.5, np.exp(beta[3]))
    return (lambd[0] * (rv.cdf(1) - rv.cdf(0) + np.exp(beta[1])) + lambd[1] / (
            rv.cdf(1) - rv.cdf(0) + np.exp(beta[1])) + lambd[2] * norm(-3, 2).pdf(np.exp(beta[3])))
