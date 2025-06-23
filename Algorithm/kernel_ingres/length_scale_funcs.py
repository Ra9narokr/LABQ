import numpy as np
from scipy.stats import norm



def l_linear(x, beta, n):
    """
    linear interpolation of n points uniformly spread out
    对均匀分布的n个点进行线性插值。
    """
    cond_list = [np.logical_and(x >= i / n, x <= (i + 1) / n) for i in range(n)]
    choice_list = [n * (np.exp(beta[i + 1]) * ((i + 1) / n - x) + np.exp(beta[i + 2]) * (x - (i) / n)) for i in
                   range(n)]
    return np.select(cond_list, choice_list)



def l_pconst(x, beta, n):
    """
    piecewise constant of n points uniformly spread out
    由 n 个均匀分布的点组成的分段常数。
    """
    cond_list = [np.logical_and(x >= i / n, x <= (i + 1) / n) for i in range(n)]
    choice_list = [np.exp(beta[i + 1]) for i in range(n)]
    return np.select(cond_list, choice_list)


def l_explinear(x, beta, n):
    """
    piecewise distributed points of n points uniformly spread out
    由 n 个均匀分布的点组成的分段指数插值。
    """
    cond_list = [np.logical_and(x >= i / n, x <= (i + 1) / n) for i in range(n)]
    choice_list = [np.exp(n * (beta[i + 1] * ((i + 1) / n - x) + beta[i + 2] * (x - (i) / n))) for i in range(n)]
    return np.select(cond_list, choice_list)


def l_const(x, beta):
    """
    常数
    constant
    """
    return x - x + np.exp(beta[1])


def l_gauss_1D(x, beta):
    rv = norm(np.arctan(beta[2]) / np.pi + 0.5, np.exp(beta[3]))
    return 1 / (np.exp(beta[1]) + rv.pdf(x))