import numpy as np
from scipy.integrate import quad, dblquad
from Algorithm.kernel_ingres import integration_funcs as ls

def int_2D(fun, beta, n, simp = False):
    kern = lambda x, y: fun(x, y, beta)
    total = 0
    for i in range(n):
        for j in range(n):
            if simp is False:
                total = total + dblquad(kern, i / n, (i + 1) / n, j / n, (j + 1) / n)[0]
            else:
                total = total + dblquad(kern, i / n, (i + 1) / n, j / n, (j + 1) / n, epsabs=1e-5)[0]
    return total


def int_lin(fun, x, beta, n, simp = False):
    kern = lambda y: fun(y, x, beta)
    total = 0
    for i in range(n):
        if simp is False:
            total = total + quad(kern, i / n, (i + 1) / n)[0]
        else:
            total = total + quad(kern, i / n, (i + 1) / n, epsabs=1e-5)[0]
    return total


def int_pconst_mat12(fun, x, beta, n):
    c_x = ls.l_pconst(x, beta, n)
    total = 0
    for i in range(n):
        c_y = np.exp(beta[i + 1])
        a = np.sqrt(c_y * c_x)
        b = np.sqrt(c_y ** 2 + c_x ** 2)
        exp1 = np.exp(-np.abs(x - (i + 1) / n) / b)
        exp2 = np.exp(-np.abs(x - i / n) / b)
        if x < i / n:
            total = total + a * (-exp1 + exp2)
        elif i / n <= x < (i + 1) / n:
            total = total + a * (2 - (exp1 + exp2))
        else:
            total = total + a * (exp1 - exp2)
    return np.exp(2 * beta[0]) * total


def int_pconst_mat32(fun, x, beta, n):
    c_x = ls.l_pconst(x, beta, n)
    total = 0
    for i in range(n):
        c_y = np.exp(beta[i + 1])
        a = np.sqrt(c_y * c_x)
        b = np.sqrt(c_y ** 2 + c_x ** 2)
        exp1 = np.exp(-np.sqrt(3) * np.abs(x - (i + 1) / n) / b)
        exp2 = np.exp(-np.sqrt(3) * np.abs(x - i / n) / b)
        if x < i / n:
            lin1 = -3 * x + 2 * np.sqrt(3) * b + 3 * (i + 1) / n
            lin2 = -3 * x + 2 * np.sqrt(3) * b + 3 * i / n
            total = total + a / (3 * b) * (-exp1 * lin1 + exp2 * lin2)
        elif i / n <= x < (i + 1) / n:
            lin1 = -3 * x + 2 * np.sqrt(3) * b + 3 * (i + 1) / n
            lin2 = 3 * x + 2 * np.sqrt(3) * b - 3 * i / n
            total = total + 4 * a / np.sqrt(3) + a / (3 * b) * (- (lin1 * exp1 + lin2 * exp2))
        else:
            lin1 = 3 * x + 2 * np.sqrt(3) * b - 3 * (i + 1) / n
            lin2 = 3 * x + 2 * np.sqrt(3) * b - 3 * i / n
            total = total + a / (3 * b) * (exp1 * lin1 - exp2 * lin2)
    return np.exp(2 * beta[0]) * total


def int_const_mat12(fun, x, beta):
    c = np.exp(beta[1])
    return np.exp(2 * beta[0]) * c * (2 - np.exp(-x / c) - np.exp((x - 1) / c))


def int_const_mat32(fun, x, beta):
    c = np.exp(beta[1])
    b = np.sqrt(2) * c
    exp1 = np.exp(-np.sqrt(3) * (1 - x) / b)
    exp2 = np.exp(-np.sqrt(3) * x / b)
    lin1 = -3 * x + 2 * np.sqrt(3) * b + 3
    lin2 = 3 * x + 2 * np.sqrt(3) * b

    return np.exp(2 * beta[0]) * (4 * c / np.sqrt(3) + c / (3 * b) * (- (lin1 * exp1 + lin2 * exp2)))

def int_const(fun, x, beta):
    kern = lambda y: fun(y, x, beta)
    return quad(kern, 0, 1)[0]

