import numpy as np

# Matern smoothness 1/2
def matern_12(x):
    return np.exp(-x)


# Matern smoothness 3/2
def matern_32(x):
    return (1 + np.sqrt(3) * x) * np.exp(-np.sqrt(3) * x)


# Matern smoothness 5/2
def matern_52(x):
    return (1 + np.sqrt(5) * x + 5 / 3 * x ** 2) * np.exp(-np.sqrt(5) * x)


# Gaussian kernel
def gauss(x):
    return np.exp(-x ** 2)