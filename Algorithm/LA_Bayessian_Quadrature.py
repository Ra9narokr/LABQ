from os import times

from .Gaussian_Process import beta_processing
from .Matrix import block_cholesky, to_SPDM
from .Mesh import get_points_Sobol_D_adaptive

import copy
import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from qmcpy import Sobol
from SALib.analyze import sobol as analyze_sobol
from SALib.sample import sobol as sample_sobol



# Checks if optimisation of kernel parameters is necessary (only if new point is unexpected)
def get_optim(GP, new_x, new_y, chol, K_Xx, tol):
    solved = la.solve_triangular(chol, K_Xx, check_finite=False, lower=True)
    mu = np.dot(la.solve_triangular(chol, GP.Y, check_finite=False, lower=True),
                la.solve_triangular(chol, K_Xx, check_finite=False, lower=True))
    var = GP.kernel(new_x, new_x, GP.beta) - np.dot(solved, solved)
    z = (new_y - mu) / np.sqrt(var)

    if norm.cdf(np.abs(z)) > (1 + tol) / 2:
        return True
    else:
        return False


# Calculates integral parameters using mean and variance vector
def get_int_param(GP, mean_ints, chol):
    solved = la.solve_triangular(chol, mean_ints, check_finite=False, lower=True)

    mean_int = np.dot(la.solve_triangular(chol, GP.Y, check_finite=False, lower=True), solved)  # mean

    var_int = 1
    if GP.D == 1:
        var_int = GP.Kernel.int_2D(GP.kernel, GP.beta) - np.dot(solved, solved)  # variance
    else:
        beta_D = beta_processing(GP.beta, GP.D)
        for i in range(GP.D):
            var_int = var_int * GP.Kernel.int_2D(GP.kernel, beta_D[i])
        var_int = var_int - np.dot(solved, solved)  # variance

    return np.array([mean_int, var_int])


# ABC algorithm in 1D
def LABQ(func, _GP, get_points, n, lambd=[30, 1], point_mesh=False,
        options={"tol": False, "adapt": False, "info": False}):
    GP_list = []
    GP = copy.deepcopy(_GP)
    D = GP.D

    tol = options.get("tol")
    if tol is None:
        tol = False
    adapt = options.get("adapt")
    if adapt is None:
        adapt = False
    info = options.get("info")
    if info is None:
        info = False


    int_params = []  # list of integral estimates

    if tol == False and info == True:
        print("No threshold given, optimising beta at each step.")
    optim = True  # only if optim is true and we have a given tolerance we optimise the beta

    if point_mesh is not False:
        point_set = point_mesh


    for i in range(n + 1):  # The main loop
        print("Step: ", i + 1, " / ", n + 1)
        if optim is True or tol is False:
            if info == True and i > 0 and tol != False:
                print("Optimising beta since new point outside tolerance ")
            GP.beta = GP.fit(GP.beta, lambd, adapt=adapt, info = info)
            GP.cov_matrix = GP.cov_matrix_(GP.mesh, GP.mesh, GP.beta)
        elif info == True:
            print("Not optimising beta since new point within tolerance.")

        GP_list.append(copy.deepcopy(GP))


        if point_mesh is False:
            point_set = get_points(GP.X)

        kern1D_X_ints = np.zeros(len(GP.X) + 1)

        vars_i = []

        for j in range(len(GP.X)):
            kern1D_X_ints[j] = GP.Kernel.int_kern(GP.kernel, GP.X[j][0], GP.beta)  # this cannot be simplified otherwise our integral estimate would be incorrect

        mean_ints = kern1D_X_ints[:-1]
        kern = to_SPDM(GP.cov_matrix_(GP.X, GP.X, GP.beta))
        chol = la.cholesky(kern, lower=True)

        # mean and variance of integral
        solved = la.solve_triangular(chol, mean_ints, check_finite=False, lower=True)
        mean_int = np.dot(la.solve_triangular(chol, GP.Y, check_finite=False, lower=True), solved)
        var_int = GP.Kernel.int_2D(GP.kernel, GP.beta) - np.dot(solved, solved)
        int_params.append(np.array([mean_int, var_int]))

        # if at last step then no need to find new point!
        if i == n:
            break

        for j in point_set:
            kern1D_X_ints[-1] = GP.Kernel.int_kern(GP.kernel, j,
                                                   GP.beta)  # This is the step we can reduce computation on integrals
            X_j = np.concatenate((GP.X, np.array([[j]])))
            K_Xj = GP.cov_matrix_(np.array([j]), X_j, GP.beta).flatten()
            chol_j = block_cholesky(chol, K_Xj)  # Computes cholesky decomp using cholesky decomp of previous kern matrix
            solved = la.solve_triangular(chol_j, kern1D_X_ints, check_finite=True,
                                         lower=True)  # This could break if points are not removed!
            var_j = -np.dot(solved, solved)

            vars_i.append(var_j)  # this is for plotting costs


        new_x = point_set[np.argmin(vars_i)]
        if info == True:
            print(new_x)
            print(func(new_x))
        new_y = func(new_x)

        if not isinstance(new_y, float):  # checks if output is an integer or a numpy array with one element in
            new_y = new_y[0]

        if point_mesh is not False:
            point_set = get_points(point_set, new_x)
            print(point_set)

        # Check if optimisation is required:
        if tol != False:
            optim = get_optim(GP, new_x, new_y, chol, GP.cov_matrix_(np.array([new_x]), GP.X, GP.beta).flatten(), tol)

        # Update Gaussian process data:
        GP.X = np.concatenate((GP.X, np.array([[new_x]])))
        GP.Y = np.concatenate((GP.Y, np.array([new_y])))

    return GP_list, np.array(int_params)




# Performs ABC method in higher dimensions with assumed tensor product kernel
def LABQ_D(func, _GP, get_points, n, lambd=[30, 1],
          point_mesh=False,
          options={"tol": False, "n_subset": "", "adapt": False}):
    GP_list = []  # List of Gaussian processes used in output (a GP object for each i = 1,...,N)
    int_params = []  # list of posterior integral parameters (mean and variance) for each GP
    times_used = []
    uncertainties = []
    GP = copy.deepcopy(_GP)  # To ensure immutability of input GP
    D = GP.D
    print("Dimensions:", D, ", Steps:", n+1)

    tol = options.get("tol")
    if tol is None:
        tol = False
    n_subset = options.get("n_subset")
    if n_subset is None:
        n_subset = ""
    adapt = options.get("adapt")
    if adapt is None:
        adapt = False

    optim = True  # only if optim is true and we have a given tolerance we optimise the beta

    if point_mesh is not False:
        point_set = point_mesh
    start_time = time.time()
    for i in range(n + 1):  # The main loop
        check_point_start = time.time()
        if optim == True or tol == False:
            if i > 0 and tol != False:
                print("Optimising beta since new point outside tolerance ")
            GP.beta = GP.fit(GP.beta, lambd, adapt=adapt)
            GP.cov_matrix = GP.cov_matrix_(GP.mesh, GP.mesh, GP.beta)
        else:
            print("Not optimising beta since new point within tolerance")

        GP_list.append(copy.deepcopy(GP))


        if point_mesh is False:
            point_set = get_points(GP.X, point_set)

        # Calculate all integrals at once:

        kernD_X_ints = np.zeros(len(GP.X) + 1) + 1
        point_set_ints = np.zeros(len(point_set)) + 1

        beta_D = beta_processing(GP.beta, D)
        X_T = GP.X.T
        p_T = point_set.T
        for j in range(D):
            X_j = X_T[j]
            intsX_j = np.concatenate((X_j, np.array([0])))
            for k in np.unique(X_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])
                intsX_j = np.where(intsX_j == k, int_k, intsX_j)
            kernD_X_ints = kernD_X_ints * intsX_j

            p_j = p_T[j]
            intsp_j = p_j
            for k in np.unique(p_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])  # could potentially make this cheaper
                intsp_j = np.where(intsp_j == k, int_k, intsp_j)
            point_set_ints = point_set_ints * intsp_j

        chol = la.cholesky(to_SPDM(GP.cov_matrix_(GP.X, GP.X, GP.beta)), lower=True)

        int_param = get_int_param(GP, kernD_X_ints[:-1], chol)
        int_params.append(int_param)  # Add integral parameters to int_params

        uncertainty = np.sqrt(int_param[1])/(int_param[0])
        uncertainties.append(uncertainty)

        check_point_end = time.time()
        times_used.append(check_point_end - check_point_start)

        print(f"Step {i+1} of {n+1}, Time: {check_point_end - check_point_start:.2f}, "
              f"Result: {int_param}, Uncertainty Rate: {uncertainty*100:.2f}%")


        # if at last step then no need to find new point!
        if i == n:
            break

        # Compute optimisation on point set:
        vars_i = []
        if isinstance(n_subset, int):
            subset = np.random.choice(len(point_set), n_subset, replace=False)
            for j in subset:
                kernD_X_ints[-1] = point_set_ints[j]
                X_j = np.concatenate((GP.X, np.array([point_set[j]])))

                K_Xj = GP.cov_matrix_(np.array([point_set[j]]), X_j, GP.beta).flatten()
                chol_j = block_cholesky(chol, K_Xj)
                solved = la.solve_triangular(chol_j, kernD_X_ints, check_finite=False, lower=True)
                var_j = -np.dot(solved, solved)

                vars_i.append(var_j)
            n_subset = n_subset - 1
            new_x = point_set[subset[np.argmin(vars_i)]]
            new_y = func(new_x)
        else:
            for j in range(len(point_set)):
                kernD_X_ints[-1] = point_set_ints[j]
                X_j = np.concatenate((GP.X, np.array([point_set[j]])))

                K_Xj = GP.cov_matrix_(np.array([point_set[j]]), X_j, GP.beta).flatten()
                chol_j = block_cholesky(chol, K_Xj)
                solved = la.solve_triangular(chol_j, kernD_X_ints, check_finite=False, lower=True)
                var_j = -np.dot(solved, solved)

                vars_i.append(var_j)
            new_x = point_set[np.argmin(vars_i)]
            new_y = func(new_x)

        # Check if optimisation is required:
        if tol != False:
            optim = get_optim(GP, new_x, new_y, chol, GP.cov_matrix_(np.array([new_x]), GP.X, GP.beta).flatten(), tol)

        # Add new point to Gaussian process:
        GP.X = np.concatenate((GP.X, np.array([new_x])))
        GP.Y = np.concatenate((GP.Y, np.array([new_y])))
        end_time = time.time()
    totaltime = end_time - start_time
    print(f"{D}Dimensions Total_Time: {totaltime:.2f}")

    return GP_list, np.array(int_params), times_used, uncertainties


def LABQ_D_Sobol(func, _GP, get_points, rate, min_n, max_n, point_mesh,
                lambd=[30, 1], sobol = 7):

    GP_list = []  # List of Gaussian processes used in output (a GP object for each i = 1,...,N)
    int_params = []  # list of posterior integral parameters (mean and variance) for each GP
    times_used = []
    uncertainties = []
    GP = copy.deepcopy(_GP)  # To ensure immutability of input GP
    D = GP.D
    print("Dimensions:", D, ", Sobol:", sobol,)

    optim = True  # only if optim is true and we have a given tolerance we optimise the beta

    point_set = point_mesh
    start_time = time.time()

    uncertainty = 1
    n = 0

    while True:  # The main loop
        check_point_start = time.time()
        if optim == True:
            GP.beta = GP.fit(GP.beta, lambd)
            GP.cov_matrix = GP.cov_matrix_(GP.mesh, GP.mesh, GP.beta)

        GP_list.append(copy.deepcopy(GP))



        # Calculate all integrals at once:

        kernD_X_ints = np.zeros(len(GP.X) + 1) + 1
        point_set_ints = np.zeros(len(point_set)) + 1

        beta_D = beta_processing(GP.beta, D)
        X_T = GP.X.T
        p_T = point_set.T
        for j in range(D):
            X_j = X_T[j]
            intsX_j = np.concatenate((X_j, np.array([0])))
            for k in np.unique(X_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])
                intsX_j = np.where(intsX_j == k, int_k, intsX_j)
            kernD_X_ints = kernD_X_ints * intsX_j

            p_j = p_T[j]
            intsp_j = p_j
            for k in np.unique(p_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])  # could potentially make this cheaper
                intsp_j = np.where(intsp_j == k, int_k, intsp_j)
            point_set_ints = point_set_ints * intsp_j

        chol = la.cholesky(to_SPDM(GP.cov_matrix_(GP.X, GP.X, GP.beta)), lower=True)

        int_param = get_int_param(GP, kernD_X_ints[:-1], chol)
        int_params.append(int_param)  # Add integral parameters to int_params

        uncertainty = np.sqrt(int_param[1])/(int_param[0])
        uncertainties.append(uncertainty)

        check_point_end = time.time()
        times_used.append(check_point_end - check_point_start)

        print(f"Step {n+1}, Time: {check_point_end - check_point_start:.2f}, "
              f"Result: {int_param}, Uncertainty Rate: {uncertainty*100:.2f}%")


        # if at last step then no need to find new point!
        if ( uncertainty < rate or n > max_n ) and n > min_n:
            break

        # Compute optimisation on point set:
        vars_i = []

        for j in range(len(point_set)):
            kernD_X_ints[-1] = point_set_ints[j]
            X_j = np.concatenate((GP.X, np.array([point_set[j]])))

            K_Xj = GP.cov_matrix_(np.array([point_set[j]]), X_j, GP.beta).flatten()
            chol_j = block_cholesky(chol, K_Xj)
            solved = la.solve_triangular(chol_j, kernD_X_ints, check_finite=False, lower=True)
            var_j = -np.dot(solved, solved)

            vars_i.append(var_j)
        new_x = point_set[np.argmin(vars_i)]
        new_y = func(new_x)

        # Check if optimisation is required:

        point_set = get_points(point_set, new_x, sobol)

        # Add new point to Gaussian process:
        GP.X = np.concatenate((GP.X, np.array([new_x])))
        GP.Y = np.concatenate((GP.Y, np.array([new_y])))
        end_time = time.time()
        n = n + 1
    totaltime = end_time - start_time
    print(f"{D}Dimensions Total_Time: {totaltime:.2f}")

    return GP_list, np.array(int_params), times_used, uncertainties


def LABQ_D_alter2(func, _GP, rate, min_n, max_n):
    lambd = [30, 1]
    GP_list = []  # List of Gaussian processes used in output (a GP object for each i = 1,...,N)
    int_params = []  # list of posterior integral parameters (mean and variance) for each GP
    times_used = []
    uncertainties = []
    GP = copy.deepcopy(_GP)  # To ensure immutability of input GP
    D = GP.D
    print("Dimensions:", D)


    start_time = time.time()

    uncertainty = 1
    n = 0
    problem = {
        'num_vars': D,
        'names': [f"x{i + 1}" for i in range(D)],
        'bounds': [[0, 1]] * D
    }
    N = 32
    param_values = sample_sobol.sample(problem, N, calc_second_order=False)

    result_values = np.array([func(x) for x in param_values])
    Si = analyze_sobol.analyze(problem, result_values, calc_second_order=False)
    print("一阶Sobol指数：", Si['S1'])
    point_set = get_points_Sobol_D_adaptive(Si['S1'])

    while True:  # The main loop
        check_point_start = time.time()
        GP.beta = GP.fit(GP.beta, lambd)
        GP.cov_matrix = GP.cov_matrix_(GP.mesh, GP.mesh, GP.beta)

        GP_list.append(copy.deepcopy(GP))


        # Calculate all integrals at once:

        kernD_X_ints = np.zeros(len(GP.X) + 1) + 1
        point_set_ints = np.zeros(len(point_set)) + 1

        beta_D = beta_processing(GP.beta, D)
        X_T = GP.X.T
        p_T = point_set.T
        for j in range(D):
            X_j = X_T[j]
            intsX_j = np.concatenate((X_j, np.array([0])))
            for k in np.unique(X_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])
                intsX_j = np.where(intsX_j == k, int_k, intsX_j)
            kernD_X_ints = kernD_X_ints * intsX_j

            p_j = p_T[j]
            intsp_j = p_j
            for k in np.unique(p_j):
                int_k = GP.Kernel.int_kern(GP.kernel, k, beta_D[j])  # could potentially make this cheaper
                intsp_j = np.where(intsp_j == k, int_k, intsp_j)
            point_set_ints = point_set_ints * intsp_j

        chol = la.cholesky(to_SPDM(GP.cov_matrix_(GP.X, GP.X, GP.beta)), lower=True)

        int_param = get_int_param(GP, kernD_X_ints[:-1], chol)
        int_params.append(int_param)  # Add integral parameters to int_params

        uncertainty = np.sqrt(int_param[1])/(int_param[0])
        uncertainties.append(uncertainty)

        check_point_end = time.time()
        times_used.append(check_point_end - check_point_start)

        print(f"Step {n+1}, Time: {check_point_end - check_point_start:.2f}, "
              f"Result: {int_param}, Uncertainty Rate: {uncertainty*100:.2f}%")


        # if at last step then no need to find new point!
        if ( uncertainty < rate or n > max_n ) and n > min_n:
            break

        # Compute optimisation on point set:
        vars_i = []

        for j in range(len(point_set)):
            kernD_X_ints[-1] = point_set_ints[j]
            X_j = np.concatenate((GP.X, np.array([point_set[j]])))

            K_Xj = GP.cov_matrix_(np.array([point_set[j]]), X_j, GP.beta).flatten()
            chol_j = block_cholesky(chol, K_Xj)
            solved = la.solve_triangular(chol_j, kernD_X_ints, check_finite=False, lower=True)
            var_j = -np.dot(solved, solved)

            vars_i.append(var_j)
        new_x = point_set[np.argmin(vars_i)]
        new_y = func(new_x)

        # Check if optimisation is required:

        point_set = get_points_Sobol_D_adaptive(Si['S1'])

        # Add new point to Gaussian process:
        GP.X = np.concatenate((GP.X, np.array([new_x])))
        GP.Y = np.concatenate((GP.Y, np.array([new_y])))
        end_time = time.time()
        n = n + 1
    totaltime = end_time - start_time
    print(f"{D}Dimensions Total_Time: {totaltime:.2f}")

    return GP_list, np.array(int_params), times_used, uncertainties
