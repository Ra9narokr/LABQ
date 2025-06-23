import numpy as np
import scipy.linalg as la

def is_PDM(matrix):
    """
    检查输入矩阵是否为正定矩阵
    Checks if input matrix is Positive-Definite Matrix
    """

    try:
        _ = la.cholesky(matrix)
        return True

    except la.LinAlgError:
        return False

def to_SPDM(Matrix):
    """
    基于弗罗贝尼乌斯范数计算最近的半正定矩阵
    算法基于尼克·海姆的“计算最近的相关矩阵——一个来自金融的问题”
    Calculates nearest semi-positive definite matrix w.r.t. the Frobenius norm
    algorithm based on Nick Higham's "Computing the nearest correlation matrix - a problem from finance"

    在高斯过程中，反复用到的协方差矩阵K必须是半正定的，以确保：
    - 能做 Cholesky 分解（即 K = L L^T）
    - 能定义多元高斯分布（用于最大似然和采样）
    In the Gaussian process, the covariance matrix K that is used repeatedly must be semi-positive definite to ensure that:
    - Cholesky decomposition can be performed (i.e., K = L L^T)
    - Multivariate Gaussian distribution can be defined (used for maximum likelihood and sampling)

    但由于数值误差，有时 K 可能会“看起来不是半正定”，这时我们就要强制修正它。

    However, due to numerical errors, K may sometimes “appear not to be semi-positive definite.”
    In this case, we must force it to be corrected.
    """

    M = (Matrix + Matrix.T) / 2
    _, s, V = la.svd(M)
    K = ( M + np.dot(V.T, np.dot(np.diag(s), V)) ) / 2
    Result = (K + K.T) / 2

    if is_PDM(Result):
        return Result + 1e-8 * np.eye(Matrix.shape[0])

    spacing = np.spacing(la.norm(Matrix))

    k = 1
    while not is_PDM(Result):
        mineig = np.min(np.real(la.eigvals(Result)))
        Result += np.eye(Matrix.shape[0]) * (-mineig * k ** 2 + spacing)
        k += 1
    distance = la.norm(Matrix - Result, ord='fro') / la.norm(Result, ord='fro')
    if distance > 10:
        print("Matrix to SPDM failed, distance in Frobenius norm: ", distance)
    return Result + 1e-8 * np.eye(Matrix.shape[0])

def block_cholesky(L,x):
    """
    增量式更新 Cholesky 分解
    用于在加入新样本点之后快速更新原来的协方差矩阵分解结果
    Incrementally updating the Cholesky decomposition
    quickly update the original covariance matrix decomposition results after new sample points are added

    在每一步迭代采样新点, 为了避免重复计算 Cholesky 分解，我们希望用上一步的结果 L 快速更新。

    At each iteration of sampling new points,
    we want to quickly update with the result L from the previous step,
    in order to avoid double-counting the Cholesky decomposition.
    """

    B = x[:-1]
    d = x[-1]
    tri = la.solve_triangular(L, B, check_finite = False, lower = True)
    return(np.block([
        [L, np.zeros((len(B),1))],
        [tri,np.sqrt(d - np.dot(tri,tri))]
    ]))