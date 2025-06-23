import numpy as np
from scipy.stats import qmc

### 1D get points function


def get_points(X):
    X = X.flatten()
    return (np.sort(X)[1:] + np.sort(X)[:-1]) / 2


# D>1 get points function - uses Voronoi cells
from scipy.spatial import Voronoi


def get_points_D(X, p):
    verts = Voronoi(X).vertices
    pointSet = []
    for i in verts:
        inCube = True
        for j in i:
            if j < 0 or j > 1:
                inCube = False
                break
        if inCube:
            pointSet.append(i)
    return np.array(pointSet)


# initial point sets for recursive points
points_2D = np.array(
    [[0, 0.25], [0, 0.5], [0, 0.75], [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1], [0.5, 0],
     [0.5, 0.25], [0.5, 0.75], [0.5, 1], [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1], [1, 0.25],
     [1, 0.5], [1, 0.75]])
mesh_2D = np.array(np.meshgrid(np.arange(0, 1.02, 0.02), np.arange(0, 1.02, 0.02))).reshape(2, -1).T


# add "bisected" points
def new_points_D(point_set, new_x):
    point_set = point_set[np.any(point_set != new_x, axis=1)]
    D = len(new_x)

    if D == 2:
        S = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])
        ll = np.array([0, 0])
        ur = np.array([1, 1])
    if D == 3:
        S = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])
        ll = np.array([0, 0, 0])
        ur = np.array([1, 1, 1])

    m = 0
    for i in new_x:
        n = len(str(i)) - 2
        if n > m:
            m = float(n)

    new_points = np.round(new_x + S * 2 ** (-m - 1), 10)

    inidx = np.all(np.logical_and(ll <= new_points, new_points <= ur), axis=1)
    new_points = new_points[inidx]

    return np.unique(np.concatenate((point_set, new_points)), axis=0)


# mesh search
def mesh_points_D(point_set, new_x):
    return point_set[np.any(point_set != new_x, axis=1)]


# mesh search 1D
def mesh_points_1(point_set, new_x):
    return point_set[point_set != new_x]




def get_points_lhs(X, n_points=100, bounds=None):
    """
    使用 Latin Hypercube 生成一组新点。

    参数:
        X         : 当前已采样点（shape = [N, D]）
        n_points  : 生成多少个候选点
        bounds    : 每一维的取值范围，形如 [[a1, b1], [a2, b2], ..., [aD, bD]]
                    如果为 None，默认用 [0, 1]^D
    返回:
        points    : shape = [n_points, D] 的 numpy 数组
    """

    D = X.shape[1]  # 自动识别维度
    sampler = qmc.LatinHypercube(d=D)
    points = sampler.random(n=n_points)

    if bounds is None:
        return points  # 默认 [0,1]^D

    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    return qmc.scale(points, l_bounds, u_bounds)


def get_points_sobol(X, n_points=128, bounds=None, scramble=False):
    """
    使用 Sobol 序列生成一组新点。

    参数:
        X         : 当前已采样点（shape = [N, D]）
        n_points  : 生成多少个候选点（必须是 2^m）
        bounds    : 每一维的取值范围，形如 [[a1, b1], ..., [aD, bD]]
        scramble  : 是否打乱序列（推荐 True）
    返回:
        points    : shape = [n_points, D] 的 numpy 数组
    """

    D = X.shape[1]
    m = int(np.log2(n_points))
    if 2 ** m != n_points:
        raise ValueError("n_points for Sobol must be a power of 2 (e.g., 128, 256, etc.)")

    sampler = qmc.Sobol(d=D, scramble=scramble)
    points = sampler.random_base2(m=m)

    if bounds is None:
        return points

    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    return qmc.scale(points, l_bounds, u_bounds)
