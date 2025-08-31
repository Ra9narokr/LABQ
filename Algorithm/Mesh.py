import numpy as np
from scipy.stats import qmc
from scipy.spatial import Voronoi


def get_points(X):
    X = X.flatten()
    return (np.sort(X)[1:] + np.sort(X)[:-1]) / 2

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


def get_points_LHS(X, ):

    D = X.shape[1]
    sampler = qmc.LatinHypercube(d=D)
    points = sampler.random(n=100)
    return points


def get_points_Sobol(X,p,m = 7):

    D = X.shape[1]
    sampler = qmc.Sobol(d=D, scramble=True)
    points = sampler.random_base2(m = m)
    return points


def get_points_Sobol_D_adaptive(sobol_indices, n_points=8, max_segments=10, min_segments=2):
    """

      Parameters:
          sobol_indices : ，shape = (D,)
          n_points      : Total Points
          max_segments  :
          min_segments  :

      Return:
          X_candidates  : shape = (n_points, D)
      """
    D = len(sobol_indices)
    base_segments = int(np.ceil(n_points ** (1/D)))
    segments_per_dim = np.full(D, base_segments)

    sorted_indices = np.argsort(-sobol_indices)
    for i in sorted_indices[:D//2]:
        segments_per_dim[i] = min(max_segments, segments_per_dim[i] + 2)

    grids = [np.linspace(0, 1, num=s, endpoint=False) + 0.5/s for s in segments_per_dim]
    mesh = np.meshgrid(*grids)
    X_all = np.vstack([m.ravel() for m in mesh]).T

    np.random.shuffle(X_all)
    if n_points > len(X_all):
        raise ValueError(f"Total Points {len(X_all)} is less then{n_points}.")

    return X_all[:n_points]
