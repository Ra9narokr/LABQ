# LABQ
Locally Adaptive Bayesian Quadrature

Python implementation of **Locally Adaptive Bayesian Quadrature** (LABQ), an adaptive Gaussian‑process–based numerical integration method.

---

## Project Layout

```
LABQ/
├── Algorithm/
│   ├── LA_Bayessian_Quadrature.py   # Core LABQ algorithms (ABC, ABC_D, etc.)
│   ├── Gaussian_Process.py          # Gaussian process model and hyperparameter fitting
│   ├── Kernels.py                   # Kernel definitions and length‑scale tuning
│   ├── Matrix.py                    # Positive definite matrix utilities and incremental Cholesky
│   ├── Mesh.py                      # Sampling strategies (Sobol, Latin hypercube, …)
│   └── kernel_ingres/               # Kernel integration helpers
├── Target_Function.py               # Option portfolio Delta example target function
├── LABQ_*.ipynb                     # Experiments and figures (Jupyter Notebooks)
├── LABQ_*.csv                       # Example experiment results
├── LICENSE                          # MIT license
└── README.md                        # This file
```

---

## Features

- Adaptive Bayesian quadrature driven by posterior variance of a Gaussian process.
- Works for one‑ and multi‑dimensional integrals with Sobol sequence or Latin hypercube candidate points.
- Locally adaptive length scales tailored to each dimension and kernel.
- Includes an option‑portfolio Delta example as a ready‑to‑use integrand.

---

## Installation

Requires **Python 3.9+**.

```bash
pip install numpy scipy matplotlib qmcpy SALib
```

`qmcpy` and `SALib` provide Sobol sequences and sensitivity analysis utilities.

---

## Quick Start

The following snippet estimates a 2‑D integral over the unit square using LABQ:

```python
import numpy as np
from Algorithm.LA_Bayessian_Quadrature import ABC_D
from Algorithm.Gaussian_Process import GaussianProcess
from Algorithm.Kernels import K_lin_mat2_10
from Algorithm.Mesh import get_points_Sobol

# integrand
def f(x):
    return np.exp(-np.sum(x**2))

D = 2
# initial observations
X0 = np.random.rand(5, D)
Y0 = np.array([f(x) for x in X0])

# hyperparameters (amplitude + length scales; adjust for chosen kernel)
beta0 = np.zeros(12)

gp = GaussianProcess(K_lin_mat2_10, beta0, X0, Y0)

gp_list, int_params, times, uncertainties = ABC_D(
    func=f,
    _GP=gp,
    get_points=get_points_Sobol,
    n=20,
    lambd=[30, 1],
    options={"tol": False, "adapt": False}
)

estimate, variance = int_params[-1]
print("Integral ≈", estimate, "±", np.sqrt(variance))
```

- Use `ABC` for one‑dimensional problems.
- For adaptive Sobol sequences or sensitivity‑analysis‑driven sampling, see `LABQ_D_Sobol`.

---

## Option Portfolio Example

`Target_Function.py` supplies analytic Delta formulas for single options and option portfolios. Pass `OptionPortfolio.target_function` as `func` in LABQ algorithms to evaluate the portfolio’s average Delta across the price domain.

---

## Data and Notebooks

- `LABQ_original*.csv`, `LABQ_Sobol*.csv`: experimental results for various dimensions.
- `LABQ_original.ipynb`, `LABQ_for_*.ipynb`, `Figures.ipynb`: for experiments and figures in the paper.

---

## License

Released under the [MIT License](LICENSE).

---