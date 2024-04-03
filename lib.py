import math

import numpy as np
from scipy.optimize import minimize as scp_minimize
from gekko import GEKKO
from dwave_qbsolv import QBSolv
import itertools


MAX_BORDER = 100
EPS = 0.1


def to_QP(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    return A.T @ A, -2 * b.T @ A


def substitution_kx_plus_d(A: np.ndarray, b: np.ndarray, k: np.ndarray, d: np.ndarray) -> (
        np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    assert k.size == np.size(A, axis=1)
    assert d.size == np.size(A, axis=1)
    n = np.size(A, axis=1)
    return A * k, b - A @ d


def QUBO_bruteforce(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert A.shape == A.shape[::-1]
    n = np.size(A, axis=1)
    minimum = np.inf
    argmin = None
    for x in itertools.product(range(2), repeat=n):
        x = np.array(x)
        val = x.T @ A @ x
        if val < minimum:
            minimum = val
            argmin = x
    return argmin


def QUBO_gekko(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert A.shape == A.shape[::-1]
    n = np.size(A, axis=1)
    m = GEKKO(remote=False)
    x = m.Array(m.Var, (n,), value=0, lb=0, ub=1, integer=True)
    m.Obj(x.T @ A @ x)
    try:
        m.solve(disp=False)
    except Exception:
        print(f"gekko unable to solve QUBO")
        return np.zeros(n)
    return np.vectorize(lambda x: x[0])(x)


def QUBO_dwave(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert A.shape == A.shape[::-1]
    n = np.size(A, axis=1)
    response = QBSolv().sample_qubo({(i, j): A[i][j] for (i, j) in itertools.product(range(n), repeat=2)})
    samples = response.samples()
    return np.array([samples[0][i] for i in range(n)])


def solve_QUBO(A: np.ndarray, option: str) -> np.ndarray:
    match option:
        case 'bruteforce':
            return QUBO_bruteforce(A)
        case 'gekko':
            return QUBO_gekko(A)
        case 'dwave':
            return QUBO_dwave(A)
        case _:
            raise NotImplemented()


def solve_exact(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    return np.linalg.solve(A, b)


def solve_slsqp(A: np.ndarray, b: np.ndarray, tol: float = 0.01, verbose: bool = False) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    n = np.size(A, axis=1)
    assert b.size == n
    if verbose:
        print(f"SLSQP solving SLE with A={A}, b={b}")
    AA, bb = to_QP(A, b)
    fun = lambda x, *args: x.T @ AA @ x + bb @ x
    callback = lambda intermediate_result: print(intermediate_result) \
        if verbose else \
        lambda intermediate_result: ...
    result = scp_minimize(fun, np.zeros(n), method='SLSQP', tol=tol, callback=callback)
    if verbose:
        print(f"Exiting SLSQP")
    return result['x']


def solve_DNC_QUBO(A: np.ndarray, b: np.ndarray, lb: np.ndarray = None, ub: np.ndarray = None, option: str = 'bruteforce',
                         tol: float = 0.01, verbose: bool = False) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    n = np.size(A, axis=1)
    assert b.size == n
    if lb is not None:
        assert lb.size == n
    else:
        lb = -MAX_BORDER * np.ones(n)
    if ub is not None:
        assert ub.size == n
    else:
        ub = MAX_BORDER * np.ones(n)
    if verbose:
        print(f"Bisection QUBO solving SLE with A={A}, b={b}, lb={lb}, ub={ub}")
    it = math.ceil(math.log2(max(ub - lb)/tol))
    for i in range(it):
        k = (ub - lb) / 2
        d = lb + (ub - lb) / 4
        A_sub, b_sub = substitution_kx_plus_d(A, b, k, d)
        AA, bb = to_QP(A_sub, b_sub)
        x = solve_QUBO(AA + bb, option)
        ub -= k * (1 - x)
        lb += k * x
        if verbose:
            print(f"x={x}, lb={lb}, ub={ub}")
    if verbose:
        print(f"Exiting Bisection QUBO")
    return lb
