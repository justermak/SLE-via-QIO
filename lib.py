import numpy as np
from scipy.optimize import minimize as scp_minimize
from gekko import GEKKO


def solve_exact(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    return np.linalg.solve(A, b)


def to_QP(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    return A.T @ A, -2 * b.T @ A


def solve_slsqp(A: np.ndarray, b: np.ndarray, tol: float = 0.001, verbose: bool = False) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    if verbose:
        print(f"SLSQP solving SLE with A={A}, b={b}")
    n = np.size(A, axis=1)
    AA, bb = to_QP(A, b)
    fun = lambda x, *args: x.T @ AA @ x + bb @ x
    callback = lambda intermediate_result: print(intermediate_result) \
        if verbose else \
        lambda intermediate_result: ...
    result = scp_minimize(fun, np.zeros(n), method='SLSQP', tol=tol, callback=callback)
    if verbose:
        print(f"Exiting SLSQP")
    return result['x']


def QUBO_gekko(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert A.shape == A.shape[::-1]
    n = np.size(A, axis=1)
    m = GEKKO()
    x = m.Array(m.Var, (n,), value=0, lb=0, ub=1, integer=True)
    m.Obj(x.T @ A @ x)
    m.solve(disp=False)
    return np.vectorize(lambda x: x[0])(x)


def solve_QUBO(A: np.ndarray, option: str = 'gekko') -> np.ndarray:
    match option:
        case 'gekko':
            return QUBO_gekko(A)
        case _:
            raise NotImplemented()


def substitution_kx_plus_d(A: np.ndarray, b: np.ndarray, k: np.ndarray, d: np.ndarray) -> (
        np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    assert k.size == np.size(A, axis=1)
    assert d.size == np.size(A, axis=1)
    n = np.size(A, axis=1)
    return A * k, b - A @ d

