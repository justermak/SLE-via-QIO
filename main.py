import numpy as np
import lib


def solve_via_QUBO(A: np.ndarray, b: np.ndarray, lb: float = -10, ub: float = 10, option: str = 'gekko', tol: float = 0.1) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    n = b.size
    lb = lb * np.ones(n)
    ub = ub * np.ones(n)
    while ub[0] - lb[0] > tol:
        k = (ub - lb) / 2
        d = lb + (ub - lb) / 4
        A_sub, b_sub = lib.substitution_kx_plus_d(A, b, k, d)
        AA, bb = lib.to_QP(A_sub, b_sub)
        x = lib.solve_QUBO(AA + bb, option)
        ub -= k * (1 - x)
        lb += k * x
    return lb


if __name__ == '__main__':
    A = np.array([[-1, 1], [1, 0]])
    b = np.array([2, 3])
    print(lib.solve_exact(A, b))
    print(lib.solve_slsqp(A, b))
    print(solve_via_QUBO(A, b))
