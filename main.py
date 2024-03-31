import numpy as np
import lib


def solve_via_QUBO(A: np.ndarray, b: np.ndarray, option: str = 'gekko', tol: float = 0.001) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    assert np.size(A, axis=1) == np.size(b)
    n = b.size
    lb = -100 * np.ones(n, dtype='float')
    ub = 100 * np.ones(n, dtype='float')
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
    # print(solve_exact(A, b))
    # print(solve_slsqp(A, b, verbose=True))
    print(solve_via_QUBO(A, b))
