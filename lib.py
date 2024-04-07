import math

import numpy as np
from scipy.optimize import minimize as scp_minimize
from gekko import GEKKO
from dwave.samplers import SimulatedAnnealingSampler
import itertools

max_border = 1e5
eps = 1e-5
base_prec = 3


def to_QP(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    return A.T @ A, -2 * b.T @ A


def substitute_kx_plus_d(A: np.ndarray, b: np.ndarray, k: np.ndarray, d: np.ndarray) -> (
        np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    assert k.size == m
    assert d.size == m
    AA = A * k
    bb = b - A @ d
    return AA, bb


def substitute_linear_form(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> (np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    assert c.shape[0] == m
    # x = c[0] + c[1] * x_1 + ... + c[n-1] * x_{n-1}
    n = np.size(A, axis=1)
    k = np.size(c, axis=1) - 1
    const, coef = c[:, 0], c[:, 1:]
    AA = np.repeat(A, k, axis=1) * coef.flatten()
    bb = b - A @ const
    return AA, bb


def extract_solution_from_substitution(res: np.ndarray, c: np.ndarray) -> np.ndarray:
    assert res.ndim == 1
    assert c.ndim == 2
    n = np.size(c, axis=0)
    k = np.size(c, axis=1) - 1
    assert res.size == n * k
    const, coef = c[:, 0], c[:, 1:]
    res.reshape((n, k))
    vals = const + (coef * res.reshape(n, k)).sum(axis=1)
    return vals


def QUBO_bruteforce(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    n, m = A.shape
    assert n == m
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
    n, m = A.shape
    assert n == m
    solver = GEKKO(remote=False)
    x = solver.Array(solver.Var, (n,), lb=0, ub=1, integer=True)
    solver.Obj(x.T @ A @ x)
    try:
        solver.solve(disp=False)
    except Exception:
        print(f"gekko can't solve QUBO")
        return np.zeros(n)
    return np.vectorize(lambda v: v[0])(x)


def QUBO_dwave(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    n, m = A.shape
    assert n == m
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo({(i, j): A[i][j] for (i, j) in itertools.product(range(n), repeat=2)})
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


def solve_np(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    assert n == m, "A should be square matrix"
    return np.linalg.solve(A, b)


def solve_slsqp(A: np.ndarray, b: np.ndarray, tol: float = 0.01, verbose: bool = False) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    if verbose:
        print(f"SLSQP solving SLE with\nA={A}\nb={b}")
    AA, bb = to_QP(A, b)

    def fun(x, *args):
        return (x.T @ AA + bb) @ x

    def callback(xk):
        if verbose:
            print(f"xk={xk}")

    result = scp_minimize(fun, np.zeros(m), method='SLSQP', tol=tol, callback=callback)
    if verbose:
        print(f"Exiting SLSQP")
    return result['x']


def solve_reference(A: np.ndarray, b: np.ndarray, option: str) -> np.ndarray:
    match option:
        case 'np':
            return solve_np(A, b)
        case 'slsqp':
            return solve_slsqp(A, b)
        case _:
            raise NotImplemented()


def solve_DNC_QUBO(A: np.ndarray, b: np.ndarray, option: str = 'bruteforce', lb: np.ndarray = None,
                   ub: np.ndarray = None, tol: float = eps, verbose: bool = False) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    if lb is not None:
        assert lb.size == m
    else:
        lb = -max_border * np.ones(m)
    if ub is not None:
        assert ub.size == m
    else:
        ub = max_border * np.ones(m)
    if verbose:
        print(f"DNC QUBO solving SLE with\nA={A}\nb={b}\nlb={lb}\nub={ub}\ntol={tol}")
    it = math.ceil(math.log2(max(ub - lb) / tol))
    for i in range(it):
        k = (ub - lb) / 2
        d = lb + (ub - lb) / 4
        A_sub, b_sub = substitute_kx_plus_d(A, b, k, d)
        AA, bb = to_QP(A_sub, b_sub)
        x = solve_QUBO(AA + np.diag(bb), option)
        ub -= k * (1 - x)
        lb += k * x
        if verbose:
            print(f"iter={i}\nk={k}\nd={d}\nA_sub={A_sub}\nb_sub={b_sub}\nAA={AA}\nbb={bb}\nx={x}\nub={ub}\nlb={lb}\n")
    if verbose:
        print(f"Exiting DNC QUBO")
    return lb


def to_finite_precision_form(lb: np.ndarray, ub: np.ndarray, prec: int) -> (np.ndarray, np.ndarray):
    assert lb.ndim == 1
    assert ub.ndim == 1
    assert lb.size == ub.size
    assert prec > 0
    n = lb.size
    k = np.log2((ub - lb)).astype(int) + 1
    c = np.array([[lb[i]] + [2 ** j for j in range(k[i] - prec, k[i])] for i in range(n)])
    return c


def solve_one_step_QUBO(A: np.ndarray, b: np.ndarray, option: str = 'bruteforce', lb: np.ndarray = None,
                        ub: np.ndarray = None, prec: int = None, verbose: bool = False) -> np.ndarray:
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    if lb is not None:
        assert lb.size == m
    else:
        lb = -max_border * np.ones(m)
    if ub is not None:
        assert ub.size == m
    else:
        ub = max_border * np.ones(m)
    if prec is not None:
        assert prec > 0
    else:
        prec = base_prec
    if verbose:
        print(f"One step QUBO solving SLE with\nA={A}\nb={b}\nlb={lb}\nub={ub}\nprec={prec}")
    c = to_finite_precision_form(lb, ub, prec)
    A_sub, b_sub = substitute_linear_form(A, b, c)
    AA, bb = to_QP(A_sub, b_sub)
    x = solve_QUBO(AA + np.diag(bb), option)
    vals = extract_solution_from_substitution(x, c)
    if verbose:
        print(f"c={c}\nA_sub={A_sub}\nb_sub={b_sub}\nAA={AA}\nbb={bb}\nx={x}\nvals={vals}\n")
    if verbose:
        print(f"Exiting one step QUBO")
    return vals
