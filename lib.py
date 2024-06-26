import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
import itertools


rng = np.random.default_rng()
default_niter = 1
bound = 1000
eps = 1e-1


def to_QP(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    AA = A.T @ A
    bb = -2 * b.T @ A
    return AA, bb


def eval(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return (x.T @ A + b) @ x


def substitute(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> (np.ndarray, np.ndarray):
    k = c.shape[1]
    k -= 1
    # obj: x^T A x + b^T x
    # x_i = c[i][0] + c[i][1] * x'_{i 1} + ... + c[i][m-1] * x'_{i m-1}
    const, coef = c[:, 0].reshape(1, -1), c[:, 1:].reshape(1, -1)
    AA = coef.T * np.repeat(np.repeat(A, k, axis=1), k, axis=0) * coef
    bb = ((np.repeat(b, k, axis=0) + (2 * np.repeat(A, k, axis=0) * const).sum(axis=1)) * coef).ravel()
    return AA, bb


def substitute_shift(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # obj: x^T A x + b^T x
    # x_i = c[i] + x'_i
    bb = (b + (2 * A * c.reshape(1, -1)).sum(axis=1)).ravel()
    return bb


def extract_solution_box(q: np.ndarray, prec: int) -> np.ndarray:
    q = q.reshape(-1, prec)
    powers = np.array([2 ** i for i in range(prec)])
    vals = (q * powers).sum(axis=1)
    return vals


def solve_QUBO(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo({(i, j): A[i][j] for (i, j) in itertools.product(range(n), repeat=2)})
    samples = response.samples()
    return np.array([samples[0][i] for i in range(n)])


def solve_one(A: np.ndarray, b: np.ndarray, prec: int, neighbors: int, lb: np.ndarray, ub: np.ndarray, tol: float,
              verbose: bool) -> np.ndarray:
    if verbose:
        print(f"Solving\nA={A}\nb={b}\n")
    it = 0
    while (ub - lb).max() > tol:
        it += 1
        lengths = (ub - lb) / 2 ** prec
        const = lengths / 2 + lb
        coefs = np.array([2 ** k for k in range(prec)]).reshape(1, -1) * lengths.reshape(-1, 1)
        c = np.hstack((const[:, None], coefs))
        AA, bb = substitute(A, b, c)
        q = solve_QUBO(AA + np.diag(bb))
        box = extract_solution_box(q, prec)
        lbd = np.clip(box - neighbors, 0, 2 ** prec - 1)
        ubd = np.clip(2 ** prec - 1 - box - neighbors, 0, 2 ** prec - 1)
        lb += lbd * lengths
        ub -= ubd * lengths
        if verbose:
            print(f"it={it}\nc={c}\nq={q}\nbox={box}\nlbd={lbd}\nubd={ubd}\nlb={lb}\nub={ub}\n")
    return lb


def solve_iter(A: np.ndarray, b: np.ndarray, prec: int, neighbors: int, niter: int, lb: np.ndarray, ub: np.ndarray,
               tol: float, verbose: bool) -> np.ndarray:
    n = A.shape[0]
    res = [np.zeros(n)]
    if verbose:
        print(f"Solving iteratively\nA={A}\nb={b}\n")
    for iter in range(niter):
        res += [solve_one(A, b, prec, neighbors, lb, ub, tol, verbose)]
        y = eval(A, b, res[-1])
        if verbose:
            print(f"iter={iter}\ny={y}\nres={res[-1]}\n")
        if y >= 0:
            res.pop()
        else:
            b = substitute_shift(A, b, res[-1])
    return np.array(res).sum(axis=0)


def solve(A: np.ndarray, b: np.ndarray, prec: int, neighbors: int, niter: int = default_niter,
          lb: np.ndarray = None, ub: np.ndarray = None, tol: float = eps, verbose: bool = False) -> np.ndarray:
    n = A.shape[0]
    if lb is None:
        lb = -bound * np.ones(n)
    if ub is None:
        ub = bound * np.ones(n)
    return solve_iter(A, b, prec, neighbors, niter, lb, ub, tol, verbose)
