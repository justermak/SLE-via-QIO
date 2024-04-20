import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize as scp_minimize
from gekko import GEKKO
from dwave.samplers import SimulatedAnnealingSampler
import itertools

rng = np.random.default_rng()
max_border = 1e3
eps = 1e-2
base_prec = 4
base_neighbors = 6
base_iter = 10
sigma = 1 / 4


def to_QP(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    AA = A.T @ A
    bb = -2 * b.T @ A
    return AA, bb


def eval(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    AA, bb = to_QP(A, b)
    return (x.T @ AA + bb) @ x


def substitute(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> (np.ndarray, np.ndarray):
    assert A.ndim == 2
    assert b.ndim == 1
    n, m = A.shape
    assert b.size == n
    assert c.ndim == 2
    _m, k = c.shape
    k -= 1
    assert _m == m
    # x = c[0] + c[1] * x_1 + ... + c[m-1] * x_{m-1}
    const, coef = c[:, 0], c[:, 1:]
    AA = np.repeat(A, k, axis=1) * coef.flatten()
    bb = b - A @ const
    return AA, bb


def extract_solution_box(q: np.ndarray, prec: int) -> np.ndarray:
    assert q.ndim == 1
    n = q.size
    q.reshape(-1, prec)
    powers = np.array([2 ** i for i in range(prec)])
    vals = (q.reshape(-1, prec) * powers).sum(axis=1)
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

    def obj(x, *args):
        return (x.T @ AA + bb) @ x

    def callback(xk):
        if verbose:
            print(f"xk={xk}")

    result = scp_minimize(obj, np.zeros(m), method='SLSQP', tol=tol, callback=callback)
    if verbose:
        print(f"Exiting SLSQP")
    return result['x']


def solve_reference(A: np.ndarray, b: np.ndarray, option: str = 'np') -> np.ndarray:
    match option:
        case 'np':
            return solve_np(A, b)
        case 'slsqp':
            return solve_slsqp(A, b)
        case _:
            raise NotImplemented()


def solve_one(A: np.ndarray, b: np.ndarray, option: str = 'dwave', lb: np.ndarray = None,
              ub: np.ndarray = None, prec: int = base_prec, neighbors: int = base_neighbors, tol: float = eps,
              stir: bool = True, verbose: bool = False) -> np.ndarray:
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
    assert neighbors >= 0
    assert prec > 0
    assert 2 * neighbors + 1 < 2 ** prec
    if verbose:
        print(f"Solving SLE with\nA={A}\nb={b}\nlb={lb}\nub={ub}\nprec={prec}\nneighbors={neighbors}\ntol={tol}\nstir={stir}\n")
    it = np.ceil(np.log(max(ub - lb) / tol) / np.log(2 ** prec/(2 * neighbors + 1))).astype(int)
    for _ in range(it):
        lengths = (ub - lb) / 2 ** prec
        c = []
        for i in range(m):
            rnd = ss.truncnorm(-1 / 2 / sigma, 1 / 2 / sigma, 1 / 2, sigma).rvs() if stir else 1/2
            mn, mx = rnd, rnd
            const = rnd * lengths[i] + lb[i]
            coefs = []
            for k in range(prec):
                rnd = ss.truncnorm(-mn / sigma, (1 - mx) / sigma, 0, sigma).rvs() if stir else 0
                coefs += [(rnd + 2 ** k) * lengths[i]]
                mn = min(mn, mn + rnd)
                mx = max(mx, mx + rnd)
            c += [[const] + coefs]
        c = np.array(c)
        A_sub, b_sub = substitute(A, b, c)
        AA, bb = to_QP(A_sub, b_sub)
        q = solve_QUBO(AA + np.diag(bb), option)
        box = extract_solution_box(q, prec)
        lbd = np.clip(box - neighbors, 0, 2 ** prec - 1)
        ubd = np.clip(2 ** prec - 1 - box - neighbors, 0, 2 ** prec - 1)
        lb += lbd * lengths
        ub -= ubd * lengths
        if verbose:
            print(f"c={c}\nq={q}\nbox={box}\nlbd={lbd}\nubd={ubd}\nlb={lb}\nub={ub}\n")
    return lb


def solve(A: np.ndarray, b: np.ndarray, niter: int = base_iter, option: str = 'dwave', lb: np.ndarray = None,
               ub: np.ndarray = None, prec: int = base_prec, neighbors: int = base_neighbors, tol: float = eps,
               stir: bool = False, verbose: bool = False) -> np.ndarray:
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
    res = [np.zeros(m)]
    if verbose:
        print(f"Solving SLE with\nA={A}\nb={b}\nniter={niter}\noption={option}\nlb={lb}\nub={ub}\nprec={prec}\nneighbors={neighbors}\ntol={tol}\nstir={stir}\n")
    for _ in range(niter):
        res += [solve_one(A, b, option, lb, ub, prec, neighbors, tol, stir, verbose)]
        y = eval(A, b, res[-1])
        if verbose:
            print(f"y={y}\nres={res[-1]}\n")
        if y >= 0:
            res.pop()
        else:
            b = b - A @ res[-1]
    return np.array(res).sum(axis=0)
