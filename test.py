import numpy as np
import itertools
import os
from pathlib import Path
import matplotlib.pyplot as plt
import lib


EPS = 1e-1
rng = np.random.default_rng()
directory = Path(f"{os.path.dirname(__file__)}")
tests_path = directory / Path("tests.txt")


def write_eqs(A, b, filename):
    np.savez(filename, A + b)


def read_eqs(filename):
    d = np.load(filename)
    x = d.values()
    return x[:len(x)/2], x[len(x)/2:]


def generate_normal_distr(mu: float, sigma: float, mu2: float, sigma2: float, size: int, n: int):
    for i in range(n):
        A = rng.normal(mu, sigma, (size, size))
        b = rng.normal(mu2, sigma2, size)
        yield A, b


def generate_all(n: int):
    for e in generate_normal_distr(0, 1, 10, 10, 2, n):
        yield e


def test_and_plot(n: int):
    tests = [*generate_all(n)]
    np.savez(tests_path, *[item for pair in tests for item in pair])
    norms = [np.linalg.norm(lib.solve_exact(A, b) - lib.solve_DNC_QUBO(A, b)) for A, b in tests]
    plt.scatter(range(n), norms, c='red')
    plt.xlabel("Test number")
    plt.ylabel("Norm difference")
    plt.title("Accuracy of DNC QUBO")
    plt.savefig(directory / Path('test_result.png'))
    norms = np.array(norms)
    return np.size(norms[norms > EPS])
