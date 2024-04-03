import numpy as np
import itertools
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import lib


rng = np.random.default_rng()
directory = Path(f"{os.path.dirname(__file__)}")
tests_path = directory / Path("tests")


def write_eqs(A, b, filename):
    np.savez(filename, A + b)


def read_eqs(filename):
    d = np.load(filename)
    x = d.values()
    return x[:len(x)/2], x[len(x)/2:]


def generate_uniform(l1: float, r1: float, l2: float, r2: float, size: int, n: int):
    for i in range(n):
        A = rng.uniform(l1, r1, (size, size))
        b = rng.uniform(l2, r2, size)
        yield A, b


def generate_all(n: int):
    for e in generate_uniform(-5, 5, -10, 10, 10, n):
        yield e


def solve_all(qubo_option: str, n: int) -> pd.DataFrame:
    tests = [*generate_all(n)]
    np.savez(tests_path, *[item for pair in tests for item in pair])
    exact_solutions = np.array([lib.solve_exact(A, b) for A, b in tests])
    DNC_QUBO_solutions = np.array([lib.solve_DNC_QUBO(A, b, option=qubo_option) for A, b in tests])
    df = pd.DataFrame()
    df['exact_norms'] = np.apply_along_axis(np.linalg.norm, 1, exact_solutions)
    df['DNC_QUBO_norms'] = np.apply_along_axis(np.linalg.norm, 1, DNC_QUBO_solutions)
    df['difference_norms'] = np.apply_along_axis(np.linalg.norm, 1, exact_solutions - DNC_QUBO_solutions)
    df['ratios'] = df['difference_norms'] / df['exact_norms']
    df = df[df['DNC_QUBO_norms'] < 1e9]
    return df


def test_and_plot(qubo_option: str, test_size: str):
    match test_size:
        case 'big': n = 100
        case 'small': n = 10
        case _: raise Exception('Incorrect test_size. Should be big / small')
    df = solve_all(qubo_option, n)
    n = df.shape[0]
    plt.scatter(range(n), df['difference_norms'], c='red')
    plt.title("Absolute accuracy of DNC QUBO")
    plt.xlabel("Test number")
    plt.ylabel("Norm difference")
    plt.savefig(directory / Path('!absolute_errors.png'))
    plt.scatter(range(n), df['ratios'], c='red')
    plt.xlabel("Test number")
    plt.ylabel("Relative norm difference")
    plt.title("Relative accuracy of DNC QUBO")
    plt.savefig(directory / Path('!relative_errors.png'))
    plt.scatter(range(n), df['difference_norms'], c='red')
