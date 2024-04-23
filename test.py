import numpy as np
from scipy.stats import ortho_group
import pandas as pd
from typing import Iterator, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt
import lib

rng = np.random.default_rng()
directory = Path(f"{os.path.dirname(__file__)}")
tests_path = directory / Path("tests")
test_groups = 9
group_size = 10
matrix_size = 3
eigenvalues_ratios = [1, 2, 5, 10, 100, 1000, 10000, 100000, 1000000]
tested_params = [(1, 0), (2, 1), (3, 2), (3, 3), (4, 5), (4, 6), (4, 7)]
tested_algorithms = ['QUBO_stable', 'QUBO_perturb']


def generate_equations_result_eigenvalues(res_bound, eigenvalues_ratio, size: int, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for i in range(n):
        res = rng.uniform(-res_bound, res_bound, size)
        eigenvalues = np.exp(rng.uniform(0, np.log(eigenvalues_ratio), size))
        a = ortho_group.rvs(size)
        A = a.T @ np.diag(eigenvalues) @ a
        b = lib.substitute_shift(A, np.zeros(size), -res)
        yield A, b, res


def generate_tests() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    res_bound = lib.bound
    for eigenvalues_ration in eigenvalues_ratios:
        for e in generate_equations_result_eigenvalues(res_bound, eigenvalues_ration, matrix_size, group_size):
            yield e


def solve_tests(tests: [Tuple[np.ndarray, np.ndarray, np.ndarray]], prec: int, neighbors: int) -> pd.DataFrame:
    df = pd.DataFrame()
    solutions = [[res, lib.solve(A, b, prec=prec, neighbors=neighbors),
                  lib.solve(A, b, prec=prec, neighbors=neighbors, perturb=True)] for A, b, res in tests]
    df[[f'{name}_solution' for name in ['reference'] + tested_algorithms]] = solutions
    for name in tested_algorithms:
        df[f'{name}_diff'] = np.vectorize(np.linalg.norm)(df['reference_solution'] - df[f'{name}_solution'])
    return df


def plot(data: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.tight_layout(pad=2.5)
    for i in range(test_groups):
        axs[i // 3, i % 3].set_yscale('log')
        axs[i // 3, i % 3].scatter(range(group_size), data[i * group_size:(i + 1) * group_size])
        axs[i // 3, i % 3].set_title(f'eigenvalues ratio: {eigenvalues_ratios[i]}')
    if not os.path.exists(tests_path):
        os.makedirs(tests_path)
    plt.savefig(tests_path / path)
    plt.clf()


def test_all() -> None:
    global tests_path
    tests = list(generate_tests())
    for prec, neighbors in tested_params:
        print(prec, neighbors)
        tests_path = directory / Path(f"tests_{2 * neighbors + 1}_{2 ** prec}")
        df = solve_tests(tests, prec, neighbors)
        for name in tested_algorithms:
            plot(df[f'{name}_diff'], Path(f'{name}_diff.png'))
