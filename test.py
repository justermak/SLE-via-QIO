import numpy as np
from scipy.stats import ortho_group
from typing import Iterator, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt
import lib

rng = np.random.default_rng()
directory = Path(f"{os.path.dirname(__file__)}")
tests_path = directory / Path("tests")
group_size = 16
matrix_size = 3
eigenvalues_ratios = [50, 60, 70, 80, 90, 100, 110]
tested_params = [(12, int(2 ** 12 * i / 20 * 0.5)) for i in range(10, 20)]


def generate_equations_result_eigenvalues(res_bound, eigenvalues_ratio, size: int, n: int) -> Iterator[
    Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for i in range(n):
        res = rng.uniform(-res_bound, res_bound, size)
        eigenvalues = np.hstack(([1, eigenvalues_ratio], rng.uniform(1, eigenvalues_ratio, size - 2)))
        a = ortho_group.rvs(size)
        A = a.T @ np.diag(eigenvalues) @ a
        b = lib.substitute_shift(A, np.zeros(size), -res)
        yield A, b, res


def generate_tests() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    res_bound = lib.bound
    for eigenvalues_ration in eigenvalues_ratios:
        for e in generate_equations_result_eigenvalues(res_bound, eigenvalues_ration, matrix_size, group_size):
            yield e


def solve_tests(tests: [Tuple[np.ndarray, np.ndarray, np.ndarray]], prec: int, neighbors: int) -> np.ndarray:
    ref, sol = map(lambda x: x.reshape(-1, group_size, matrix_size),
                   map(np.array, zip(*[[res, lib.solve(A, b, prec=prec, neighbors=neighbors)] for A, b, res in tests])))
    res = (np.count_nonzero(np.apply_along_axis(np.linalg.norm, 2, ref - sol) < 1, axis=1) >= group_size / 2)
    return res


def plot(total_solved: np.ndarray, max_group: np.ndarray) -> None:
    if not os.path.exists(tests_path):
        os.makedirs(tests_path)
    plt.title("Total algorithms to solve test case")
    plt.xticks(np.arange(len(eigenvalues_ratios)), eigenvalues_ratios)
    plt.yticks(np.arange(-1, len(tested_params) + 1))
    plt.scatter(np.arange(len(eigenvalues_ratios)), total_solved)
    plt.savefig(tests_path / Path("total"))
    plt.clf()

    plt.title("Hardest test case solved")
    plt.xlabel("Neighbors percentage")
    plt.ylabel("Condition number")
    plt.xticks(np.arange(len(tested_params)), list(map(lambda x: f"{(2 * x[1] + 1)/(2 ** x[0]):.2f}", tested_params)))
    plt.yticks(np.arange(len(eigenvalues_ratios)), eigenvalues_ratios)
    plt.scatter(np.arange(len(tested_params)), max_group)
    plt.savefig(tests_path / Path("hardest"))
    plt.clf()


def test_all() -> None:
    tests = list(generate_tests())
    total_solved = np.zeros(len(eigenvalues_ratios))
    max_group = np.zeros(len(tested_params))
    for i, (prec, neighbors) in enumerate(tested_params):
        print(prec, neighbors)
        res = solve_tests(tests, prec, neighbors)
        total_solved += res
        if np.all(res == 0):
            max_group[i] = -1
        else:
            max_group[i] = np.max(np.nonzero(res))
    plot(total_solved, max_group)
