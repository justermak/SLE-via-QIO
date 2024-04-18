import numpy as np
from scipy.stats import ortho_group
import itertools
import pandas as pd
from typing import Iterator, Tuple, Optional
import os
from pathlib import Path
import matplotlib.pyplot as plt
import lib

rng = np.random.default_rng()
directory = Path(f"{os.path.dirname(__file__)}")
tests_path = directory / Path("tests")
test_groups = 4
group_size = 10
matrix_size = 4
tested_algorithms = ['QUBO_standard', 'QUBO_stir', 'QUBO_non_iter']


def write_eqs(eqs: Iterator[Tuple[np.ndarray, np.ndarray]], filename: str) -> None:
    As, bs = zip(*eqs)
    np.savez(filename, *(list(As) + list(bs)))


def read_eqs(filename: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    assert os.path.exists(filename + ".npz")
    d = np.load(filename + ".npz")
    x = d.values().__iter__()
    return zip(x.__next__(), x.__next__())


def generate_equations_result_eigenvalues(res: Optional[np.ndarray], eigenvalues: Optional[np.ndarray], size: int,
                                          n: int):
    gen_res = res is None
    gen_eigenvalues = eigenvalues is None
    for i in range(n):
        if gen_res:
            res = rng.uniform(-1000, 1000, size)
        if gen_eigenvalues:
            eigenvalues = rng.uniform(1, 10, size)
        a = ortho_group.rvs(size)
        A = a.T @ np.diag(eigenvalues) @ a
        b = A @ res
        yield A, b


def generate_all():
    ones = np.ones(matrix_size)
    for e in generate_equations_result_eigenvalues(ones, ones, matrix_size, group_size):
        yield e
    for e in generate_equations_result_eigenvalues(ones, None, matrix_size, group_size):
        yield e
    for e in generate_equations_result_eigenvalues(None, ones, matrix_size, group_size):
        yield e
    for e in generate_equations_result_eigenvalues(None, None, matrix_size, group_size):
        yield e


def get_tests(option: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    match option:
        case 'load':
            path = (tests_path / Path("tests"))
            return read_eqs(path.absolute().as_posix())
        case 'generate':
            return generate_all()
        case _:
            raise Exception('Incorrect option. Should be load / generate')


def solve_all(tests: Iterator[Tuple[np.ndarray, np.ndarray]], save_option: str) -> pd.DataFrame:
    match save_option:
        case 'save':
            tests, write_tests = itertools.tee(tests)
            path = (tests_path / Path("tests"))
            if not os.path.exists(tests_path):
                os.makedirs(tests_path)
            write_eqs(write_tests, path.absolute().as_posix())
        case 'discard':
            pass
        case _:
            raise Exception('Incorrect option. Should be save / discard')
    df = pd.DataFrame()
    tests = list(tests)
    solutions = [[lib.solve_reference(A, b), lib.solve(A, b),
                  lib.solve(A, b, stir=True), lib.solve_one(A, b)] for A, b in tests]
    df[[f'{name}_solution' for name in ['reference'] + tested_algorithms]] = solutions
    As, bs = zip(*tests)
    df['reference_norm'] = np.vectorize(np.linalg.norm)(df['reference_solution'])
    for name in tested_algorithms:
        df[f'{name}_norm'] = np.vectorize(np.linalg.norm)(df[f'{name}_solution'])
        df[f'{name}_value'] = [np.linalg.norm(A @ x - b) for A, b, x in zip(As, bs, df[f'{name}_solution'].tolist())]
        df[f'{name}_diff'] = np.vectorize(np.linalg.norm)(df['reference_solution'] - df[f'{name}_solution'])
        df[f'{name}_ratio'] = df[f'{name}_diff'] / df['reference_norm']
    return df


def plot(title: str, data: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout(pad=5)
    fig.suptitle(title)
    for i in range(test_groups):
        axs[i // 2, i % 2].scatter(range(group_size), data[i * group_size:(i + 1) * group_size])
        axs[i // 2, i % 2].set_title(
            f'{"fixed" if i // 2 == 0 else "random"} solution, {"normal" if i % 2 == 0 else "random"} matrix')
    if not os.path.exists(tests_path):
        os.makedirs(tests_path)
    plt.savefig(tests_path / path)
    plt.clf()


def test_and_plot(test_option: str, save_option: str) -> None:
    tests = get_tests(test_option)
    df = solve_all(tests, save_option)
    plot('Norm ||x*||', df['reference_norm'], Path('reference_norm.png'))

    for name in tested_algorithms:
        plot('Norm ||x||', df[f'{name}_norm'], Path(f'{name}_norm.png'))
        plot('Value ||Ax - b||', df[f'{name}_value'], Path(f'{name}_value.png'))
        plot('Absolute error (||x - x*||)', df[f'{name}_diff'], Path(f'{name}_diff.png'))
        plot('Relative error (||x - x*|| / ||x*||)', df[f'{name}_ratio'], Path(f'{name}_ratio.png'))

