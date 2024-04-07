import numpy as np
from scipy.stats import ortho_group
import itertools
import pandas as pd
from typing import Iterator, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt
import lib

rng = np.random.default_rng()
directory = Path(f"{os.path.dirname(__file__)}")
tests_path = directory / Path("tests")
test_groups = 4
group_size = 10
matrix_size = 5


def write_eqs(eqs: Iterator[Tuple[np.ndarray, np.ndarray]], filename: str) -> None:
    As, bs = zip(*eqs)
    np.savez(filename, *(list(As) + list(bs)))


def read_eqs(filename: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    assert os.path.exists(filename + ".npz")
    d = np.load(filename + ".npz")
    x = d.values().__iter__()
    return zip(x.__next__(), x.__next__())


def generate_by_solution(res: np.ndarray, size: int, n: int):
    for i in range(n):
        A = ortho_group.rvs(size)
        b = A @ res
        yield A, b


def generate_by_eigenvalues(eigenvalues: np.ndarray, size: int, n: int):
    for i in range(n):
        a = ortho_group.rvs(size)
        A = a @ np.diag(eigenvalues) @ a.T
        res = rng.uniform(-1, 1, size)
        b = A @ res
        yield A, b


def generate_all():
    for e in generate_by_solution(np.array([1] * matrix_size), matrix_size, group_size):
        yield e
    for e in generate_by_solution(np.array([1000] + [1] * (matrix_size - 1)), matrix_size, group_size):
        yield e
    for e in generate_by_eigenvalues(np.array([1] * matrix_size), matrix_size, group_size):
        yield e
    for e in generate_by_eigenvalues(np.array([1000] + [1] * (matrix_size - 1)), matrix_size, group_size):
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


def solve_all(tests: Iterator[Tuple[np.ndarray, np.ndarray]], qubo_option: str = 'bruteforce',
              reference_option: str = 'np', option: str = 'save') -> pd.DataFrame:
    match option:
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
    solutions = [[lib.solve_reference(A, b, reference_option), lib.solve_DNC_QUBO(A, b, qubo_option),
                  lib.solve_one_step_QUBO(A, b, qubo_option, lb=np.array([-2] * matrix_size),
                                          ub=np.array([2] * matrix_size))] for A, b in tests]
    df[['reference_solution', 'DNC_QUBO_solution', 'One_step_QUBO_solutions']] = solutions
    df['reference_norm'] = np.vectorize(np.linalg.norm)(df['reference_solution'])
    df['DNC_QUBO_norm'] = np.vectorize(np.linalg.norm)(df['DNC_QUBO_solution'])
    df['One_step_QUBO_norm'] = np.vectorize(np.linalg.norm)(df['One_step_QUBO_solutions'])
    df['DNC_diff'] = np.vectorize(np.linalg.norm)(df['reference_solution'] - df['DNC_QUBO_solution'])
    df['One_step_diff'] = np.vectorize(np.linalg.norm)(df['reference_solution'] - df['One_step_QUBO_solutions'])
    df['DNC_ratio'] = df['DNC_diff'] / df['reference_norm']
    df['One_step_ratio'] = df['One_step_diff'] / df['reference_norm']
    return df


def plot(title: str, data: pd.DataFrame, path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout(pad=5)
    fig.suptitle(title)
    for i in range(test_groups):
        axs[i // 2, i % 2].scatter(range(group_size), data[i * group_size:(i + 1) * group_size])
        axs[i // 2, i % 2].set_title(f'Group {i}')
    if not os.path.exists(tests_path):
        os.makedirs(tests_path)
    plt.savefig(tests_path / path)
    plt.clf()


def test_and_plot(qubo_option: str, refecence_option: str, test_option: str, save_option: str):
    tests = get_tests(test_option)
    df = solve_all(tests, qubo_option=qubo_option, reference_option=refecence_option, option=save_option)

    plot('Absolute error (||x - x*||)', df['DNC_diff'], Path('DNC_diff.png'))
    plot('Absolute error (||x - x*||)', df['One_step_diff'], Path('One_step_diff.png'))
    plot('Relative error (||x - x*|| / ||x*||)', df['DNC_ratio'], Path('DNC_ratio.png'))
    plot('Relative error (||x - x*|| / ||x*||)', df['One_step_ratio'], Path('One_step_ratio.png'))
    plot('Norm of reference solution', df['reference_norm'], Path('reference_norm.png'))
    plot('Norm of DNC solution', df['DNC_QUBO_norm'], Path('DNC_norm.png'))
    plot('Norm of One step solution', df['One_step_QUBO_norm'], Path('One_step_norm.png'))
