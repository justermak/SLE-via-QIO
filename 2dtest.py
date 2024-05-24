import numpy as np
import os
from matplotlib import pyplot as plt

EPS = 1e-6


def eval(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return (x.T @ A + b) @ x


def check(A: np.ndarray, p: np.ndarray, r: float) -> bool:
    b = np.zeros(2)
    x, y = p[0], p[1]
    points = np.array([[x + r + 2 * i * r, y + r + 2 * j * r] for i in range(-3, 3) for j in range(-3, 3)])
    mn_val = np.inf
    mn_val_idx = -1
    mn_dist = np.inf
    mn_dist_idx = -1
    for point in points:
        val = eval(A, b, point)
        if val < mn_val - EPS:
            mn_val = val
            mn_val_idx = point
        dist = np.linalg.norm(point)
        if dist < mn_dist - EPS:
            mn_dist = dist
            mn_dist_idx = point
    return np.all(mn_val_idx == mn_dist_idx)


def check_top_left(A: np.ndarray, p: np.ndarray, r: float) -> bool:
    b = np.zeros(2)
    x, y = p[0], p[1]
    return eval(A, b, np.array([x - r, y + r])) < eval(A, b, np.array([x + r, y + r]))


def plot_area(A: np.ndarray, r: float) -> float:
    xs = np.linspace(-2 * r, 2 * r, 300)
    ys = np.linspace(-2 * r, 2 * r, 300)
    res = []
    for x in xs:
        for y in ys:
            if check(A, np.array([x, y]), r):
                res.append([x, y])
    plt.scatter(*zip(*res), s=10, c="cyan")
    return len(res)/90000


def plot_area_top_left(A: np.ndarray, r: float) -> float:
    xs = np.linspace(-2 * r, 2 * r, 100)
    ys = np.linspace(-2 * r, 2 * r, 100)
    res = []
    for x in xs:
        for y in ys:
            if check_top_left(A, np.array([x, y]), r):
                res.append([x, y + 2])
    plt.scatter(*zip(*res), s=10, c="cyan")
    return len(res)/10000


def plot_ellipse(A: np.ndarray, c: float,  r: float) -> None:
    xs = np.linspace(-2 * r, 2 * r, 100)
    ys = np.linspace(-2 * r, 2 * r, 100)
    res = []
    for x in xs:
        for y in ys:
            if abs(eval(A, np.zeros(2), np.array([x, y])) - c) < 0.1 * c:
                res.append([x, y])
    plt.scatter(*zip(*res), s=10, c="red")


def rotate(A: np.ndarray, phi: float) -> np.ndarray:
    c, s = np.cos(2 * np.pi / 360 * phi), np.sin(2 * np.pi / 360 * phi)
    R = np.array([[c, s], [-s, c]])
    return R.T @ A @ R


if __name__ == "__main__":
    A = np.array([[1, 0], [0, 20]])
    if not os.path.exists("tests"):
        os.makedirs("tests")
    p = []
    for phi in range(0, 50, 5):
        print(phi)
        AA = rotate(A, phi)
        p.append(plot_area(AA, 2))
        plot_ellipse(AA, 10, 2)
        plt.savefig(f"tests/{phi}.png")
        plt.clf()
    plt.scatter(range(0, 50, 5), p)
    plt.xlabel("Angle")
    plt.ylabel("Probability")
    plt.savefig("tests/plot.png")
    plt.clf()
