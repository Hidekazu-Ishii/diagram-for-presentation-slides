"""Illustrate the decomposition of received power into large-scale propagation and shadowing.

P(x) (measured average received power) is split into an estimated large-scale
propagation trend L_hat(x) (fitted from the log-distance path-loss model) and the
residual shadowing W(x) = P(x) - L_hat(x). A few block-shaped buildings are placed
in the field so that the shadowing pattern differs near/behind buildings versus in
open space.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.size"] = 18

SHOW_BUILDINGS = True  # toggle whether buildings are drawn (they always affect the physics)

AREA_SIZE = 1000.0
UNIT = 50.0  # side length of the minimum square unit that buildings are built from
TX_POS = np.array([50.0, 500.0])
N_POINTS = 1000
PATH_LOSS_EXPONENT = 3.0
P0_DBM = -30.0  # reference power at d0
D0 = 1.0
BUILDING_ATTENUATION_DB = 12.0
RNG_SEED = 0

# Buildings as unions of unit squares (lower-left corner, in units of UNIT).
BUILDINGS_UNITS = [
    [(5, 12), (6, 12), (6, 13), (6, 14), (7, 14), (8, 14)],  # 1 building
    [(4, 9), (5, 8)],  # 2 building
    [(7, 4), (9, 5), (8, 5), (7, 5), (6, 6)],  # 3 building
    [(9, 9), (8, 8), (10, 9)],  # 4 building
    [(12, 12), (13, 12), (14, 12), (13, 13)],  # 5 building
    [(16, 14), (17, 14)],  # 6 building
    [(2, 2), (3, 2), (3, 3)],  # 7 building
    [(17, 4), (17, 5)],  # 8 building
    [(12, 4), (13, 4)],
    [(12, 16)],  # 10 building
]


def building_squares() -> list[tuple[float, float, float, float]]:
    """Return each unit square as (xmin, xmax, ymin, ymax)."""
    squares = []
    for building in BUILDINGS_UNITS:
        for ux, uy in building:
            xmin, ymin = ux * UNIT, uy * UNIT
            squares.append((xmin, xmin + UNIT, ymin, ymin + UNIT))
    return squares


def point_in_any_square(points: np.ndarray, squares: list[tuple[float, float, float, float]]) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for xmin, xmax, ymin, ymax in squares:
        inside |= (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
        )
    return inside


def segment_intersects_rect(
    p0: np.ndarray, p1: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float
) -> bool:
    """Liang-Barsky segment-vs-axis-aligned-box intersection test."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t_min, t_max = 0.0, 1.0
    for p, q in (
        (-dx, p0[0] - xmin),
        (dx, xmax - p0[0]),
        (-dy, p0[1] - ymin),
        (dy, ymax - p0[1]),
    ):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
        if t_min > t_max:
            return False
    return True


def line_of_sight_blocked(points: np.ndarray, squares: list[tuple[float, float, float, float]]) -> np.ndarray:
    blocked = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        blocked[i] = any(segment_intersects_rect(TX_POS, p, *sq) for sq in squares)
    return blocked


def correlated_shadowing(points: np.ndarray, rng: np.random.Generator, n_blobs: int = 25) -> np.ndarray:
    """Sum of random Gaussian blobs to create spatially correlated shadowing."""
    centers = rng.uniform(0, AREA_SIZE, size=(n_blobs, 2))
    amplitudes = rng.normal(0.0, 4.0, size=n_blobs)
    length_scale = 150.0
    values = np.zeros(len(points))
    for center, amp in zip(centers, amplitudes, strict=True):
        dist_sq = np.sum((points - center) ** 2, axis=1)
        values += amp * np.exp(-dist_sq / (2 * length_scale**2))
    return values


def large_scale_estimate(points: np.ndarray, power_dbm: np.ndarray) -> np.ndarray:
    """Fit the classic log-distance path-loss model P = a + b * log10(d) by least squares."""
    distance = np.linalg.norm(points - TX_POS, axis=1)
    distance = np.maximum(distance, D0)
    log_d = np.log10(distance / D0)
    b, a = np.polyfit(log_d, power_dbm, deg=1)
    return a + b * log_d


def draw_buildings(ax: Axes, squares: list[tuple[float, float, float, float]]) -> None:
    if not SHOW_BUILDINGS:
        return
    for xmin, xmax, ymin, ymax in squares:
        ax.add_patch(
            Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, facecolor="#808080", edgecolor="none"
            )
        )


def save_panel(points: np.ndarray, values: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(points[:, 0], points[:, 1], c=values, cmap="viridis", s=25, edgecolors="none")
    draw_buildings(ax, building_squares())
    ax.scatter(TX_POS[0], TX_POS[1], c="red", marker="+", s=200, linewidths=2)
    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "26_09_mid/figure"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    squares = building_squares()

    points = np.empty((0, 2))
    while len(points) < N_POINTS:
        candidates = rng.uniform(0, AREA_SIZE, size=(N_POINTS, 2))
        candidates = candidates[~point_in_any_square(candidates, squares)]
        points = np.vstack([points, candidates])
    points = points[:N_POINTS]

    distance = np.maximum(np.linalg.norm(points - TX_POS, axis=1), D0)
    large_scale_true = P0_DBM - 10 * PATH_LOSS_EXPONENT * np.log10(distance / D0)

    shadowing_random = correlated_shadowing(points, rng)
    shadowing_building = np.where(line_of_sight_blocked(points, squares), -BUILDING_ATTENUATION_DB, 0.0)

    power_dbm = large_scale_true + shadowing_random + shadowing_building
    large_scale_hat = large_scale_estimate(points, power_dbm)
    measured_shadowing = power_dbm - large_scale_hat

    save_panel(points, power_dbm, output_dir / "received_power.svg")
    save_panel(points, large_scale_hat, output_dir / "large_scale_pattern.svg")
    save_panel(points, measured_shadowing, output_dir / "shadowing.svg")


if __name__ == "__main__":
    main()
