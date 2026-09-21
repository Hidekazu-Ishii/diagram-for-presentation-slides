"""Plot the exponential correlation kernel used to model spatially correlated shadowing.

The kernel follows the empirical rule of Gudmundson [Gudmundson-EL1991]:

    k(x_i, x_j) = sigma^2 * exp(-||x_i - x_j|| / d_cor * ln2)

This script plots the covariance k(x_i, x_j) as a function of distance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.size"] = 18

SIGMA2 = 1.0  # shadowing variance [dB^2]
D_COR = 50.0  # correlation distance [m]
MAX_DISTANCE = 400.0
N_POINTS = 500


def kernel(distance: np.ndarray, sigma2: float, d_cor: float) -> np.ndarray:
    return sigma2 * np.exp(-(distance / d_cor) * np.log(2))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "26_09_mid/theory"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    distance = np.linspace(0, MAX_DISTANCE, N_POINTS)
    cov = kernel(distance, SIGMA2, D_COR)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(distance, cov, color="red", linewidth=2)
    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_ylim(0, SIGMA2)
    ax.set_xlabel("Distance[m]")
    ax.set_ylabel("Covariance")

    half_cov = SIGMA2 / 2
    ax.plot([D_COR, D_COR], [0, half_cov], linestyle="--", color="#595959", linewidth=1)
    ax.plot([0, D_COR], [half_cov, half_cov], linestyle="--", color="#595959", linewidth=1)
    ax.set_xticks([*ax.get_xticks(), D_COR])
    ax.set_yticks([*ax.get_yticks(), half_cov])
    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_ylim(0, SIGMA2)

    fig.tight_layout()
    fig.savefig(output_dir / "gudmundson.svg")
    plt.close(fig)
    print(f"Saved figure to {output_dir}")


if __name__ == "__main__":
    main()
