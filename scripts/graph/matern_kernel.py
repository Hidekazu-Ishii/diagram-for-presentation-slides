"""グラフMaternカーネル(nu=1/2)の説明用デモ図を作成する.

目的: グラフ距離(ホップ数)だけではカーネル値が決まらないこと
(同じホップ数でも経路の多重性で値が変わること)を1枚の図と1枚の散布図で示す.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.size"] = 18


def build_demo_graph() -> tuple[nx.Graph, dict[int, tuple[float, float]], dict[int, str]]:
    """単一経路の枝と多重経路の枝を持つ最小構成のデモグラフを作る.

    0 が観測点(参照ノード). 上側の枝(1→2→7)は経路が1本のみ,
    下側の枝(3,4→5→6)は 0→5 に至る経路が2本(菱形)ある.
    """
    edges = [
        (0, 1), (1, 2), (2, 7),  # 単一経路の枝
        (0, 3), (0, 4), (3, 5), (4, 5), (5, 6),  # 多重経路(菱形)の枝
    ]
    graph = nx.Graph()
    graph.add_edges_from(edges)

    positions = {
        0: (0.0, 0.0),
        1: (1.0, 1.2), 2: (2.0, 1.6), 7: (3.0, 1.9),
        3: (1.0, -0.6), 4: (1.0, -1.4), 5: (2.2, -1.0), 6: (3.3, -1.0),
    }
    arm_labels = {n: "single-path" for n in (1, 2, 7)}
    arm_labels.update({n: "multi-path" for n in (3, 4, 5, 6)})
    arm_labels[0] = "source"
    return graph, positions, arm_labels


def matern_graph_kernel(laplacian: np.ndarray, kappa: float, nu: float = 2.0) -> np.ndarray:
    """スペクトル関数 Phi(lambda) = (2*nu/kappa**2 + lambda)**(-nu) によるグラフMaternカーネル."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    spectral_response = (2.0 * nu / kappa**2 + eigenvalues) ** (-nu)
    return eigenvectors @ np.diag(spectral_response) @ eigenvectors.T


def to_correlation_row(kernel: np.ndarray, source: int) -> np.ndarray:
    """自己分散の違いに影響されないよう, 参照ノード行を相関形に正規化する."""
    variances = np.diag(kernel)
    return kernel[source, :] / np.sqrt(variances[source] * variances)


def plot_graph_with_kernel_coloring(
    graph: nx.Graph,
    positions: dict[int, tuple[float, float]],
    source: int,
    correlation_row: np.ndarray,
    output_path: Path,
) -> None:
    """参照ノードからの相関値をノードの色とサイズで表したグラフ図を1枚保存する."""
    node_order = list(graph.nodes())
    values = np.array([correlation_row[n] for n in node_order])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#808080", width=1.5)
    nodes = nx.draw_networkx_nodes(
        graph, positions, ax=ax, nodelist=node_order,
        node_color=values, cmap="YlOrRd", vmin=values.min(), vmax=values.max(),
        node_size=300 + 900 * values,
        edgecolors=["black" if n == source else "#595959" for n in node_order],
        linewidths=[2.5 if n == source else 0.8 for n in node_order],
    )
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=16)
    fig.colorbar(nodes, ax=ax, label="correlation", shrink=0.85)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_distance_vs_kernel_scatter(
    graph: nx.Graph,
    source: int,
    correlation_row: np.ndarray,
    arm_labels: dict[int, str],
    output_path: Path,
) -> None:
    """ホップ距離を横軸, カーネル値(相関)を縦軸にした散布図を1枚保存する."""
    hop_distances = nx.shortest_path_length(graph, source=source)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = {"single-path": "o", "multi-path": "^"}
    for arm_name, marker in markers.items():
        nodes_in_arm = [n for n, label in arm_labels.items() if label == arm_name]
        x_values = [hop_distances[n] for n in nodes_in_arm]
        y_values = [correlation_row[n] for n in nodes_in_arm]
        ax.scatter(x_values, y_values, marker=marker, s=90, label=arm_name)

    ax.set_xlabel("graph distance")
    ax.set_ylabel("correlation")
    ax.set_xticks(sorted(set(hop_distances.values())))
    ax.legend(loc="lower left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "26_09_mid/theory"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = 0
    kappa = 6.0
    graph, positions, arm_labels = build_demo_graph()
    # ノードラベルが 0..N-1 の整数なので, sorted 順で並べると行列index=ノードラベルになる
    laplacian = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray().astype(float)
    kernel = matern_graph_kernel(laplacian, kappa=kappa, nu=0.5)
    correlation_row = to_correlation_row(kernel, source)

    plot_graph_with_kernel_coloring(
        graph, positions, source, correlation_row,
        output_dir / "graph_kernel_correlation.svg",
    )
    plot_distance_vs_kernel_scatter(
        graph, source, correlation_row, arm_labels,
        output_dir / "graph_kernel_scatter.svg",
    )

    for node in sorted(graph.nodes()):
        hop = nx.shortest_path_length(graph, source=source, target=node)
        print(f"node={node} hop={hop} corr={correlation_row[node]:.3f} arm={arm_labels[node]}")


if __name__ == "__main__":
    main()
