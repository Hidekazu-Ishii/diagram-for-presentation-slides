"""グラフ上の観測値(dbm)を色分けし, 観測なしノードを黒で示すデモ図を作成する.

共分散(Maternカーネル)からの変形は行わず, dbm値は直接与える.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.size"] = 18


def build_demo_graph() -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    """観測点(色付き)と観測なし点(黒)が混在する最小構成のデモグラフを作る."""
    edges = [
        (0, 1), (1, 2), (2, 7),
        (0, 3), (0, 4), (3, 5), (4, 5), (5, 6),
    ]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 1.2), 2: (2.0, 1.6), 7: (3.0, 1.9),
        3: (1.0, -0.6), 4: (1.0, -1.4), 5: (2.2, -1.0), 6: (3.3, -1.0),
    }
    return graph, positions


def add_value_bars(
    ax: Axes,
    positions: dict[int, tuple[float, float]],
    dbm_by_node: dict[int, float],
    node_radius_offset: float = 0.15,
    bar_width: float = 0.15,
    length_scale: float = 0.04,
) -> None:
    """値の絶対値に比例した棒をノードの外側から伸ばす. プラスは上向き, マイナスは下向き."""
    for node, value in dbm_by_node.items():
        x, y = positions[node]
        length = abs(value) * length_scale
        bottom = y + node_radius_offset if value >= 0 else y - node_radius_offset
        height = length if value >= 0 else -length
        ax.bar(x, height, bottom=bottom, width=bar_width, color="#595959", zorder=2)


def plot_graph_with_dbm_coloring(
    graph: nx.Graph,
    positions: dict[int, tuple[float, float]],
    dbm_by_node: dict[int, float],
    output_path: Path,
    show_value_bars: bool = True,
) -> None:
    """観測値(dbm)をviridisで色分けする. dbm_by_node に値がないノードは観測なし(黒塗り)扱い.

    show_value_bars=True のとき, 観測値を持つノードに棒(プラスは上, マイナスは下)を重ねる.
    """
    node_order = list(graph.nodes())
    norm = mcolors.Normalize(vmin=-20.0, vmax=20.0)
    colormap = plt.get_cmap("viridis")
    node_colors = [
        colormap(norm(dbm_by_node[n])) if n in dbm_by_node else (0.0, 0.0, 0.0, 1.0)
        for n in node_order
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#999999", width=1.5)
    if show_value_bars:
        add_value_bars(ax, positions, dbm_by_node)
    nodes_collection = nx.draw_networkx_nodes(
        graph, positions, ax=ax, nodelist=node_order,
        node_color=node_colors, node_size=500,
        edgecolors="#808080", linewidths=0.8,
    )
    nodes_collection.set_zorder(3)
    colorbar_source = cm.ScalarMappable(norm=norm, cmap=colormap)
    colorbar_source.set_array([])
    fig.colorbar(colorbar_source, ax=ax, label="dbm", shrink=0.85)

    x_values = [p[0] for p in positions.values()]
    y_values = [p[1] for p in positions.values()]
    ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    ax.set_ylim(min(y_values) - 1.1, max(y_values) + 1.1)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "26_09_mid/theory"
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, positions = build_demo_graph()
    dbm_by_node = {0: 15.0, 1: 5.0, 3: 12.0, 6: 6.0, 7: -10.0}  # ここにないノードは自動で黒塗り(観測なし)
    show_value_bars = True  # False にすると棒グラフなしで色分けのみ表示

    plot_graph_with_dbm_coloring(
        graph, positions, dbm_by_node,
        output_dir / "graph_dbm.svg",
        show_value_bars=show_value_bars,
    )


if __name__ == "__main__":
    main()
