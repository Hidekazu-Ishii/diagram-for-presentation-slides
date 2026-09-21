"""疑似道路グラフを生成しPowerPoint向けのSVG図として可視化するスクリプト．"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Int
from matplotlib.patches import Rectangle

Node = tuple[int, int]  # (row, col)
Edge = tuple[Node, Node]

FIGSIZE = (5, 5)
FONT_FAMILY = "Segoe UI"
FONT_SIZE = 18

# 8近傍のオフセット（上下左右＋斜め）
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (d_row, d_col) for d_row in (-1, 0, 1) for d_col in (-1, 0, 1) if (d_row, d_col) != (0, 0)
)


@dataclass(frozen=True, slots=True)
class GridConfig:
    """建物/非建物セルからなるグリッドの設定．occupancy: 1=非建物（道路）, 0=建物．"""

    cell_size: float
    occupancy: Int[np.ndarray, "rows cols"]

    @property
    def n_rows(self) -> int:
        return self.occupancy.shape[0]

    @property
    def n_cols(self) -> int:
        return self.occupancy.shape[1]

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """セル(row, col)の中心座標(x, y)を返す．"""
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return x, y


@dataclass(frozen=True, slots=True)
class RoadGraph:
    """非建物セルを頂点，8近傍で隣接する頂点同士を辺で結んだグラフ（最大次数8）．"""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


@dataclass(frozen=True, slots=True)
class StyleConfig:
    """図の配色の設定．"""

    building_color: str
    non_building_color: str
    grid_line_color: str
    node_color: str
    edge_color: str


def build_road_graph(grid: GridConfig) -> RoadGraph:
    """非建物セル(occupancy==1)を頂点とし，直接隣接するセル同士を辺で結ぶ．

    斜め方向の接続は，両脇のセル（flank）のどちらかが建物だと建物の角を
    横切ってしまうため，その場合は辺を引かない．
    """
    nodes = tuple(
        (row, col)
        for row in range(grid.n_rows)
        for col in range(grid.n_cols)
        if grid.occupancy[row, col] == 1
    )
    node_set = set(nodes)

    edges: set[Edge] = set()
    for row, col in nodes:
        for d_row, d_col in NEIGHBOR_OFFSETS:
            neighbor = (row + d_row, col + d_col)
            if neighbor not in node_set:
                continue
            if d_row != 0 and d_col != 0:
                flank_a = grid.occupancy[row + d_row, col]
                flank_b = grid.occupancy[row, col + d_col]
                if flank_a == 0 or flank_b == 0:
                    continue
            edge = (row, col), neighbor
            edges.add(edge if edge[0] < edge[1] else (edge[1], edge[0]))

    return RoadGraph(nodes=nodes, edges=tuple(sorted(edges)))


def plot_road_graph(
    grid: GridConfig,
    graph: RoadGraph,
    style: StyleConfig,
    output_path: Path,
) -> None:
    """疑似道路グラフを描画しSVGとして保存する（1図1ファイル，タイトルなし）．"""
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONT_SIZE

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            is_building = grid.occupancy[row, col] == 0
            ax.add_patch(
                Rectangle(
                    (col * grid.cell_size, row * grid.cell_size),
                    grid.cell_size,
                    grid.cell_size,
                    facecolor=style.building_color if is_building else style.non_building_color,
                    edgecolor=style.grid_line_color,
                    linewidth=1.0,
                )
            )

    for (row1, col1), (row2, col2) in graph.edges:
        x1, y1 = grid.cell_center(row1, col1)
        x2, y2 = grid.cell_center(row2, col2)
        ax.plot([x1, x2], [y1, y2], color=style.edge_color, linewidth=1.0, zorder=2)

    for row, col in graph.nodes:
        x, y = grid.cell_center(row, col)
        ax.plot(x, y, marker="o", color=style.node_color, markersize=10, zorder=3)

    ax.set_xlim(0, grid.n_cols * grid.cell_size)
    ax.set_ylim(0, grid.n_rows * grid.cell_size)
    ax.set_xticks(np.arange(0, grid.n_cols * grid.cell_size + 1, grid.cell_size))
    ax.set_yticks(np.arange(0, grid.n_rows * grid.cell_size + 1, grid.cell_size))
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelsize=FONT_SIZE, labelfontfamily=FONT_FAMILY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # 1: 非建物（道路）セル, 0: 建物セル。建物は2つの塊（各2セル）にまとめる。
    occupancy = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    grid = GridConfig(cell_size=5.0, occupancy=occupancy)
    style = StyleConfig(
        building_color="#CDCDCD",
        non_building_color="#FFFFFF",
        grid_line_color="#7F7F7F",
        node_color="#000000",
        edge_color="#404040",
    )

    graph = build_road_graph(grid)

    plot_road_graph(
        grid=grid,
        graph=graph,
        style=style,
        output_path=root / "26_09_mid" / "figure" / "road_graph.svg",
    )

if __name__ == "__main__":
    main()
