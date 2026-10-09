
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

# Project paths
ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "graphs" / "saved_graphs"
OUTPUT_DIR = ROOT / "results" / "plots"
OUTPUT_FILE = OUTPUT_DIR / "evolving_graph_animation.gif"

# Animation settings
MAX_FRAMES = 60
FPS = 2
LAYOUT_SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph_files = sorted(GRAPH_DIR.glob("graph_*.pkl"))

    if not graph_files:
        raise FileNotFoundError(
            f"No graph snapshots found in {GRAPH_DIR}"
        )

    # Limit the number of frames to keep the GIF manageable.
    if len(graph_files) > MAX_FRAMES:
        indices = np.linspace(
            0, len(graph_files) - 1, MAX_FRAMES, dtype=int
        )
        graph_files = [graph_files[i] for i in indices]

    graphs = []
    all_nodes = set()
    node_attributes = {}

    # Load graph snapshots and collect all nodes.
    for path in graph_files:
        with path.open("rb") as file:
            graph = pickle.load(file)

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                f"{path.name} is not a NetworkX graph"
            )

        graphs.append(graph)
        all_nodes.update(graph.nodes())

        for node, attributes in graph.nodes(data=True):
            node_attributes[node] = {
                **node_attributes.get(node, {}),
                **attributes,
            }

    nodes = sorted(all_nodes)

    # Combine snapshots to calculate one consistent layout.
    combined = nx.Graph()
    combined.add_nodes_from(
        (node, node_attributes[node]) for node in nodes
    )

    for graph in graphs:
        combined.add_edges_from(graph.edges())

    positions = nx.spring_layout(
        combined,
        seed=LAYOUT_SEED,
        iterations=100,
    )

    # Assign colors according to each stock's sector.
    sectors = {
        node: combined.nodes[node].get("sector", "Unknown")
        for node in nodes
    }

    sector_names = sorted(set(sectors.values()))
    palette = plt.get_cmap("tab10")

    sector_colors = {
        sector: palette(i % 10)
        for i, sector in enumerate(sector_names)
    }

    node_colors = [
        sector_colors[sectors[node]]
        for node in nodes
    ]

    # Larger figure with dedicated space for the legend.
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(
        left=0.04,
        right=0.78,
        top=0.88,
        bottom=0.06,
    )

    def draw_frame(frame_index):
        ax.clear()
        graph = graphs[frame_index]

        # Only draw nodes present in the current snapshot.
        current_nodes = list(graph.nodes())

        edges = list(graph.edges(data=True))

        edge_widths = [
            0.5 + 3 * abs(
                float(
                    data.get(
                        "correlation",
                        data.get("weight", 0),
                    )
                )
            )
            for _, _, data in edges
        ]

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=current_nodes,
            node_color=[
                sector_colors[sectors[node]]
                for node in current_nodes
            ],
            node_size=700,
            ax=ax,
        )

        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node: node for node in current_nodes},
            font_size=9,
            ax=ax,
        )

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[
                (u, v) for u, v, _ in edges
            ],
            width=edge_widths,
            edge_color="gray",
            alpha=0.55,
            ax=ax,
        )

        snapshot_date = graph.graph.get("date", graph_files[frame_index].stem)
        ax.set_title(
            "Finora: Evolving Stock Correlation Network\n"
            f"Snapshot {frame_index + 1}/{len(graphs)} | "
            f"Date: {snapshot_date}",
            fontsize=15,
            pad=18,
        )

        ax.margins(0.12)
        ax.axis("off")

        # Place the legend outside the graph but inside the figure.
        legend_items = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=sector,
                markerfacecolor=sector_colors[sector],
                markersize=10,
            )
            for sector in sector_names
        ]

        ax.legend(
            handles=legend_items,
            title="Stock Sector",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
        )

        return []

    animation = FuncAnimation(
        fig,
        draw_frame,
        frames=len(graphs),
        interval=1000 // FPS,
        blit=False,
    )

    # Save the GIF with a white background.
    animation.save(
        OUTPUT_FILE,
        writer=PillowWriter(fps=FPS),
        savefig_kwargs={"facecolor": "white"},
    )

    plt.close(fig)

    print(f"Animation saved to: {OUTPUT_FILE}")
    print(f"Frames generated: {len(graphs)}")
    print(f"Stocks: {len(nodes)}")
    print(f"Sectors: {', '.join(sector_names)}")


if __name__ == "__main__":
    main()