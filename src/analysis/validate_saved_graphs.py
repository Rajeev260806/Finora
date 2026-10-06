import os
import pickle
import networkx as nx
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "graphs",
    "saved_graphs"
)

graph_files = sorted(
    f for f in os.listdir(GRAPHS_DIR)
    if f.endswith(".pkl")
)

print("=" * 60)
print("SAVED GRAPH VALIDATION")
print("=" * 60)

print("Total graph files:", len(graph_files))

edge_counts = []
node_counts = []
isolated_counts = []
dates = []

invalid_graphs = []

for filename in graph_files:

    path = os.path.join(GRAPHS_DIR, filename)

    try:
        with open(path, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, nx.Graph):
            invalid_graphs.append(filename)
            continue

        nodes = G.number_of_nodes()
        edges = G.number_of_edges()

        isolated = len(list(nx.isolates(G)))

        node_counts.append(nodes)
        edge_counts.append(edges)
        isolated_counts.append(isolated)
        dates.append(G.graph.get("date"))

        if nodes != 20:
            invalid_graphs.append(filename)

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        invalid_graphs.append(filename)


print("\n===== NODE VALIDATION =====")

print("Minimum nodes:", min(node_counts))
print("Maximum nodes:", max(node_counts))

print("\n===== EDGE VALIDATION =====")

print("Minimum edges:", min(edge_counts))
print("Maximum edges:", max(edge_counts))
print("Average edges:", sum(edge_counts) / len(edge_counts))

print("\n===== ISOLATED NODE VALIDATION =====")

print("Minimum isolated nodes:", min(isolated_counts))
print("Maximum isolated nodes:", max(isolated_counts))
print(
    "Graphs with isolated nodes:",
    sum(x > 0 for x in isolated_counts)
)

print("\n===== DATE VALIDATION =====")

print("First graph date:", dates[0])
print("Last graph date:", dates[-1])

print("\n===== INVALID GRAPHS =====")

print("Invalid graphs:", len(invalid_graphs))

if invalid_graphs:
    print(invalid_graphs[:20])

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)