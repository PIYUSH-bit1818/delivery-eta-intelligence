"""Smoke-run graph analytics pipeline (notebook 03 logic)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed" / "delivery_logistics_processed.parquet"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
TABLES = PROJECT_ROOT / "outputs" / "tables"


def main():
    df = pd.read_parquet(PROCESSED)
    edges = (
        df.groupby(["source_center", "destination_center"], as_index=False)
        .agg(
            weight=("route_lane", "count"),
            avg_delay=("segment_delay_ratio", "mean"),
            avg_actual_time=("segment_actual_time", "mean"),
            source_name=("source_name", "first"),
            destination_name=("destination_name", "first"),
        )
    )
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(
            r.source_center,
            r.destination_center,
            weight=int(r.weight),
            avg_delay=float(r.avg_delay),
            avg_actual_time=float(r.avg_actual_time),
        )
    for u, v, d in G.edges(data=True):
        d["flow_distance"] = 1.0 / max(d.get("weight", 1), 1)

    print("nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    bc = nx.betweenness_centrality(G, weight="flow_distance", normalized=True)
    print("betweenness ok, top hub", max(bc, key=bc.get))

    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)
    comms = nx.community.louvain_communities(U, weight="weight", seed=42)
    print("communities", len(comms))
    print("OK")


if __name__ == "__main__":
    main()
