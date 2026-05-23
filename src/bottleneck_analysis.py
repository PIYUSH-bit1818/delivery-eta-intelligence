"""Identify congestion and capacity bottlenecks in the logistics network."""

from typing import Any

import networkx as nx
import pandas as pd


def node_betweenness(
    G: nx.Graph,
    *,
    weight: str = "weight",
    normalized: bool = True,
) -> pd.DataFrame:
    """Rank nodes by betweenness centrality."""
    scores = nx.betweenness_centrality(G, weight=weight, normalized=normalized)
    return (
        pd.DataFrame({"node_id": list(scores.keys()), "betweenness": list(scores.values())})
        .sort_values("betweenness", ascending=False)
        .reset_index(drop=True)
    )


def high_degree_nodes(G: nx.Graph, top_k: int = 10) -> pd.DataFrame:
    """Return the top-k nodes by degree."""
    degrees = dict(G.degree())
    return (
        pd.DataFrame({"node_id": list(degrees.keys()), "degree": list(degrees.values())})
        .sort_values("degree", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
