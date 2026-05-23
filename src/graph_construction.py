"""Build logistics network graphs from shipment and route data."""

from typing import Any, Optional

import networkx as nx
import pandas as pd


def build_directed_graph(
    edges: pd.DataFrame,
    *,
    source_col: str = "origin_id",
    target_col: str = "destination_id",
    weight_col: Optional[str] = "transit_minutes",
    edge_attrs: Optional[dict[str, Any]] = None,
) -> nx.DiGraph:
    """Construct a directed graph from an edge list DataFrame."""
    G = nx.DiGraph()
    attrs = edge_attrs or {}
    for row in edges.itertuples(index=False):
        u = getattr(row, source_col)
        v = getattr(row, target_col)
        data = dict(attrs)
        if weight_col and hasattr(row, weight_col):
            data["weight"] = getattr(row, weight_col)
        G.add_edge(u, v, **data)
    return G
