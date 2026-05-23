"""Build and merge logistics network hub features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.config import GRAPH_COVERAGE_THRESHOLD, RANDOM_STATE, ProjectPaths, get_paths


def build_lane_graph_tables(df: pd.DataFrame, *, seed: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """Compute full-hub graph tables from segment-level logistics data."""
    lane_col = "route_lane" if "route_lane" in df.columns else None
    if lane_col is None:
        df = df.copy()
        df["route_lane"] = (
            df["source_center"].astype(str) + " → " + df["destination_center"].astype(str)
        )
        lane_col = "route_lane"

    agg: dict[str, Any] = {"weight": (lane_col, "count")}
    if "segment_delay_ratio" in df.columns:
        agg["avg_delay"] = ("segment_delay_ratio", "mean")

    edges = df.groupby(["source_center", "destination_center"], as_index=False).agg(**agg)

    G = nx.DiGraph()
    for row in edges.itertuples(index=False):
        attrs: dict[str, Any] = {"weight": int(row.weight)}
        if hasattr(row, "avg_delay"):
            attrs["avg_delay"] = float(row.avg_delay)
        G.add_edge(row.source_center, row.destination_center, **attrs)

    for _, _, d in G.edges(data=True):
        d["flow_distance"] = 1.0 / max(d.get("weight", 1), 1)

    traffic: dict[Any, int] = {}
    for n in G.nodes():
        out_w = sum(d.get("weight", 1) for _, _, d in G.out_edges(n, data=True))
        in_w = sum(d.get("weight", 1) for _, _, d in G.in_edges(n, data=True))
        traffic[n] = out_w + in_w

    degree = nx.degree_centrality(G)
    between = nx.betweenness_centrality(G, weight="flow_distance", normalized=True)
    pagerank = nx.pagerank(G, weight="weight")

    bottleneck = pd.DataFrame({
        "hub_id": list(between.keys()),
        "betweenness_centrality": list(between.values()),
        "traffic_volume": [traffic.get(n, 0) for n in between.keys()],
    })
    lo_b, hi_b = bottleneck["betweenness_centrality"].min(), bottleneck["betweenness_centrality"].max()
    lo_t, hi_t = bottleneck["traffic_volume"].min(), bottleneck["traffic_volume"].max()
    bottleneck["bottleneck_score"] = (
        0.6 * (bottleneck["betweenness_centrality"] - lo_b) / (hi_b - lo_b + 1e-9)
        + 0.4 * (bottleneck["traffic_volume"] - lo_t) / (hi_t - lo_t + 1e-9)
    )

    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)

    communities = nx.community.louvain_communities(U, weight="weight", seed=seed)
    comm_rows = [
        {"hub_id": h, "community_id": cid, "community_size": len(members)}
        for cid, members in enumerate(communities)
        for h in members
    ]

    return {
        "pagerank": pd.DataFrame({"hub_id": list(pagerank.keys()), "pagerank": list(pagerank.values())}),
        "degree": pd.DataFrame({"hub_id": list(degree.keys()), "degree_centrality": list(degree.values())}),
        "bottleneck": bottleneck[["hub_id", "bottleneck_score", "betweenness_centrality", "traffic_volume"]],
        "communities": pd.DataFrame(comm_rows),
    }


def load_graph_tables_from_exports(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Load notebook 03 CSV exports."""
    pr = pd.read_csv(tables_dir / "03_hub_rankings_pagerank.csv")
    deg = pd.read_csv(tables_dir / "03_hub_rankings_degree.csv")
    bn = pd.read_csv(tables_dir / "03_bottleneck_hubs.csv")
    comm = pd.read_csv(tables_dir / "03_hub_communities.csv")
    return {
        "pagerank": pr[["hub_id", "pagerank"]].copy(),
        "degree": deg[["hub_id", "degree_centrality"]].copy(),
        "bottleneck": bn[["hub_id", "bottleneck_score"]].copy(),
        "communities": comm[["hub_id", "community_id", "community_size"]].copy(),
    }


def graph_hub_coverage(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> float:
    hubs = pd.unique(df[["source_center", "destination_center"]].values.ravel("K"))
    covered = set(tables["pagerank"]["hub_id"].astype(str))
    return len([h for h in hubs if str(h) in covered]) / max(len(hubs), 1)


def load_or_build_graph_tables(
    df: pd.DataFrame,
    paths: ProjectPaths | None = None,
) -> dict[str, pd.DataFrame]:
    """Prefer persisted graph store, then CSV exports, else rebuild."""
    paths = paths or get_paths()

    if paths.graph_features_store.exists():
        import joblib
        return joblib.load(paths.graph_features_store)

    graph_files = [
        "03_bottleneck_hubs.csv",
        "03_hub_rankings_pagerank.csv",
        "03_hub_rankings_degree.csv",
        "03_hub_communities.csv",
    ]
    if all((paths.outputs_tables / f).exists() for f in graph_files):
        tables = load_graph_tables_from_exports(paths.outputs_tables)
        if graph_hub_coverage(df, tables) >= GRAPH_COVERAGE_THRESHOLD:
            return tables

    return build_lane_graph_tables(df)


def merge_hub_features(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    *,
    hub_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Attach hub-level graph metrics with column prefix (source_ / destination_)."""
    out = df.copy()
    pr = tables["pagerank"].rename(columns={"hub_id": hub_col, "pagerank": f"{prefix}_pagerank"})
    deg = tables["degree"].rename(columns={"hub_id": hub_col, "degree_centrality": f"{prefix}_degree"})
    bn = tables["bottleneck"][["hub_id", "bottleneck_score"]].rename(
        columns={"hub_id": hub_col, "bottleneck_score": f"{prefix}_bottleneck_score"}
    )
    comm = tables["communities"][["hub_id", "community_id"]].rename(
        columns={"hub_id": hub_col, "community_id": f"{prefix}_community"}
    )
    for extra in (pr, deg, bn, comm):
        out = out.merge(extra, on=hub_col, how="left")
    return out


def enrich_with_graph_features(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Merge source and destination graph features."""
    tables = tables or load_or_build_graph_tables(df, paths)
    out = merge_hub_features(df, tables, hub_col="source_center", prefix="source")
    out = merge_hub_features(out, tables, hub_col="destination_center", prefix="destination")
    return out


def persist_graph_tables(tables: dict[str, pd.DataFrame], paths: ProjectPaths | None = None) -> Path:
    import joblib

    paths = paths or get_paths()
    paths.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(tables, paths.graph_features_store)
    return paths.graph_features_store
