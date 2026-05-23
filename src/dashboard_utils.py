"""Dashboard-ready figures, KPIs, and tables (Plotly dark theme + Streamlit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard_theme import COLORS, RISK_COLORS, apply_dark_theme, risk_color

# Chart heights (laptop-friendly, full-width)
CHART_HEIGHT_SM = 380
CHART_HEIGHT_MD = 480
CHART_HEIGHT_LG = 560


def _finalize(fig: go.Figure, height: int, *, showlegend: bool | None = None) -> go.Figure:
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return apply_dark_theme(fig, height=height)


def extended_summary_cards(df: pd.DataFrame) -> dict[str, float | int]:
    base = {
        "trips_scored": len(df),
        "mean_predicted_eta": float(df["predicted_eta"].mean()),
        "critical_risk_count": int((df["risk_level"] == "CRITICAL").sum()),
        "bottleneck_alerts": int(df["bottleneck_warning"].sum()) if "bottleneck_warning" in df.columns else 0,
    }
    high_crit = df["risk_level"].isin(["HIGH", "CRITICAL"]).sum() if "risk_level" in df.columns else 0
    base["sla_breach_pct"] = float(round(100 * high_crit / max(len(df), 1), 1))
    if "confidence_low" in df.columns and "confidence_high" in df.columns:
        base["avg_confidence_half_width"] = float((df["confidence_high"] - df["confidence_low"]).mean() / 2)
    else:
        base["avg_confidence_half_width"] = float(df.get("predicted_eta", pd.Series([0])).std() * 0.15)
    base["high_risk_count"] = int(high_crit)
    return base


def eta_summary_cards(df: pd.DataFrame) -> dict[str, float | int]:
    return extended_summary_cards(df)


def plot_risk_distribution(df: pd.DataFrame, title: str = "Operational risk distribution") -> go.Figure:
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = df["risk_level"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                name="",
                marker=dict(
                    color=[RISK_COLORS.get(str(k), COLORS["cyan"]) for k in counts.index],
                    line=dict(width=0),
                ),
                text=counts.values.astype(int),
                textposition="outside",
                textfont=dict(color=COLORS["text_muted"], size=11),
                hovertemplate="<b>%{x}</b><br>Trips: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="Risk level", yaxis_title="Trips", showlegend=False)
    return _finalize(fig, CHART_HEIGHT_MD)


def plot_eta_vs_osrm(df: pd.DataFrame, title: str = "Predicted ETA vs OSRM baseline") -> go.Figure:
    fig = px.scatter(
        df,
        x="osrm_time",
        y="predicted_eta",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        opacity=0.65,
        title=title,
        labels={"osrm_time": "OSRM (min)", "predicted_eta": "Predicted ETA (min)"},
        hover_data=["route_lane"] if "route_lane" in df.columns else None,
    )
    mx = max(float(df["osrm_time"].max()), float(df["predicted_eta"].max())) * 1.05
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=mx,
        y1=mx,
        line=dict(dash="dash", color=COLORS["text_muted"], width=1),
        layer="below",
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="rgba(255,255,255,0.15)")))
    fig.update_layout(legend_title_text="Risk level")
    return _finalize(fig, CHART_HEIGHT_LG)


def plot_congestion_by_lane(df: pd.DataFrame, top_n: int = 12, title: str | None = None) -> go.Figure:
    lane = (
        df.groupby("route_lane", as_index=False)
        .agg(mean_eta=("predicted_eta", "mean"), trips=("predicted_eta", "count"))
        .sort_values("mean_eta", ascending=True)
        .tail(top_n)
    )
    fig = go.Figure(
        go.Bar(
            x=lane["mean_eta"],
            y=lane["route_lane"],
            orientation="h",
            name="",
            marker=dict(
                color=lane["mean_eta"],
                colorscale=[[0, COLORS["cyan"]], [0.5, COLORS["purple"]], [1, COLORS["red"]]],
                showscale=True,
                colorbar=dict(title=dict(text="ETA (min)", font=dict(color=COLORS["text_muted"]))),
            ),
            hovertemplate="<b>%{y}</b><br>Mean ETA: %{x:.1f} min<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or f"Route congestion — top {top_n} lanes by mean ETA",
        xaxis_title="Mean predicted ETA (min)",
        showlegend=False,
    )
    return _finalize(fig, CHART_HEIGHT_MD)


def plot_bottleneck_hubs(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    col = "source_bottleneck_score"
    if col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Bottleneck scores not available in batch",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS["text_muted"]),
        )
        return _finalize(fig, CHART_HEIGHT_SM, showlegend=False)

    hubs = (
        df.groupby("source_center", as_index=False)
        .agg(bottleneck=(col, "max"), trips=("predicted_eta", "count"))
        .nlargest(top_n, "bottleneck")
        .sort_values("bottleneck", ascending=True)
    )
    fig = go.Figure(
        go.Bar(
            x=hubs["bottleneck"],
            y=hubs["source_center"],
            orientation="h",
            name="",
            marker=dict(color=COLORS["purple"], line=dict(color=COLORS["cyan"], width=1)),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Bottleneck hub intensity (source)",
        xaxis_title="Bottleneck score",
        showlegend=False,
    )
    return _finalize(fig, CHART_HEIGHT_MD)


def plot_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    if "trip_hour" not in work.columns:
        work["trip_hour"] = 12
    work["is_high_risk"] = work["risk_level"].isin(["HIGH", "CRITICAL"]).astype(int)
    pivot = work.pivot_table(
        index="route_type" if "route_type" in work.columns else "risk_level",
        columns="trip_hour",
        values="is_high_risk",
        aggfunc="mean",
        fill_value=0,
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale=[[0, COLORS["bg_secondary"]], [0.5, COLORS["purple"]], [1, COLORS["red"]]],
            hovertemplate="Route: %{y}<br>Hour: %{x}<br>High-risk rate: %{z:.1%}<extra></extra>",
            colorbar=dict(title=dict(text="High-risk %", font=dict(color=COLORS["text_muted"])), tickformat=".0%"),
        )
    )
    fig.update_layout(
        title="Risk heatmap — hour × route type",
        xaxis_title="Trip hour (UTC)",
        yaxis_title="Route type",
        showlegend=False,
    )
    return _finalize(fig, CHART_HEIGHT_MD)


def plot_route_type_donut(df: pd.DataFrame) -> go.Figure:
    if "route_type" not in df.columns or df["route_type"].nunique() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No route type data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=COLORS["text_muted"]),
        )
        return _finalize(fig, CHART_HEIGHT_SM, showlegend=False)

    counts = df["route_type"].value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[str(x) for x in counts.index],
                values=counts.values,
                hole=0.55,
                name="",
                marker=dict(colors=[COLORS["cyan"], COLORS["purple"]], line=dict(color=COLORS["bg_primary"], width=2)),
                textinfo="label+percent",
                textfont=dict(color=COLORS["text_primary"], size=11),
                hovertemplate="<b>%{label}</b><br>%{value} trips (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(title="Route type mix", showlegend=False)
    return _finalize(fig, CHART_HEIGHT_SM)


def plot_eta_percentile_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["predicted_eta"],
            nbinsx=40,
            name="",
            marker=dict(color=COLORS["cyan"], line=dict(color=COLORS["bg_primary"], width=1)),
            opacity=0.88,
            hovertemplate="ETA: %{x} min<br>Count: %{y}<extra></extra>",
        )
    )
    for p, color in [(50, COLORS["purple"]), (90, COLORS["red"])]:
        val = float(df["predicted_eta"].quantile(p / 100))
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            annotation=dict(text=f"P{p}", font=dict(size=10, color=color), bgcolor="rgba(0,0,0,0)"),
        )
    fig.update_layout(
        title="Predicted ETA distribution",
        xaxis_title="Minutes",
        yaxis_title="Frequency",
        showlegend=False,
    )
    return _finalize(fig, CHART_HEIGHT_SM)


GAUGE_TITLE_DEFAULT = "Network operational risk index"


def plot_operational_risk_gauge(risk_score: float, title: str | None = None) -> go.Figure:
    score = float(np.clip(risk_score, 0, 100))
    gauge_title = (title or GAUGE_TITLE_DEFAULT).strip() or GAUGE_TITLE_DEFAULT

    fig = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=score,
                name="",
                title={"text": gauge_title, "font": {"size": 14, "color": COLORS["text_muted"]}},
                number={"font": {"size": 42, "color": COLORS["text_primary"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_muted"]},
                    "bar": {"color": COLORS["cyan"], "thickness": 0.72},
                    "bgcolor": "rgba(21,34,56,0.55)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16,185,129,0.28)"},
                        {"range": [35, 65], "color": "rgba(34,211,238,0.22)"},
                        {"range": [65, 85], "color": "rgba(167,139,250,0.28)"},
                        {"range": [85, 100], "color": "rgba(244,63,94,0.38)"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["red"], "width": 3},
                        "thickness": 0.78,
                        "value": 85,
                    },
                },
                domain={"x": [0.1, 0.9], "y": [0.15, 0.85]},
            )
        ]
    )
    fig.update_layout(title=None)
    return _finalize(fig, CHART_HEIGHT_SM, showlegend=False)


def compute_operational_risk_index(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    weights = {"LOW": 0, "MEDIUM": 25, "HIGH": 60, "CRITICAL": 100}
    risk_part = df["risk_level"].map(lambda x: weights.get(str(x), 30)).mean()
    bn_part = 0.0
    if "source_bottleneck_score" in df.columns:
        bn_part = df["source_bottleneck_score"].fillna(0).mean() * 40
    return float(np.clip(0.6 * risk_part + bn_part, 0, 100))


def plot_risky_lanes_leaderboard(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    lanes = (
        df.groupby("route_lane", as_index=False)
        .agg(
            critical_rate=("risk_level", lambda s: (s == "CRITICAL").mean()),
            trips=("predicted_eta", "count"),
        )
        .query("trips >= 3")
        .sort_values("critical_rate", ascending=False)
        .head(top_n)
        .sort_values("critical_rate", ascending=True)
    )
    if lanes.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient lane volume for leaderboard",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=COLORS["text_muted"]),
        )
        return _finalize(fig, CHART_HEIGHT_MD, showlegend=False)

    fig = go.Figure(
        go.Bar(
            x=lanes["critical_rate"],
            y=lanes["route_lane"],
            orientation="h",
            name="",
            marker=dict(color=COLORS["red"], line=dict(color=COLORS["orange"], width=1)),
            text=[f"{v:.0%}" for v in lanes["critical_rate"]],
            textposition="outside",
            textfont=dict(color=COLORS["text_muted"], size=10),
            hovertemplate="<b>%{y}</b><br>Critical rate: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top risky lanes (critical risk rate)",
        xaxis_title="Critical %",
        xaxis_tickformat=".0%",
        showlegend=False,
    )
    return _finalize(fig, CHART_HEIGHT_MD)


def hub_hotspot_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """High-risk hubs, or busiest hubs if none in HIGH/CRITICAL."""
    risky = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].copy()
    if not risky.empty:
        src = risky.groupby("source_center").size()
        dst = risky.groupby("destination_center").size()
        trips = src.add(dst, fill_value=0).reset_index()
        trips.columns = ["hub_id", "risky_trips"]
        eta_src = risky.groupby("source_center")["predicted_eta"].mean()
        eta_dst = risky.groupby("destination_center")["predicted_eta"].mean()
        trips["avg_eta"] = trips["hub_id"].map(eta_src.add(eta_dst, fill_value=0) / 2)
        return trips.sort_values("risky_trips", ascending=False).head(top_n)

    if "source_center" not in df.columns:
        return pd.DataFrame(columns=["hub_id", "risky_trips", "avg_eta"])
    fallback = (
        df.groupby("source_center", as_index=False)
        .agg(risky_trips=("predicted_eta", "count"), avg_eta=("predicted_eta", "mean"))
        .rename(columns={"source_center": "hub_id"})
        .sort_values("risky_trips", ascending=False)
        .head(top_n)
    )
    return fallback


def critical_lanes_table(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Lanes with highest HIGH/CRITICAL exposure for monitoring tab."""
    if "route_lane" not in df.columns or "risk_level" not in df.columns:
        return pd.DataFrame()
    risky = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].copy()
    if risky.empty:
        return pd.DataFrame()
    out = (
        risky.groupby("route_lane", as_index=False)
        .agg(
            alert_trips=("predicted_eta", "count"),
            mean_eta=("predicted_eta", "mean"),
            worst_risk=("risk_level", lambda s: "CRITICAL" if (s == "CRITICAL").any() else "HIGH"),
        )
        .sort_values(["worst_risk", "alert_trips"], ascending=[True, False])
        .head(top_n)
    )
    return out


def monitoring_ops_summary(df: pd.DataFrame) -> dict[str, int]:
    """Compact SLA / anomaly counters for monitoring header."""
    high_crit = int(df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if "risk_level" in df.columns else 0
    critical_lanes = 0
    if "route_lane" in df.columns and "risk_level" in df.columns:
        critical_lanes = int(
            df.loc[df["risk_level"].isin(["HIGH", "CRITICAL"]), "route_lane"].nunique()
        )
    bn = int(df["bottleneck_warning"].sum()) if "bottleneck_warning" in df.columns else 0
    return {
        "sla_at_risk_trips": high_crit,
        "critical_lane_count": critical_lanes,
        "bottleneck_alerts": bn,
        "critical_trips": int((df["risk_level"] == "CRITICAL").sum()) if "risk_level" in df.columns else 0,
    }


def congestion_hotspot_table(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if "route_lane" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("route_lane", as_index=False)
        .agg(
            trips=("predicted_eta", "count"),
            mean_eta=("predicted_eta", "mean"),
            max_risk=("risk_level", lambda s: s.mode().iloc[0] if len(s) else "LOW"),
            bottleneck=("source_bottleneck_score", "max") if "source_bottleneck_score" in df.columns else ("predicted_eta", "count"),
        )
        .sort_values("mean_eta", ascending=False)
        .head(top_n)
    )


def network_intelligence_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "unique_hubs": int(pd.unique(df[["source_center", "destination_center"]].values.ravel("K")).size)
        if {"source_center", "destination_center"}.issubset(df.columns)
        else 0,
        "unique_lanes": int(df["route_lane"].nunique()) if "route_lane" in df.columns else 0,
        "avg_source_bottleneck": float(df["source_bottleneck_score"].mean())
        if "source_bottleneck_score" in df.columns
        else 0.0,
        "same_community_pct": float((df["source_community"] == df["destination_community"]).mean() * 100)
        if {"source_community", "destination_community"}.issubset(df.columns)
        else 0.0,
        "graph_enriched_trips": len(df),
    }


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if "eta" in col.lower() or "time" in col.lower():
            out[col] = out[col].round(1)
        elif "score" in col.lower() or "ratio" in col.lower():
            out[col] = out[col].round(3)
    return out


def plot_confidence_band_bar(predicted: float, low: float, high: float, osrm: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[high - low],
            y=["ETA range"],
            orientation="h",
            base=[low],
            name="",
            marker=dict(color="rgba(34,211,238,0.4)", line=dict(color=COLORS["cyan"], width=2)),
            hovertemplate=f"{low:.0f} – {high:.0f} min<extra></extra>",
        )
    )
    fig.add_vline(
        x=predicted,
        line_color=COLORS["purple"],
        line_width=3,
        annotation=dict(text="Pred", font=dict(size=10, color=COLORS["purple"])),
    )
    fig.add_vline(
        x=osrm,
        line_color=COLORS["green"],
        line_dash="dot",
        annotation=dict(text="OSRM", font=dict(size=10, color=COLORS["green"])),
    )
    fig.update_layout(
        title="Prediction confidence band",
        showlegend=False,
        margin=dict(l=40, r=40, t=56, b=40),
    )
    return _finalize(fig, 240, showlegend=False)


def save_plotly(fig: go.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")))
    except Exception:
        pass
    return path
