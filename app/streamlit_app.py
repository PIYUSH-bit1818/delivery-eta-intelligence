"""
Delivery ETA Intelligence — enterprise dark-theme operations dashboard.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup before Streamlit ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="ETA Intelligence Platform",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import get_paths
from src.dashboard_theme import (
    PLOTLY_CHART_CONFIG,
    get_custom_css,
    render_compact_table_html,
    render_empty_state_card,
    render_kpi_card,
    render_prediction_card,
)
from src.dashboard_utils import (
    CHART_HEIGHT_LG,
    CHART_HEIGHT_MD,
    CHART_HEIGHT_SM,
    compute_operational_risk_index,
    congestion_hotspot_table,
    critical_lanes_table,
    extended_summary_cards,
    hub_hotspot_table,
    monitoring_ops_summary,
    network_intelligence_summary,
    plot_bottleneck_hubs,
    plot_confidence_band_bar,
    plot_congestion_by_lane,
    plot_eta_percentile_distribution,
    plot_eta_vs_osrm,
    plot_operational_risk_gauge,
    plot_risk_heatmap,
    plot_risk_distribution,
    plot_risky_lanes_leaderboard,
    plot_route_type_donut,
    prepare_display_table,
)
from src.inference import ETAInferencePipeline

# Permanent dark theme CSS (injected on every rerun — overrides Streamlit light toggle)
st.markdown(get_custom_css(), unsafe_allow_html=True)

paths = get_paths(ROOT)


def _html(content: str) -> None:
    """Render HTML reliably (never use st.write for markup)."""
    st.markdown(content, unsafe_allow_html=True)


def _section(title: str, *, top: bool = False) -> None:
    cls = "section-title section-title-first" if top else "section-title"
    _html(f'<p class="{cls}">{title}</p>')


def _empty_state(message: str) -> None:
    _html(f'<div class="empty-state">{message}</div>')


def _empty_card(icon: str, title: str, hint: str) -> None:
    _html(render_empty_state_card(icon, title, hint))


def _footer() -> None:
    _html(
        '<footer class="app-footer">'
        "Delivery ETA Intelligence • Graph-aware ML • Enterprise demo build"
        "</footer>"
    )


@st.cache_resource
def load_pipeline() -> ETAInferencePipeline:
    return ETAInferencePipeline(paths)


@st.cache_data
def load_model_metadata() -> dict:
    if paths.model_metadata.exists():
        return json.loads(paths.model_metadata.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_batch_predictions() -> pd.DataFrame | None:
    batch_path = paths.outputs_predictions / "sample_predictions.csv"
    if batch_path.exists():
        return pd.read_csv(batch_path)
    return None


def render_hero() -> None:
    _html('<p class="hero-title">Delivery ETA Intelligence Platform</p>')
    _html('<p class="hero-sub">Graph-aware congestion prediction & logistics risk analytics</p>')
    _html(
        """
        <span class="status-badge live">● Live inference</span>
        <span class="status-badge">Graph-enhanced XGBoost</span>
        <span class="status-badge">Trip-safe ML pipeline</span>
        <span class="status-badge">Network risk scoring</span>
        """
    )
    _html('<hr class="divider">')


def render_kpi_dashboard(cards: dict) -> None:
    """
    Render KPI cards via Streamlit columns — avoids large HTML blocks
    being escaped as code (common Streamlit markdown issue).
    """
    specs = [
        (f"{cards['trips_scored']:,}", "Trips scored", "📦", "cyan", "Batch sample"),
        (f"{cards['mean_predicted_eta']:.0f}", "Mean ETA (min)", "⏱️", "purple", ""),
        (str(cards["critical_risk_count"]), "Critical risks", "🔴", "red", "Needs review"),
        (str(cards["bottleneck_alerts"]), "Bottleneck alerts", "⚠️", "red", ""),
        (f"{cards['sla_breach_pct']:.1f}%", "SLA breach risk", "📉", "purple", "HIGH+CRITICAL"),
        (f"±{cards['avg_confidence_half_width']:.0f}", "Avg confidence (min)", "🎯", "green", ""),
    ]
    cols = st.columns(6, gap="small")
    for col, spec in zip(cols, specs):
        with col:
            _html(render_kpi_card(*spec))


def render_network_panel(summary: dict) -> None:
    _html(
        f"""
        <div class="network-panel">
            <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;margin-bottom:0.75rem;">
                Network intelligence summary
            </div>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;">
                <div><div class="network-stat">{summary['unique_hubs']}</div><div style="color:#94a3b8;font-size:0.8rem;">Active hubs</div></div>
                <div><div class="network-stat">{summary['unique_lanes']}</div><div style="color:#94a3b8;font-size:0.8rem;">Unique lanes</div></div>
                <div><div class="network-stat">{summary['avg_source_bottleneck']:.2f}</div><div style="color:#94a3b8;font-size:0.8rem;">Avg bottleneck score</div></div>
                <div><div class="network-stat">{summary['same_community_pct']:.0f}%</div><div style="color:#94a3b8;font-size:0.8rem;">Same-community trips</div></div>
            </div>
        </div>
        """
    )


def plot_chart(fig, *, height: int | None = None) -> None:
    """Full-width Plotly charts — figures are pre-themed in dashboard_utils."""
    if height is not None and (fig.layout.height is None or fig.layout.height < height):
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CHART_CONFIG)


def render_compact_table(
    df: pd.DataFrame | None,
    *,
    max_rows: int = 15,
    caption: str | None = None,
) -> bool:
    """HTML table — no large empty Streamlit dataframe shells."""
    if df is None or df.empty:
        return False
    display = prepare_display_table(df)
    table_html = render_compact_table_html(display, max_rows=max_rows)
    if not table_html:
        return False
    if caption:
        _html(f'<p class="table-caption">{caption}</p>')
    _html(table_html)
    return True


def _render_ops_alert_chips(summary: dict) -> None:
    _html(
        f"""
        <div class="ops-alert-row">
            <div class="ops-alert-chip warn">
                <div class="chip-value">{summary['sla_at_risk_trips']}</div>
                <div class="chip-label">SLA at-risk trips</div>
            </div>
            <div class="ops-alert-chip warn">
                <div class="chip-value">{summary['critical_lane_count']}</div>
                <div class="chip-label">Lanes under alert</div>
            </div>
            <div class="ops-alert-chip">
                <div class="chip-value">{summary['bottleneck_alerts']}</div>
                <div class="chip-label">Bottleneck flags</div>
            </div>
            <div class="ops-alert-chip warn">
                <div class="chip-value">{summary['critical_trips']}</div>
                <div class="chip-label">Critical trips</div>
            </div>
        </div>
        """
    )


def render_sidebar(meta: dict):
    with st.sidebar:
        _html('<p class="sidebar-brand">🚚 ETA Intelligence</p>')
        _html(
            '<p class="sidebar-desc">Production ML ops console for graph-aware '
            "delivery ETA prediction, bottleneck detection, and corridor risk monitoring.</p>"
        )
        _html('<hr class="divider">')

        _html('<p class="section-title sidebar-section">Trip parameters</p>')
        source = st.text_input("Source center", "IND000000ACB", help="Origin hub / DC code")
        destination = st.text_input("Destination center", "IND562132AAA", help="Destination hub code")
        osrm_distance = st.number_input("OSRM distance (km)", min_value=0.1, value=120.0, step=1.0)
        trip_hour = st.slider("Trip hour (UTC)", 0, 23, 10)
        route_type = st.selectbox("Route type", ["FTL", "Carting"])
        osrm_time = st.number_input("OSRM time (min, optional)", min_value=0.0, value=0.0, step=1.0)

        _html('<hr class="divider">')
        run = st.button("⚡ Predict ETA", type="primary", use_container_width=True)

        _html('<hr class="divider">')
        _html('<p class="section-title sidebar-section">Model status</p>')
        trained = meta.get("trained_at", "—")
        if trained != "—":
            try:
                trained = datetime.fromisoformat(trained.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
            except ValueError:
                pass
        mae = meta.get("metrics", {}).get("MAE")
        _html(
            f"""
            <div class="model-status-item">Model loaded <span class="model-status-ok">✓</span></div>
            <div class="model-status-item">Engine <span class="model-status-ok">XGBoost</span></div>
            <div class="model-status-item">Graph features <span class="model-status-ok">Enabled</span></div>
            <div class="model-status-item">Features <span class="model-status-ok">{len(meta.get('feature_columns', []))}</span></div>
            <div class="model-status-item">Test MAE <span class="model-status-ok">{f'{mae:.1f} min' if mae else '—'}</span></div>
            <div class="model-status-item">Last trained<br><span style="color:#94a3b8;font-size:0.75rem;">{trained}</span></div>
            """
        )
    return run, source, destination, osrm_distance, trip_hour, route_type, osrm_time


# ── Load ML pipeline ───────────────────────────────────────────────────────
try:
    pipeline = load_pipeline()
    meta = load_model_metadata()
except FileNotFoundError:
    st.error("Production model not found. Run: `python scripts/train_production_model.py`")
    st.stop()

batch = load_batch_predictions()
run, source, destination, osrm_distance, trip_hour, route_type, osrm_time = render_sidebar(meta)

if run:
    st.session_state["last_prediction"] = pipeline.predict_eta(
        source,
        destination,
        osrm_distance,
        trip_hour,
        route_type,
        osrm_time=osrm_time if osrm_time > 0 else None,
    )

render_hero()

tab_overview, tab_risk, tab_network, tab_predictions, tab_monitoring = st.tabs(
    ["📊 Overview", "🔥 Risk Analytics", "🕸️ Network Intelligence", "🎯 Predictions", "📡 Monitoring"]
)

with tab_overview:
    if batch is not None and len(batch):
        _section("Executive KPIs", top=True)
        render_kpi_dashboard(extended_summary_cards(batch))
        risk_idx = compute_operational_risk_index(batch)

        _section("Operational snapshot")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            plot_chart(plot_operational_risk_gauge(risk_idx), height=CHART_HEIGHT_SM)
        with c2:
            plot_chart(plot_route_type_donut(batch), height=CHART_HEIGHT_SM)
        with c3:
            plot_chart(plot_eta_percentile_distribution(batch), height=CHART_HEIGHT_SM)

        _section("Risk & congestion")
        c4, c5 = st.columns(2, gap="medium")
        with c4:
            plot_chart(plot_risk_distribution(batch), height=CHART_HEIGHT_MD)
        with c5:
            plot_chart(plot_congestion_by_lane(batch), height=CHART_HEIGHT_MD)
    else:
        st.info(
            "Batch analytics unavailable. Run notebook 05 or generate "
            "`outputs/predictions/sample_predictions.csv`."
        )

with tab_risk:
    if batch is not None and len(batch):
        _section("ETA calibration")
        plot_chart(plot_eta_vs_osrm(batch), height=CHART_HEIGHT_LG)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            plot_chart(plot_risk_heatmap(batch), height=CHART_HEIGHT_MD)
        with c2:
            plot_chart(plot_risky_lanes_leaderboard(batch), height=CHART_HEIGHT_MD)
        _section("Congestion hotspot corridors")
        corridors = congestion_hotspot_table(batch)
        if not render_compact_table(corridors, max_rows=12, caption="Top lanes by mean predicted ETA"):
            _empty_card("🛣️", "No corridor data", "Batch export has no route lanes to rank.")
    else:
        st.warning("Load batch predictions to enable risk analytics.")

with tab_network:
    if batch is not None and len(batch):
        c1, c2 = st.columns([1, 1.5], gap="medium")
        with c1:
            render_network_panel(network_intelligence_summary(batch))
        with c2:
            plot_chart(plot_bottleneck_hubs(batch), height=CHART_HEIGHT_MD)
        _section("Hub congestion hotspots")
        hubs = hub_hotspot_table(batch)
        hub_caption = (
            "Hubs with HIGH/CRITICAL trip exposure"
            if batch["risk_level"].isin(["HIGH", "CRITICAL"]).any()
            else "Busiest source hubs (no HIGH/CRITICAL trips in sample)"
        )
        if not render_compact_table(hubs, max_rows=15, caption=hub_caption):
            _empty_card("🏭", "No hub hotspots", "Source hub fields missing or empty in batch.")
    else:
        st.warning("Network panels require batch prediction data.")

with tab_predictions:
    _section("Live inference")
    if "last_prediction" in st.session_state:
        r = st.session_state["last_prediction"]
        _html(
            render_prediction_card(
                r.predicted_eta,
                r.risk_level,
                r.confidence_low,
                r.confidence_high,
                r.osrm_baseline,
                r.route_lane,
                r.insights,
                r.bottleneck_warning,
            )
        )
        plot_chart(
            plot_confidence_band_bar(r.predicted_eta, r.confidence_low, r.confidence_high, r.osrm_baseline),
            height=220,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("ETA / OSRM ratio", f"{r.eta_vs_osrm_ratio:.2f}")
        m2.metric("Source bottleneck", f"{r.source_bottleneck_score:.3f}")
        m3.metric("Dest bottleneck", f"{r.destination_bottleneck_score:.3f}")
    else:
        _empty_card(
            "🎯",
            "Run a live ETA prediction from the sidebar",
            "Set source, destination, distance, and route type — then click Predict ETA to score this trip.",
        )

    if batch is not None and len(batch):
        _section("Sample batch preview")
        preview_cols = [
            c
            for c in [
                "route_lane",
                "predicted_eta",
                "risk_level",
                "osrm_time",
                "bottleneck_warning",
            ]
            if c in batch.columns
        ]
        if preview_cols:
            preview = batch[preview_cols].head(8)
            render_compact_table(
                preview,
                max_rows=8,
                caption="Recent scored trips (preview while live inference is idle)",
            )

with tab_monitoring:
    _section("SLA & model health")
    if batch is not None and len(batch):
        cards = extended_summary_cards(batch)
        ops = monitoring_ops_summary(batch)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scored trips", f"{cards['trips_scored']:,}")
        m2.metric("SLA at-risk %", f"{cards['sla_breach_pct']:.1f}%")
        m3.metric("Critical count", cards["critical_risk_count"])
        m4.metric("Test MAE (offline)", f"{meta.get('metrics', {}).get('MAE', 0):.1f} min")

        _section("Live ops signals")
        _render_ops_alert_chips(ops)

        _section("Recent critical lanes")
        crit_lanes = critical_lanes_table(batch, top_n=10)
        if render_compact_table(
            crit_lanes,
            max_rows=10,
            caption="Lanes with HIGH/CRITICAL trips in current batch",
        ):
            pass
        else:
            _empty_card(
                "✅",
                "No critical lane alerts",
                "All lanes in the batch sample are below HIGH risk — continue monitoring.",
            )

        plot_chart(plot_risk_distribution(batch, title="Risk mix — monitoring snapshot"), height=CHART_HEIGHT_MD)
        risk_summary = (
            batch.groupby("risk_level", as_index=False)
            .agg(trips=("predicted_eta", "count"), mean_eta=("predicted_eta", "mean"))
            .sort_values("trips", ascending=False)
        )
        _section("Risk level breakdown")
        render_compact_table(risk_summary, max_rows=8, caption="Trip counts and mean ETA by risk band")
        with st.expander("Monitoring playbook"):
            st.markdown(
                "| Signal | Threshold | Action |\n"
                "|--------|-----------|--------|\n"
                "| Data drift | Input distribution shift | Retrain review |\n"
                "| Concept drift | MAE ↑ 15% WoW | Challenger test |\n"
                "| SLA degradation | HIGH+CRITICAL > 25% | Ops escalation |\n"
                "| Graph staleness | Hub tables > 7d | Refresh graph store |"
            )
    else:
        st.info("Monitoring tables populate when batch predictions are available.")

_footer()
