"""Dark-theme design tokens and Streamlit enterprise UI primitives."""

from __future__ import annotations

import html
import plotly.graph_objects as go

# ── Enterprise palette (permanent dark) ───────────────────────────────────
COLORS = {
    "bg_primary": "#081225",
    "bg_secondary": "#111827",
    "bg_card": "#152238",
    "bg_chart_card": "#172554",
    "bg_card_glass": "rgba(21, 34, 56, 0.92)",
    "border": "rgba(255, 255, 255, 0.10)",
    "border_strong": "rgba(255, 255, 255, 0.14)",
    "border_focus": "rgba(34, 211, 238, 0.45)",
    "text_primary": "#f8fafc",
    "text_secondary": "#e2e8f0",
    "text_muted": "#94a3b8",
    "cyan": "#22d3ee",
    "cyan_dim": "#0891b2",
    "blue": "#3b82f6",
    "purple": "#a78bfa",
    "purple_deep": "#7c3aed",
    "green": "#10b981",
    "red": "#f43f5e",
    "orange": "#fb923c",
    "gradient_start": "#06b6d4",
    "gradient_end": "#8b5cf6",
}

RISK_COLORS = {
    "LOW": COLORS["green"],
    "MEDIUM": COLORS["cyan"],
    "HIGH": COLORS["purple"],
    "CRITICAL": COLORS["red"],
}

# Streamlit Plotly — no floating toolbar on any chart
PLOTLY_CHART_CONFIG: dict = {
    "displayModeBar": False,
    "displaylogo": False,
    "staticPlot": False,
    "responsive": True,
}

PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=COLORS["text_secondary"], size=12),
    margin=dict(l=56, r=40, t=72, b=56),
    legend=dict(
        bgcolor="rgba(21, 34, 56, 0.95)",
        bordercolor=COLORS["border"],
        borderwidth=1,
        font=dict(color=COLORS["text_secondary"]),
    ),
    xaxis=dict(
        title=dict(font=dict(color="#cbd5e1", size=12)),
        tickfont=dict(color="#cbd5e1", size=11),
        gridcolor="rgba(255,255,255,0.10)",
        zerolinecolor="rgba(255,255,255,0.14)",
        linecolor="rgba(255,255,255,0.16)",
    ),
    yaxis=dict(
        title=dict(font=dict(color="#cbd5e1", size=12)),
        tickfont=dict(color="#cbd5e1", size=11),
        gridcolor="rgba(255,255,255,0.10)",
        zerolinecolor="rgba(255,255,255,0.14)",
        linecolor="rgba(255,255,255,0.16)",
    ),
    colorway=[
        COLORS["cyan"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["red"],
    ],
)


def _clean_text(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("undefined", "null", "nan"):
        return None
    return s


def sanitize_figure(fig: go.Figure) -> go.Figure:
    """Strip undefined/null trace names and legend entries."""
    for trace in fig.data:
        name = getattr(trace, "name", None)
        if name is not None:
            cleaned = _clean_text(name)
            trace.name = cleaned if cleaned else ""
        # Indicator traces: keep only explicit title text on the trace
    return fig


def _is_gauge_figure(fig: go.Figure) -> bool:
    return bool(fig.data) and all(getattr(t, "type", None) == "indicator" for t in fig.data)


def apply_dark_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply dark styling; gauge charts use trace title only (no layout title slot)."""
    is_gauge = _is_gauge_figure(fig)
    title_text = None
    if not is_gauge and fig.layout.title is not None:
        title_text = _clean_text(fig.layout.title.text)

    layout = dict(PLOTLY_DARK_LAYOUT)
    if is_gauge:
        layout["margin"] = dict(l=20, r=24, t=44, b=24)
    fig.update_layout(**layout)

    if title_text and not is_gauge:
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=17, color=COLORS["text_primary"]),
                x=0.5,
                xanchor="center",
                y=0.98,
            ),
        )
    else:
        # Empty layout title — prevents Streamlit/Plotly rendering literal "undefined"
        fig.update_layout(title=dict(text=""))

    if height:
        fig.update_layout(height=height)

    if not is_gauge:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color=COLORS["text_muted"]),
            ),
        )
    return sanitize_figure(fig)


def risk_color(level: str) -> str:
    return RISK_COLORS.get(str(level).upper(), COLORS["text_muted"])


def get_custom_css() -> str:
    """Permanent enterprise dark overrides — survives Streamlit theme toggle."""
    c = COLORS
    p = c["bg_primary"]
    s = c["bg_secondary"]
    card = c["bg_card"]
    chart_card = c["bg_chart_card"]
    border = c["border"]
    border_strong = c["border_strong"]
    radius = "12px"
    shadow = "0 4px 18px rgba(0,0,0,0.22)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Force dark on ALL theme states (light toggle cannot break UI) ── */
    :root {{
        --de-bg: {p};
        --de-panel: {s};
        --de-card: {card};
        --de-border: {border};
        --de-text: {c['text_primary']};
        --de-muted: {c['text_muted']};
    }}

    .stApp,
    .stApp[data-theme="light"],
    .stApp[data-theme="dark"] {{
        background: linear-gradient(160deg, {p} 0%, {s} 42%, #0f172a 100%) !important;
        color: {c['text_secondary']} !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    }}

    /* Remove white top chrome — match header to app background */
    header[data-testid="stHeader"] {{
        background: {p} !important;
        background-color: {p} !important;
        border-bottom: 1px solid {border} !important;
        box-shadow: none !important;
    }}
    header[data-testid="stHeader"] > div {{
        background: transparent !important;
    }}
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    .stAppToolbar {{
        background: {p} !important;
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; height: 0; }}

    /* App shell — no white gaps / overlay artifacts */
    [data-testid="stAppViewContainer"] {{
        background: {p} !important;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background: transparent !important;
    }}
    section.main > div {{
        background: transparent !important;
    }}
    .main .block-container {{
        max-width: 100% !important;
        padding: 0.5rem 2.25rem 1rem 2.5rem !important;
        background: transparent !important;
    }}

    /* Sidebar — full viewport height */
    section[data-testid="stSidebar"] {{
        min-height: 100vh !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(180deg, {p} 0%, {s} 100%) !important;
        border-right: 1px solid {border_strong} !important;
        min-width: 19rem !important;
        width: 19rem !important;
        min-height: 100vh !important;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding: 1.35rem 1rem 1.75rem !important;
        background: transparent !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] label span {{
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: {c['text_primary']} !important;
    }}
    [data-testid="stSidebar"] .stSlider label {{
        margin-bottom: 0.35rem !important;
    }}
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox {{
        margin-bottom: 0.65rem !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .model-status-item {{
        color: {c['text_muted']} !important;
    }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {card} !important;
        color: {c['text_primary']} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {border} !important;
        margin: 1rem 0 !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, {c['cyan_dim']} 0%, {c['purple_deep']} 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
        padding: 0.65rem 1rem !important;
        margin-top: 0.35rem !important;
        box-shadow: 0 4px 20px rgba(34, 211, 238, 0.25) !important;
        transition: box-shadow 0.2s ease, transform 0.15s ease !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"]:hover {{
        box-shadow: 0 6px 28px rgba(34, 211, 238, 0.35) !important;
        transform: translateY(-1px);
    }}

    /* Main content typography */
    .main h1, .main h2, .main h3,
    .main .stMarkdown, .main p, .main span, .main label {{
        color: {c['text_secondary']} !important;
    }}
    .main h1, .main h2, .main h3 {{ color: {c['text_primary']} !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: transparent !important;
        padding: 0.15rem 0 0.65rem !important;
        border-bottom: 1px solid {border};
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 0.75rem !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        color: {c['text_muted']} !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 600 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(167,139,250,0.12)) !important;
        color: {c['text_primary']} !important;
        border-color: {c['border_focus']} !important;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.08) !important;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: {card} !important;
        border: 1px solid {border_strong} !important;
        border-radius: {radius} !important;
        padding: 0.65rem 0.85rem !important;
        box-shadow: {shadow} !important;
        margin-bottom: 0.35rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {c['text_muted']} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {c['text_primary']} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {c['text_primary']} !important;
    }}

    /* Alerts */
    [data-testid="stAlert"],
    .stAlert {{
        background: {card} !important;
        border: 1px solid {border} !important;
        color: {c['text_secondary']} !important;
        border-radius: 12px !important;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    .dvn-scroller {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stDataFrame"] div[data-testid="glideDataEditor"] {{
        background: {card} !important;
    }}

    /* Plotly containers — brighter chart cards */
    [data-testid="stPlotlyChart"] {{
        background: {chart_card} !important;
        border: 1px solid {border_strong} !important;
        border-radius: {radius} !important;
        padding: 0.35rem 0.5rem 0.45rem !important;
        margin-bottom: 0.4rem !important;
        box-shadow: {shadow} !important;
        overflow: hidden !important;
        min-height: 0 !important;
    }}
    [data-testid="stPlotlyChart"] iframe {{
        background: transparent !important;
    }}
    .js-plotly-plot .plotly .modebar {{
        display: none !important;
    }}

    /* Sliders / inputs in main area */
    .main input, .main textarea, .main select,
    .main [data-baseweb="select"] > div {{
        background: {card} !important;
        color: {c['text_primary']} !important;
        border-color: {border} !important;
    }}

    /* Kill light-mode overlay / decoration layers */
    [data-testid="stDecoration"] {{
        display: none !important;
    }}
    div[data-testid="collapsedControl"] {{
        color: {c['text_primary']} !important;
        background: {card} !important;
    }}

    /* Custom components ───────────────────────────────────────────── */
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(92deg, {c['cyan']} 0%, {c['purple']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }}
    .hero-sub {{
        color: {c['text_muted']};
        font-size: 1.02rem;
        margin-bottom: 0.5rem;
    }}
    .status-badge {{
        display: inline-block;
        padding: 0.32rem 0.8rem;
        margin: 0 0.4rem 0.4rem 0;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid {border};
        background: {card};
        color: {c['text_muted']};
    }}
    .status-badge.live {{
        border-color: rgba(16, 185, 129, 0.35);
        color: {c['green']};
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.12);
    }}

    .kpi-card {{
        background: linear-gradient(145deg, {card} 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid {border_strong};
        border-radius: {radius};
        padding: 0.8rem 0.85rem;
        min-height: 96px;
        box-shadow: {shadow}, inset 0 1px 0 rgba(255,255,255,0.04);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(34, 211, 238, 0.18), 0 0 0 1px rgba(34, 211, 238, 0.12);
        border-color: rgba(34, 211, 238, 0.25);
    }}
    .kpi-card.accent-cyan {{ border-top: 3px solid {c['cyan']}; }}
    .kpi-card.accent-purple {{ border-top: 3px solid {c['purple']}; }}
    .kpi-card.accent-red {{ border-top: 3px solid {c['red']}; }}
    .kpi-card.accent-green {{ border-top: 3px solid {c['green']}; }}
    .kpi-icon {{ font-size: 1.25rem; margin-bottom: 0.3rem; }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {c['text_primary']};
        line-height: 1.15;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        color: {c['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.3rem;
    }}
    .kpi-trend {{ font-size: 0.68rem; color: {c['cyan']}; margin-top: 0.2rem; }}

    .section-title {{
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: {c['text_primary']} !important;
        margin: 0.5rem 0 0.35rem 0 !important;
        letter-spacing: -0.01em;
    }}
    .section-title-first {{
        margin-top: 0.25rem !important;
    }}
    .sidebar-section {{
        margin-top: 0 !important;
        font-size: 0.95rem !important;
    }}
    .empty-state {{
        color: {c['text_muted']};
        font-size: 0.86rem;
        padding: 0.65rem 0.9rem;
        background: {card};
        border: 1px dashed {border_strong};
        border-radius: {radius};
        margin: 0.15rem 0 0.35rem;
    }}
    .empty-state-card {{
        text-align: center;
        padding: 1.5rem 1.25rem;
        margin: 0.35rem 0 0.5rem;
        background: {card};
        border: 1px solid {border_strong};
        border-radius: {radius};
        box-shadow: {shadow};
    }}
    .empty-state-card .empty-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }}
    .empty-state-card .empty-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {c['text_primary']};
        margin-bottom: 0.35rem;
    }}
    .empty-state-card .empty-hint {{
        font-size: 0.82rem;
        color: {c['text_muted']};
        line-height: 1.45;
        max-width: 28rem;
        margin: 0 auto;
    }}
    .compact-table-wrap {{
        background: {card};
        border: 1px solid {border_strong};
        border-radius: {radius};
        box-shadow: {shadow};
        padding: 0.5rem 0.65rem;
        margin: 0.2rem 0 0.4rem;
        overflow-x: auto;
    }}
    table.compact-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }}
    table.compact-table th {{
        text-align: left;
        color: {c['text_muted']};
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.68rem;
        letter-spacing: 0.05em;
        padding: 0.4rem 0.55rem;
        border-bottom: 1px solid {border_strong};
    }}
    table.compact-table td {{
        color: {c['text_secondary']};
        padding: 0.38rem 0.55rem;
        border-bottom: 1px solid {border};
    }}
    table.compact-table tr:last-child td {{
        border-bottom: none;
    }}
    .table-caption {{
        font-size: 0.72rem;
        color: {c['text_muted']};
        margin: 0.15rem 0 0.25rem 0.1rem;
    }}
    .ops-alert-row {{
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin: 0.25rem 0 0.45rem;
    }}
    .ops-alert-chip {{
        flex: 1;
        min-width: 140px;
        background: {card};
        border: 1px solid {border_strong};
        border-radius: {radius};
        padding: 0.55rem 0.75rem;
        box-shadow: {shadow};
    }}
    .ops-alert-chip .chip-value {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {c['cyan']};
    }}
    .ops-alert-chip .chip-label {{
        font-size: 0.68rem;
        color: {c['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .ops-alert-chip.warn .chip-value {{ color: {c['red']}; }}
    .app-footer {{
        text-align: center;
        color: {c['text_muted']};
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        padding: 1.25rem 0 0.5rem;
        margin-top: 0.5rem;
        border-top: 1px solid {border};
    }}
    .app-footer strong {{
        color: {c['text_secondary']};
        font-weight: 600;
    }}

    .pred-card {{
        background: linear-gradient(135deg, rgba(34,211,238,0.06) 0%, rgba(124,58,237,0.08) 100%);
        border: 1px solid {border_strong};
        border-radius: {radius};
        padding: 1.25rem;
        box-shadow: {shadow};
    }}
    .pred-eta {{ font-size: 2.75rem; font-weight: 800; }}
    .risk-pill {{
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
    }}
    .alert-box {{
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        color: #fecdd3;
        margin: 0.65rem 0;
        font-size: 0.95rem;
        line-height: 1.45;
    }}
    .insight-box {{
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #a855f7;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        color: #dbeafe;
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    .insight-box .insight-label {{
        display: block;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #a855f7;
        margin-bottom: 0.35rem;
    }}
    .insight-box p {{
        margin: 0;
        color: #dbeafe;
    }}
    .network-panel {{
        background: {card};
        border: 1px solid {border_strong};
        border-radius: {radius};
        padding: 0.95rem;
        box-shadow: {shadow};
    }}
    .network-stat {{ font-size: 1.45rem; font-weight: 700; color: {c['cyan']}; }}
    .sidebar-brand {{
        font-size: 1.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, {c['cyan']}, {c['purple']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .sidebar-desc {{ font-size: 0.78rem; color: {c['text_muted']}; line-height: 1.5; }}
    .model-status-item {{
        font-size: 0.8rem;
        color: {c['text_muted']};
        padding: 0.3rem 0;
        border-bottom: 1px solid {border};
    }}
    .model-status-ok {{ color: {c['green']} !important; font-weight: 600; }}

    hr.divider {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {c['cyan']}, {c['purple']}, transparent);
        margin: 0.55rem 0 0.75rem;
    }}

    /* Hide bulky dataframe shells when used sparingly */
    [data-testid="stDataFrame"] {{
        display: none !important;
    }}
    </style>
    """


def render_empty_state_card(icon: str, title: str, hint: str) -> str:
    return (
        f'<div class="empty-state-card">'
        f'<div class="empty-icon">{html.escape(icon)}</div>'
        f'<div class="empty-title">{html.escape(title)}</div>'
        f'<div class="empty-hint">{html.escape(hint)}</div>'
        f"</div>"
    )


def render_compact_table_html(df, *, max_rows: int = 15) -> str:
    """HTML table — no Streamlit dataframe empty shell."""
    import pandas as pd

    if df is None or (hasattr(df, "empty") and df.empty):
        return ""
    display = df.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in display.columns)
    body_rows = []
    for _, row in display.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="compact-table-wrap">'
        f'<table class="compact-table"><thead><tr>{headers}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_kpi_card(value: str, label: str, icon: str = "📊", accent: str = "cyan", trend: str = "") -> str:
    trend_html = f'<div class="kpi-trend">{trend}</div>' if trend else ""
    return (
        f'<div class="kpi-card accent-{accent}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f"{trend_html}</div>"
    )


def render_kpi_row(cards: list[str]) -> str:
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def render_bottleneck_alert() -> str:
    """HTML for bottleneck warning banner (single-line — safe for Streamlit markdown)."""
    return (
        '<div class="alert-box">⚠️ <strong>Bottleneck alert</strong> — '
        "trip touches high-stress hub(s). Consider alternate routing or capacity buffer."
        "</div>"
    )


def render_insight_box(insights: str, *, label: str = "Route insights") -> str:
    """Styled insight card for prediction commentary."""
    safe = html.escape(str(insights).strip())
    return (
        f'<div class="insight-box">'
        f'<span class="insight-label">{html.escape(label)}</span>'
        f"<p>💡 {safe}</p></div>"
    )


def render_prediction_card(
    predicted_eta: float,
    risk_level: str,
    confidence_low: float,
    confidence_high: float,
    osrm_baseline: float,
    route_lane: str,
    insights: str,
    bottleneck_warning: bool,
) -> str:
    """Core prediction metrics card (alert + insights rendered separately in the app)."""
    del insights, bottleneck_warning
    rc = risk_color(risk_level)
    return (
        '<div class="pred-card">'
        '<div style="color:#94a3b8;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;">Predicted ETA</div>'
        f'<div class="pred-eta" style="color:{rc};">{predicted_eta:.1f}'
        f'<span style="font-size:1.1rem;color:#94a3b8;"> min</span></div>'
        f'<span class="risk-pill" style="background:{rc}22;border:1px solid {rc};color:{rc};">{html.escape(str(risk_level))} RISK</span>'
        f'<p style="color:#94a3b8;margin-top:0.85rem;">Lane: <strong style="color:#f8fafc;">{html.escape(str(route_lane))}</strong></p>'
        f'<p style="color:#94a3b8;">OSRM: <strong>{osrm_baseline:.1f}</strong> min · Band: '
        f"<strong>{confidence_low:.1f} – {confidence_high:.1f}</strong> min</p>"
        "</div>"
    )