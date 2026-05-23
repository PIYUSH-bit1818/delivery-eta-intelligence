"""Generate notebooks/05_production_pipeline_and_dashboard.ipynb."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_production_pipeline_and_dashboard.ipynb"


def md(t: str):
    return new_markdown_cell(dedent(t).strip())


def code(t: str):
    return new_code_cell(dedent(t).strip())


def build():
    return [
        md(
            """
            # 05 — Production pipeline & dashboard readiness

            **Delivery ETA Intelligence** · ML engineering phase

            ---

            ### Why production ETA systems matter

            Exploratory notebooks prove value; **operations** need a repeatable system: versioned features, trip-safe training, batch scoring, and monitoring hooks. This notebook wires the **production path** from processed logistics data to saved model artifacts, inference API, risk scoring, and dashboard-ready outputs.

            | Capability | Business value |
            |------------|----------------|
            | **Real-time routing** | Score ETAs when a trip is created or rerouted |
            | **Congestion-aware inference** | Graph hub features adjust OSRM-only estimates |
            | **Operational deployment** | Same code in training, batch, and Streamlit |

            **Inputs (API):** `source_center`, `destination_center`, `osrm_distance`, `trip_hour`, `route_type`  
            **Outputs:** `predicted_eta`, `risk_level`, `bottleneck_warning`, `confidence_band`, route insights
            """
        ),
        md(
            """
            ## 1. Project architecture

            ```
            delivery_eta/
            │
            ├── notebooks/          # Research & validation (01–05)
            ├── src/                # Production Python modules
            │   ├── config.py       # Paths, constants, risk thresholds
            │   ├── features.py     # Feature generation
            │   ├── graph_features.py
            │   ├── preprocessing.py
            │   ├── train.py
            │   ├── inference.py
            │   └── dashboard_utils.py
            ├── models/             # final_eta_model.pkl, graph_hub_features.pkl
            ├── app/                # streamlit_app.py
            ├── outputs/
            │   ├── figures/
            │   ├── tables/
            │   └── predictions/
            └── data/
                ├── raw/
                └── processed/
            ```

            | Module | Responsibility |
            |--------|----------------|
            | `features` | Temporal, route dummies, OSRM ratios, graph interactions |
            | `graph_features` | Build/load hub tables; merge source & destination |
            | `preprocessing` | Fit clip/fill; transform at inference |
            | `train` | Trip-level split; train XGBoost; save bundle |
            | `inference` | `predict_eta()` + risk categories |
            | `dashboard_utils` | Plotly charts for Streamlit |
            """
        ),
        md("## 2. Environment & imports"),
        code(
            """
            import json
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, display

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            sns.set_theme(style="whitegrid", context="notebook")

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            sys.path.insert(0, str(PROJECT_ROOT))

            from src.config import get_paths
            from src.dashboard_utils import (
                eta_summary_cards,
                hub_hotspot_table,
                plot_congestion_by_lane,
                plot_eta_vs_osrm,
                plot_risk_distribution,
                save_plotly,
            )
            from src.features import build_ml_features, get_all_feature_names
            from src.graph_features import enrich_with_graph_features, load_or_build_graph_tables
            from src.inference import ETAInferencePipeline, predict_eta
            from src.train import train_production_model, load_production_bundle

            paths = get_paths(PROJECT_ROOT)
            NOTEBOOK_TAG = "05"
            paths.outputs_figures.mkdir(parents=True, exist_ok=True)
            paths.outputs_predictions.mkdir(parents=True, exist_ok=True)
            paths.outputs_tables.mkdir(parents=True, exist_ok=True)

            def save_figure(fig, name: str):
                out = paths.outputs_figures / f"{NOTEBOOK_TAG}_{name}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return out

            def save_table(df, name: str):
                out = paths.outputs_tables / f"{NOTEBOOK_TAG}_{name}.csv"
                df.to_csv(out, index=False)
                return out

            print("Project root:", PROJECT_ROOT)
            """
        ),
        md("## 3. Train final production model"),
        code(
            """
            if not paths.final_model.exists():
                report = train_production_model(paths=paths)
            else:
                bundle = load_production_bundle(paths)
                report = {"metrics": bundle["metrics"], "bundle_path": str(paths.final_model)}
                print("Loaded existing production model (delete models/final_eta_model.pkl to retrain)")

            display(pd.DataFrame([report["metrics"]]))
            with open(paths.model_metadata) as f:
                meta = json.load(f)
            print(f"Features: {len(meta['feature_columns'])}")
            print(f"Artifact: {paths.final_model}")
            """
        ),
        md(
            """
            **Engineering note:** Training uses **trip-level holdout** (no segment leakage). Artifacts: `models/final_eta_model.pkl`, `models/model_metadata.json`, `models/graph_hub_features.pkl`.
            """
        ),
        md("## 4. Inference pipeline — single trip API"),
        code(
            """
            pipeline = ETAInferencePipeline(paths)

            demo = pipeline.predict_eta(
                source_center="IND000000ACB",
                destination_center="IND562132AAA",
                osrm_distance=85.0,
                trip_hour=14,
                route_type="FTL",
            )
            pd.DataFrame([demo.to_dict()])
            """
        ),
        md(
            """
            ### Risk scoring (operational)

            | Level | Typical triggers |
            |-------|------------------|
            | **LOW** | Normal bottleneck scores; ETA near OSRM |
            | **MEDIUM** | Elevated hub score or moderate ETA/OSRM ratio |
            | **HIGH** | High bottleneck + delay ratio |
            | **CRITICAL** | Top-decile bottleneck hubs and large ETA overrun |

            `bottleneck_warning=True` when either hub exceeds the medium bottleneck threshold — route for capacity review.
            """
        ),
        md("## 5. Batch prediction simulation"),
        code(
            """
            hist = pd.read_parquet(paths.processed_parquet)
            sample = hist.drop_duplicates("trip_uuid").sample(500, random_state=RANDOM_STATE)

            batch_input = sample[[
                "source_center", "destination_center", "osrm_distance",
                "trip_hour", "route_type", "osrm_time",
            ]].copy()
            if "trip_dayofweek" in sample.columns:
                batch_input["trip_dayofweek"] = sample["trip_dayofweek"].values

            scored = pipeline.predict_batch(batch_input)
            scored["prediction_error"] = np.nan
            if "actual_time" in sample.columns:
                scored["actual_time"] = sample["actual_time"].values
                scored["prediction_error"] = (scored["actual_time"] - scored["predicted_eta"]).abs()

            out_path = paths.outputs_predictions / "sample_predictions.csv"
            scored.to_csv(out_path, index=False)
            print(f"Saved {len(scored)} predictions → {out_path}")

            top_risky = scored.sort_values("predicted_eta", ascending=False).head(10)[
                ["route_lane", "predicted_eta", "risk_level", "bottleneck_warning", "route_insights"]
            ]
            display(top_risky)
            save_table(top_risky, "top_risky_lanes")
            """
        ),
        md("## 6. Dashboard-ready visualizations"),
        code(
            """
            cards = eta_summary_cards(scored)
            for k, v in cards.items():
                print(f"{k}: {v}")

            fig1 = plot_risk_distribution(scored)
            save_plotly(fig1, paths.outputs_figures / f"{NOTEBOOK_TAG}_risk_distribution")

            fig2 = plot_eta_vs_osrm(scored)
            save_plotly(fig2, paths.outputs_figures / f"{NOTEBOOK_TAG}_eta_vs_osrm")

            fig3 = plot_congestion_by_lane(scored)
            save_plotly(fig3, paths.outputs_figures / f"{NOTEBOOK_TAG}_congestion_lanes")

            hotspots = hub_hotspot_table(scored)
            display(hotspots.head(15))
            save_table(hotspots, "hub_hotspots")

            # Matplotlib summary for static reports
            fig, ax = plt.subplots(figsize=(8, 4))
            scored["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).plot.bar(ax=ax, color="steelblue")
            ax.set_title("Risk distribution (batch sample)")
            save_figure(fig, "risk_bar")
            """
        ),
        md(
            """
            **Dashboard:** Run `streamlit run app/streamlit_app.py` — consumes `sample_predictions.csv` and live `predict_eta()` inputs.

            HTML Plotly exports saved under `outputs/figures/05_*.html` for embedding without a server.
            """
        ),
        md(
            """
            ## 7. Model monitoring concepts

            | Monitor | Definition | Action |
            |---------|------------|--------|
            | **Data drift** | Input distribution shift (distance, hour, route mix) | Alert + review retrain |
            | **Concept drift** | MAE ↑ while inputs stable | Refresh model / graph tables |
            | **SLA degradation** | % trips CRITICAL risk ↑ | Ops playbook on bottleneck hubs |
            | **Feature freshness** | Graph tables stale > 7d | Nightly `build_lane_graph_tables` job |
            | **Retraining** | Rolling 90d window monthly | Champion/challenger via trip holdout |

            Log every inference: inputs, prediction, risk, model version, graph snapshot date.
            """
        ),
        md(
            """
            ## 8. Production & deployment architecture

            ```
            [Trip event] → [Feature store / router OSRM]
                    ↓
            [Inference service: predict_eta()]
                    ↓
            [Risk engine] → [Dashboard / SLA alerts]
                    ↓
            [Kafka sink] → [Monitoring & retrain pipeline]
            ```

            - **API serving:** FastAPI wrapper around `ETAInferencePipeline` (single worker loads bundle at startup).
            - **Streaming:** Kafka topic `trip.created` → stream processor joins graph features from Redis/feature store.
            - **Graph refresh:** Nightly batch on processed segments; publish `graph_hub_features.pkl` version.
            - **Real-time inference:** p99 target < 50ms with preloaded model + in-memory hub dict lookup.
            """
        ),
        md("## 9. Executive summary"),
        code(
            """
            metrics = report["metrics"]
            print("=" * 60)
            print("PRODUCTION ETA SYSTEM — PHASE 5 SUMMARY")
            print("=" * 60)
            print(f"Model: XGBoost @ {paths.final_model.name}")
            print(f"Test MAE: {metrics.get('MAE', 'n/a'):.2f} min | R²: {metrics.get('R2', 0):.4f}")
            print(f"Batch predictions: {paths.outputs_predictions / 'sample_predictions.csv'}")
            print(f"Streamlit: streamlit run app/streamlit_app.py")
            print("Graph features: embedded in bundle (PageRank, bottleneck, community)")
            print("Next: Node2Vec embeddings · temporal graph · traffic API · streaming deploy")
            """
        ),
    ]


def main():
    nb = new_notebook(
        cells=build(),
        metadata={"kernelspec": {"display_name": "Delivery ETA (.venv)", "language": "python", "name": "delivery-eta"}},
    )
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
