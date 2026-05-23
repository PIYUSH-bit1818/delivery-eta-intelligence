"""Generate notebooks/04_eta_modeling.ipynb (nbformat 4)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "notebooks" / "04_eta_modeling.ipynb"


def md(text: str):
    return new_markdown_cell(dedent(text).strip())


def code(text: str):
    return new_code_cell(dedent(text).strip())


def build_cells():
    return [
        md(
            """
            # 04 — ETA modeling with graph-enhanced ML

            **Delivery ETA Intelligence** · Predictive operations & supply chain ML

            ---

            ### Business context

            Last-mile and linehaul operators commit to **delivery ETAs** that drive customer NPS, SLA penalties, and dock labor planning. Map routers (OSRM) provide a baseline, but **route congestion**, **hub dwell**, and **network position** (bottleneck DCs) create systematic error.

            ### Why this model matters

            | Value lever | Impact |
            |-------------|--------|
            | **Customer ETA** | Fewer missed windows → lower WISMO contacts |
            | **Capacity** | Anticipate delay on high-betweenness hubs before cutoff breach |
            | **Cost** | Reduce emergency linehaul and overtime on volatile corridors |
            | **Risk** | SLA monitoring on lanes where prediction error is structurally high |

            ### Network-aware ETA prediction

            Notebook **03** quantified **bottleneck hubs** and **critical lanes**. Here we inject those signals (PageRank, degree, bottleneck score, community) into supervised learning so the model knows *which nodes and corridors* tend to overrun OSRM—not only *how far* the shipment travels.

            **Target:** `actual_time` (minutes) · **Grain:** segment rows with **trip-level** holdout to prevent leakage.

            **Upstream:** `01` → `02` → `03` · **Artifacts:** `data/processed/delivery_logistics_processed.parquet`, `outputs/tables/03_*.csv`
            """
        ),
        md(
            """
            ## Executive framing — four questions this notebook answers

            1. **Can we predict delivery ETA accurately?** — Benchmark vs linear and tree ensembles on held-out trips.
            2. **Do graph features improve quality?** — Paired XGBoost ablation (tabular-only vs +graph).
            3. **Which features matter most?** — Gain importance + optional SHAP for operations narrative.
            4. **Which routes/hubs are hardest to predict?** — Error hotspot tables for targeted process fixes.

            | Signal class | Examples |
            |--------------|----------|
            | OSRM / distance | `osrm_time`, `osrm_distance`, segment shares |
            | Temporal | Hour (cyclic), weekend |
            | Route mode | FTL vs Carting |
            | **Graph** | `source_pagerank`, `destination_bottleneck_score`, community match |
            """
        ),
        md("## 1. Environment setup"),
        code(
            """
            from __future__ import annotations

            import sys
            import warnings
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, display
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from xgboost import XGBRegressor

            warnings.filterwarnings("ignore", category=FutureWarning)

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)

            sns.set_theme(style="whitegrid", context="notebook", palette="deep")
            plt.rcParams.update({
                "figure.figsize": (11, 6),
                "figure.dpi": 110,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
            })

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
            FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
            TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

            for d in (FIGURES_DIR, TABLES_DIR):
                d.mkdir(parents=True, exist_ok=True)

            NOTEBOOK_TAG = "04"

            sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
            from _eta_modeling_lib import (
                build_lane_graph_tables,
                merge_hub_features,
                metrics_row,
                regression_metrics,
            )
            """
        ),
        md("### Helper utilities"),
        code(
            """
            def save_figure(fig: plt.Figure, name: str) -> Path:
                out = FIGURES_DIR / f"{NOTEBOOK_TAG}_{name}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return out


            def save_table(df: pd.DataFrame, name: str) -> Path:
                out = TABLES_DIR / f"{NOTEBOOK_TAG}_{name}.csv"
                df.to_csv(out, index=False)
                return out


            def load_graph_tables_from_exports(tables_dir: Path) -> dict[str, pd.DataFrame]:
                \"\"\"Load notebook 03 CSV exports into the dict expected by merge_hub_features.\"\"\"
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
            """
        ),
        md(
            """
            ## 2. Load processed logistics data

            Regenerate with: `python scripts/build_processed_dataset.py`
            """
        ),
        code(
            """
            DATA_PATH = PROCESSED_DIR / "delivery_logistics_processed.parquet"
            if not DATA_PATH.exists():
                raise FileNotFoundError(f"Missing processed data: {DATA_PATH}")

            df = pd.read_parquet(DATA_PATH)

            required = {
                "trip_uuid",
                "source_center",
                "destination_center",
                "actual_time",
                "osrm_time",
                "osrm_distance",
                "segment_osrm_time",
                "segment_osrm_distance",
                "route_type",
                "route_lane",
                "trip_creation_time",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")

            df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], utc=True, errors="coerce")
            if "trip_hour" not in df.columns:
                df["trip_hour"] = df["trip_creation_time"].dt.hour
            if "trip_dayofweek" not in df.columns:
                df["trip_dayofweek"] = df["trip_creation_time"].dt.dayofweek

            print(f"Source: {DATA_PATH}")
            print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print("\\nNull counts (key columns):")
            display(df[list(required | {"trip_hour", "trip_dayofweek"})].isna().sum().to_frame("nulls"))
            display(df[["actual_time", "osrm_time", "segment_osrm_time", "osrm_distance"]].describe().T)
            """
        ),
        md(
            """
            ## 3. Graph features — load or rebuild

            Prefer **notebook 03 exports** (`outputs/tables/03_*.csv`). If hub coverage is below **80%**, rebuild from the full segment table via `build_lane_graph_tables`.
            """
        ),
        code(
            """
            graph_files = [
                "03_bottleneck_hubs.csv",
                "03_hub_rankings_pagerank.csv",
                "03_hub_rankings_degree.csv",
                "03_hub_communities.csv",
            ]
            if all((TABLES_DIR / f).exists() for f in graph_files):
                graph_tables = load_graph_tables_from_exports(TABLES_DIR)
                coverage = graph_hub_coverage(df, graph_tables)
                print(f"Loaded graph tables from {TABLES_DIR} — hub coverage: {coverage:.1%}")
            else:
                coverage = 0.0
                graph_tables = None
                print("One or more 03_*.csv files missing; will rebuild graph tables.")

            COVERAGE_THRESHOLD = 0.80
            if graph_tables is None or coverage < COVERAGE_THRESHOLD:
                print("Rebuilding lane graph tables (build_lane_graph_tables)...")
                graph_tables = build_lane_graph_tables(df)
                coverage = graph_hub_coverage(df, graph_tables)
                print(f"Post-rebuild hub coverage: {coverage:.1%}")

            df = merge_hub_features(df, graph_tables, hub_col="source_center", prefix="source")
            df = merge_hub_features(df, graph_tables, hub_col="destination_center", prefix="destination")

            graph_cols = [
                "source_pagerank", "destination_pagerank",
                "source_degree", "destination_degree",
                "source_bottleneck_score", "destination_bottleneck_score",
                "source_community", "destination_community",
            ]
            display(df[graph_cols].describe().T)
            """
        ),
        md(
            """
            **Why graph features may improve ETA prediction**

            OSRM encodes **geometry and speed limits**, not **operational friction** at chokepoint DCs.
            **PageRank / degree** proxy hub importance in the flow network; **bottleneck score** flags structural congestion;
            **community** captures regional sub-networks where delays covary.
            Joining these at **source** and **destination** tells the model whether a trip touches stressed parts of the backbone—
            improving predictions beyond distance and clock time alone.
            """
        ),
        md(
            """
            ## 4. Feature engineering

            - **Temporal:** hour sin/cos, weekend flag
            - **Route mode:** `route_type` dummies (FTL vs Carting)
            - **Distance structure:** ratios from OSRM / segment fields (no `delay_ratio` / `segment_delay_ratio` — those leak the target)
            - **Graph interactions:** products and contrasts of hub centrality / bottleneck
            - **Robustness:** fill NaN/inf, clip numeric features at training 99th percentile
            """
        ),
        code(
            """
            work = df.copy()

            # Temporal encodings
            work["trip_hour_sin"] = np.sin(2 * np.pi * work["trip_hour"] / 24)
            work["trip_hour_cos"] = np.cos(2 * np.pi * work["trip_hour"] / 24)
            work["is_weekend"] = (work["trip_dayofweek"] >= 5).astype(int)

            # Route type dummies
            rt = pd.get_dummies(work["route_type"].astype(str), prefix="route_type", dtype=float)
            work = pd.concat([work, rt], axis=1)

            # Distance / time ratios (OSRM-only — no post-hoc actuals)
            work["segment_time_share"] = work["segment_osrm_time"] / work["osrm_time"].replace(0, np.nan)
            work["segment_distance_share"] = work["segment_osrm_distance"] / work["osrm_distance"].replace(0, np.nan)
            work["segment_osrm_to_trip_osrm_time"] = work["segment_osrm_time"] / work["osrm_time"].replace(0, np.nan)

            # Graph interactions
            work["graph_pagerank_product"] = work["source_pagerank"] * work["destination_pagerank"]
            work["graph_bottleneck_sum"] = work["source_bottleneck_score"] + work["destination_bottleneck_score"]
            work["graph_pagerank_gap"] = (work["source_pagerank"] - work["destination_pagerank"]).abs()
            work["same_community"] = (work["source_community"] == work["destination_community"]).astype(float)

            ROUTE_DUMMY_COLS = [c for c in work.columns if c.startswith("route_type_")]

            BASE_FEATURES = [
                "osrm_time",
                "osrm_distance",
                "segment_osrm_time",
                "segment_osrm_distance",
                "trip_hour_sin",
                "trip_hour_cos",
                "is_weekend",
                "segment_time_share",
                "segment_distance_share",
                "segment_osrm_to_trip_osrm_time",
                *ROUTE_DUMMY_COLS,
            ]

            GRAPH_FEATURES = [
                "source_pagerank",
                "destination_pagerank",
                "source_degree",
                "destination_degree",
                "source_bottleneck_score",
                "destination_bottleneck_score",
                "source_community",
                "destination_community",
                "graph_pagerank_product",
                "graph_bottleneck_sum",
                "graph_pagerank_gap",
                "same_community",
            ]

            ALL_FEATURES = BASE_FEATURES + GRAPH_FEATURES
            TARGET = "actual_time"

            # Sanitize infinities / missing
            numeric_feats = [c for c in ALL_FEATURES if c in work.columns]
            work[numeric_feats] = work[numeric_feats].replace([np.inf, -np.inf], np.nan)
            work[numeric_feats] = work[numeric_feats].fillna(work[numeric_feats].median())

            print(f"BASE_FEATURES ({len(BASE_FEATURES)}): {BASE_FEATURES}")
            print(f"GRAPH_FEATURES ({len(GRAPH_FEATURES)}): {GRAPH_FEATURES}")
            """
        ),
        md(
            """
            ## 5. Train / test split (by `trip_uuid`)

            **Leakage warning:** A random row split would place segments from the same trip in both train and test, inflating scores because `actual_time` is trip-level and repeated across segments. We hold out **20% of trips** entirely.
            """
        ),
        code(
            """
            trips = work["trip_uuid"].dropna().unique()
            train_trips, test_trips = train_test_split(
                np.array(trips), test_size=0.2, random_state=RANDOM_STATE
            )
            train = work[work["trip_uuid"].isin(train_trips)].copy()
            test = work[work["trip_uuid"].isin(test_trips)].copy()

            print(
                f"Trips: train {len(train_trips):,} | test {len(test_trips):,} "
                f"(total {len(trips):,})"
            )
            print(f"Rows: train {len(train):,} | test {len(test):,}")

            # Clip numeric features at train 99th percentile
            clip_bounds = {}
            for col in numeric_feats:
                hi = train[col].quantile(0.99)
                clip_bounds[col] = hi
                train[col] = train[col].clip(upper=hi)
                test[col] = test[col].clip(upper=hi)

            X_train_base = train[BASE_FEATURES]
            X_test_base = test[BASE_FEATURES]
            X_train_all = train[ALL_FEATURES]
            X_test_all = test[ALL_FEATURES]
            y_train = train[TARGET]
            y_test = test[TARGET]
            """
        ),
        md("## 6. Linear regression (scaled)"),
        code(
            """
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train_all)
            X_te_s = scaler.transform(X_test_all)

            lin = LinearRegression()
            lin.fit(X_tr_s, y_train)
            y_pred_lin = lin.predict(X_te_s)

            lin_metrics = metrics_row("LinearRegression", y_test, y_pred_lin)
            display(pd.DataFrame([lin_metrics]))
            """
        ),
        md(
            """
            **Interpretation:** Linear regression on scaled features establishes a **transparent baseline**.
            Large gaps vs tree models indicate **non-linear interactions** (hub congestion × time-of-day) that justify XGBoost for production.
            """
        ),
        md("## 7. Random Forest & XGBoost — model comparison"),
        code(
            """
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=14,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train_all, y_train)
            y_pred_rf = rf.predict(X_test_all)

            xgb_full = XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            xgb_full.fit(X_train_all, y_train)
            y_pred_xgb = xgb_full.predict(X_test_all)

            comparison = pd.DataFrame([
                lin_metrics,
                metrics_row("RandomForest", y_test, y_pred_rf),
                metrics_row("XGBoost+graph", y_test, y_pred_xgb),
            ])
            display(comparison)
            save_table(comparison, "model_comparison")

            best_row = comparison.loc[comparison["MAE"].idxmin()]
            print(f"Best model by test MAE: {best_row['Model']} (MAE={best_row['MAE']:.2f} min, R²={best_row['R2']:.4f})")
            """
        ),
        md(
            """
            **Interpretation:** Tree ensembles typically dominate when OSRM underfits **hub dwell** and **mode effects** (FTL vs Carting).
            Persist `04_model_comparison.csv` for model governance and champion/challenger reviews.
            """
        ),
        md(
            """
            ## 8. XGBoost: without graph vs with graph

            Isolates the marginal value of network features on the **same trip holdout**.
            """
        ),
        code(
            """
            xgb_base = XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            xgb_base.fit(X_train_base, y_train)
            y_pred_base = xgb_base.predict(X_test_base)

            xgb_graph = XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            xgb_graph.fit(X_train_all, y_train)
            y_pred_graph = xgb_graph.predict(X_test_all)

            xgb_compare = pd.DataFrame([
                metrics_row("XGBoost (no graph)", y_test, y_pred_base),
                metrics_row("XGBoost (+ graph)", y_test, y_pred_graph),
            ])
            display(xgb_compare)
            save_table(xgb_compare, "xgb_graph_ablation")

            m_base = xgb_compare.loc[xgb_compare["Model"] == "XGBoost (no graph)", "MAE"].iloc[0]
            m_graph = xgb_compare.loc[xgb_compare["Model"] == "XGBoost (+ graph)", "MAE"].iloc[0]
            delta = m_base - m_graph
            print(
                f"Business insight: graph features change test MAE by {delta:+.2f} min "
                f"({'improvement' if delta > 0 else 'no improvement'} vs OSRM+tabular only)."
            )

            best_xgb = xgb_graph
            y_pred_best = y_pred_graph

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=xgb_compare, x="Model", y="MAE", ax=ax,
                hue="Model", palette="viridis", legend=False,
            )
            ax.set_title("Graph feature ablation — test MAE")
            fig.tight_layout()
            save_figure(fig, "xgb_graph_ablation_bar")
            """
        ),
        md(
            """
            **Business insight (graph impact):** If **XGBoost (+ graph)** beats **XGBoost (no graph)** on MAE, network analytics from notebook 03
            is not only descriptive—it is **predictive**. Invest in nightly graph refresh and graph-aware routing for high-bottleneck origins.
            If lift is negligible, prioritize OSRM calibration and segment temporal features before embedding complexity.
            """
        ),
        md("## 9. Feature importance (best XGBoost)"),
        code(
            """
            importances = pd.Series(best_xgb.feature_importances_, index=ALL_FEATURES)
            top_imp = importances.sort_values(ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(10, 8))
            top_imp.sort_values().plot.barh(ax=ax, color="steelblue")
            ax.set_title("Top 20 features — XGBoost (+ graph)")
            ax.set_xlabel("Gain importance")
            fig.tight_layout()
            save_figure(fig, "feature_importance_top20")
            display(top_imp.to_frame("importance"))

            graph_in_top = [f for f in top_imp.index if f in GRAPH_FEATURES]
            print(f"Graph features in top 20: {graph_in_top}")
            """
        ),
        md(
            """
            **Interpretation:** OSRM columns usually rank highest (expected—router is the prior).
            When **graph** features (e.g. `source_bottleneck_score`, `graph_bottleneck_sum`) appear in the top tier, ETA error is partly **structural**,
            not just distance-driven—align with capacity investments from notebook 03.
            """
        ),
        md("## 10. Residual diagnostics"),
        code(
            """
            residuals = y_test - y_pred_best

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(residuals, bins=60, color="steelblue", edgecolor="white")
            axes[0].set_title("Residual distribution (test)")
            axes[0].set_xlabel("actual − predicted (minutes)")

            axes[1].scatter(y_test, y_pred_best, alpha=0.15, s=8, c="steelblue")
            lim = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
            axes[1].plot(lim, lim, "r--", lw=1)
            axes[1].set_xlabel("Actual time")
            axes[1].set_ylabel("Predicted time")
            axes[1].set_title("Actual vs predicted")
            fig.tight_layout()
            save_figure(fig, "residual_diagnostics")

            worst = test.assign(
                predicted=y_pred_best,
                residual=residuals,
                abs_error=np.abs(residuals),
            ).nlargest(15, "abs_error")[
                [
                    "trip_uuid",
                    "route_lane",
                    "route_type",
                    "source_center",
                    "destination_center",
                    "actual_time",
                    "predicted",
                    "abs_error",
                ]
            ]
            display(worst)
            save_table(worst, "worst_predictions")
            """
        ),
        md(
            """
            **Interpretation:** Residual skew suggests **long-tail trips** (underprediction on very long `actual_time`).
            Review worst rows for shared lanes or hubs—systematic bias warrants segment-specific calibration or hub SLA playbooks.
            """
        ),
        md("## 11. MAE by route type (FTL vs Carting)"),
        code(
            """
            by_route = []
            for rt, grp in test.assign(pred=y_pred_best).groupby("route_type"):
                idx = grp.index
                by_route.append({
                    "route_type": rt,
                    "n": len(grp),
                    "MAE": mean_absolute_error(y_test.loc[idx], grp["pred"]),
                })
            mae_route = pd.DataFrame(by_route).sort_values("MAE", ascending=False)
            display(mae_route)
            save_table(mae_route, "mae_by_route_type")

            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=mae_route, x="route_type", y="MAE", ax=ax, palette="muted")
            ax.set_title("Test MAE by route type")
            fig.tight_layout()
            save_figure(fig, "mae_by_route_type")
            """
        ),
        md(
            """
            **Operational implication:** Higher MAE on one mode (often **Carting** with multi-segment consolidation) implies
            **different model heads or calibration** per `route_type` in production, or separate cutoff policies by mode.
            """
        ),
        md("## 12. Error hotspots — hubs and lanes"),
        code(
            """
            eval_df = test.assign(pred=y_pred_best, abs_error=np.abs(y_test - y_pred_best))

            hub_errors = (
                eval_df.groupby("source_center", as_index=False)
                .agg(segments=("abs_error", "count"), MAE=("abs_error", "mean"))
                .rename(columns={"source_center": "hub_id"})
                .nlargest(25, "MAE")
            )
            dest_errors = (
                eval_df.groupby("destination_center", as_index=False)
                .agg(segments=("abs_error", "count"), MAE=("abs_error", "mean"))
                .rename(columns={"destination_center": "hub_id"})
                .nlargest(25, "MAE")
            )
            route_errors = (
                eval_df.groupby("route_lane", as_index=False)
                .agg(segments=("abs_error", "count"), MAE=("abs_error", "mean"))
                .nlargest(25, "MAE")
            )

            display(Markdown("**Worst source hubs (by test MAE)**"))
            display(hub_errors)
            display(Markdown("**Worst destination hubs**"))
            display(dest_errors)
            display(Markdown("**Worst lanes**"))
            display(route_errors)

            save_table(hub_errors, "error_hotspots_source_hubs")
            save_table(dest_errors, "error_hotspots_destination_hubs")
            save_table(route_errors, "error_hotspots_routes")
            """
        ),
        md(
            """
            **Hotspot narrative:** High-MAE **lanes** reflect unstable corridors (congestion unpredictability, poor OSRM fit).
            High-MAE **hubs** align with bottleneck candidates—pair with live dwell telemetry and dynamic rerouting rules.
            Filter tables with `segments >= 30` before executive escalation to avoid noise from rare lanes.
            """
        ),
        md(
            """
            ## 13. Production recommendations

            | Capability | Recommendation |
            |------------|----------------|
            | **Real-time ETA** | Score XGB (+ graph) on trip creation; refresh when segment scan events arrive |
            | **Graph-aware routing** | Penalize paths through top `bottleneck_score` hubs in route optimization |
            | **Congestion monitoring** | Dashboard MAE + volume for `04_error_hotspots_*` tables |
            | **Dynamic rerouting** | Trigger when predicted ETA exceeds SLA and alternate lane exists |
            | **SLA risk** | Alert on trips touching high-MAE hubs in top decile of `source_bottleneck_score` |

            **Engineering:** Version `ALL_FEATURES`, clip bounds, and graph tables with each model artifact in `models/`.
            """
        ),
        md(
            """
            ## 14. Executive summary

            | Dimension | Outcome (fill after run) |
            |-----------|-------------------------|
            | **Best model** | Lowest test MAE in `04_model_comparison.csv` |
            | **Best features** | Top of `04_feature_importance` / SHAP |
            | **Graph impact** | ΔMAE from `04_xgb_graph_ablation.csv` |
            | **Hard to predict** | `04_error_hotspots_routes.csv`, hub exports |

            ### Operational recommendations

            1. Deploy **graph-augmented XGBoost** when ablation shows positive MAE lift; otherwise ship OSRM + temporal baseline first.
            2. Run **trip-level** monitoring; never score with random segment splits in production backtests.
            3. Target **top bottleneck hubs** from notebook 03 that also appear in error hotspots.
            4. Separate **FTL vs Carting** calibration if MAE gap is material.

            ### Next engineering steps

            - **Node2Vec** hub embeddings → replace hand-crafted graph scalars
            - **Temporal graph learning** → weekly evolving centrality features
            - **Real-time traffic APIs** → dynamic edge weights
            - **Streaming inference** → Kafka/feature store with nightly graph batch join
            """
        ),
        md("### Optional: SHAP explainability"),
        code(
            """
            try:
                import shap

                explainer = shap.TreeExplainer(best_xgb)
                sample = X_test_all.sample(min(5000, len(X_test_all)), random_state=RANDOM_STATE)
                shap_values = explainer.shap_values(sample)

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, sample, show=False, max_display=20)
                fig = plt.gcf()
                save_figure(fig, "shap_summary")
                print("SHAP summary saved.")
            except ImportError:
                print("SHAP not installed — skip with: pip install shap")
            """
        ),
        md(
            """
            **SHAP interpretation:** Features pushing SHAP mass positive increase predicted ETA.
            Use with operations to validate whether bottleneck and OSRM fields dominate on high-delay corridors.
            """
        ),
    ]


def main() -> None:
    nb = new_notebook(
        cells=build_cells(),
        metadata={
            "kernelspec": {
                "display_name": "Delivery ETA (.venv)",
                "language": "python",
                "name": "delivery-eta",
            }
        },
    )
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    with OUT_NB.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT_NB} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
