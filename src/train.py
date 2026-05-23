"""Train and persist production ETA model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRIP_ID_COLUMN,
    ProjectPaths,
    get_paths,
)
from src.features import build_ml_features, get_all_feature_names
from src.graph_features import enrich_with_graph_features, load_or_build_graph_tables, persist_graph_tables
from src.preprocessing import FeaturePreprocessor


def default_xgb_params() -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "reg:squarederror",
    }


def trip_level_split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire trips (not segments) for leakage-safe evaluation."""
    trips = df[TRIP_ID_COLUMN].dropna().unique()
    train_trips, test_trips = train_test_split(
        np.array(trips), test_size=test_size, random_state=RANDOM_STATE
    )
    train = df[df[TRIP_ID_COLUMN].isin(train_trips)].copy()
    test = df[df[TRIP_ID_COLUMN].isin(test_trips)].copy()
    return train, test


def prepare_training_frame(df: pd.DataFrame, paths: ProjectPaths | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Enrich segments with graph features and engineered columns ready for training."""
    paths = paths or get_paths()
    if "trip_hour" not in df.columns and "trip_creation_time" in df.columns:
        df = df.copy()
        ts = pd.to_datetime(df["trip_creation_time"], utc=True, errors="coerce")
        df["trip_hour"] = ts.dt.hour
        df["trip_dayofweek"] = ts.dt.dayofweek

    graph_tables = load_or_build_graph_tables(df, paths)
    persist_graph_tables(graph_tables, paths)
    enriched = enrich_with_graph_features(df, graph_tables, paths)
    featured = build_ml_features(enriched)
    return featured, graph_tables


def train_production_model(
    df: pd.DataFrame | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """
    Train final XGBoost, fit preprocessor, save bundle to models/final_eta_model.pkl.

    Returns training report dict with metrics and artifact paths.
    """
    paths = paths or get_paths()
    if df is None:
        df = pd.read_parquet(paths.processed_parquet)

    featured, graph_tables = prepare_training_frame(df, paths)
    train_df, test_df = trip_level_split(featured)

    feature_cols = get_all_feature_names(featured.columns)
    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    model = XGBRegressor(**default_xgb_params())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "n_features": len(preprocessor.feature_columns),
    }

    preprocessor.artifact.residual_std = float(residuals.std())

    bundle = {
        "model": model,
        "preprocessor": preprocessor.artifact,
        "graph_tables": graph_tables,
        "target": TARGET_COLUMN,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    paths.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths.final_model)

    metadata = {
        "model_type": "XGBRegressor",
        "feature_columns": preprocessor.feature_columns,
        "preprocessing": preprocessor.artifact.to_dict(),
        "metrics": metrics,
        "trained_at": bundle["trained_at"],
    }
    paths.model_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"bundle_path": str(paths.final_model), "metadata_path": str(paths.model_metadata), "metrics": metrics}


def load_production_bundle(paths: ProjectPaths | None = None) -> dict[str, Any]:
    """
    Load the production joblib bundle: XGBoost model, preprocessor artifact, and graph tables.

    Raises:
        FileNotFoundError: If ``models/final_eta_model.pkl`` has not been trained yet.
    """
    paths = paths or get_paths()
    if not paths.final_model.exists():
        raise FileNotFoundError(f"Model not found: {paths.final_model}. Run train_production_model() first.")
    return joblib.load(paths.final_model)
