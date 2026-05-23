"""End-to-end smoke test for ETA modeling pipeline."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from _eta_modeling_lib import build_lane_graph_tables, merge_hub_features, metrics_row, regression_metrics

RANDOM_STATE = 42
df = pd.read_parquet(ROOT / "data/processed/delivery_logistics_processed.parquet")
df = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], utc=True)
df["trip_hour"] = df["trip_creation_time"].dt.hour
df["trip_dayofweek"] = df["trip_creation_time"].dt.dayofweek

tables = build_lane_graph_tables(df)
df = merge_hub_features(df, tables, hub_col="source_center", prefix="source")
df = merge_hub_features(df, tables, hub_col="destination_center", prefix="destination")

trips = df["trip_uuid"].dropna().unique()
tr, te = train_test_split(np.array(trips), test_size=0.2, random_state=RANDOM_STATE)
train, test = df[df.trip_uuid.isin(tr)], df[df.trip_uuid.isin(te)]

BASE = ["osrm_time", "osrm_distance", "segment_osrm_time", "segment_osrm_distance", "trip_hour"]
GRAPH = ["source_pagerank", "destination_pagerank", "source_bottleneck_score", "destination_bottleneck_score"]
ALL = BASE + GRAPH

for c in ALL:
    if c not in train.columns:
        train[c] = 0
        test[c] = 0

Xtr, ytr = train[ALL].fillna(0), train["actual_time"]
Xte, yte = test[ALL].fillna(0), test["actual_time"]

m = XGBRegressor(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
m.fit(Xtr, ytr)
print(metrics_row("XGB+graph", yte, m.predict(Xte)))
print("OK")
