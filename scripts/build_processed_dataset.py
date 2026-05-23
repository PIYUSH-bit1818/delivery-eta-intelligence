"""Build cleaned logistics dataset for graph and ML pipelines."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "delivery_data.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "delivery_logistics_processed.parquet"


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    datetime_cols = [
        "trip_creation_time",
        "od_start_time",
        "od_end_time",
        "cutoff_timestamp",
    ]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["delay_ratio"] = safe_ratio(df["actual_time"], df["osrm_time"])
    df["segment_delay_ratio"] = safe_ratio(
        df["segment_actual_time"], df["segment_osrm_time"]
    )
    df["route_lane"] = (
        df["source_center"].astype(str) + " → " + df["destination_center"].astype(str)
    )

    # Quality filters for analytics / modeling
    valid = (
        (df["segment_actual_time"] >= 0)
        & (df["actual_time"] >= 0)
        & (df["segment_osrm_time"] > 0)
        & (df["osrm_time"] > 0)
        & df["source_center"].notna()
        & df["destination_center"].notna()
    )
    df = df.loc[valid].copy()

    keep_cols = [
        "data",
        "trip_uuid",
        "route_schedule_uuid",
        "route_type",
        "source_center",
        "destination_center",
        "source_name",
        "destination_name",
        "trip_creation_time",
        "od_start_time",
        "od_end_time",
        "actual_time",
        "osrm_time",
        "segment_actual_time",
        "segment_osrm_time",
        "delay_ratio",
        "segment_delay_ratio",
        "route_lane",
        "actual_distance_to_destination",
        "osrm_distance",
        "segment_osrm_distance",
        "factor",
        "segment_factor",
        "is_cutoff",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
