"""Production ETA inference and operational risk scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CONFIDENCE_Z,
    RISK_BOTTLENECK_CRITICAL,
    RISK_BOTTLENECK_HIGH,
    RISK_BOTTLENECK_MEDIUM,
    ProjectPaths,
    get_paths,
)
from src.features import build_ml_features, expand_minimal_request
from src.graph_features import enrich_with_graph_features
from src.preprocessing import FeaturePreprocessor, PreprocessingArtifact
from src.train import load_production_bundle


@dataclass
class ETAPrediction:
    """Structured inference response."""

    predicted_eta: float
    risk_level: str
    bottleneck_warning: bool
    confidence_low: float
    confidence_high: float
    osrm_baseline: float
    eta_vs_osrm_ratio: float
    source_bottleneck_score: float
    destination_bottleneck_score: float
    route_lane: str
    insights: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_eta": self.predicted_eta,
            "risk_level": self.risk_level,
            "bottleneck_warning": self.bottleneck_warning,
            "confidence_band": (self.confidence_low, self.confidence_high),
            "osrm_baseline": self.osrm_baseline,
            "eta_vs_osrm_ratio": self.eta_vs_osrm_ratio,
            "source_bottleneck_score": self.source_bottleneck_score,
            "destination_bottleneck_score": self.destination_bottleneck_score,
            "route_lane": self.route_lane,
            "insights": self.insights,
        }


class ETAInferencePipeline:
    """Load once; score trips in real time or batch."""

    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or get_paths()
        self.bundle = load_production_bundle(self.paths)
        self.model = self.bundle["model"]
        self.artifact: PreprocessingArtifact = self.bundle["preprocessor"]
        self.graph_tables = self.bundle["graph_tables"]
        self.preprocessor = FeaturePreprocessor(self.artifact.feature_columns)
        self.preprocessor.artifact = self.artifact

    def _frame_from_inputs(
        self,
        source_center: str,
        destination_center: str,
        osrm_distance: float,
        trip_hour: int,
        route_type: str,
        *,
        osrm_time: float | None = None,
        trip_dayofweek: int | None = None,
    ) -> pd.DataFrame:
        row = {
            "source_center": source_center,
            "destination_center": destination_center,
            "osrm_distance": float(osrm_distance),
            "trip_hour": int(trip_hour),
            "route_type": route_type,
        }
        if osrm_time is not None:
            row["osrm_time"] = float(osrm_time)
        if trip_dayofweek is not None:
            row["trip_dayofweek"] = int(trip_dayofweek)

        df = pd.DataFrame([row])
        df = expand_minimal_request(
            df,
            osrm_minutes_per_km=self.artifact.osrm_minutes_per_km,
            default_trip_dayofweek=trip_dayofweek if trip_dayofweek is not None else 2,
        )
        return df

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score many trips; input must include minimal or full feature columns."""
        work = df.copy()
        if "osrm_time" not in work.columns:
            work = expand_minimal_request(work, osrm_minutes_per_km=self.artifact.osrm_minutes_per_km)
        if "trip_hour" in work.columns and "trip_dayofweek" not in work.columns:
            work["trip_dayofweek"] = 2

        work = enrich_with_graph_features(work, self.graph_tables, self.paths)
        work = build_ml_features(work)
        X = self.preprocessor.transform(work)
        preds = self.model.predict(X)
        work["predicted_eta"] = preds
        return self._attach_risk_columns(work)

    def predict_eta(
        self,
        source_center: str,
        destination_center: str,
        osrm_distance: float,
        trip_hour: int,
        route_type: str,
        *,
        osrm_time: float | None = None,
        trip_dayofweek: int | None = None,
    ) -> ETAPrediction:
        """Single-trip production inference API."""
        df = self._frame_from_inputs(
            source_center,
            destination_center,
            osrm_distance,
            trip_hour,
            route_type,
            osrm_time=osrm_time,
            trip_dayofweek=trip_dayofweek,
        )
        scored = self.predict_batch(df)
        row = scored.iloc[0]
        return ETAPrediction(
            predicted_eta=float(row["predicted_eta"]),
            risk_level=str(row["risk_level"]),
            bottleneck_warning=bool(row["bottleneck_warning"]),
            confidence_low=float(row["confidence_low"]),
            confidence_high=float(row["confidence_high"]),
            osrm_baseline=float(row["osrm_time"]),
            eta_vs_osrm_ratio=float(row["eta_vs_osrm_ratio"]),
            source_bottleneck_score=float(row.get("source_bottleneck_score", 0) or 0),
            destination_bottleneck_score=float(row.get("destination_bottleneck_score", 0) or 0),
            route_lane=str(row.get("route_lane", f"{source_center} → {destination_center}")),
            insights=str(row.get("route_insights", "")),
        )

    def _attach_risk_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["route_lane"] = (
            out["source_center"].astype(str) + " → " + out["destination_center"].astype(str)
        )

        src_bn = out.get("source_bottleneck_score", pd.Series(0, index=out.index)).fillna(0)
        dst_bn = out.get("destination_bottleneck_score", pd.Series(0, index=out.index)).fillna(0)
        max_bn = np.maximum(src_bn, dst_bn)

        out["osrm_time"] = out.get("osrm_time", self.artifact.osrm_time_median)
        out["eta_vs_osrm_ratio"] = out["predicted_eta"] / out["osrm_time"].replace(0, np.nan)
        out["eta_vs_osrm_ratio"] = out["eta_vs_osrm_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

        std = self.artifact.residual_std or 30.0
        out["confidence_low"] = out["predicted_eta"] - CONFIDENCE_Z * std
        out["confidence_high"] = out["predicted_eta"] + CONFIDENCE_Z * std

        out["bottleneck_warning"] = max_bn >= RISK_BOTTLENECK_MEDIUM
        out["risk_level"] = [
            _risk_category(bn, ratio)
            for bn, ratio in zip(max_bn, out["eta_vs_osrm_ratio"])
        ]
        out["route_insights"] = [
            _route_insight(bn, ratio, warn)
            for bn, ratio, warn in zip(max_bn, out["eta_vs_osrm_ratio"], out["bottleneck_warning"])
        ]
        return out


def _risk_category(max_bottleneck: float, eta_ratio: float) -> str:
    score = 0
    if max_bottleneck >= RISK_BOTTLENECK_CRITICAL:
        score += 3
    elif max_bottleneck >= RISK_BOTTLENECK_HIGH:
        score += 2
    elif max_bottleneck >= RISK_BOTTLENECK_MEDIUM:
        score += 1

    if eta_ratio >= 1.8:
        score += 2
    elif eta_ratio >= 1.4:
        score += 1

    if score >= 4:
        return "CRITICAL"
    if score >= 3:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


def _route_insight(max_bottleneck: float, eta_ratio: float, warning: bool) -> str:
    parts = []
    if warning:
        parts.append("Touches high-bottleneck hub; expect elevated dwell risk.")
    if eta_ratio >= 1.5:
        parts.append("Predicted ETA materially above OSRM baseline.")
    if not parts:
        parts.append("Within normal network operating envelope.")
    return " ".join(parts)


def predict_eta(
    source_center: str,
    destination_center: str,
    osrm_distance: float,
    trip_hour: int,
    route_type: str,
    *,
    model_path: Path | None = None,
    osrm_time: float | None = None,
    trip_dayofweek: int | None = None,
) -> dict[str, Any]:
    """
    Score a single trip and return a JSON-serializable dict.

    Keys include ``predicted_eta``, ``risk_level``, ``bottleneck_warning``,
    confidence band, OSRM baseline, and route insights.
    """
    pipeline = ETAInferencePipeline(get_paths())
    return pipeline.predict_eta(
        source_center,
        destination_center,
        osrm_distance,
        trip_hour,
        route_type,
        osrm_time=osrm_time,
        trip_dayofweek=trip_dayofweek,
    ).to_dict()
