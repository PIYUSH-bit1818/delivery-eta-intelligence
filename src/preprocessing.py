"""Preprocessing transformers for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLIP_PERCENTILE
from src.features import get_all_feature_names


@dataclass
class PreprocessingArtifact:
    """Fitted preprocessing state shipped with the production model."""

    feature_columns: list[str]
    fill_values: dict[str, float] = field(default_factory=dict)
    clip_upper: dict[str, float] = field(default_factory=dict)
    route_type_columns: list[str] = field(default_factory=list)
    osrm_minutes_per_km: float = 1.0
    residual_std: float = 0.0
    osrm_time_median: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "fill_values": self.fill_values,
            "clip_upper": self.clip_upper,
            "route_type_columns": self.route_type_columns,
            "osrm_minutes_per_km": self.osrm_minutes_per_km,
            "residual_std": self.residual_std,
            "osrm_time_median": self.osrm_time_median,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingArtifact:
        return cls(**data)


def drop_missing_targets(
    df: pd.DataFrame,
    target_col: str,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    out = df if inplace else df.copy()
    return out.dropna(subset=[target_col])


def coerce_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    utc: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=utc, errors="coerce")
    return out


class FeaturePreprocessor:
    """Fit on training data; apply consistent transforms at inference."""

    def __init__(self, feature_columns: list[str] | None = None):
        self.feature_columns = feature_columns or []
        self.artifact: PreprocessingArtifact | None = None

    def fit(self, df: pd.DataFrame, feature_columns: list[str] | None = None) -> PreprocessingArtifact:
        """Learn imputation, clipping bounds, and OSRM calibration from training rows."""
        cols = feature_columns or get_all_feature_names(df.columns)
        self.feature_columns = [c for c in cols if c in df.columns]

        work = df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        fill_values = {c: float(work[c].median()) for c in self.feature_columns}
        clip_upper = {c: float(work[c].quantile(CLIP_PERCENTILE)) for c in self.feature_columns}
        route_cols = sorted(c for c in self.feature_columns if c.startswith("route_type_"))

        ratio = (df["osrm_time"] / df["osrm_distance"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        osrm_mpk = float(ratio.median()) if ratio.notna().any() else 1.0

        self.artifact = PreprocessingArtifact(
            feature_columns=self.feature_columns,
            fill_values=fill_values,
            clip_upper=clip_upper,
            route_type_columns=route_cols,
            osrm_minutes_per_km=osrm_mpk,
            osrm_time_median=float(df["osrm_time"].median()) if "osrm_time" in df.columns else 0.0,
        )
        return self.artifact

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation and clipping; returns model-ready feature matrix."""
        if self.artifact is None:
            raise RuntimeError("FeaturePreprocessor.fit() must be called first.")

        art = self.artifact
        out = df.copy()

        for col in art.route_type_columns:
            if col not in out.columns:
                out[col] = 0.0

        missing_feats = [c for c in art.feature_columns if c not in out.columns]
        if missing_feats:
            raise ValueError(f"Missing features at inference: {missing_feats}")

        X = out[art.feature_columns].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(art.fill_values)
        for col, hi in art.clip_upper.items():
            X[col] = X[col].clip(upper=hi)
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
