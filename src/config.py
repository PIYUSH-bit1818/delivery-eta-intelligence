"""Central configuration for Delivery ETA Intelligence production pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "actual_time"
TRIP_ID_COLUMN = "trip_uuid"
TEST_SIZE = 0.2
CLIP_PERCENTILE = 0.99
GRAPH_COVERAGE_THRESHOLD = 0.80

# Risk thresholds (tuned on training residual / bottleneck distributions)
RISK_BOTTLENECK_MEDIUM = 0.35
RISK_BOTTLENECK_HIGH = 0.55
RISK_BOTTLENECK_CRITICAL = 0.75
CONFIDENCE_Z = 1.28  # ~80% interval


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project directories."""

    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def outputs_figures(self) -> Path:
        return self.root / "outputs" / "figures"

    @property
    def outputs_tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def outputs_predictions(self) -> Path:
        return self.root / "outputs" / "predictions"

    @property
    def processed_parquet(self) -> Path:
        return self.data_processed / "delivery_logistics_processed.parquet"

    @property
    def final_model(self) -> Path:
        return self.models / "final_eta_model.pkl"

    @property
    def model_metadata(self) -> Path:
        return self.models / "model_metadata.json"

    @property
    def graph_features_store(self) -> Path:
        return self.models / "graph_hub_features.pkl"


def get_paths(root: Path | None = None) -> ProjectPaths:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    return ProjectPaths(root=root)
