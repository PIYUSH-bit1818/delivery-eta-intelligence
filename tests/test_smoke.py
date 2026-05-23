"""Smoke tests for package importability."""

import importlib

PRODUCTION_MODULES = [
    "src.config",
    "src.features",
    "src.graph_features",
    "src.preprocessing",
    "src.train",
    "src.inference",
    "src.dashboard_utils",
    "src.dashboard_theme",
]

LEGACY_MODULES = [
    "src.data_loading",
    "src.graph_construction",
    "src.bottleneck_analysis",
    "src.feature_engineering",
    "src.machine_learning",
    "src.evaluation",
    "src.visualization",
]


def test_production_modules_import():
    for name in PRODUCTION_MODULES:
        importlib.import_module(name)


def test_legacy_modules_import():
    for name in LEGACY_MODULES:
        importlib.import_module(name)


def test_metadata_present():
    from pathlib import Path

    meta = Path(__file__).resolve().parents[1] / "models" / "model_metadata.json"
    assert meta.exists(), "model_metadata.json should be committed for portfolio metrics"
