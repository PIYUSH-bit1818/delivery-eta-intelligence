"""
Delivery ETA Intelligence — production ML package.

Primary entry points:
  - ``src.train.train_production_model``
  - ``src.inference.ETAInferencePipeline`` / ``predict_eta``
"""

from src.config import RANDOM_STATE, get_paths
from src.inference import ETAInferencePipeline, predict_eta
from src.train import load_production_bundle, train_production_model

__all__ = [
    "RANDOM_STATE",
    "ETAInferencePipeline",
    "get_paths",
    "load_production_bundle",
    "predict_eta",
    "train_production_model",
]
