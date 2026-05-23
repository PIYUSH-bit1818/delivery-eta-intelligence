"""Train and persist ETA prediction models."""

from pathlib import Path
from typing import Any, Optional, Union

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline


def build_default_regressor(**kwargs: Any) -> GradientBoostingRegressor:
    """Return a sensible default gradient boosting regressor for ETA."""
    defaults = dict(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    defaults.update(kwargs)
    return GradientBoostingRegressor(**defaults)


def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    estimator: Optional[BaseEstimator] = None,
) -> BaseEstimator:
    """Fit an estimator on feature matrix X and target y."""
    model = estimator or build_default_regressor()
    model.fit(X, y)
    return model


def save_model(model: Union[BaseEstimator, Pipeline], path: Union[str, Path]) -> None:
    """Serialize a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Union[str, Path]) -> Union[BaseEstimator, Pipeline]:
    """Load a serialized model from disk."""
    return joblib.load(path)
