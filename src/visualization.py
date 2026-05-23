"""Plots for ETA distributions, model residuals, and network views."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    title: str = "Actual vs Predicted ETA",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Scatter plot of actual vs predicted delivery times."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=12)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Actual ETA (minutes)")
    ax.set_ylabel("Predicted ETA (minutes)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
