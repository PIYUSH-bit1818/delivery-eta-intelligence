"""Load raw and external logistics datasets."""

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


def load_csv(
    path: Union[str, Path],
    *,
    parse_dates: Optional[list[str]] = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, parse_dates=parse_dates, **read_csv_kwargs)


def load_parquet(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)
