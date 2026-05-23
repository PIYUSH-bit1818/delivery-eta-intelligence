"""CLI: train and persist production ETA model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train_production_model


def main() -> None:
    report = train_production_model()
    print("Training complete.")
    print(f"  Model: {report['bundle_path']}")
    print(f"  Metadata: {report['metadata_path']}")
    print(f"  Test MAE: {report['metrics']['MAE']:.2f} min")
    print(f"  Test R2:  {report['metrics']['R2']:.4f}")


if __name__ == "__main__":
    main()
