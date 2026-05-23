"""Validate production inference API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import ETAInferencePipeline, predict_eta


def main():
    p = ETAInferencePipeline()
    r = p.predict_eta("IND000000ACB", "IND562132AAA", 85.0, 14, "FTL")
    assert r.predicted_eta > 0
    assert r.risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    d = predict_eta("IND000000ACB", "IND562132AAA", 85.0, 14, "Carting")
    assert "predicted_eta" in d
    print("predicted_eta:", round(r.predicted_eta, 2))
    print("risk_level:", r.risk_level)
    print("OK")


if __name__ == "__main__":
    main()
