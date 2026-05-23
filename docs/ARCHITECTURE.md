# System architecture

High-level view of the Delivery ETA Intelligence platform. The canonical diagram lives in the [README](../README.md#architecture).

## Components

### Data pipeline

- **Input:** segment-level delivery logs (`data/raw/delivery_data.csv`)
- **Processing:** `scripts/build_processed_dataset.py` → `data/processed/delivery_logistics_processed.parquet`
- **Grain:** one row per trip segment with OSRM baselines, hubs, timestamps, route type

### Graph analytics

- **Module:** `src/graph_features.py` (+ notebook 03)
- **Graph:** directed hub graph from aggregated lanes
- **Outputs:** hub rankings, communities, bottleneck scores → CSV tables and in-model feature store

### ML training

- **Module:** `src/train.py`
- **Split:** trip-level holdout (`trip_uuid`)
- **Artifact:** `models/final_eta_model.pkl` (XGBoost + `PreprocessingArtifact` + graph tables)
- **Metadata:** `models/model_metadata.json` (metrics, feature manifest)

### Inference

- **Module:** `src/inference.py`
- **Class:** `ETAInferencePipeline` — load bundle once, score trips
- **Output:** `ETAPrediction` — ETA, risk tier, confidence band, bottleneck flags, insights

### Dashboard

- **Entry:** `app/streamlit_app.py`
- **Support:** `src/dashboard_utils.py`, `src/dashboard_theme.py`
- **Data:** live inference + optional `outputs/predictions/sample_predictions.csv`

## Dependency flow

```
config → features → graph_features → preprocessing → train → inference
                                                      ↘ dashboard_utils
```

Legacy exploratory modules (`data_loading`, `machine_learning`, etc.) remain for early notebooks; production path uses `features`, `graph_features`, `preprocessing`, `train`, `inference`.
