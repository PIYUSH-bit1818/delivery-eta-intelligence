# Model artifacts

| File | Purpose | In Git? |
|------|---------|--------|
| `final_eta_model.pkl` | XGBoost + preprocessor + graph tables | No (train locally) |
| `graph_hub_features.pkl` | Cached hub graph features | No |
| `model_metadata.json` | Metrics, feature list, training timestamp | **Yes** |

## Train the production model

```bash
python scripts/build_processed_dataset.py
python scripts/train_production_model.py
```

Expected offline test metrics (reference run): **MAE ~44 min**, **R² ~0.97** on held-out trips.

The Streamlit dashboard requires `final_eta_model.pkl`. Batch analytics also use `outputs/predictions/sample_predictions.csv` (regenerate via notebook 05 or inference batch scoring).
