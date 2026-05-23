# Scripts

| Script | Description |
|--------|-------------|
| `build_processed_dataset.py` | Raw CSV → cleaned Parquet |
| `train_production_model.py` | Trip-level XGBoost + graph features → `models/` |
| `build_notebook_04.py` / `build_notebook_05.py` | Regenerate analysis notebooks from `src/` |
| `validate_*.py` | Smoke checks for graph, modeling, and inference |

Run from repository root with the virtual environment activated.
