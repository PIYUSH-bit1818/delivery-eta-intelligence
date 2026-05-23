# Raw data

Place the logistics delivery dataset here as `delivery_data.csv`.

This file is **not** committed to Git (size / licensing). After adding the CSV, run:

```bash
python scripts/build_processed_dataset.py
```

The processed Parquet is written to `data/processed/`.
