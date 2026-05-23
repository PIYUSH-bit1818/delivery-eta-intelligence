# Windows development environment setup

This guide walks you through creating a Python virtual environment for **Delivery ETA Intelligence** on Windows 10/11.

---

## Prerequisites

1. **Python 3.10 or newer**  
   Download from [python.org](https://www.python.org/downloads/windows/).  
   During installation, check **“Add python.exe to PATH”**.

2. **Verify Python in PowerShell**

   ```powershell
   python --version
   pip --version
   ```

   If `python` is not found, try `py --version` instead (Windows launcher).

---

## Step 1 — Open a terminal in the project folder

```powershell
cd "D:\delivery eta"
```

Use your actual project path if it differs.

---

## Step 2 — Create a virtual environment

A virtual environment keeps this project’s packages separate from other Python projects.

```powershell
python -m venv .venv
```

This creates a `.venv` folder (already ignored by `.gitignore`).

---

## Step 3 — Activate the virtual environment

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If you see an execution policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the activate command again.

**Command Prompt (cmd.exe):**

```cmd
.venv\Scripts\activate.bat
```

When activation succeeds, your prompt shows `(.venv)` at the beginning.

---

## Step 4 — Upgrade pip (recommended)

```powershell
python -m pip install --upgrade pip
```

---

## Step 5 — Install project dependencies

```powershell
pip install -r requirements.txt

# Optional — Jupyter notebooks, SHAP, extra viz
pip install -r requirements-dev.txt
```

Installation may take a few minutes (especially `xgboost` and `node2vec`).

---

## Step 6 — Register the Jupyter kernel (optional but helpful)

So notebooks can use this environment:

```powershell
python -m ipykernel install --user --name=delivery-eta --display-name="Delivery ETA (.venv)"
```

In Jupyter or VS Code/Cursor, select the kernel **“Delivery ETA (.venv)”**.

---

## Step 7 — Run the first notebook

**Option A — Jupyter Lab**

```powershell
jupyter lab
```

Open `notebooks/01_dataset_understanding.ipynb`.

**Option B — VS Code / Cursor**

Open the notebook file and pick the `.venv` Python interpreter  
(`Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`).

---

## Step 8 — Quick sanity check

```powershell
python -c "import pandas, numpy, sklearn, xgboost, networkx, streamlit; print('OK')"
```

You should see `OK` with no errors.

---

## Daily workflow

| Action | Command |
|--------|---------|
| Go to project | `cd "D:\delivery eta"` |
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Deactivate venv | `deactivate` |
| Run dashboard | `streamlit run dashboard/app.py` |

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `python` not recognized | Reinstall Python with “Add to PATH”, or use `py -m venv .venv` |
| `pip install` fails on xgboost | Update pip; ensure 64-bit Python |
| Notebook can’t find CSV | Run notebook from project root; path is `../data/raw/delivery_data.csv` from `notebooks/` |
| Wrong packages in notebook | Confirm `(.venv)` in prompt and correct kernel/interpreter |

---

## Next steps

1. Run `notebooks/01_dataset_understanding.ipynb`
2. Explore data in `notebooks/`
3. Move reusable code into `src/` as the pipeline matures
