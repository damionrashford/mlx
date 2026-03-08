---
name: notebook
description: >
  Clean, organize, optimize, and convert Jupyter notebooks. Extract reusable functions,
  add documentation, generate requirements.txt, and convert to scripts.
  Part of the mlx workbench. Use when the user wants to clean a notebook, organize cells,
  extract functions, convert to script, or optimize a notebook for production.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to .ipynb file (e.g. "analysis.ipynb")
---

# Jupyter Notebook Management

Reference for cleaning, organizing, and converting Jupyter notebooks.

## Capabilities

| Action | What |
|--------|------|
| Clean | Remove empty cells, clear stale outputs, fix order |
| Organize | Add section headers, TOC, logical grouping |
| Extract | Pull reusable code into `utils.py` |
| Document | Add docstrings, markdown, type hints |
| Optimize | Memory management, chunked processing |
| Reproduce | Set seeds, pin versions, freeze requirements |
| Convert | Export to `.py` script |

## Recommended notebook structure

```
1.  Title & Description
2.  Table of Contents
3.  Setup & Imports
4.  Configuration & Constants
5.  Data Loading
6.  EDA
7.  Data Preparation
8.  Feature Engineering
9.  Model Training
10. Evaluation
11. Conclusions
```

## Assess a notebook

```python
import json
with open('$ARGUMENTS') as f:
    nb = json.load(f)
cells = nb.get('cells', [])
code = [c for c in cells if c['cell_type'] == 'code']
md = [c for c in cells if c['cell_type'] == 'markdown']
empty = [c for c in code if not ''.join(c['source']).strip()]
print(f"Total: {len(cells)} cells ({len(code)} code, {len(md)} markdown, {len(empty)} empty)")
print(f"Has outputs: {sum(1 for c in code if c.get('outputs'))}")
```

## Cleaning checklist

1. Remove empty cells
2. Clear old outputs (bloat, stale results)
3. Consolidate imports to first code cell
4. Insert section headers between logical groups
5. Order: imports -> config -> data -> analysis -> results

## Extract reusable code

```python
# Before (scattered in cells):
df['age_binned'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100])
df['income_log'] = np.log1p(df['income'])

# After (in utils.py):
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard feature engineering."""
    df = df.copy()
    df['age_binned'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100])
    df['income_log'] = np.log1p(df['income'])
    return df
```

## Generate requirements.txt

```python
import re, json
with open('notebook.ipynb') as f:
    nb = json.load(f)
imports = set()
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for line in cell['source']:
            m = re.match(r'^(?:from|import)\s+(\w+)', line)
            if m: imports.add(m.group(1))
```

## Convert to script

Structure output as:
```python
#!/usr/bin/env python3
"""Converted from notebook: {name}."""

# --- Imports ---
import pandas as pd

# --- Config ---
DATA_PATH = "data/input.csv"

# --- Functions ---
def load_data(path): ...
def preprocess(df): ...
def analyze(df): ...

# --- Main ---
def main():
    df = load_data(DATA_PATH)
    df = preprocess(df)
    results = analyze(df)
    print(results)

if __name__ == "__main__":
    main()
```

## Memory optimization

- `del df_temp; gc.collect()` after intermediates
- `df.astype({'col': 'category'})` for low-cardinality
- `pd.read_csv(chunksize=10000)` for large files

## Reproducibility

- Set seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- Pin versions in requirements.txt
- Use relative paths
- Clear outputs before committing

## Automated tools

```bash
# Format code in notebooks
pip install black[jupyter] && black notebook.ipynb

# Lint notebooks
pip install nbqa && nbqa flake8 notebook.ipynb

# Version-control friendly sync (.ipynb <-> .py)
pip install jupytext && jupytext --set-formats ipynb,py notebook.ipynb

# Convert formats
jupyter nbconvert --to html notebook.ipynb
jupyter nbconvert --to python notebook.ipynb
jupyter nbconvert --to notebook --execute notebook.ipynb
```
