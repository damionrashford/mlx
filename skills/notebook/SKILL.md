---
name: notebook
description: >
  Create, clean, organize, optimize, and convert Jupyter notebooks. Build new notebooks
  from scratch with proper cell structure, cell IDs, and Colab compatibility. Extract
  reusable functions, add documentation, generate requirements.txt, and convert to scripts.
  Use when the user wants to create a notebook, clean a notebook, organize cells, extract
  functions, convert to script, or optimize a notebook for production.
allowed-tools: >
  Bash(uv run * scripts/assess.py *)
  Read Write Glob Grep
argument-hint: path to .ipynb file (e.g. "analysis.ipynb")
model: haiku
effort: low
paths: "**/*.ipynb"
compatibility: ">=1.0"
metadata:
  category: tooling
  tags: [jupyter, notebook, cleanup, refactor, script-conversion, documentation]
  phase: any
---

# Jupyter Notebook Management

Reference for cleaning, organizing, and converting Jupyter notebooks.

## Available scripts

| Script                         | Usage                                                         |
| ------------------------------ | ------------------------------------------------------------- |
| [assess.py](scripts/assess.py) | `uv run ${CLAUDE_SKILL_DIR}/scripts/assess.py notebook.ipynb` |

### Assess a notebook

```bash
# Text report
uv run ${CLAUDE_SKILL_DIR}/scripts/assess.py $ARGUMENTS

# JSON output
uv run ${CLAUDE_SKILL_DIR}/scripts/assess.py notebook.ipynb --json
```

The [assess.py](scripts/assess.py) script analyzes notebook structure, detects issues (empty cells, scattered imports, missing documentation, hardcoded paths, missing seeds), and returns a quality score out of 10.

## Capabilities

| Action    | What                                               |
| --------- | -------------------------------------------------- |
| Clean     | Remove empty cells, clear stale outputs, fix order |
| Organize  | Add section headers, TOC, logical grouping         |
| Extract   | Pull reusable code into `utils.py`                 |
| Document  | Add docstrings, markdown, type hints               |
| Optimize  | Memory management, chunked processing              |
| Reproduce | Set seeds, pin versions, freeze requirements       |
| Convert   | Export to `.py` script                             |

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

## Creating notebooks

### .ipynb structure

When creating a notebook from scratch, the JSON structure is:

```json
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": { "name": "python3", "display_name": "Python 3" },
    "colab": { "provenance": [] }
  },
  "cells": []
}
```

Use the `NotebookEdit` tool to create and modify cells — it handles JSON serialization correctly. Only fall back to raw `json.load/dump` when the tool is unavailable.

### Cell source format

`source` is an **array of strings**, each ending with `\n` (except possibly the last):

```json
"source": ["import pandas as pd\n", "import numpy as np\n", "\n", "df = pd.read_csv('data.csv')\n"]
```

NOT a single string. This is the most common formatting mistake when writing notebook JSON directly.

### Cell IDs

Every cell needs a unique `metadata.id`. Use descriptive names:

```
"load_data"      "clean_features"    "train_model"
"eda_overview"   "split_dataset"     "eval_results"
"setup_imports"  "config"            "plot_curves"
```

### Common ML cell templates

**Setup cell:**

```python
#@title Setup
%pip install -q scikit-learn pandas numpy matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("✓ Setup complete")
```

**Config cell (Colab form):**

```python
#@title Configuration { display-mode: "form" }

MODEL_TYPE = "xgboost"   #@param ["xgboost", "lightgbm", "random_forest"]
LEARNING_RATE = 0.1      #@param {type:"number"}
MAX_DEPTH = 6            #@param {type:"integer"}
TEST_SIZE = 0.2          #@param {type:"number"}
RANDOM_SEED = 42         #@param {type:"integer"}
```

**Training progress cell:**

```python
from tqdm.notebook import tqdm

for epoch in tqdm(range(n_epochs), desc="Training"):
    # train step
    pass
```

### Programmatic cell operations

**Insert cell at position:**

```python
import json

with open('notebook.ipynb') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "source": ["# new code\n"],
    "metadata": {"id": "new_cell_id"},
    "execution_count": None,
    "outputs": []
}
nb['cells'].insert(index, new_cell)

with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
```

**Delete cell by ID:**

```python
nb['cells'] = [c for c in nb['cells'] if c.get('metadata', {}).get('id') != 'cell_to_delete']
```

**Find cell by ID:**

```python
cell = next((c for c in nb['cells'] if c.get('metadata', {}).get('id') == 'target_id'), None)
```

### Colab collapsible sections

Code cells with `#@title` become collapsible when run in Colab — use for long setup or helper sections:

```python
#@title Helper functions { display-mode: "form" }
def preprocess(df): ...
def evaluate(model, X, y): ...
```

## Cleaning checklist

1. Remove empty cells
2. Clear old outputs (bloat, stale results)
3. Consolidate imports to first code cell
4. Insert section headers between logical groups
5. Order: imports → config → data → analysis → results

## Quality checklist

Before finalizing any notebook (created or cleaned):

- [ ] All cells have unique, descriptive `metadata.id` values
- [ ] Markdown cells have proper headers and section dividers
- [ ] Code cells are logically ordered — no forward references
- [ ] All imports in setup cell or first code cell
- [ ] Config values use Colab form fields (`#@param`) where appropriate
- [ ] Random seeds set (`np.random.seed`, `torch.manual_seed`)
- [ ] Relative paths only — no hardcoded absolute paths
- [ ] Error handling for common failures (file not found, OOM, missing deps)
- [ ] Progress indicators (`tqdm`) on long loops
- [ ] Clear output messages (`✓ success`, `⚠ warning`) in setup cells
- [ ] Outputs cleared before committing to git

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

## Code style reference

When cleaning or converting notebooks to scripts, apply the conventions in [references/ml-code-style.md](references/ml-code-style.md):

- Tensor shape annotations in docstrings (B, T, D, C symbols)
- Two-line module header + section dividers for long files
- Variable-first inline comments (what it IS, not what the op does)
- Import ordering: stdlib → third-party → local
