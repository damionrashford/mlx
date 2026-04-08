# Notebook Output Style

Code blocks first, markdown prose after. Suitable for insertion as Jupyter notebook cells.

## Structure

Each response maps to one or more notebook cells:

1. **Code cell** — complete, runnable code block first
2. **Markdown cell** — explanation, caveats, and interpretation after

## Code cell requirements

- Runnable as-is (includes all necessary imports in the cell or assumes prior cells)
- Outputs are meaningful (use `print()`, `display()`, or assign to show result)
- Comments inline for non-obvious logic only

## Markdown cell requirements

- Short: 2-5 sentences per concept
- Explain WHY, not WHAT (the code shows what)
- Use `$LaTeX$` for mathematical notation
- Use `**bold**` for key terms on first use

## Format example

```python
# Cell 1: Load and inspect
import pandas as pd
df = pd.read_csv("data.csv")
print(df.shape)
df.head()
```

**Shape**: `(n_rows, n_cols)` confirms the dataset loaded correctly.
Check `df.dtypes` next if any columns are unexpectedly `object`.

## No

- Prose paragraphs before showing the code
- Code that requires manual editing to run (use variables at the top)
- Reproducing the entire notebook structure in one response
