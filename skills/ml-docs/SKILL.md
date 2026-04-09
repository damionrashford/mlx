---
name: ml-docs
description: >
  On-demand ML/data science library expert. Use when the user asks how to use any
  function, class, or method from NumPy, Pandas, scikit-learn, Matplotlib, TensorFlow,
  Keras, PyTorch, Seaborn, SciPy, statsmodels, XGBoost, LightGBM, Hugging Face
  Transformers, OpenCV, NLTK, spaCy, Plotly, Dask, PySpark, SQLAlchemy, or Jupyter.
  Fetches and synthesizes official API docs, parameter reference, and working code
  examples. Also use when the user asks "how do I do X in pandas/numpy/torch/sklearn",
  needs to understand a deep learning layer or training loop, asks about NLP pipelines,
  computer vision transforms, statistical tests, SQL ORM patterns, or big data ops.
allowed-tools: >
  Bash(uv run * scripts/process.py *)
  Read Write WebFetch Glob Grep
argument-hint: "[library] [topic or function name]"
model: haiku
effort: low
compatibility: ">=1.0"
metadata:
  category: reference
  tags: [docs, api-reference, numpy, pandas, pytorch, sklearn, huggingface, xgboost, lightgbm, statsmodels]
  phase: any
---

# ML Docs

**Context:** $ARGUMENTS

## Quick start

- **Look up a specific function/class:** → Step 1, then Step 2
- **Understand a concept or workflow:** → Step 1, pick a topic URL, fetch it
- **Compare options across libraries:** → run Step 1 for each library, fetch in parallel

## When to use

- User asks how to use any function, class, or method from a supported library
- User asks "what parameters does X take" or "what does Y return"
- User needs working code examples for a task (groupby, cross-validation, fine-tuning, etc.)
- User asks about a concept: broadcasting, autograd, attention, sparse matrices, etc.
- User wants to know which library/function to use for a task
- User hits an error and needs to check expected behavior from the official docs

## Step 1 — Resolve the documentation URL

Run `process.py resolve` to get the prioritized list of URLs to fetch:

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve \
  --library <library-name-or-alias> \
  --query "<function, class, or topic>"
```

The script returns a JSON object with `fetch_in_order` — a list of URLs ranked by specificity.
Start with priority 1. If it returns 404 or empty content, move to priority 2, then 3.

**Library aliases accepted:** See `references/guide.md` for the full alias table.

Examples:
```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library pandas --query "DataFrame.groupby"
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library torch --query "autograd"
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library sklearn --query "cross_val_score"
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library transformers --query "pipeline"
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library scipy --query "hypothesis testing"
```

To list all supported libraries:
```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py list
```

## Step 2 — Fetch and synthesize

Use WebFetch on the URLs returned by Step 1, in order. Stop at the first URL that contains
useful content. Synthesize into a direct answer including:

1. **What it does** — one sentence
2. **Signature** — function/class signature with parameter types if available
3. **Key parameters** — name, type, default, what it controls
4. **Return value** — type and meaning
5. **Working example** — copy-pasteable, minimal, correct
6. **Common gotchas** — anything non-obvious from the docs (deprecations, defaults that surprise)

Do not paste raw documentation. Synthesize it.

## Step 3 — Cross-library questions

When the question spans multiple libraries (e.g. "how do I use a PyTorch model with
scikit-learn's cross-validation"), run Step 1 for each library in parallel, fetch both,
then synthesize a combined answer that shows how they integrate.

## Gotchas

- **Keras vs tf.keras:** Keras 3.x (keras.io) is standalone and backend-agnostic. `tf.keras`
  is the older TF-bundled version. If the user has TF < 2.16, they likely have `tf.keras`.
  Check which one they're importing before advising.

- **PyTorch `torch.` vs `torch.nn.functional`:** Many operations exist in both places with
  different call conventions. `torch.nn.Conv2d` is a module (stateful), `F.conv2d` is a
  function. The docs are on different pages — resolve for the right one.

- **scikit-learn class paths:** API URLs use the full dotted path, e.g.
  `sklearn.linear_model.LogisticRegression`, not just `LogisticRegression`. Include the
  module prefix in the `--query` argument for direct API lookups.

- **Pandas 2.x breaking changes:** `DataFrame.append` is removed in 2.0. `df.swaplevel`
  behavior changed. If the user shows old code, check the 2.0 migration guide:
  https://pandas.pydata.org/docs/whatsnew/v2.0.0.html

- **Hugging Face `pipeline` task names:** They changed between versions. Always fetch the
  current docs rather than recalling task names from memory (e.g. `"text-generation"` vs
  `"text2text-generation"`).

- **spaCy model names:** `en_core_web_sm/md/lg/trf` are not installed by default. The docs
  show `nlp = spacy.load("en_core_web_sm")` but users need `python -m spacy download en_core_web_sm`
  first. Always mention this.

- **OpenCV Python bindings:** The Python docs at docs.opencv.org are C++ first. Prefer
  fetching the Python tutorials (`/tutorial_py_*`) over the raw C++ API pages.

- **PySpark version:** API paths differ between Spark 3.x and older versions. The default
  URL targets `latest`. If the user specifies a version, adjust the URL.

- **Dask DataFrame is NOT Pandas:** Dask DataFrames don't support all Pandas operations.
  Always check the Dask API index rather than assuming Pandas parity.

- **LightGBM vs XGBoost parameter names:** They use different names for the same concept
  (e.g. `num_leaves` in LightGBM vs `max_leaves` in XGBoost). When helping with both,
  always fetch both sets of parameter docs.

## Examples

### Example 1: Look up a Pandas function

User: "How do I use pandas pivot_table?"

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library pandas --query "DataFrame.pivot_table"
```

Fetch priority-1 URL → synthesize parameters, example, gotchas about `aggfunc` defaulting to mean.

### Example 2: PyTorch training loop

User: "How does autograd work in PyTorch?"

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library torch --query "autograd"
```

Fetch the autograd topic page → explain `requires_grad`, `.backward()`, `zero_grad()`, with example.

### Example 3: Hugging Face fine-tuning

User: "How do I fine-tune BERT with the Trainer API?"

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library transformers --query "trainer"
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library transformers --query "bert"
```

Fetch both in parallel → synthesize a complete fine-tuning workflow.

### Example 4: Statistical test

User: "How do I run a t-test in Python?"

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py resolve --library scipy --query "hypothesis testing"
```

Fetch stats topic page → show `scipy.stats.ttest_ind` and `ttest_rel` with example and interpretation.

### Example 5: Unknown library scope

User: "How do I do X?"

Run:
```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/process.py list
```

Scan the list for the relevant library. If the task spans multiple (e.g. plot a pandas DataFrame
with Seaborn), resolve for both, fetch in parallel, synthesize combined answer.

## Reference docs

Read [`references/guide.md`](references/guide.md) for:
- Full alias table for all 19 libraries
- Direct topic URL index (skip the script for common topics)
- Library selection guidance (which library to use for a given task)
