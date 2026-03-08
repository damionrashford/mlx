---
name: train
description: >
  Train ML models and iterate systematically with experiment tracking. Covers data splitting,
  model selection, cross-validation, metrics, persistence, hyperparameter search, and TSV-based
  run tracking. Supports scikit-learn, XGBoost, LightGBM, and PyTorch. Use when the user
  wants to train a model, fit a classifier, evaluate performance, do cross-validation, run
  experiments, tune hyperparameters, compare runs, or track results.
allowed-tools: Bash, Read, Write, Glob, Grep
disable-model-invocation: true
argument-hint: path to feature-engineered dataset or results.tsv (e.g. "data/features.csv")
---

# Model Training & Evaluation

Templates and reference for training, evaluating, and persisting ML models.

## Data splitting

### Standard (>10k rows)
```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

### Cross-validation (<10k rows)
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### Time series
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

## Model selection guide

### Classification
| Data size | Models |
|-----------|--------|
| Small (<1k) | Logistic Regression, SVM |
| Medium (1k-100k) | Random Forest, XGBoost |
| Large (>100k) | LightGBM, Neural Net |

### Regression
| Relationship | Models |
|-------------|--------|
| Linear | Ridge, Lasso, ElasticNet |
| Non-linear | Random Forest, XGBoost |
| Complex | LightGBM, Neural Net |

## Templates

### sklearn pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)
print(classification_report(y_val, y_pred))
joblib.dump(pipe, 'model.joblib')
```

### XGBoost
```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'seed': 42,
}
model = xgb.train(params, dtrain, num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')], early_stopping_rounds=50, verbose_eval=100)
model.save_model('model.xgb')
```

### PyTorch
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128, output=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, output),
        )
    def forward(self, x): return self.net(x)

model = MLP(X_train.shape[1])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(100):
    model.train()
    for bx, by in loader:
        opt.zero_grad()
        loss_fn(model(bx).squeeze(), by).backward()
        opt.step()
torch.save(model.state_dict(), 'model.pt')
```

## Metrics reference

| Classification | When to use |
|---------------|-------------|
| Accuracy | Balanced classes |
| F1 | Imbalanced classes |
| AUC-ROC | Ranking tasks |
| Precision | FP costly |
| Recall | FN costly |

| Regression | When to use |
|-----------|-------------|
| RMSE | Penalize large errors |
| MAE | Robust to outliers |
| R-squared | Variance explained |

## Evaluation report format

```
=== Training Report ===
Task: Binary Classification
Model: XGBoost (1000 rounds, early stopped at 347)
Split: 35k train / 7.5k val / 7.5k test

Val:  Accuracy=0.8634, F1=0.8521, AUC=0.9234
Test: Accuracy=0.8601, F1=0.8489

Top features: feature_a (0.234), feature_b (0.189), feature_c (0.156)
Saved: model.xgb, metrics.json
```

## Rules

- Never evaluate on training data
- Set random seeds everywhere
- Use early stopping for iterative models
- Save both model and preprocessing (use Pipeline or save scaler separately)
- Stratify splits for classification
- Only touch test set ONCE for final evaluation

---

## Experiment Tracking & Iteration

### Setup

```bash
echo -e "id\tmetric\tval_score\ttest_score\tmemory_mb\tstatus\tdescription" > results.tsv
echo -e "exp000\taccuracy\t0.8523\t0.8401\t4096\tKEEP\tbaseline" >> results.tsv
```

### Results format (TSV)

```
id        metric    val_score  test_score  memory_mb  status   description
exp000    accuracy  0.8523     0.8401      4096       KEEP     baseline
exp001    accuracy  0.8612     0.8498      4096       KEEP     lr=0.001
exp002    accuracy  0.8590     -           4096       DISCARD  lr=0.003 (overfit)
exp003    accuracy  0.0000     -           0          CRASH    lr=0.01 (diverged)
exp004    accuracy  0.8634     0.8521      4352       KEEP     dropout=0.1
```

Status: `KEEP` (improved), `DISCARD` (same or worse), `CRASH` (error/OOM/NaN)

### Experiment cycle

```
1. Hypothesize (what change, why it might help)
2. Modify (one variable at a time)
3. Run (fixed budget: time or epochs)
4. Record (append to results.tsv)
5. Decide: KEEP or DISCARD
6. Repeat
```

### What to try (priority order)

**High impact (try first)**
1. Learning rate (3x and 0.3x current)
2. Model capacity (layers, hidden size)
3. Batch size (double or halve)
4. Regularization (dropout, weight decay)

**Medium impact**
5. Optimizer (Adam → AdamW → SGD+momentum)
6. LR schedule (cosine, warmup, step decay)
7. Data augmentation
8. Feature selection

**Low impact (try last)**
9. Activation functions
10. Normalization layers
11. Initialization schemes
12. Gradient clipping

### Search strategies

#### Grid (small spaces)
```python
from itertools import product
params = {'lr': [1e-4, 3e-4, 1e-3], 'dropout': [0.0, 0.1, 0.3]}
for combo in product(*params.values()):
    config = dict(zip(params.keys(), combo))
```

#### Random (large spaces)
```python
import random
def sample():
    return {
        'lr': 10 ** random.uniform(-5, -2),
        'dropout': random.uniform(0, 0.5),
        'hidden': random.choice([64, 128, 256, 512]),
    }
```

#### Informed (after several runs)
Analyze results.tsv: which LR range works? Did more capacity help? Narrow search.

### Analyze results

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/analyze_results.py results.tsv
```

Or inline:
```python
import pandas as pd
r = pd.read_csv("results.tsv", sep="\t")
kept = r[r.status == "KEEP"]
print(f"Total: {len(r)}, Kept: {len(kept)}, Best: {kept.val_score.max():.6f}")
print(kept.nlargest(5, 'val_score')[['id', 'val_score', 'description']])
```

### Experiment rules

- ONE variable per experiment
- Validation set for decisions, test set only at the end
- Track memory — OOM means not viable
- Fixed random seeds
- Log everything (stdout/stderr to run.log)
- Commit code before each experiment
