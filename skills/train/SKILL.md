---
name: train
description: >
  Train and evaluate ML models with proper splits, cross-validation, metrics, and persistence.
  Supports scikit-learn, XGBoost, LightGBM, and PyTorch.
  Part of the mlx workbench. Use when the user wants to train a model, fit a classifier,
  evaluate performance, do cross-validation, or build a prediction pipeline.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to feature-engineered dataset (e.g. "data/features.csv")
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
