---
name: train
description: >
  Train ML models and iterate systematically with experiment tracking. Full coverage of
  supervised learning: Naive Bayes, KNN, Discriminant Analysis (LDA/QDA), SVM/SVR, Decision
  Trees, Ensemble Methods (Random Forest, XGBoost, LightGBM), GLM (Poisson, Gamma, Tweedie),
  Gaussian Process, Ridge/Lasso/ElasticNet, and Neural Networks (PyTorch). Covers data
  splitting, cross-validation, metrics, persistence, hyperparameter search, and TSV-based
  experiment tracking. Use when the user wants to train a model, fit a classifier or regressor,
  evaluate performance, do cross-validation, run experiments, tune hyperparameters, or compare runs.
allowed-tools: >
  Bash(uv run * scripts/analyze_results.py *)
  Read Write Glob Grep
disable-model-invocation: true
argument-hint: path to feature-engineered dataset or results.tsv (e.g. "data/features.csv")
model: sonnet
effort: high
paths: "**/results.tsv"
compatibility: ">=1.0"
metadata:
  category: model-training
  tags: [sklearn, xgboost, lightgbm, pytorch, naive-bayes, knn, svm, decision-tree, glm, gaussian-process, experiment-tracking]
  phase: train
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
| Situation | Best choices | Why |
|-----------|-------------|-----|
| Small data (<1k), interpretability needed | Naive Bayes, LDA, Decision Tree | Low variance, fast, explainable |
| Small data, metric learning | KNN | Non-parametric, no assumptions |
| Small/medium, max accuracy | SVM (RBF kernel) | Effective in high-dimensional space |
| Medium (1k–100k) | Random Forest, XGBoost | Handles mixed types, robust to noise |
| Large (>100k) | LightGBM, Neural Net | Scales efficiently |
| Text / sparse features | Naive Bayes, Logistic Regression | Works well on high-dim sparse input |
| Classes linearly separable | LDA, Logistic Regression | Efficient, calibrated probabilities |

### Regression
| Situation | Best choices | Why |
|-----------|-------------|-----|
| Linear relationship | Ridge, Lasso, ElasticNet | Regularized, interpretable coefficients |
| Count / rate data, non-Gaussian target | GLM (Poisson, Gamma) | Correct distributional assumptions |
| Uncertainty quantification needed | Gaussian Process | Outputs full posterior distribution |
| Non-linear, tabular | Random Forest, XGBoost | Captures interactions automatically |
| Complex / large data | LightGBM, Neural Net | Scales, highest ceiling |

### Decision rule: start simple
```
Linear/Naive Bayes → SVM/KNN → Tree ensembles → Neural Net
     ↑                                                 ↑
 interpretable                                   highest capacity
```
Always record a linear baseline as exp000 before trying complex models.

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

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report
import joblib

# GaussianNB — continuous features (assumes Gaussian distribution per class)
model = GaussianNB()
model.fit(X_train, y_train)
print(classification_report(y_val, model.predict(X_val)))
joblib.dump(model, 'model_nb.joblib')

# MultinomialNB — count/frequency features (e.g. TF-IDF, bag-of-words)
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing

# BernoulliNB — binary features (word presence/absence)
# from sklearn.naive_bayes import BernoulliNB
# model = BernoulliNB(alpha=1.0)

# Tuning: var_smoothing (GaussianNB), alpha (Multinomial/Bernoulli)
# When to use: text classification, spam detection, small data, fast baseline
```

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Classification
pipe = Pipeline([
    ('scaler', StandardScaler()),   # KNN is distance-based — scaling is mandatory
    ('model', KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',         # 'uniform' or 'distance' (closer = more weight)
        metric='minkowski',         # Euclidean when p=2, Manhattan when p=1
        n_jobs=-1,
    )),
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'model_knn.joblib')

# Regression
# pipe = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=5))])

# Tuning: n_neighbors (odd to avoid ties), weights, metric
# Weakness: O(n) prediction time — slow at inference on large datasets
# When to use: small/medium data, non-linear boundaries, anomaly detection
```

### Discriminant Analysis (LDA / QDA)
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
import joblib

# LDA — assumes equal covariance across classes; also works as dimensionality reduction
lda = LinearDiscriminantAnalysis(
    solver='svd',          # 'svd' (default), 'lsqr', 'eigen'
    n_components=None,     # reduce to min(n_classes-1, n_features) components
    store_covariance=False,
)
lda.fit(X_train, y_train)
print(classification_report(y_val, lda.predict(X_val)))
joblib.dump(lda, 'model_lda.joblib')

# For dimensionality reduction (supervised):
# X_reduced = lda.transform(X_train)  # reduces to n_classes-1 dimensions

# QDA — allows different covariance per class; more flexible but needs more data
# qda = QuadraticDiscriminantAnalysis(reg_param=0.0)  # reg_param adds regularization

# When to use: Gaussian class distributions, interpretable decision boundary,
#              dimensionality reduction to n_classes-1, well-separated classes
```

### Support Vector Machine (SVM / SVR)
```python
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Classification — RBF kernel (best general-purpose choice)
pipe = Pipeline([
    ('scaler', StandardScaler()),   # SVM is distance-based — scaling is mandatory
    ('model', SVC(
        C=1.0,              # Regularization: high C = low bias/high variance
        kernel='rbf',       # 'linear', 'poly', 'rbf', 'sigmoid'
        gamma='scale',      # 'scale' = 1/(n_features*X.var()), 'auto' = 1/n_features
        probability=True,   # enables predict_proba (slower fitting)
        random_state=42,
        class_weight='balanced',  # handles class imbalance
    )),
])
pipe.fit(X_train, y_train)
print(pipe.predict_proba(X_val)[:5])
joblib.dump(pipe, 'model_svm.joblib')

# For large datasets (>10k): use LinearSVC (much faster, linear kernel only)
# pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearSVC(C=1.0, max_iter=2000))])

# Regression — SVR
# pipe = Pipeline([('scaler', StandardScaler()), ('model', SVR(C=1.0, epsilon=0.1, kernel='rbf'))])

# Tuning priority: C first, then gamma (for RBF), then kernel
# When to use: small/medium data (<50k), high-dimensional (text, images), clear margin of separation
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import classification_report
import joblib

# Classification
tree = DecisionTreeClassifier(
    max_depth=5,            # constrain depth to prevent overfitting
    min_samples_split=20,   # min samples to split a node
    min_samples_leaf=10,    # min samples in a leaf
    criterion='gini',       # 'gini' or 'entropy'
    class_weight='balanced',
    random_state=42,
)
tree.fit(X_train, y_train)
print(classification_report(y_val, tree.predict(X_val)))

# Print interpretable rules
rules = export_text(tree, feature_names=list(X_train.columns))
print(rules[:2000])   # first 2000 chars of rule set
joblib.dump(tree, 'model_tree.joblib')

# Regression
# tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)

# Tuning priority: max_depth → min_samples_leaf → criterion
# Weakness: high variance (small data changes flip the tree) — prefer ensemble unless interpretability required
# When to use: interpretability is mandatory, rule extraction, feature selection proxy
```

### Generalized Linear Model (GLM)
```python
import statsmodels.api as sm
import numpy as np

# Poisson GLM — count data (events per unit time/area)
X_train_sm = sm.add_constant(X_train)   # statsmodels needs explicit intercept
glm_poisson = sm.GLM(
    y_train,
    X_train_sm,
    family=sm.families.Poisson(link=sm.families.links.Log()),
)
result = glm_poisson.fit()
print(result.summary())
y_pred = result.predict(sm.add_constant(X_val))

# Gamma GLM — positive continuous, right-skewed (insurance claims, durations)
# glm_gamma = sm.GLM(y_train, X_train_sm, family=sm.families.Gamma(link=sm.families.links.Log()))

# Tweedie GLM — flexible family (p=0: Normal, p=1: Poisson, p=2: Gamma, 1<p<2: compound)
# glm_tweedie = sm.GLM(y_train, X_train_sm, family=sm.families.Tweedie(var_power=1.5))

# Negative Binomial — overdispersed count data (variance > mean)
# glm_nb = sm.GLM(y_train, X_train_sm, family=sm.families.NegativeBinomial())

# Save coefficients
import json
coefs = dict(zip(['intercept'] + list(X_train.columns), result.params))
with open('model_glm_coefs.json', 'w') as f:
    json.dump(coefs, f, indent=2)

# When to use: count data, rate data, insurance/actuarial, non-Gaussian errors,
#              heteroscedastic residuals, log/logit link needed for interpretability
```

### Gaussian Process (GP)
```python
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# Regression — returns mean prediction AND uncertainty (std dev)
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,          # numerical stability
    normalize_y=True,    # subtract mean of y_train
    n_restarts_optimizer=5,  # restarts to find global kernel hyperparams
    random_state=42,
)
gpr.fit(X_train, y_train)
y_pred, y_std = gpr.predict(X_val, return_std=True)
print(f"Val RMSE: {np.sqrt(np.mean((y_pred - y_val)**2)):.4f}")
print(f"Mean uncertainty (std): {y_std.mean():.4f}")
joblib.dump(gpr, 'model_gp.joblib')

# Classification — probabilistic predictions
# kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
# gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5, random_state=42)
# gpc.fit(X_train, y_train)
# probs = gpc.predict_proba(X_val)

# Weakness: O(n³) training, O(n²) memory — not viable above ~5k samples
# When to use: uncertainty quantification is required, small data (<5k),
#              spatial/temporal data (use Matern kernel), active learning, Bayesian optimization
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
uv run ${CLAUDE_SKILL_DIR}/scripts/analyze_results.py results.tsv
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

## Code style

Follow the ML code style conventions in [references/ml-code-style.md](references/ml-code-style.md) when writing or reviewing training code. Key rules:

- Annotate tensor shapes in docstrings using standard dimension symbols (B, T, D, C, etc.)
- Two-line file header: purpose + author
- Section dividers (`# ── Name ──`) for files > 200 lines
- No defensive validation inside trusted internals — only at system boundaries
- Document what a variable IS, not what the operation does
