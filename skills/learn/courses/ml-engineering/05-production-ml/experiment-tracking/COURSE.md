# Experiment Tracking

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- What to track in ML experiments (code, data, hyperparameters, metrics, artifacts, environment) and why each element is necessary for reproducibility
- The core concepts and trade-offs between MLflow (open-source, self-hosted) and Weights & Biases (managed SaaS, richer visualization)
- How the experiment tracking to model registry to serving pipeline works end to end

**Apply:**
- Implement structured experiment tracking using MLflow or W&B, including parameter logging, metric logging, artifact storage, and model registration
- Design and execute systematic hyperparameter searches using grid search, random search, and Bayesian optimization (Optuna / W&B sweeps)

**Analyze:**
- Evaluate whether a model improvement is statistically significant and deployment-worthy by considering confidence intervals, slice-level performance, and practical trade-offs (latency, cost, interpretability)

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- **Training mechanics** -- how models are trained, including loss functions, optimizers, learning rate schedules, and overfitting (see [Training Mechanics](../02-neural-networks/training-mechanics/COURSE.md))
- **Evaluation metrics** -- precision, recall, AUC-ROC, AUC-PR, F1, RMSE, and when to use each (see [Evaluation Metrics](../04-classical-ml/evaluation-metrics/COURSE.md))

---

## Why This Matters

Every ML project involves running hundreds of experiments: different features, different hyperparameters, different data splits, different model architectures. Without systematic tracking, you lose reproducibility, waste time re-running experiments, and can't justify model choices to stakeholders.

At production scale, ML teams need to collaborate on shared models. Experiment tracking is the foundation of reproducible, auditable ML.

---

## What Experiment Tracking Solves

### Without Tracking

```
"Which model did we deploy?"
"The one from Tuesday... or was it Wednesday?"
"What hyperparameters did it use?"
"I think learning_rate was 0.01... or 0.1?"
"What data was it trained on?"
"The latest data... from some date."
"Can we reproduce it?"
"No."
```

### With Tracking

```
Model: merchant_churn_v3.2.1
Run ID: abc123
Date: 2024-01-15
Data: gs://bucket/training_data/v2024.01.14 (SHA256: 8f3a...)
Code: git commit 7e4b2c1
Hyperparameters: {learning_rate: 0.01, max_depth: 6, n_estimators: 500}
Metrics: {AUC: 0.892, precision@0.5: 0.78, recall@0.5: 0.85}
Artifacts: model.joblib, feature_importance.png, confusion_matrix.png
```

---

## What to Track

### The Complete Tracking Checklist

| Category | What to Log | Why |
|----------|-------------|-----|
| **Code** | Git commit hash, branch name | Reproduce the exact code that generated the model |
| **Data** | Data version, row count, hash, date range | Know exactly what data was used |
| **Features** | Feature list, feature engineering code version | Reproduce feature computation |
| **Hyperparameters** | All model config values | Reproduce training |
| **Metrics** | Train/val/test metrics, per-slice metrics | Compare experiments objectively |
| **Artifacts** | Model file, plots, feature importance | Deploy and analyze |
| **Environment** | Python version, library versions, hardware | Reproduce the runtime |
| **Duration** | Training time, data loading time | Cost estimation |
| **Notes** | Hypothesis, observations, decisions | Context for future you |

### Metrics to Always Track

```python
# For classification
metrics = {
    'auc_roc': roc_auc_score(y_test, y_pred_proba),
    'auc_pr': average_precision_score(y_test, y_pred_proba),
    'precision_at_50': precision_score(y_test, y_pred_binary),
    'recall_at_50': recall_score(y_test, y_pred_binary),
    'f1_at_50': f1_score(y_test, y_pred_binary),
    'log_loss': log_loss(y_test, y_pred_proba),
}

# For regression
metrics = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'mae': mean_absolute_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred),
    'mape': mean_absolute_percentage_error(y_test, y_pred),
}

# Always include slice metrics
for segment in ['small_merchants', 'medium_merchants', 'large_merchants']:
    mask = test_df['merchant_size'] == segment
    metrics[f'auc_{segment}'] = roc_auc_score(y_test[mask], y_pred_proba[mask])
```

---

## MLflow

MLflow is the most widely used open-source experiment tracking tool. It's the default choice for most ML teams.

### Core Concepts

- **Experiment**: a named group of related runs (e.g., "merchant_churn_v3")
- **Run**: a single training execution with parameters, metrics, and artifacts
- **Parameter**: input configuration (hyperparameters, data version)
- **Metric**: output measurement (AUC, loss, training time)
- **Artifact**: output file (model, plots, data samples)

### Basic Usage

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Set experiment
mlflow.set_experiment("merchant_churn_v3")

# Define hyperparameters
params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'min_samples_leaf': 50,
}

with mlflow.start_run(run_name="gbm_baseline_v3"):
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_param("data_version", "2024.01.14")
    mlflow.log_param("feature_count", len(feature_columns))
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    # Train
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Log metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("auc_roc_train", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))

    # Log per-slice metrics
    for size in ['small', 'medium', 'large']:
        mask = test_df['merchant_size'] == size
        slice_auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
        mlflow.log_metric(f"auc_{size}", slice_auc)

    # Log artifacts
    # Feature importance plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    importance = pd.Series(model.feature_importances_, index=feature_columns)
    importance.nlargest(20).plot(kind='barh', ax=ax)
    ax.set_title("Top 20 Feature Importances")
    fig.savefig("feature_importance.png", bbox_inches='tight')
    mlflow.log_artifact("feature_importance.png")

    # Log the model itself
    mlflow.sklearn.log_model(model, "model")

    # Log git info
    mlflow.log_param("git_commit", subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"AUC: {auc:.4f}")
```

### MLflow Tracking Server

```bash
# Start a tracking server (shared across team)
mlflow server \
    --backend-store-uri postgresql://mlflow:password@db:5432/mlflow \
    --default-artifact-root gs://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000

# Point clients to the server
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

### MLflow Model Registry

Promote experiment runs to registered models with stages.

```python
# Register the best model from experiments
import mlflow

client = mlflow.tracking.MlflowClient()

# Register model from a specific run
model_uri = f"runs:/{best_run_id}/model"
registered = mlflow.register_model(model_uri, "merchant_churn")

# Transition to staging
client.transition_model_version_stage(
    name="merchant_churn",
    version=registered.version,
    stage="Staging",
)

# After validation, promote to production
client.transition_model_version_stage(
    name="merchant_churn",
    version=registered.version,
    stage="Production",
)

# Load the production model for serving
model = mlflow.pyfunc.load_model("models:/merchant_churn/Production")
```

---

### Check Your Understanding: Tracking Fundamentals and MLflow

**1. Why is it important to log the git commit hash and data version alongside model metrics?**

<details>
<summary>Answer</summary>

Model metrics alone are not sufficient for reproducibility. To reproduce a model, you need the exact code (git commit) and exact data (data version) that produced it. Without the git commit, you cannot recreate the feature engineering or training logic. Without the data version, you cannot guarantee the same training examples. Two runs with identical hyperparameters but different code or data versions can produce very different results.
</details>

**2. In the MLflow Model Registry workflow, why is there a "Staging" stage between experiment runs and "Production"?**

<details>
<summary>Answer</summary>

The Staging stage provides a gate for automated and manual validation before a model reaches production. In Staging, the model is tested against held-out data, checked for performance regressions on specific segments, validated for serving compatibility (latency, input format), and reviewed by the team. This prevents deploying models that look good on experiment metrics but fail in production due to data issues, latency problems, or segment-level regressions.
</details>

**3. Why should you always log per-slice metrics (e.g., AUC by merchant size) in addition to aggregate metrics?**

<details>
<summary>Answer</summary>

Aggregate metrics can hide critical segment-level failures. A model may improve overall AUC from 0.87 to 0.89 while regressing on small merchants (AUC 0.83 to 0.75). If small merchants represent a strategically important segment, this regression is unacceptable despite the aggregate improvement. Per-slice metrics reveal these disparities and are essential for fairness evaluation and stakeholder trust.
</details>

---

## Weights & Biases (W&B)

W&B is a managed platform with richer visualization and collaboration features than MLflow.

### Basic Usage

```python
import wandb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Initialize a run
wandb.init(
    project="merchant-churn",
    name="gbm_baseline_v3",
    config={
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'data_version': '2024.01.14',
    },
    tags=["baseline", "v3", "gbm"],
)

# Train
model = GradientBoostingClassifier(**wandb.config)
model.fit(X_train, y_train)

# Log metrics
y_pred_proba = model.predict_proba(X_test)[:, 1]
wandb.log({
    'auc_roc': roc_auc_score(y_test, y_pred_proba),
    'precision': precision_score(y_test, (y_pred_proba > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred_proba > 0.5).astype(int)),
})

# Log training curves (logged per step during training)
for epoch in range(n_epochs):
    train_loss = train_one_epoch(model, X_train, y_train)
    val_loss = evaluate(model, X_val, y_val)
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
    })

# Log plots
wandb.log({
    'roc_curve': wandb.plot.roc_curve(y_test, y_pred_proba_all_classes),
    'confusion_matrix': wandb.plot.confusion_matrix(y_true=y_test, preds=y_pred),
    'feature_importance': wandb.Table(
        data=[[f, i] for f, i in zip(feature_columns, model.feature_importances_)],
        columns=["feature", "importance"]
    ),
})

# Save model artifact
artifact = wandb.Artifact("merchant-churn-model", type="model")
artifact.add_file("model.joblib")
wandb.log_artifact(artifact)

wandb.finish()
```

### W&B Sweeps (Hyperparameter Search)

```python
# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # bayesian optimization
    'metric': {
        'name': 'auc_roc',
        'goal': 'maximize',
    },
    'parameters': {
        'n_estimators': {'values': [100, 300, 500, 1000]},
        'max_depth': {'min': 3, 'max': 10},
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.1},
        'subsample': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
        'min_samples_leaf': {'values': [10, 25, 50, 100]},
    },
}

# Define training function
def train():
    wandb.init()
    config = wandb.config

    model = GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        min_samples_leaf=config.min_samples_leaf,
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    wandb.log({'auc_roc': auc})

# Run sweep (50 trials)
sweep_id = wandb.sweep(sweep_config, project="merchant-churn")
wandb.agent(sweep_id, function=train, count=50)
```

---

## MLflow vs W&B: Choosing

| Factor | MLflow | W&B |
|--------|--------|-----|
| Cost | Free (open source) | Free tier, paid for teams |
| Hosting | Self-hosted or Databricks | Managed SaaS |
| UI | Functional, basic | Rich, interactive |
| Collaboration | Basic sharing | Team dashboards, reports |
| Sweeps | Not built-in (use Optuna) | Built-in Bayesian sweeps |
| Model registry | Built-in | Built-in |
| Setup effort | Medium (need to host server) | Low (SaaS) |
| Enterprise | Databricks managed MLflow | W&B Enterprise |

**Recommendation:** Use MLflow if your company is on Databricks or prefers open source. Use W&B if you want better visualization and don't mind SaaS.

---

## Experiment Design

### Systematic Hyperparameter Search

**Grid Search:** Try all combinations. Exhaustive but expensive.
```python
# 4 x 4 x 3 = 48 combinations
grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
}
```

**Random Search:** Sample randomly from distributions. More efficient than grid search.
```python
# 50 random combinations — often finds better results than grid with same budget
distributions = {
    'max_depth': randint(3, 12),
    'learning_rate': loguniform(0.001, 0.1),
    'n_estimators': randint(100, 2000),
}
```

**Bayesian Optimization:** Use past results to guide search. Most sample-efficient.
```python
# Optuna example
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Search Strategy Decision Tree

```
Budget < 20 trials     → Random search (simple, decent results)
Budget 20-100 trials   → Bayesian optimization (Optuna or W&B sweeps)
Budget > 100 trials    → Bayesian + early stopping (prune bad trials quickly)
Continuous parameters  → Bayesian (exploits smoothness)
Mostly categorical     → Random or grid (Bayesian less helpful)
```

---

### Check Your Understanding: W&B and Hyperparameter Search

**1. In the W&B sweeps example, why is Bayesian optimization (`method: 'bayes'`) preferred over grid search for a budget of 50 trials?**

<details>
<summary>Answer</summary>

With 50 trials, grid search can only explore a sparse grid (e.g., 4 x 4 x 3 = 48 combinations for just 3 parameters), missing potentially good regions between grid points. Bayesian optimization uses results from completed trials to build a probabilistic model of the objective function and intelligently samples the next trial from regions likely to yield improvements. This is far more sample-efficient -- it finds better hyperparameters with fewer trials, especially for continuous parameters where grid search wastes trials on arbitrary grid points.
</details>

**2. What is the advantage of using `log=True` (log-uniform distribution) for the learning rate in a hyperparameter search?**

<details>
<summary>Answer</summary>

Learning rate typically matters on a logarithmic scale -- the difference between 0.001 and 0.01 is much more significant than the difference between 0.091 and 0.1. A uniform distribution would spend most of its samples in the range 0.05-0.1 (which all behave similarly) and very few in the range 0.001-0.01 (where small changes matter a lot). A log-uniform distribution samples uniformly in log space, giving equal attention to each order of magnitude (0.001-0.01, 0.01-0.1), which aligns with how learning rate actually affects training.
</details>

---

## Comparing Experiments

### Comparison Table

```
┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Run             │ AUC      │ F1       │ Train Time│ Model Size│
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ gbm_baseline    │ 0.872    │ 0.81     │ 45 min   │ 12 MB    │
│ gbm_tuned       │ 0.891    │ 0.84     │ 90 min   │ 24 MB    │
│ xgboost_v1      │ 0.895    │ 0.85     │ 30 min   │ 18 MB    │
│ lightgbm_v1     │ 0.893    │ 0.84     │ 15 min   │ 8 MB     │ ← best trade-off
│ deep_model_v1   │ 0.898    │ 0.86     │ 4 hours  │ 200 MB   │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘
```

### What to Compare Beyond Metrics

| Dimension | Question | Impact |
|-----------|----------|--------|
| Accuracy | Which model has the best metric? | Primary selection criterion |
| Latency | How fast is inference? | Determines serving feasibility |
| Model size | How large is the artifact? | Storage and loading time |
| Training cost | How long and how expensive to train? | Budget and iteration speed |
| Interpretability | Can you explain predictions? | Stakeholder trust, debugging |
| Slice performance | Does it work for all segments? | Fairness, reliability |
| Robustness | How sensitive to input perturbations? | Production reliability |

### Statistical Significance

Don't declare a winner based on a single metric difference. Test whether the difference is significant.

```python
from scipy import stats

# Compare two models using paired bootstrap test
def bootstrap_comparison(y_true, pred_a, pred_b, metric_fn, n_bootstrap=1000):
    """Test whether model B is significantly better than model A."""
    n = len(y_true)
    improvements = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        metric_a = metric_fn(y_true[indices], pred_a[indices])
        metric_b = metric_fn(y_true[indices], pred_b[indices])
        improvements.append(metric_b - metric_a)

    improvements = np.array(improvements)

    # 95% confidence interval for the improvement
    ci_lower = np.percentile(improvements, 2.5)
    ci_upper = np.percentile(improvements, 97.5)

    return {
        'mean_improvement': improvements.mean(),
        'ci_95': (ci_lower, ci_upper),
        'significant': ci_lower > 0,  # True if B is significantly better
        'p_value': (improvements <= 0).mean(),
    }

result = bootstrap_comparison(y_test, pred_model_a, pred_model_b, roc_auc_score)
print(f"Improvement: {result['mean_improvement']:.4f} (95% CI: {result['ci_95']})")
print(f"Significant: {result['significant']}")
```

---

## Model Registry Integration

The experiment tracking to deployment pipeline:

```
┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Experiment      │────→│ Model Registry    │────→│ Serving          │
│ Tracking        │     │                  │     │                  │
│ (many runs)     │     │ Registered model │     │ Production model │
│                 │     │ + version        │     │                  │
│ MLflow / W&B    │     │ + stage          │     │ FastAPI / Triton │
└────────────────┘     └──────────────────┘     └─────────────────┘

Workflow:
1. Data scientist runs 50 experiments
2. Best run is registered as model version 3.2
3. Version 3.2 moves to "Staging" → automated tests run
4. Tests pass → version 3.2 moves to "Production"
5. Serving system loads the new production version
6. Previous version archived (rollback available)
```

---

### Check Your Understanding: Comparing Experiments

**1. In the bootstrap comparison test, why is it important that the confidence interval's lower bound is above zero (rather than just checking if the mean improvement is positive)?**

<details>
<summary>Answer</summary>

A positive mean improvement alone could be due to random chance. The bootstrap confidence interval quantifies uncertainty -- if the 95% CI lower bound is above zero, you can be 95% confident that the improvement is real and not a statistical artifact. For example, a mean improvement of 0.005 AUC with a 95% CI of (-0.002, 0.012) means the improvement could actually be negative, so you should not deploy based on it. This prevents deploying models based on noise in the evaluation set.
</details>

**2. In the experiment comparison table, the `lightgbm_v1` model is marked as "best trade-off" despite not having the highest AUC. Why might you choose it over `deep_model_v1`?**

<details>
<summary>Answer</summary>

LightGBM v1 achieves AUC 0.893 vs. deep model's 0.898 -- a difference of only 0.005. But LightGBM trains in 15 minutes vs. 4 hours (16x faster iteration), produces an 8 MB model vs. 200 MB (25x smaller, faster deployment and loading), and likely has lower inference latency. The 0.005 AUC difference may not be statistically significant, and the deep model's higher resource requirements reduce iteration speed and increase serving costs. Unless that 0.005 translates to meaningful business impact, the simpler model is the better engineering choice.
</details>

---

## Best Practices

### Naming Conventions

```
Experiment name: {project}_{version}
  merchant_churn_v3
  fraud_detection_v2
  search_ranking_v1

Run name: {model_type}_{description}_{date}
  xgb_baseline_20240115
  lgbm_tuned_more_features_20240120
  deep_transformer_embeddings_20240125

Tags: [model_type, data_version, experiment_phase]
  ["xgboost", "data_v2024.01", "hyperparameter_search"]
```

### Documentation Per Experiment

```markdown
## Experiment: merchant_churn_v3

### Hypothesis
Adding 30-day rolling revenue features will improve churn prediction for small merchants.

### Changes from v2
- Added 5 new features: revenue_7d, revenue_30d, revenue_7d_30d_ratio, order_frequency_change, product_count_change
- Updated data split: temporal split at 2024-01-01 (was 2023-10-01)
- Same model architecture (XGBoost)

### Results
- AUC improved from 0.872 (v2) to 0.891 (v3)
- Improvement is statistically significant (p < 0.01)
- Large improvement for small merchants: AUC 0.83 → 0.89
- Slight regression for enterprise: AUC 0.91 → 0.90

### Decision
Promote v3 to production. The small merchant improvement justifies the minor enterprise regression.
```

### Anti-Patterns to Avoid

1. **Not tracking at all.** "I'll remember what I tried." You won't.
2. **Tracking only final metrics.** Log hyperparameters, data version, environment too.
3. **No baseline comparison.** Always compare to the current production model.
4. **Cherry-picking metrics.** Report all relevant metrics, not just the one that improved.
5. **Not versioning data.** Two runs with the same code but different data are not comparable.
6. **Overfitting to validation.** If you run 500 experiments and pick the best validation score, you've overfit to validation. Use a held-out test set.

---

## Common Pitfalls

**1. Overfitting to the validation set through excessive experimentation.** Running 500 experiments and selecting the best validation score effectively uses the validation set as a training signal. The "best" model may simply be the one that got lucky on validation data. Always hold out a final test set that is only evaluated once, after you have selected your best model.

**2. Not tracking the data version.** Two runs with identical code and hyperparameters but different training data are not comparable. If you change the data (add new rows, fix labels, change the date range) without logging which version was used, you lose the ability to understand why metrics changed and cannot reproduce past results.

**3. Cherry-picking metrics that improved while ignoring regressions.** A model that improves recall by 5% but drops precision by 10% is not an obvious win. Report all relevant metrics, including segment-level breakdowns, and make the trade-off explicit. Stakeholders lose trust when they discover unreported regressions after deployment.

**4. Skipping the production baseline comparison.** Comparing a new model against a previous experiment run instead of the actual production model leads to incorrect deployment decisions. The production model is the true baseline -- always compare against it using the same evaluation data and metrics.

---

## Hands-On Exercises

### Exercise 1: Structured Experiment with MLflow

Using a dataset of your choice (e.g., sklearn's breast cancer or iris dataset), implement a complete experiment tracking workflow:

1. Set up an MLflow experiment with a descriptive name
2. Train 3 different model types (e.g., logistic regression, random forest, gradient boosting)
3. For each, log: all hyperparameters, train/val/test metrics, feature importance plot, model artifact, and git commit hash
4. Use Optuna to run a Bayesian hyperparameter search (20 trials) for the best model type, logging each trial to MLflow
5. Register the best model in the MLflow Model Registry and transition it through None -> Staging -> Production

### Exercise 2: Statistical Significance Analysis

Using the bootstrap comparison function from this lesson:

1. Train two models (e.g., random forest and gradient boosting) on the same data
2. Run the paired bootstrap test with 1000 iterations
3. Report the mean improvement, 95% confidence interval, and p-value
4. Based on your results, write a one-paragraph recommendation for whether to deploy the new model, considering both statistical significance and practical factors (training cost, inference latency, model size)

---

## Practice Interview Questions

1. "How do you decide when a new model is good enough to deploy?"
2. "You've run 200 experiments and the best AUC is 0.005 better than baseline. Is it worth deploying?"
3. "How do you ensure reproducibility of ML experiments?"
4. "Walk me through your experiment workflow for improving a production model."
5. "How do you handle the case where a model improves overall metrics but regresses on a specific segment?"

---

## Key Takeaways

1. Track everything: code version, data version, hyperparameters, metrics, artifacts, environment.
2. MLflow is the industry standard open-source tool. W&B offers richer visualization.
3. Use Bayesian optimization (Optuna or W&B sweeps) for efficient hyperparameter search.
4. Statistical significance matters. Don't deploy based on a 0.001 AUC improvement on one eval set.
5. Slice-based evaluation is critical. A model that improves globally but fails for a segment is dangerous.
6. Document your experiments. Future you (and your team) will thank you.
7. The experiment-to-registry-to-serving pipeline must be automated for production ML.

---

## Summary and What's Next

This lesson covered the complete experiment tracking lifecycle: what to track and why, hands-on usage of MLflow and Weights & Biases, systematic hyperparameter search strategies (grid, random, Bayesian), rigorous experiment comparison with statistical significance testing, model registry integration, and best practices for naming, documentation, and avoiding common anti-patterns. Experiment tracking is the foundation of reproducible, auditable ML.

**Where to go from here:**
- **Model Serving** (./model-serving/COURSE.md) -- learn how registered models are deployed to production serving infrastructure
- **Monitoring and Drift** (./monitoring-drift/COURSE.md) -- understand what happens after deployment: how to detect when your carefully tracked model starts degrading
- **System Design** (./system-design/COURSE.md) -- see how experiment tracking fits into the broader ML system design framework
