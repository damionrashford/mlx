# 04 — Classical Machine Learning

> The foundation models that still power most production ML. Transformers get the hype, XGBoost ships the features.

## Why This Matters

If you have tabular ML experience with XGBoost, you already have a strong starting point. Now you need breadth across classical ML — production recommendation systems use ensemble methods, collaborative filtering, and embedding-based approaches alongside LLMs.

## Subdirectories

```
04-classical-ml/
├── supervised/             # Regression, classification, trees, ensembles, SVMs
├── unsupervised/           # Clustering, PCA, anomaly detection
├── evaluation-metrics/     # Precision, recall, F1, AUC-ROC, confusion matrices
└── feature-engineering/    # Creating features, selection, importance
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| Machine Learning Specialization (Stanford/DeepLearning.AI) | Comprehensive classical ML + intro DL | ~3 months |
| StatQuest YouTube | Individual algorithm explanations | Ongoing |

## Algorithm Decision Guide

### "When do I use what?"

| Algorithm | Best for | Strengths | Weaknesses |
|---|---|---|---|
| Logistic Regression | Binary classification, baseline | Interpretable, fast, works with small data | Can't capture non-linear patterns |
| Random Forest | Tabular data, feature importance | Robust, little tuning needed | Slower inference than linear models |
| XGBoost / LightGBM | Tabular data, competitions, production | Best tabular performance, handles missing data | Harder to interpret than linear |
| SVM | Small datasets, high-dimensional | Strong with good kernels | Doesn't scale to large datasets |
| K-Nearest Neighbors | Simple classification | No training needed | Slow at inference, curse of dimensionality |
| K-Means | Clustering unlabeled data | Simple, scalable | Must specify K, assumes spherical clusters |
| PCA | Dimensionality reduction | Unsupervised, fast | Linear only, loses interpretability |
| Collaborative Filtering | Recommendations | Works without content features | Cold start problem |

## Key Concepts

### Supervised Learning
- [ ] Bias-variance tradeoff — underfitting vs overfitting
- [ ] Cross-validation — why holdout isn't enough
- [ ] Regularization (L1/L2) — preventing overfitting by penalizing complexity
- [ ] Ensemble methods — bagging (Random Forest) vs boosting (XGBoost)
- [ ] Hyperparameter tuning — grid search, random search, Bayesian optimization

### Evaluation (Critical for interviews)
- [ ] Accuracy — why it's misleading with imbalanced classes
- [ ] Precision — of what you predicted positive, how many were right?
- [ ] Recall — of actual positives, how many did you catch?
- [ ] F1 Score — harmonic mean of precision and recall
- [ ] AUC-ROC — model's ability to distinguish classes across thresholds
- [ ] Confusion matrix — the full picture of predictions vs reality
