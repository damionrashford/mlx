# Model Selection Guide

## By Data Size and Type

| Data | Rows | Start with | Then try |
|------|------|-----------|----------|
| Tabular, <10K | Small | Logistic/Ridge | Random Forest |
| Tabular, 10K-1M | Medium | XGBoost/LightGBM | Tuned ensemble |
| Tabular, >1M | Large | LightGBM | Neural net (tabular) |
| Text | Any | TF-IDF + LogReg | Transformers |
| Image | Any | Pretrained CNN | Fine-tuned ViT |
| Time series | Any | ARIMA/Prophet | LSTM/Temporal fusion |

## Hyperparameter Priorities

### XGBoost / LightGBM
1. learning_rate (0.01-0.3)
2. max_depth (3-10)
3. n_estimators (100-1000)
4. min_child_weight / min_data_in_leaf
5. subsample (0.6-1.0)
6. colsample_bytree (0.6-1.0)

### Random Forest
1. n_estimators (100-500)
2. max_depth (None, 10-30)
3. min_samples_leaf (1-10)

### Neural Networks
1. learning_rate (1e-5 to 1e-2)
2. batch_size (16-256)
3. architecture (layers, hidden size)
4. dropout (0.1-0.5)
5. weight_decay (1e-5 to 1e-2)

## Experiment Tracking (results.tsv)

```
exp_id	model	params	val_score	decision	notes
exp000	ridge	alpha=1.0	0.782	KEEP	baseline
exp001	xgb	lr=0.1,depth=6	0.831	KEEP	+6.3%
```
