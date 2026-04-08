# Model Explainability Guide

## SHAP: SHapley Additive exPlanations

### Which explainer to use

| Model type | Explainer | Speed |
|------------|-----------|-------|
| sklearn tree (RF, XGBoost, LightGBM) | TreeExplainer | Fast |
| sklearn linear (Ridge, Lasso, Logistic) | LinearExplainer | Fast |
| PyTorch/TensorFlow | DeepExplainer | Medium |
| Any black-box | KernelExplainer | Slow |

### TreeExplainer

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For classification: shap_values is list (one per class)
# For regression: shap_values is array
```

### LinearExplainer

```python
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
```

### DeepExplainer (PyTorch/TF)

```python
background = X_train[torch.randperm(len(X_train))[:100]]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test[:50])
```

### KernelExplainer (any model)

```python
# Slow but works for any model
background = shap.kmeans(X_train, 50)  # summarize training data
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer.shap_values(X_test[:20], nsamples=200)
```

## SHAP plots

```python
# Global feature importance (beeswarm — best for overview)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Single prediction breakdown
shap.waterfall_plot(shap.Explanation(
    values=shap_values[i],
    base_values=explainer.expected_value,
    data=X_test[i],
    feature_names=feature_names
))

# Interactive force plot (HTML)
shap.force_plot(explainer.expected_value, shap_values[i], X_test[i])

# Feature dependence (one feature vs another)
shap.dependence_plot("feature_name", shap_values, X_test)

# Bar plot (mean |SHAP|)
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

## LIME: Local Interpretable Model-agnostic Explanations

### Tabular data

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",  # or "regression"
)

exp = explainer.explain_instance(
    X_test[i],
    model.predict_proba,
    num_features=10,
)
exp.show_in_notebook()
exp.as_pyplot_figure()
```

### Text data

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(
    text_sample,
    pipeline.predict_proba,
    num_features=10,
)
```

## Integrated Gradients (captum — PyTorch only)

```python
from captum.attr import IntegratedGradients, LayerConductance

ig = IntegratedGradients(model)
attributions = ig.attribute(
    inputs=X_test_tensor,
    target=predicted_class,
    n_steps=200,
)

# Layer conductance: contribution of each neuron in a layer
lc = LayerConductance(model, model.layer2)
layer_attrs = lc.attribute(X_test_tensor, target=predicted_class)
```

## Permutation importance

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)
```

## Partial dependence

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_train,
    features=["feature1", "feature2", ("feature1", "feature2")],  # 2D for interactions
    feature_names=feature_names,
)
```

## Attention visualization (transformers)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("model_name", output_attentions=True)
outputs = model(**inputs)
attentions = outputs.attentions  # tuple of (batch, heads, seq, seq) per layer

# Average across heads, last layer
avg_attention = attentions[-1].mean(dim=1).squeeze()  # (seq, seq)

# Visualize as heatmap
import seaborn as sns
sns.heatmap(avg_attention.numpy(), xticklabels=tokens, yticklabels=tokens)
```
