#!/usr/bin/env python3
# shap_explain.py — Auto-detect model type, run correct SHAP explainer,
# save summary plot to explanations/shap_summary.png.

import sys
import os
import argparse
from pathlib import Path


def load_model(model_path: str):
    """Load sklearn or torch model from file."""
    p = Path(model_path)
    if p.suffix in (".joblib", ".pkl"):
        import joblib
        return joblib.load(model_path), "sklearn"
    elif p.suffix == ".pt":
        import torch
        model = torch.load(model_path, map_location="cpu")
        return model, "torch"
    else:
        sys.exit(f"Unsupported model format: {p.suffix}. Use .joblib, .pkl, or .pt")


def detect_sklearn_type(model) -> str:
    """Detect sklearn model type for explainer selection."""
    class_name = type(model).__name__
    # Check for pipeline
    if hasattr(model, "steps"):
        # Get final estimator
        model = model.steps[-1][1]
        class_name = type(model).__name__

    tree_types = {"RandomForestClassifier", "RandomForestRegressor",
                  "GradientBoostingClassifier", "GradientBoostingRegressor",
                  "DecisionTreeClassifier", "DecisionTreeRegressor",
                  "XGBClassifier", "XGBRegressor",
                  "LGBMClassifier", "LGBMRegressor",
                  "CatBoostClassifier", "CatBoostRegressor"}
    linear_types = {"LinearRegression", "Ridge", "Lasso", "ElasticNet",
                    "LogisticRegression", "SGDClassifier", "SGDRegressor"}

    if class_name in tree_types:
        return "tree"
    elif class_name in linear_types:
        return "linear"
    else:
        return "kernel"


def main():
    parser = argparse.ArgumentParser(description="SHAP model explanations")
    parser.add_argument("model", help="Model file (.joblib, .pkl, .pt)")
    parser.add_argument("data", help="Dataset CSV file (test set)")
    parser.add_argument("--target-col", help="Target column name (excluded from features)")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples for explanation")
    parser.add_argument("--output-dir", default="explanations", help="Output directory")
    args = parser.parse_args()

    try:
        import shap
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install shap pandas matplotlib")

    # Load data
    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data)
    if args.target_col and args.target_col in df.columns:
        X = df.drop(columns=[args.target_col])
    else:
        X = df
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)

    # Sample if needed
    if len(X) > args.max_samples:
        X = X.sample(args.max_samples, random_state=42)
    X_np = X.to_numpy().astype(np.float32)

    # Load model
    print(f"Loading model: {args.model}")
    model, framework = load_model(args.model)

    # Select and run explainer
    os.makedirs(args.output_dir, exist_ok=True)

    if framework == "sklearn":
        model_type = detect_sklearn_type(model)
        print(f"Model type: sklearn/{model_type} → using SHAP {model_type}Explainer")

        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_np)
            # For multiclass, shap_values is a list — use class 1 or squeeze
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_np)
            shap_values = explainer.shap_values(X_np)

        else:
            # KernelExplainer for any black-box
            print("Using KernelExplainer (may be slow for large samples)")
            background = shap.kmeans(X_np, min(50, len(X_np)))
            if hasattr(model, "predict_proba"):
                fn = model.predict_proba
            else:
                fn = model.predict
            explainer = shap.KernelExplainer(fn, background)
            shap_values = explainer.shap_values(X_np[:50], nsamples=200)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

    elif framework == "torch":
        import torch
        print("Using SHAP DeepExplainer for PyTorch model")
        X_tensor = torch.from_numpy(X_np)
        background = X_tensor[:min(100, len(X_tensor))]
        model.eval()
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_tensor)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.numpy() if hasattr(shap_values, "numpy") else shap_values

    # Summary plot
    output_path = os.path.join(args.output_dir, "shap_summary.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_np, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Bar plot
    bar_path = os.path.join(args.output_dir, "shap_bar.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_np, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")

    # Print top features
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    print("\nTop features by mean |SHAP value|:")
    for name, importance in ranked[:10]:
        print(f"  {name:<30} {importance:.4f}")


if __name__ == "__main__":
    main()
