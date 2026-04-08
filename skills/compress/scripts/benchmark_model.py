#!/usr/bin/env python3
# benchmark_model.py — Latency, throughput, accuracy comparison for two models.

import sys
import time
import argparse
import os


def load_model(path: str):
    """Load model from file (joblib or torch)."""
    from pathlib import Path
    p = Path(path)
    if p.suffix in (".joblib", ".pkl"):
        import joblib
        return joblib.load(path), "sklearn"
    elif p.suffix == ".pt":
        import torch
        model = torch.load(path, map_location="cpu")
        model.eval()
        return model, "torch"
    elif p.suffix == ".onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return session, "onnx"
    else:
        sys.exit(f"Unsupported format: {p.suffix}")


def make_predict_fn(model, framework: str):
    """Return a callable predict function."""
    if framework == "sklearn":
        return model.predict
    elif framework == "torch":
        import torch
        def predict(X):
            with torch.no_grad():
                return model(torch.from_numpy(X)).numpy()
        return predict
    elif framework == "onnx":
        input_name = model.get_inputs()[0].name
        def predict(X):
            import numpy as np
            return model.run(None, {input_name: X.astype("float32")})[0]
        return predict


def benchmark(predict_fn, X, n_warmup: int = 10, n_runs: int = 100) -> dict:
    """Benchmark latency and throughput."""
    import numpy as np

    # Warmup
    for _ in range(n_warmup):
        predict_fn(X[:1])

    # Single-sample latency
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X[:1])
        latencies.append((time.perf_counter() - t0) * 1000)

    # Batch throughput
    t0 = time.perf_counter()
    predict_fn(X)
    batch_elapsed = time.perf_counter() - t0
    throughput = len(X) / batch_elapsed

    # Memory (resident set size)
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        rss_mb = 0.0

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_rps": throughput,
        "memory_rss_mb": rss_mb,
    }


def accuracy(predict_fn, X, y) -> float:
    """Compute accuracy (classification) or 1 - normalized RMSE (regression)."""
    import numpy as np
    preds = predict_fn(X)
    if preds.ndim > 1 and preds.shape[1] > 1:
        # Classification probabilities
        preds = preds.argmax(axis=1)
    preds = preds.flatten()
    y = y.flatten()

    if set(y.tolist()).issubset({0, 1}) or len(set(y.tolist())) <= 20:
        # Classification
        return float(np.mean(preds.round() == y))
    else:
        # Regression — return 1 - normalized RMSE
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        return float(1.0 - rmse / (y.std() + 1e-8))


def print_comparison(name_a: str, r_a: dict, name_b: str, r_b: dict, acc_a: float, acc_b: float):
    def delta(a, b, higher_is_better=True):
        if a == 0:
            return "N/A"
        pct = (b - a) / abs(a) * 100
        sign = "+" if pct > 0 else ""
        better = (pct > 0) == higher_is_better
        marker = " ✓" if better else " ✗"
        return f"{sign}{pct:.1f}%{marker}"

    w = 22
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {name_a:<20} {name_b:<20} {'Delta'}")
    print("=" * 70)
    print(f"{'p50 latency (ms)':<25} {r_a['p50_ms']:<20.2f} {r_b['p50_ms']:<20.2f} {delta(r_a['p50_ms'], r_b['p50_ms'], False)}")
    print(f"{'p95 latency (ms)':<25} {r_a['p95_ms']:<20.2f} {r_b['p95_ms']:<20.2f} {delta(r_a['p95_ms'], r_b['p95_ms'], False)}")
    print(f"{'Throughput (req/s)':<25} {r_a['throughput_rps']:<20.1f} {r_b['throughput_rps']:<20.1f} {delta(r_a['throughput_rps'], r_b['throughput_rps'], True)}")
    print(f"{'Memory RSS (MB)':<25} {r_a['memory_rss_mb']:<20.1f} {r_b['memory_rss_mb']:<20.1f} {delta(r_a['memory_rss_mb'], r_b['memory_rss_mb'], False)}")
    print(f"{'Accuracy/Score':<25} {acc_a:<20.4f} {acc_b:<20.4f} {delta(acc_a, acc_b, True)}")
    print("=" * 70)

    acc_drop = abs(acc_a - acc_b)
    speedup = r_a["p50_ms"] / max(r_b["p50_ms"], 1e-6)
    mem_reduction = (r_a["memory_rss_mb"] - r_b["memory_rss_mb"]) / max(r_a["memory_rss_mb"], 1e-6)

    print(f"\nSummary:")
    print(f"  Latency speedup: {speedup:.2f}×")
    print(f"  Memory reduction: {mem_reduction*100:.1f}%")
    print(f"  Accuracy delta: {acc_drop:.4f} ({'acceptable' if acc_drop < 0.01 else 'SIGNIFICANT DROP'})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark original vs compressed model")
    parser.add_argument("original", help="Original model path")
    parser.add_argument("compressed", help="Compressed model path")
    parser.add_argument("data", help="Test CSV with features and optional target")
    parser.add_argument("--target-col", help="Target column name")
    parser.add_argument("--n-runs", type=int, default=100, help="Latency measurement runs")
    parser.add_argument("--n-warmup", type=int, default=10, help="Warmup runs")
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        sys.exit("Install: pip install numpy pandas")

    # Load data
    print(f"Loading test data: {args.data}")
    df = pd.read_csv(args.data)
    if args.target_col and args.target_col in df.columns:
        y = df[args.target_col].to_numpy()
        X = df.drop(columns=[args.target_col]).select_dtypes(include=[np.number]).to_numpy().astype("float32")
    else:
        X = df.select_dtypes(include=[np.number]).to_numpy().astype("float32")
        y = None

    print(f"Test set: {X.shape}")

    # Load models
    print(f"\nLoading original: {args.original}")
    model_a, fw_a = load_model(args.original)
    predict_a = make_predict_fn(model_a, fw_a)

    print(f"Loading compressed: {args.compressed}")
    model_b, fw_b = load_model(args.compressed)
    predict_b = make_predict_fn(model_b, fw_b)

    # Benchmark
    print(f"\nBenchmarking original ({args.n_runs} runs)...")
    results_a = benchmark(predict_a, X, args.n_warmup, args.n_runs)

    print(f"Benchmarking compressed ({args.n_runs} runs)...")
    results_b = benchmark(predict_b, X, args.n_warmup, args.n_runs)

    # Accuracy
    acc_a = accuracy(predict_a, X, y) if y is not None else 0.0
    acc_b = accuracy(predict_b, X, y) if y is not None else 0.0

    # Print comparison
    orig_name = os.path.basename(args.original)
    comp_name = os.path.basename(args.compressed)
    print_comparison(orig_name, results_a, comp_name, results_b, acc_a, acc_b)


if __name__ == "__main__":
    main()
