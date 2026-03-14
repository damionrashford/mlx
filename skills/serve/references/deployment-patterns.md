# ML Deployment Patterns

## Serving Options

| Pattern | Latency | Complexity | Best for |
|---------|---------|------------|----------|
| FastAPI + joblib | Low | Low | scikit-learn, XGBoost, small models |
| TorchServe | Low | Medium | PyTorch models |
| TF Serving | Low | Medium | TensorFlow/Keras models |
| ONNX Runtime | Very low | Medium | Cross-framework, optimized inference |
| Triton | Very low | High | GPU inference, multi-model |
| BentoML | Low | Low | Any framework, batteries included |

## Container Base Images

| Image | Size | Use |
|-------|------|-----|
| python:3.11-slim | ~150MB | Default for most models |
| nvidia/cuda:12.x-runtime | ~3GB | GPU inference |
| python:3.11-alpine | ~50MB | Minimal, no ML libs |

## Health Check Endpoints

- `GET /health` — liveness (is the process running?)
- `GET /ready` — readiness (is the model loaded?)
- `GET /metrics` — Prometheus-format metrics

## CI/CD Pipeline Stages

1. Test (unit + integration)
2. Build (Docker image)
3. Validate (smoke test against staging)
4. Deploy (rolling update / blue-green)
5. Monitor (latency, error rate, drift)
