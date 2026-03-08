---
name: serve
description: >
  Deploy and serve trained ML models in production: inference APIs, containerization,
  CI/CD pipelines, monitoring, health endpoints, model versioning, and reproducibility
  packaging. Use when the user has a trained model and wants to deploy it, serve it,
  containerize it, build an inference API, set up monitoring, write a model card, create
  a CI/CD pipeline, or package for reproducibility.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
disable-model-invocation: true
argument-hint: path to model file or project root (e.g. "model.joblib" or ".")
---

# Model Serving & Deployment

Reference for taking trained models to production. Follow phases in order.

---

## Phase 1: Pre-deployment checklist

Before writing any serving code:

- [ ] Model format confirmed (joblib, .pt, .xgb, .onnx, SavedModel, HuggingFace)
- [ ] Preprocessing pipeline bundled or saved separately
- [ ] Input schema documented (feature names, types, expected ranges)
- [ ] Output schema documented (class names, probability range, regression bounds)
- [ ] Inference latency measured (single prediction + batch of 100)
- [ ] Memory footprint measured (`python3 -c "import joblib, tracemalloc; tracemalloc.start(); joblib.load('model.joblib')"`)
- [ ] `requirements.txt` pinned (`pip freeze > requirements.txt`)
- [ ] Training provenance recorded (experiment ID, dataset version, hyperparams)

---

## Phase 2: Inference API patterns

### Standard FastAPI template

```python
# serve.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, numpy as np, logging, time

logger = logging.getLogger(__name__)
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("model.joblib")
    logger.info("Model loaded")
    yield

app = FastAPI(title="Model API", version="1.0.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)

class PredictResponse(BaseModel):
    prediction: float
    probability: float | None = None
    latency_ms: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    X = np.array(req.features).reshape(1, -1)
    pred = float(model.predict(X)[0])
    proba = float(model.predict_proba(X).max()) if hasattr(model, 'predict_proba') else None
    return PredictResponse(prediction=pred, probability=proba,
                           latency_ms=(time.perf_counter() - start) * 1000)

@app.post("/predict/batch")
def predict_batch(requests: list[PredictRequest]):
    X = np.array([r.features for r in requests])
    preds = model.predict(X).tolist()
    return {"predictions": preds, "count": len(preds)}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready"}
```

### Async with background batching (high throughput)

```python
import asyncio
from collections import deque

request_queue: deque = deque()

async def batch_processor():
    while True:
        if len(request_queue) >= 32 or (request_queue and await asyncio.sleep(0.005) is None):
            batch = [request_queue.popleft() for _ in range(min(32, len(request_queue)))]
            features = np.array([b['features'] for b in batch])
            preds = model.predict(features)
            for item, pred in zip(batch, preds):
                item['future'].set_result(float(pred))
        await asyncio.sleep(0.001)
```

### Model caching (multiple models)

```python
from functools import lru_cache

@lru_cache(maxsize=4)
def load_model(version: str):
    return joblib.load(f"models/{version}/model.joblib")

@app.post("/predict/{version}")
def predict_versioned(version: str, req: PredictRequest):
    m = load_model(version)
    return {"prediction": float(m.predict([req.features])[0])}
```

---

## Phase 3: Containerization

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib .
COPY serve.py .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### docker-compose.yml

```yaml
services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.joblib
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      retries: 3
```

### Build and run

```bash
docker build -t model-api:v1.0 .
docker run -p 8000:8000 model-api:v1.0
docker compose up -d
```

---

## Phase 4: CI/CD (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Build & Deploy

on:
  push:
    branches: [main]
    paths: ['serve.py', 'requirements.txt', 'model.joblib']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt pytest httpx
      - run: pytest tests/ -v

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -t model-api:${{ github.sha }} .
          docker tag model-api:${{ github.sha }} model-api:latest
      - name: Smoke test
        run: |
          docker run -d -p 8000:8000 --name test-api model-api:latest
          sleep 5
          curl -f http://localhost:8000/health
          docker stop test-api
```

---

## Phase 5: Monitoring

### Structured request logging

```python
import json, time
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(json.dumps({
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": round(duration_ms, 2),
    }))
    return response
```

### Prometheus metrics

```python
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

predict_counter = Counter('predictions_total', 'Total predictions', ['status'])
predict_latency = Histogram('prediction_latency_seconds', 'Prediction latency',
                             buckets=[.005, .01, .025, .05, .1, .25, .5])

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Data drift detection (simple)

```python
import numpy as np

class DriftMonitor:
    def __init__(self, reference_stats: dict):
        self.ref = reference_stats  # {"feature": {"mean": x, "std": y}}
        self.window = []

    def record(self, features: list[float]):
        self.window.append(features)
        if len(self.window) >= 1000:
            self._check_drift()
            self.window = []

    def _check_drift(self):
        arr = np.array(self.window)
        for i, (feat, stats) in enumerate(self.ref.items()):
            z = abs(arr[:, i].mean() - stats['mean']) / (stats['std'] + 1e-8)
            if z > 3:
                logger.warning(f"Drift detected in {feat}: z-score={z:.2f}")
```

---

## Phase 6: Model card template

```markdown
# Model Card: {model_name}

## Model details
- **Type**: {classification|regression}
- **Algorithm**: {e.g. XGBoost, RandomForest}
- **Version**: {semantic version}
- **Training date**: {date}

## Intended use
- **Primary use**: {task description}
- **Out-of-scope**: {what this model should NOT be used for}

## Performance
| Metric | Validation | Test |
|--------|-----------|------|
| {metric} | {val_score} | {test_score} |

## Data
- **Training data**: {description, size, date range}
- **Features**: {N} input features
- **Target**: {description}

## Limitations
- {List known limitations}

## Ethical considerations
- {Bias considerations, fairness assessment}
```

---

## Phase 7: Reproducibility package

```
{project}/
├── model.joblib          # trained model
├── config.json           # hyperparameters + training config
├── data_manifest.json    # dataset checksums + version
├── requirements.txt      # pinned dependencies
├── serve.py              # inference API
├── Dockerfile
├── README.md             # how to reproduce training + how to serve
└── tests/
    └── test_serve.py     # API smoke tests
```

### config.json

```json
{
  "model": "xgboost",
  "experiment_id": "exp004",
  "hyperparameters": {"max_depth": 6, "learning_rate": 0.1},
  "training_date": "2026-03-07",
  "python_version": "3.11",
  "val_score": 0.8634,
  "test_score": 0.8601,
  "feature_count": 42
}
```

---

## Rules

- Pin ALL dependencies in requirements.txt before containerizing
- Always have `/health` and `/ready` endpoints — they are required for load balancers
- Log every prediction with input hash, output, and latency
- Never load models on every request — load once at startup
- Test the Docker image locally before pushing to registry
- Save config.json alongside every model artifact
