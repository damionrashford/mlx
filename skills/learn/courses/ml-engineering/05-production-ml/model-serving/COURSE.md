# Model Serving

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The four main serving patterns (REST API, gRPC, embedded, batch) and when each is appropriate
- Production serving frameworks (FastAPI, TorchServe, Triton, TGI, vLLM) and their trade-offs
- Deployment strategies (A/B testing, canary deployments, shadow mode) and their risk profiles

**Apply:**
- Build a containerized ML model serving endpoint with FastAPI, including health checks, caching, and proper error handling
- Design a scaling strategy using horizontal scaling, auto-scaling, and appropriate load balancing for a given latency and throughput requirement

**Analyze:**
- Evaluate end-to-end serving latency budgets and identify optimization opportunities (caching, batching, quantization, ONNX conversion) for a production ML system

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- **ML system design fundamentals** -- serving patterns, architectural trade-offs, and how model serving fits within the overall ML system (see [System Design](./system-design/COURSE.md))
- **Training mechanics** -- how models are trained, serialized, and exported for inference (see [Training Mechanics](../02-neural-networks/training-mechanics/COURSE.md))

---

## Why This Matters

Training a model is step one. Getting it to serve predictions reliably at scale — with low latency, high throughput, proper monitoring, and safe deployments — is the engineering challenge. This is what separates ML research from applied ML engineering.

At large-scale platforms, models serve predictions for millions of merchants and hundreds of millions of consumers. Serving infrastructure must handle traffic spikes (Black Friday), fail gracefully, and allow safe rollouts.

---

## Serving Patterns

### Pattern 1: REST API

The most common pattern. Model wrapped in an HTTP service.

```
Client → HTTP POST /predict → Model Service → JSON response
```

**When to use:** General-purpose, works with any client, easy to load balance.
**Latency:** 10-100ms overhead from HTTP + serialization.

### Pattern 2: gRPC

Binary protocol, faster than REST. Used for service-to-service communication.

```
Client → gRPC call → Model Service → Protobuf response
```

**When to use:** Internal services where latency matters, streaming predictions.
**Latency:** 2-5x lower overhead than REST.

### Pattern 3: Embedded Model

Model runs inside the application process. No network call.

```
Application code → Load model into memory → Call predict() directly
```

**When to use:** Ultra-low latency (< 1ms), simple models (decision trees, logistic regression).
**Tradeoff:** Tight coupling between model and application deployment.

### Pattern 4: Batch Prediction

Pre-compute predictions for all entities. Store in a database. Serve from cache.

```
Batch job → Compute predictions for all merchants → Write to Redis/DB
Application → Read pre-computed prediction from Redis
```

**When to use:** Predictions don't change within the batch window, high coverage needed.
**Tradeoff:** Stale predictions, wasted compute on unqueried entities.

### Choosing a Pattern

```
┌───────────────────────────────────────────────────────────────┐
│ Latency requirement < 1ms?                                    │
│   YES → Embedded model                                        │
│   NO  → Latency requirement < 100ms?                         │
│           YES → Online serving (REST/gRPC)                    │
│           NO  → Latency requirement < 1 hour?                │
│                   YES → Near-real-time (micro-batch)          │
│                   NO  → Batch prediction                     │
└───────────────────────────────────────────────────────────────┘
```

---

## FastAPI for ML Models

FastAPI is the simplest way to serve an ML model. Start here, graduate to specialized frameworks when needed.

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI(title="Merchant Churn Predictor", version="1.0.0")

# Load model at startup (not per-request)
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("models/churn_model_v3.joblib")
    logging.info("Model loaded successfully")

class PredictionRequest(BaseModel):
    merchant_id: str
    features: dict  # {"revenue_30d": 5000, "order_count_7d": 12, ...}

class PredictionResponse(BaseModel):
    merchant_id: str
    churn_probability: float
    risk_tier: str  # "low", "medium", "high"
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract features in expected order
    feature_names = model.feature_names_in_
    feature_vector = np.array([[request.features.get(f, 0) for f in feature_names]])

    # Predict
    probability = model.predict_proba(feature_vector)[0][1]

    # Map to risk tier
    if probability > 0.7:
        risk_tier = "high"
    elif probability > 0.3:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    return PredictionResponse(
        merchant_id=request.merchant_id,
        churn_probability=round(probability, 4),
        risk_tier=risk_tier,
        model_version="v3.2.1",
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

```bash
# Run it
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
```

**When to graduate beyond FastAPI:**
- You need GPU inference (use Triton, TGI, vLLM)
- You need dynamic batching (use TorchServe, Triton)
- You need model ensembles (use Triton)
- You need multi-model serving (use Seldon, KServe)

---

## Production Serving Frameworks

### TorchServe (PyTorch models)

```
Features: Dynamic batching, model versioning, multi-model, metrics
Best for: PyTorch models in production, CPU or GPU
Limitation: PyTorch only
```

### NVIDIA Triton Inference Server

```
Features: Multi-framework (PyTorch, TensorFlow, ONNX, XGBoost),
          dynamic batching, model ensemble, GPU optimization
Best for: High-throughput GPU serving, multi-model deployments
Limitation: Complex setup, overkill for simple models
```

### TGI (Text Generation Inference) by Hugging Face

```
Features: Optimized for LLM serving, continuous batching, tensor parallelism
Best for: Serving LLMs (GPT-like models)
Limitation: Text generation models only
```

### vLLM

```
Features: PagedAttention for memory-efficient LLM serving, OpenAI-compatible API
Best for: High-throughput LLM serving
Limitation: Text generation models only
```

### Comparison Matrix

| Framework | Model Types | Batching | GPU Opt | Complexity | Use Case |
|-----------|------------|----------|---------|------------|----------|
| FastAPI | Any | Manual | No | Low | Simple models, prototypes |
| TorchServe | PyTorch | Dynamic | Yes | Medium | PyTorch production |
| Triton | Multi | Dynamic | Yes | High | Multi-model, high throughput |
| TGI | LLMs | Continuous | Yes | Medium | LLM serving |
| vLLM | LLMs | Continuous | Yes | Medium | High-throughput LLM |

---

### Check Your Understanding: Serving Patterns and Frameworks

**1. When would you choose gRPC over REST for model serving, and what is the primary performance advantage?**

<details>
<summary>Answer</summary>

Choose gRPC for internal service-to-service communication where latency matters and both client and server are controlled by the same team. gRPC uses a binary protocol (protobuf) instead of text-based JSON, resulting in 2-5x lower serialization/deserialization overhead compared to REST. It also supports streaming, which is useful for models that produce sequential outputs. REST is preferred when you need broad client compatibility (browsers, third-party integrations) or simpler debugging.
</details>

**2. What is dynamic batching, and what is the fundamental trade-off it introduces?**

<details>
<summary>Answer</summary>

Dynamic batching accumulates multiple incoming requests and processes them together in a single forward pass through the model. This dramatically increases throughput (fewer total forward passes) and improves GPU utilization. The trade-off is increased latency for individual requests -- each request must wait for the batch to fill (or for a max wait timeout). You configure a max wait time (e.g., 10ms) to bound the added latency. Dynamic batching is most beneficial for GPU models where the forward pass cost is high and batching amortizes it.
</details>

**3. Looking at the serving framework comparison, when would FastAPI be insufficient and you would need to graduate to Triton?**

<details>
<summary>Answer</summary>

FastAPI is insufficient when you need: (1) GPU-optimized inference with dynamic batching -- FastAPI requires manual batching implementation, (2) multi-framework model serving -- running PyTorch, TensorFlow, and XGBoost models from a single server, (3) model ensembles -- chaining multiple models in a pipeline, or (4) high-throughput GPU serving where you need to maximize GPU utilization. FastAPI is ideal for simple CPU models, prototypes, and low-traffic endpoints.
</details>

---

## Containerization for ML

Docker is the standard for packaging ML models for deployment.

### ML-Specific Dockerfile

```dockerfile
# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim as runtime

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy model artifact
COPY models/churn_model_v3.joblib /app/models/

# Copy application code
COPY app.py .

# Non-root user for security
RUN useradd -m appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### Key Practices

1. **Pin all dependency versions.** `scikit-learn==1.3.2`, not `scikit-learn>=1.0`.
2. **Multi-stage builds.** Build dependencies in one stage, copy only runtime artifacts to the final image.
3. **Model artifact outside the image (optional).** Download from model registry at startup for flexibility.
4. **GPU images.** Use `nvidia/cuda:12.0-runtime-ubuntu22.04` as base for GPU models.
5. **Image size matters.** Large images = slow deployments. Target < 1GB for CPU, < 5GB for GPU.

```
Image size comparison:
  python:3.11           → 1.0 GB (avoid)
  python:3.11-slim      → 130 MB (use this)
  python:3.11-alpine    → 50 MB (glibc issues with numpy/scipy — avoid)
```

---

## Scaling

### Horizontal Scaling

Run multiple instances of your model service behind a load balancer.

```
                    ┌─── Model Instance 1
Load Balancer ──────┼─── Model Instance 2
                    ├─── Model Instance 3
                    └─── Model Instance 4
```

**CPU models (XGBoost, sklearn):** Scale horizontally easily. Each instance handles concurrent requests.

**GPU models:** More complex. Each instance needs a GPU. GPU utilization matters.

### Auto-Scaling

Scale based on metrics:
- **CPU/Memory:** Scale when CPU > 70% for 5 minutes
- **Request latency:** Scale when p99 latency exceeds SLA
- **Queue depth:** Scale when pending requests exceed threshold
- **Custom metrics:** Scale based on GPU utilization, batch queue size

```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-predictor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-predictor
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: request_latency_p99
      target:
        type: AverageValue
        averageValue: 100m  # 100ms
```

### Load Balancing Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| Round Robin | Rotate through instances | Uniform request cost |
| Least Connections | Route to instance with fewest active requests | Variable request cost |
| Weighted | Route based on instance capacity | Mixed hardware (CPU + GPU) |

---

## Caching

Caching is the cheapest way to reduce latency and cost.

### What to Cache

| Cache Target | Example | TTL | Impact |
|-------------|---------|-----|--------|
| Full predictions | "Recommendations for merchant X" | 5-60 min | Eliminates inference entirely |
| Embeddings | Product embedding for product ID 123 | 24 hours | Eliminates embedding computation |
| Feature vectors | Pre-assembled features for merchant X | 1-5 min | Eliminates feature lookup |
| Model artifacts | Loaded model in memory | Until new version | Eliminates model loading |

### Cache Implementation

```python
import hashlib
import json
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_prediction_cached(merchant_id: str, features: dict, ttl: int = 300):
    """Cache predictions in Redis with TTL."""
    # Create cache key from inputs
    cache_key = f"pred:{merchant_id}:{hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute prediction
    prediction = model.predict(features)

    # Store in cache
    redis_client.setex(cache_key, ttl, json.dumps(prediction))

    return prediction
```

### Cache Invalidation Strategy

```
Event: New model deployed → Flush all prediction caches
Event: Feature update      → Invalidate affected entity caches
Event: TTL expires         → Re-compute on next request (lazy)
```

---

### Check Your Understanding: Containerization, Scaling, and Caching

**1. Why should you avoid `python:3.11-alpine` as a Docker base image for ML models?**

<details>
<summary>Answer</summary>

Alpine Linux uses musl libc instead of glibc. NumPy, SciPy, and many other scientific Python libraries are compiled against glibc and will either fail to install or require compilation from source on Alpine, dramatically increasing build times and image size. Use `python:3.11-slim` instead -- it is based on Debian with glibc, is only 130 MB (vs 1 GB for the full image), and works with all ML libraries out of the box.
</details>

**2. In the caching implementation, why does the cache key include a hash of the features rather than just the merchant_id?**

<details>
<summary>Answer</summary>

The same merchant may have different features at different times (features update as new data arrives). If you cache only by merchant_id, you would serve stale predictions based on outdated features even after the feature store updates. By including a hash of the feature values in the cache key, the cache automatically invalidates when features change -- a request with updated features will be a cache miss and trigger a fresh prediction.
</details>

**3. Why is "Least Connections" load balancing preferred over "Round Robin" when model inference times vary per request?**

<details>
<summary>Answer</summary>

With variable request costs (e.g., different input sizes causing different inference times), round robin can overload an instance that is processing slow requests while other instances are idle. Least Connections routes new requests to the instance with the fewest active requests, naturally balancing load based on actual processing capacity rather than assuming all requests are equal.
</details>

---

## A/B Testing Infrastructure

### Traffic Splitting

```
Incoming Request → Router → 90% → Model A (current production)
                          → 10% → Model B (challenger)
                                    ↓
                              Log predictions from both
                                    ↓
                              Compare metrics after N days
```

### Implementation

```python
import hashlib

def route_request(user_id: str, experiment_config: dict) -> str:
    """Deterministic routing based on user ID hash."""
    # Hash ensures same user always gets same model (consistent experience)
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = hash_value % 100  # 0-99

    if bucket < experiment_config['control_pct']:  # e.g., 90
        return 'model_a'  # control
    else:
        return 'model_b'  # treatment
```

### Canary Deployments

Gradual rollout with automated rollback:

```
Step 1: Deploy Model B to 1% of traffic. Monitor for 1 hour.
        → If error rate increases > 0.1%: automatic rollback
        → If latency p99 increases > 20%: automatic rollback

Step 2: Increase to 10%. Monitor for 6 hours.

Step 3: Increase to 50%. Monitor for 24 hours.

Step 4: Full rollout (100%).
```

### Shadow Mode

Run the new model on all traffic but don't serve its predictions. Compare outputs.

```
Request → Model A (production) → Return prediction to user
       → Model B (shadow)      → Log prediction (don't serve)

Compare: Are Model B's predictions different? Better? Does it crash?
```

**Shadow mode is the safest way to test a new model.** Zero user impact. Use it before canary.

---

## Latency Optimization

### Dynamic Batching

Instead of processing one request at a time, accumulate requests and process them together.

```
Requests arrive:  [R1] [R2] [R3]     [R4] [R5]
                   ↓    ↓    ↓         ↓    ↓
Without batching:  [P1] [P2] [P3]     [P4] [P5]    (5 forward passes)
With batching:     [P1, P2, P3]       [P4, P5]      (2 forward passes)
```

**Tradeoff:** Individual request latency increases (waiting for batch to fill), but throughput increases dramatically. Set a max wait time (e.g., 10ms).

### Model Optimization Techniques

| Technique | Speedup | Accuracy Loss | Effort |
|-----------|---------|---------------|--------|
| Quantization (FP32 → INT8) | 2-4x | < 1% | Low |
| Pruning | 2-5x | 1-3% | Medium |
| Distillation | 2-10x | 1-5% | High |
| ONNX conversion | 1.5-3x | 0% | Low |
| Feature reduction | Variable | Variable | Medium |

```python
# ONNX conversion example (scikit-learn)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Convert
initial_type = [('float_input', FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Serve with ONNX Runtime (2-3x faster than sklearn predict)
session = ort.InferenceSession("model.onnx")
prediction = session.run(None, {'float_input': features})[0]
```

### Async Processing

For non-blocking predictions where the caller doesn't need immediate results.

```python
from fastapi import BackgroundTasks

@app.post("/predict-async")
async def predict_async(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Accept request, process in background, notify via webhook."""
    job_id = str(uuid.uuid4())

    # Return immediately
    background_tasks.add_task(process_prediction, job_id, request)

    return {"job_id": job_id, "status": "processing"}

async def process_prediction(job_id: str, request: PredictionRequest):
    prediction = model.predict(request.features)
    # Store result, notify caller via webhook/polling
    await store_result(job_id, prediction)
    await notify_webhook(request.callback_url, job_id, prediction)
```

---

### Check Your Understanding: Deployment Strategies and Optimization

**1. Why should you run shadow mode before a canary deployment, rather than going straight to canary?**

<details>
<summary>Answer</summary>

Shadow mode runs the new model on all production traffic but does not serve its predictions to users, so there is zero user impact. This lets you verify that the new model (1) does not crash on real production inputs, (2) has acceptable latency, (3) produces reasonable predictions, and (4) handles edge cases in production data. Only after confirming the model is operationally sound in shadow mode should you proceed to a canary deployment, which does affect a small percentage of real users. This two-stage approach minimizes risk.
</details>

**2. Quantization (FP32 to INT8) provides 2-4x speedup with less than 1% accuracy loss. Why would you not always quantize?**

<details>
<summary>Answer</summary>

Quantization works well for models with robust learned representations (e.g., large neural networks) but can cause more significant accuracy loss in models that rely on precise numerical boundaries (e.g., small decision trees near threshold boundaries). Additionally, not all operations are supported in INT8 on all hardware, some models require post-training calibration data to quantize well, and the accuracy loss may be unevenly distributed across data segments (e.g., acceptable globally but problematic for a critical subgroup). Always evaluate quantized model performance on your specific use case and data before deploying.
</details>

---

## Edge Deployment

Serving models on the user's device instead of a server.

### When to Use Edge

| Factor | Server | Edge |
|--------|--------|------|
| Latency | Network round-trip | Zero network latency |
| Privacy | Data leaves device | Data stays on device |
| Offline | Requires connectivity | Works offline |
| Model size | Unlimited | Constrained (< 50MB typical) |
| Updates | Instant (server-side) | Requires app update or OTA |
| Cost | Server compute costs | Free (user's device) |

**Example:** A product image classifier that runs in a merchant's mobile app, allowing merchants to auto-categorize products by taking a photo — no server round-trip needed.

### Edge Frameworks

- **TensorFlow Lite**: Mobile/embedded, Android/iOS
- **Core ML**: iOS/macOS native
- **ONNX Runtime Mobile**: Cross-platform
- **MediaPipe**: Google's on-device ML toolkit

---

## End-to-End Serving Architecture

Putting it all together for a recommendation system at production scale:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Online Serving Path                            │
│                                                                        │
│  Storefront Request                                                    │
│       ↓                                                                │
│  CDN / Edge Cache (check for cached recommendations)                   │
│       ↓ (cache miss)                                                   │
│  API Gateway (rate limiting, auth)                                     │
│       ↓                                                                │
│  Recommendation Service                                                │
│       ├── Feature Store (Redis) → merchant features, shopper features  │
│       ├── Candidate Generation (ANN index in memory)                   │
│       └── Ranking Model (XGBoost, in-process)                         │
│       ↓                                                                │
│  Post-processing (filtering, diversity, business rules)                │
│       ↓                                                                │
│  Response (top 10 products, < 200ms total)                            │
│       ↓                                                                │
│  Log prediction (async, to Kafka for monitoring + feedback)            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Offline Pipeline                               │
│                                                                        │
│  Daily: Retrain model → Validate → Push to model registry              │
│  Daily: Recompute ANN index → Push to serving instances                │
│  Daily: Update feature store from BigQuery                             │
│  Hourly: Update trending/popular items cache                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Common Pitfalls

**1. Loading the model per request instead of at startup.** Deserializing a model file on every inference request adds hundreds of milliseconds to latency and wastes compute. Always load the model once at server startup (or on first request with lazy loading) and reuse the in-memory object for all subsequent requests.

**2. Not pinning dependency versions in Docker images.** Using `scikit-learn>=1.0` instead of `scikit-learn==1.3.2` means your container may build with a different library version than what the model was trained with, causing silent prediction differences or outright crashes due to incompatible serialization formats.

**3. Deploying without a fallback mechanism.** If the model fails to load, times out, or returns an error, the system should degrade gracefully -- for example, returning popular items for recommendations or approving transactions with enhanced monitoring for fraud. Without a fallback, model failures become user-facing outages.

**4. Ignoring the latency budget breakdown.** Teams often optimize model inference time while ignoring that feature lookup, network overhead, or post-processing dominate the total latency. Always profile the full request path end to end and optimize the actual bottleneck.

---

## Hands-On Exercises

### Exercise 1: Build and Profile a Model Serving Endpoint

Using the FastAPI example in this lesson as a starting point:

1. Add Redis-based prediction caching with a configurable TTL
2. Add request logging that records merchant_id, prediction, latency, and timestamp
3. Add a `/metrics` endpoint that returns cache hit rate, average latency, and request count
4. Profile the endpoint under load (use a tool like `wrk` or `locust`) and identify the bottleneck

### Exercise 2: Design a Deployment Plan

You have a new fraud detection model that is 3% better on AUC than the production model. Design a complete deployment plan:

1. What do you validate in shadow mode, and for how long?
2. What are your canary rollout stages and what metrics trigger automatic rollback?
3. How do you handle the transition period where some traffic uses the old model and some uses the new one?
4. What is your rollback procedure if the model degrades after full rollout?

---

## Practice Interview Questions

1. "Our recommendation model takes 500ms per request. The SLA is 200ms. How do you reduce latency?"
2. "We're deploying a new fraud model. How do you roll it out safely?"
3. "A model serving 10K requests/second needs to be updated without downtime. How?"
4. "When would you choose batch prediction over real-time inference? Give examples."
5. "How do you handle a model that works great in A/B test but the team wants to change the feature pipeline?"

---

## Key Takeaways

1. Start with FastAPI. Graduate to Triton/TGI only when you need GPU optimization or dynamic batching.
2. Caching is the cheapest performance win. Cache predictions, embeddings, and features.
3. Shadow mode before canary. Canary before full rollout. Never yolo deploy a model.
4. Containerize everything. Pinned dependencies, multi-stage builds, health checks.
5. Horizontal scaling with auto-scaling solves most throughput problems for CPU models.
6. Latency budget: know where every millisecond goes (network, feature lookup, inference, post-processing).
7. Log every prediction. You need this for monitoring, debugging, and retraining.

---

## Summary and What's Next

This lesson covered the complete model serving lifecycle: four serving patterns (REST, gRPC, embedded, batch), production frameworks (FastAPI through Triton/vLLM), containerization best practices, scaling strategies, caching, deployment approaches (A/B testing, canary, shadow mode), latency optimization (batching, quantization, ONNX), and edge deployment. Together, these concepts cover what it takes to get a model from a trained artifact to a production service handling millions of requests.

**Where to go from here:**
- **Monitoring and Drift** (./monitoring-drift/COURSE.md) -- once your model is serving, learn how to detect when it starts degrading and when to retrain
- **Experiment Tracking** (./experiment-tracking/COURSE.md) -- understand how the experiment-to-registry-to-serving pipeline works end to end
- **Data Pipelines** (./data-pipelines/COURSE.md) -- deep dive into the feature pipelines that feed your serving infrastructure
