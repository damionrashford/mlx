# 05 — Production ML

> Getting models from notebooks to serving millions of users. This is what "Applied" means in Applied ML Engineering.

## Why This Matters

Applied ML roles require end-to-end experience in training, evaluating, testing, and deploying machine learning products at scale. Building a model in a notebook is 10% of the job. The other 90% is production.

## Subdirectories

```
05-production-ml/
├── system-design/          # End-to-end ML system architecture
├── data-pipelines/         # ETL, feature stores, data quality
├── model-serving/          # APIs, TorchServe, Triton, latency optimization
├── monitoring-drift/       # Data drift, model degradation, alerting
└── experiment-tracking/    # MLflow, W&B, reproducibility
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| DeepLearning.AI: Machine Learning in Production | Full MLOps lifecycle (Andrew Ng) | ~16 hrs |
| DeepLearning.AI: LLMOps | LLM-specific ops | ~2 hrs |
| DeepLearning.AI: Efficiently Serving LLMs | Inference optimization at scale | ~2 hrs |
| DeepLearning.AI: Evaluating & Debugging GenAI (W&B) | Experiment tracking | ~2 hrs |

## Key Concepts

### ML System Design Pattern
```
Data → Features → Model → Serving → Monitoring
  ↑                                      │
  └──────── Retraining trigger ──────────┘
```

### Deployment Patterns
- **Shadow deployment:** Run new model alongside old, compare outputs without serving to users
- **Canary deployment:** Serve new model to small % of traffic, monitor, then expand
- **Blue/green:** Two identical environments, switch traffic between them
- **A/B testing:** Statistical comparison of model variants on real users

### Monitoring Checklist
- [ ] Input data distribution (detect drift from training data)
- [ ] Model prediction distribution (detect output anomalies)
- [ ] Latency and throughput (SLA compliance)
- [ ] Business metrics (are predictions actually helping users?)
- [ ] Error rates and failure modes
