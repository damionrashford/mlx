# 10 — Portfolio Projects

> Capstone projects that prove you can do the job. Each project demonstrates a different competency.

## Project 1: E-Commerce Product Recommender + Agent

**Demonstrates:** Full ML pipeline + agent integration

```
Goal: Train a product recommendation model and deploy it as a tool
      inside an AI agent that can answer shopping questions.

Pipeline:
1. Data collection — e-commerce product catalog (public datasets or synthetic)
2. Feature engineering — product embeddings, category features, price normalization
3. Model training — collaborative filtering or embedding-based model
4. Evaluation — recall@K, NDCG, A/B test design
5. Serving — FastAPI endpoint
6. Agent integration — MCP tool or function call that queries the model
7. End-to-end demo — agent answers "find me a cozy sweater under $80"
```

**Status:** NOT STARTED

---

## Project 2: Tabular ML to Deep Learning Upgrade

**Demonstrates:** Upgrading from classical ML to deep learning on a real dataset

```
Goal: Take a tabular prediction task (e.g., sports outcomes, housing prices,
      or any structured dataset) and rebuild it using PyTorch instead of XGBoost.
      Compare performance. Understand when DL helps vs doesn't.

Pipeline:
1. Build or reuse an existing data pipeline for a tabular prediction task
2. Build a neural network in PyTorch for the same prediction task
3. Experiment with architectures (MLP, LSTM for sequences, transformer)
4. Compare against XGBoost baseline rigorously
5. Document when/why deep learning helps or doesn't
6. Deploy best model with experiment tracking (W&B or MLflow)
```

**Status:** NOT STARTED

---

## Project Evaluation Criteria

Each project should demonstrate:

- [ ] Data pipeline (collection, cleaning, feature engineering)
- [ ] Model selection with justification
- [ ] Proper evaluation (train/val/test, appropriate metrics)
- [ ] Experiment tracking
- [ ] Production readiness (serving, monitoring plan)
- [ ] Clear documentation of decisions and tradeoffs
