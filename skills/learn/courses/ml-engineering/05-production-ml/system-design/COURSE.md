# ML System Design — Complete Guide

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The six-step ML system design interview framework (Requirements, Data, Features, Model, Serving, Monitoring)
- The distinction between batch, real-time, near-real-time, and hybrid serving architectures
- How feature stores, model registries, and training pipelines fit into ML system architecture

**Apply:**
- Design a complete ML system for recommendation and fraud detection use cases, including data, features, model, serving, and monitoring
- Select appropriate serving patterns (synchronous, asynchronous, batch, embedded) based on latency and scale requirements

**Analyze:**
- Evaluate architectural trade-offs between Lambda, Kappa, microservice, and two-stage recommendation patterns for a given production ML problem

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- **Supervised learning fundamentals** -- understanding of classification, regression, and model training workflows (see [Supervised Learning](../04-classical-ml/supervised/COURSE.md) or [Neural Network Fundamentals](../02-neural-networks/fundamentals/COURSE.md))
- **Evaluation metrics** -- precision, recall, AUC, NDCG, and when to use each (see [Evaluation Metrics](../04-classical-ml/evaluation-metrics/COURSE.md))

---

## Overview

ML system design interviews test whether you can go from a vague business problem to a complete, production-ready ML system. This is the highest-signal interview at top tech companies — it separates ML engineers who can build models from those who can ship ML products.

---

## 1. The ML System Design Interview Framework

Every ML system design should follow this structure. Memorize it.

### Step 1: Requirements Clarification (2-3 minutes)
Ask questions before designing anything:
- What is the business objective? (increase revenue, reduce fraud, improve search)
- Who are the users? (merchants, buyers, internal teams)
- What scale? (QPS, data volume, number of items/users)
- What latency requirements? (real-time < 100ms, near-real-time < 1s, batch is fine)
- What are the constraints? (privacy, fairness, explainability)

### Step 2: Data (5 minutes)
- What data is available? (user behavior, item attributes, transactions)
- How much data? (rows, features, label availability)
- How is data collected? (event logs, databases, third-party APIs)
- Label availability: do we have labels, or do we need to create them?
- Data quality issues: missing values, noise, bias

### Step 3: Feature Engineering (5 minutes)
- What features would a human expert use?
- User features, item features, context features, interaction features
- Real-time vs batch features
- Feature freshness requirements

### Step 4: Model Selection (5 minutes)
- Start simple (logistic regression baseline)
- Propose a more complex model (gradient boosting, neural network)
- Justify the choice based on data size, latency, interpretability
- Training strategy: offline batch, online learning, transfer learning

### Step 5: Serving (5 minutes)
- How are predictions delivered? (API, pre-computed, embedded in app)
- Latency budget: how to meet it
- Caching strategy
- Fallback mechanism when model fails

### Step 6: Monitoring & Iteration (3 minutes)
- How do you know the model is working in production?
- What metrics to track (model metrics + business metrics)
- How to detect degradation
- Retraining strategy

---

## 2. Worked Example: Product Recommendation System for an E-Commerce Platform

### Requirements
- **Goal**: recommend products to buyers on e-commerce stores to increase conversion.
- **Users**: ~100M monthly active buyers across all online stores.
- **Scale**: billions of products, millions of stores, need recommendations in < 200ms.
- **Constraint**: cold-start for new stores with no purchase history.

### Data
**Available signals**:
- Purchase history (buyer_id, product_id, store_id, timestamp, amount).
- Browse history (page views, searches, clicks, add-to-cart).
- Product metadata (title, description, images, price, category, tags).
- Store metadata (industry, GMV, country).
- Buyer profile (location, device, past purchases across stores).

**Labels**: implicit feedback — purchases, add-to-cart, clicks (weighted differently).

### Feature Engineering

**Buyer features** (batch-computed):
- Purchase frequency, average order value, preferred categories.
- Browsing patterns (time of day, device, session length).
- Cross-store purchase history.

**Product features** (batch-computed):
- Category, price range, rating, number of reviews.
- Text embeddings of title + description.
- Image embeddings (pre-computed via CNN/ViT).
- Sales velocity, return rate.

**Context features** (real-time):
- Current page / category being browsed.
- Items in cart.
- Time of day, day of week.
- Device type.

**Interaction features** (batch + real-time):
- Has buyer purchased from this store before?
- Has buyer viewed this product before?
- Co-purchase patterns: "buyers who bought X also bought Y."

### Model Architecture

**Two-stage system** (standard for recommendations at scale):

**Stage 1 — Candidate Generation (retrieve)**:
- From billions of products, narrow to ~1000 candidates.
- **Approach**: two-tower model.
  - Buyer tower: maps buyer features to a dense embedding.
  - Product tower: maps product features to a dense embedding.
  - Score = dot product of buyer and product embeddings.
  - Use Approximate Nearest Neighbor (ANN) search for fast retrieval (< 10ms).
  - Alternative: collaborative filtering (matrix factorization) as a simpler baseline.

**Stage 2 — Ranking (score)**:
- Score the ~1000 candidates with a richer model.
- **Approach**: gradient boosting (XGBoost/LightGBM) or a deep ranking model.
- Uses all features: buyer, product, context, and interaction features.
- Outputs a relevance score. Rank by score, return top K.

**Cold-start handling**:
- New buyers: use content-based features (browsing behavior, popular products).
- New products: use product metadata features (category, price, description embeddings).
- New stores: use industry averages until data accumulates.

### Serving Architecture
```
Buyer request → Load buyer features (online store, < 5ms)
             → Candidate generation (ANN search, < 10ms)
             → Load candidate features (batch, < 5ms)
             → Ranking model inference (< 20ms)
             → Post-processing (diversity, business rules, < 5ms)
             → Return top K products (< 50ms total)
```

**Caching**: cache recommendations for buyers with no new activity (TTL: 1 hour).
**Fallback**: if model fails, return popular products for the store/category.

### Monitoring
- **Online metrics**: click-through rate, add-to-cart rate, conversion rate, revenue per recommendation.
- **Model metrics**: NDCG@10, Recall@50, coverage (% of catalog recommended).
- **Data quality**: feature freshness, missing features, distribution drift.
- **Business metrics**: overall store conversion rate (A/B test).

### Retraining
- Retrain daily on the latest interaction data.
- Re-index product embeddings for ANN search daily.
- Full model retraining weekly with hyperparameter tuning.

---

## 3. Worked Example: Fraud Detection for a Payment Platform

### Requirements
- **Goal**: detect fraudulent transactions before payment is processed.
- **Scale**: millions of transactions per day.
- **Latency**: < 100ms (must not slow down checkout).
- **Constraints**: minimize false positives (blocking legitimate orders loses revenue). High recall is critical (missed fraud = chargebacks).

### Data
- Transaction data: amount, currency, payment method, billing/shipping address.
- Buyer data: account age, purchase history, device fingerprint, IP geolocation.
- Merchant data: industry, average order value, chargeback rate.
- Historical labels: confirmed fraud (chargebacks) vs legitimate.
- Velocity data: number of transactions in last hour/day from same card/IP/device.

### Feature Engineering

**Transaction features**:
- Amount deviation from merchant average.
- Billing-shipping address mismatch.
- Is card country different from IP country?
- Time since card was first seen.

**Velocity features** (real-time, critical):
- Transactions per hour from this card.
- Distinct merchants per hour from this card.
- Failed transaction count in last 24 hours.
- Amount spent in last hour from this IP.

**Device/session features**:
- New device for this buyer?
- Browser fingerprint seen before?
- Proxy/VPN detected?

**Graph features** (advanced):
- Is this card connected to known fraud networks?
- Shared attributes (email domain, phone, address) with fraudulent accounts.

### Model

**Approach**: two-layer system.

**Layer 1 — Rules engine** (catches obvious fraud):
- Hard rules: amount > $10,000, impossible velocity, known bad IPs.
- Runs in < 5ms.
- High precision, moderate recall.

**Layer 2 — ML model** (catches subtle fraud):
- Gradient boosted trees (XGBoost).
- Why not deep learning? Tabular data, interpretability needed for investigations, XGBoost performance is comparable.
- Output: fraud probability.
- Action based on probability:
  - < 0.1: approve automatically.
  - 0.1-0.7: approve with enhanced monitoring.
  - > 0.7: hold for manual review.
  - > 0.95: block automatically.

**Class imbalance** (0.5% fraud rate):
- Don't use accuracy. Use AUPRC and recall at fixed precision.
- Training: use scale_pos_weight in XGBoost.
- Evaluation: PR curve, confusion matrix at the actual operating threshold.

### Serving
```
Transaction → Rules engine (< 5ms)
           → Load features (online store, < 10ms)
           → Compute velocity features (streaming, < 15ms)
           → Model inference (< 10ms)
           → Decision logic (< 5ms)
           → Approve / Review / Block (< 50ms total)
```

### Monitoring
- **False positive rate**: what percentage of blocked transactions were legitimate? Track weekly.
- **False negative rate**: what percentage of chargebacks were not caught? Track monthly.
- **Precision at recall = 0.95**: are we maintaining precision while catching 95% of fraud?
- **Feature drift**: are velocity distributions shifting? New fraud patterns emerging?
- **Alert on**: sudden spike in fraud score distribution, drop in precision, increase in chargebacks.

---

### Check Your Understanding: System Design Framework and Worked Examples

**1. In the recommendation system example, why is a two-stage architecture (candidate generation + ranking) used instead of a single model that scores all products?**

<details>
<summary>Answer</summary>

With billions of products, scoring every item with a rich ranking model is computationally infeasible within the latency budget (< 200ms). The two-stage approach first uses a lightweight model (two-tower with ANN search) to narrow billions of products to ~1000 candidates in < 10ms, then applies a richer model with more features to rank only those candidates. This balances coverage with latency.
</details>

**2. In the fraud detection example, why is AUPRC preferred over accuracy as the primary metric?**

<details>
<summary>Answer</summary>

Fraud is extremely imbalanced (0.5% fraud rate). A model that predicts "not fraud" for every transaction achieves 99.5% accuracy but catches zero fraud. AUPRC (Area Under the Precision-Recall Curve) focuses specifically on the positive (fraud) class and is not inflated by the large number of true negatives, making it a far more informative metric for imbalanced problems.
</details>

**3. Why does the fraud detection system use a rules engine in front of the ML model rather than relying on ML alone?**

<details>
<summary>Answer</summary>

The rules engine catches obvious fraud patterns (e.g., impossible velocity, known bad IPs) with very high precision in under 5ms, reducing load on the ML model. It also provides a baseline defense that does not depend on model availability, serves as a fallback if the ML model fails, and captures known fraud patterns that do not require statistical learning to detect.
</details>

---

## 4. Online vs Offline Systems

### Batch Predictions (Offline)
- Compute predictions for all entities on a schedule (hourly, daily).
- Store in a database/cache for retrieval.
- **Use when**: predictions don't need to be real-time, or the entity set is bounded.
- **Examples**: daily churn predictions, weekly product recommendations, nightly risk scores.
- **Pros**: simple infrastructure, can use complex models, easy debugging.
- **Cons**: stale predictions, can't use real-time features, wasted compute for entities never queried.

### Real-Time Inference (Online)
- Compute prediction at request time.
- **Use when**: predictions depend on real-time context or must be fresh.
- **Examples**: fraud detection, search ranking, real-time pricing.
- **Pros**: fresh predictions, uses real-time features.
- **Cons**: latency constraints limit model complexity, requires robust serving infrastructure.

### Near-Real-Time (Streaming)
- Process events as they arrive (seconds to minutes delay).
- **Use when**: you need freshness but not sub-second latency.
- **Examples**: updating user profiles after each interaction, computing velocity features.

### Hybrid Architecture
Most production systems combine approaches:
- Batch: compute user/item features daily.
- Streaming: update velocity features in real-time.
- Online: compute the final prediction at request time using both.

---

## 5. Feature Stores (in System Design Context)

### Role in the Architecture
The feature store sits between raw data and models:
```
Raw Data → Feature Pipelines → Feature Store → { Training Pipeline, Serving API }
```

### Feast (Open Source)
- Define features in Python, materialize to online/offline stores.
- Online: Redis, DynamoDB. Offline: BigQuery, Redshift, S3/Parquet.
- Point-in-time joins for training data.
- Good starting point, limited built-in monitoring.

### Tecton (Managed)
- Enterprise feature platform. Built-in streaming, monitoring, access control.
- Supports real-time feature transformations.
- Used by large ML teams that need reliability.

---

## 6. Model Registry

### What It Solves
Track which model version is in production, who trained it, what data was used, and what performance it achieved.

### Workflow
```
Train → Register (version 2.3) → Stage (staging) → Test → Promote (production)
```

### What to Store
- Model artifact (serialized model file).
- Training data version / hash.
- Hyperparameters.
- Evaluation metrics on validation + test sets.
- Code version (git commit hash).
- Environment (Python version, library versions).

### Tools
- **MLflow Model Registry**: open source, widely adopted.
- **SageMaker Model Registry**: AWS-native.
- **Vertex AI Model Registry**: GCP-native.
- **Custom**: metadata in a database + artifacts in object storage.

---

### Check Your Understanding: Serving Architectures and Feature Stores

**1. When would you choose a hybrid architecture (batch + streaming + online) over a purely batch or purely online approach?**

<details>
<summary>Answer</summary>

A hybrid approach is best when some features are expensive to compute and change slowly (batch-computed daily, e.g., user purchase history), some features must be very fresh (streaming-computed in near-real-time, e.g., velocity features like transactions in the last hour), and the final prediction must be served at request time using both types of features. Most production ML systems at scale use this hybrid pattern.
</details>

**2. What is the key advantage of point-in-time joins in a feature store, and what problem do they prevent?**

<details>
<summary>Answer</summary>

Point-in-time joins ensure that when constructing training data, each example only uses features that were available at the time the event occurred. Without this, you risk data leakage -- using future information to predict past events. For example, using a merchant's 30-day revenue that includes days after the prediction target would inflate training metrics but fail in production where future data is unavailable.
</details>

---

## 7. Training Pipelines

### Why Pipelines, Not Notebooks
Notebooks are for exploration. Production training needs:
- Reproducibility: same code + data = same model.
- Automation: retrain on schedule without human intervention.
- Monitoring: alert if training fails or performance degrades.
- Versioning: track every training run.

### Pipeline Structure
```
Data Extraction → Validation → Feature Engineering → Training → Evaluation → Registration
     ↓                ↓              ↓                  ↓           ↓             ↓
  BigQuery       Great Expect.    Feature Store      XGBoost    Metrics       MLflow
                                                               threshold
                                                               check
```

### Orchestration
- **Airflow**: industry standard DAG scheduler. Verbose but battle-tested.
- **Prefect**: modern alternative to Airflow, Python-native, easier to write.
- **Kubeflow Pipelines**: Kubernetes-native, good for GPU workloads.

---

## 8. Serving Patterns

### Synchronous (Request/Response)
Client sends request, waits for prediction, gets response.
```
Client → API Gateway → Model Server → Response
```
- Use for: real-time predictions (fraud, search, recommendations).
- Latency budget: typically < 100ms.

### Asynchronous (Message Queue)
Client sends request to queue, gets acknowledgment, prediction arrives later.
```
Client → Message Queue → Model Worker → Result Store → Client polls/webhook
```
- Use for: batch-like workloads that arrive individually (image processing, document analysis).
- No strict latency requirement.

### Batch
Pre-compute predictions for all entities.
```
Scheduler → Training Pipeline → Score all entities → Write to DB → Serve from DB
```
- Use for: daily reports, email campaigns, non-time-sensitive predictions.

### Embedded
Model runs on the client device.
```
Model (ONNX/TFLite) → Mobile App / Browser / Edge Device
```
- Use for: offline capability, privacy-sensitive applications, ultra-low latency.

---

## 9. Common Architecture Patterns for ML Systems

### Lambda Architecture
Batch layer (complete, accurate) + speed layer (fast, approximate):
```
Raw Data → Batch Pipeline → Batch View
        → Stream Pipeline → Real-time View
Query = merge(Batch View, Real-time View)
```

### Kappa Architecture
Everything through streaming, no separate batch layer:
```
Raw Data → Stream Pipeline → Serving Layer
```
Simpler but requires a capable streaming system (Kafka + Flink).

### Microservice Architecture for ML
```
API Gateway → Feature Service → Model Service → Post-processing Service
                ↓                    ↓
           Feature Store        Model Registry
```
Each component scales independently. Models can be updated without redeploying the entire system.

### The Recommendation Pattern (Two-Stage)
```
Request → Candidate Generation (fast, broad) → Ranking (accurate, narrow) → Results
```
Used by YouTube, Netflix, Amazon, Spotify, and most large-scale platforms. This is the most important pattern to know for interviews.

---

### Check Your Understanding: Architecture Patterns and Serving

**1. What is the key difference between Lambda and Kappa architectures, and when would you choose Kappa?**

<details>
<summary>Answer</summary>

Lambda architecture maintains two separate pipelines -- a batch layer for complete, accurate processing and a speed layer for fast, approximate processing -- with a query layer that merges both views. Kappa architecture eliminates the batch layer entirely, routing everything through a single streaming pipeline. Choose Kappa when you have a capable streaming system (Kafka + Flink), want to avoid maintaining two separate codebases, and can afford to reprocess the stream for corrections. Kappa is simpler to maintain but requires more sophisticated streaming infrastructure.
</details>

**2. In a microservice architecture for ML, why is it beneficial to separate the Feature Service, Model Service, and Post-processing Service into independent services?**

<details>
<summary>Answer</summary>

Independent services allow each component to scale independently (e.g., the feature service may need more instances than the model service), be updated without redeploying the entire system (models can be swapped without touching feature logic), use different technologies (feature service talks to Redis, model service runs inference), and be owned by different teams. This decoupling reduces deployment risk and increases development velocity.
</details>

---

## Common Pitfalls

**1. Skipping the baseline model.** Jumping straight to complex architectures (deep learning, two-stage retrieval) without establishing a simple baseline (logistic regression, popularity-based recommendations) makes it impossible to measure the value of complexity. Interviewers specifically watch for this.

**2. Ignoring training-serving skew.** Features computed differently in training (batch SQL) and serving (real-time code) produce silently inconsistent predictions. This is one of the most common and hardest-to-debug production ML failures. Always ensure feature computation logic is shared or validated between training and serving.

**3. Designing monitoring as an afterthought.** Many candidates describe an elegant system but mention monitoring only if prompted. In production, a model without monitoring is a ticking time bomb. Design monitoring as a first-class component of the system.

**4. Over-engineering for scale you do not have.** Proposing Kubernetes-orchestrated multi-model Triton serving for a system that handles 10 QPS is a red flag. Start with the simplest serving pattern that meets requirements and describe how to scale up if needed.

---

## Hands-On Exercises

### Exercise 1: Design a Search Ranking System

Design an ML system for ranking search results on an e-commerce platform. Follow the six-step framework:

1. Define requirements (who are the users, what is the latency budget, what are success metrics)
2. Identify data sources and labels
3. Propose at least 5 specific features across user, query, and item categories
4. Select a model architecture and justify it
5. Describe the serving architecture with a latency budget breakdown
6. Define monitoring metrics (model, prediction quality, and business)

Write your design as a structured document and compare it against the recommendation and fraud detection examples in this lesson.

### Exercise 2: Architecture Pattern Selection

For each scenario below, select the most appropriate architecture pattern (Lambda, Kappa, microservice, two-stage retrieval) and justify your choice:

1. A product recommendation system for a marketplace with 50M products and 200ms latency SLA
2. A risk scoring system that needs both real-time transaction signals and daily aggregate features
3. A content moderation system that processes user-generated images with no strict latency requirement
4. A dynamic pricing engine that adjusts prices every 5 minutes based on demand signals

---

## Key Takeaways

1. **Use the framework**: Requirements → Data → Features → Model → Serving → Monitoring. Every time.

2. **Start simple**: logistic regression baseline before complex models. Batch serving before real-time. Rules before ML.

3. **Two-stage recommendation** is the most common ML system pattern. Know it cold.

4. **Feature engineering is half the design**. Interviewers want to see you propose specific, creative features.

5. **Address cold-start explicitly**: every recommendation system interview will ask about new users/items.

6. **Monitoring is not optional**: if you skip it in the interview, you signal that you've never shipped ML to production.

7. **Connect to business metrics**: "This model improves NDCG@10 by 15%, which our A/B test showed increases conversion rate by 2.3%, translating to $X annual revenue."

---

## Summary and What's Next

This lesson covered the complete ML system design framework -- from requirements clarification through monitoring -- and applied it to two detailed worked examples (recommendations and fraud detection). You learned about batch vs. real-time vs. hybrid serving architectures, feature stores, model registries, training pipelines, and common architecture patterns like Lambda, Kappa, and two-stage retrieval.

**Where to go from here:**
- **Data Pipelines** (./data-pipelines/COURSE.md) -- deep dive into building the ETL/ELT and feature pipelines referenced throughout this lesson
- **Model Serving** (./model-serving/COURSE.md) -- detailed coverage of serving frameworks, containerization, scaling, and deployment strategies
- **Experiment Tracking** (./experiment-tracking/COURSE.md) -- learn how to systematically manage the experiment lifecycle that feeds into the model registry
- **Monitoring and Drift** (./monitoring-drift/COURSE.md) -- expand on the monitoring concepts introduced here with drift detection methods and retraining strategies
