## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The six-step framework for ML system design interviews (Clarify, Data, Features, Model, Serving, Monitoring) and the time allocation for each step
- How to structure complete ML system designs for recommendation, fraud detection, and search ranking problems
- The most common mistakes candidates make in system design interviews and how to avoid them

**Apply:**
- Walk through a complete ML system design from clarification to monitoring within 45-60 minutes
- Make and justify design decisions with explicit discussion of tradeoffs (e.g., XGBoost vs. neural networks, batch vs. real-time serving)

**Analyze:**
- Evaluate whether a proposed ML system design is appropriate for given scale, latency, and business constraints, identifying missing components or over-engineering

---

## Prerequisites

Before starting this lesson, you should be comfortable with:

- **System design for production ML** (../05-production-ml/system-design/COURSE.md) -- designing end-to-end ML systems including serving infrastructure, scaling, and deployment strategies
- **Evaluation metrics** (../04-classical-ml/evaluation-metrics/COURSE.md) -- precision, recall, AUC-ROC, NDCG, and selecting metrics appropriate to the problem
- **Data pipelines** (../05-production-ml/data-pipelines/COURSE.md) -- building data ingestion, transformation, and feature computation pipelines for training and serving

---

# ML System Design Interview Prep

## How ML System Design Interviews Work

ML system design interviews are 45-60 minutes where you design a machine learning system from scratch on a whiteboard or shared document. The interviewer gives you a vague problem ("Design a recommendation system for our product") and evaluates how you break it down, make decisions, handle tradeoffs, and communicate your thinking.

This is not a coding interview. You won't write working code. You'll draw architecture diagrams, discuss data pipelines, propose model architectures, and explain serving strategies. The interviewer wants to see that you can think through a complete ML system end-to-end.

**What interviewers evaluate:**
- Can you clarify ambiguous requirements?
- Do you understand the full ML lifecycle (data → features → model → serving → monitoring)?
- Can you make and justify design decisions?
- Do you know the tradeoffs between different approaches?
- Can you handle follow-up questions and pivot when needed?

---

## Framework for Answering

Use this six-step framework for every ML system design question. Spend approximately:
- Clarify: 5 minutes
- Data: 5-8 minutes
- Features: 5-8 minutes
- Model: 10-12 minutes
- Serving: 8-10 minutes
- Monitoring: 5 minutes

### Step 1: Clarify Requirements

**Never start designing immediately.** Ask questions to narrow the scope:

- What is the primary business metric we're optimizing? (Revenue? Engagement? Retention?)
- What is the scale? (Users, requests per second, data volume)
- What is the latency requirement? (Real-time vs batch?)
- What data do we have access to? (User behavior, product catalog, external data?)
- Is this a new system or improving an existing one?
- Any constraints? (Budget, team size, compliance requirements)

**Why this matters:** A recommendation system for a store with 1,000 products is completely different from one with 10 million. A fraud detection system that must respond in 50ms is different from one that can take 5 seconds. The interviewer wants to see you ask the right questions.

### Step 2: Data

Define what data you need and how to get it:
- What data sources exist? (Logs, databases, external APIs)
- What does the training data look like? (Schema, volume, labels)
- How do you handle labels? (Explicit labels, implicit signals, human annotation)
- Data quality issues? (Missing values, bias, noise, class imbalance)
- How is data stored and processed? (Data warehouse, streaming pipeline)

### Step 3: Features

Design the features the model will use:
- What raw features are available?
- What engineered features would help? (Aggregations, embeddings, interactions)
- How are features computed? (Batch precomputation vs real-time)
- Feature store: how are features served at prediction time?
- Feature freshness: how often do features need updating?

### Step 4: Model

Choose and justify the model architecture:
- What model type? (Tree-based, neural, linear, ensemble)
- Why this architecture? (What properties of the problem does it exploit?)
- Training procedure (loss function, optimizer, hyperparameters)
- How to handle cold start, class imbalance, or other challenges
- Baseline model vs production model (start simple, then improve)

### Step 5: Serving

Design how predictions reach users:
- Online (real-time) vs offline (batch) vs near-real-time
- Serving infrastructure (model server, API, caching)
- Latency budget and how to meet it
- How to handle model updates (blue-green, canary, shadow)
- Scaling (horizontal, load balancing)

### Step 6: Monitoring

Define how you know the system is working:
- What metrics to track (model performance, business metrics, system health)
- How to detect model degradation (data drift, prediction drift)
- Alerting thresholds
- Retraining strategy (scheduled, triggered by drift)
- A/B testing framework

---

## Worked Example 1: Design an AI Personal Shopper for E-Commerce

### Requirements Clarification

"I'd like to clarify a few things before diving in."

**Q: What is the primary goal?**
A: Help shoppers find and purchase products through conversational AI. Optimize for conversion rate and average order value.

**Q: What is the scale?**
A: The platform has millions of stores, hundreds of millions of products. Start with high-traffic stores. Target: handle 10,000 concurrent conversations, < 3 second response time.

**Q: What data is available?**
A: Product catalog (titles, descriptions, images, prices, categories, tags), user browsing history (for logged-in users), purchase history, store analytics, reviews.

**Q: What's the interaction model?**
A: Conversational — user describes what they want in natural language, the system recommends products and answers questions about them.

### Data Sources

**Product data:**
- Product catalog: title, description, price, images, categories, tags, variants
- Product metadata: reviews, ratings, sales velocity, inventory status
- Store-level data: store category, target audience, brand positioning

**User data (when available):**
- Browsing history: products viewed, time spent, add-to-cart events
- Purchase history: past orders, spending patterns, category preferences
- Demographic signals: location, device type, referral source

**Interaction data (for training):**
- Conversation logs: what users asked, what was recommended, what they clicked/purchased
- Click-through data: which recommendations led to product views
- Conversion data: which recommendations led to purchases

**Training labels:**
- Positive: user purchased the recommended product
- Weak positive: user clicked/viewed the recommended product
- Negative: user saw the recommendation but didn't engage

### Feature Engineering

**Product features:**
- Product embedding (from title + description, using a sentence transformer)
- Image embedding (from product images, using CLIP or similar)
- Price normalized within category
- Sales velocity (sales per day, log-transformed)
- Average rating, review count
- Category one-hot encoding
- Inventory status (in-stock, low-stock, out-of-stock)

**User features (logged-in):**
- User embedding (learned from purchase history using collaborative filtering)
- Category preference vector (distribution of past purchases across categories)
- Price sensitivity (average purchase price, price range)
- Recency features (days since last visit, days since last purchase)
- Session features (products viewed this session, time on site)

**Context features:**
- Query embedding (from the user's natural language request)
- Time of day, day of week (seasonality)
- Device type
- Current page/product being viewed
- Conversation turn number

**Interaction features:**
- Query-product similarity (cosine between query embedding and product embedding)
- User-product similarity (cosine between user embedding and product embedding)
- Category match (does the product match the user's preferred categories?)

### Model Architecture

**Two-stage pipeline: Retrieval + Re-ranking + LLM Generation**

**Stage 1: Candidate Retrieval**
- Architecture: Two-tower model
  - Query tower: encodes user query + context into a 256-dim embedding
  - Product tower: encodes product features into a 256-dim embedding
  - Similarity: dot product between query and product embeddings
- Training: contrastive loss (positive pairs from purchases, hard negatives from views without purchase)
- Serving: pre-compute product embeddings, use ANN (approximate nearest neighbor) search at query time
- Output: top-100 candidate products in < 50ms

**Stage 2: Re-ranking**
- Architecture: Cross-encoder (concatenate query features + product features, feed through MLP)
- Features: all user/product/context/interaction features above
- Training: pointwise cross-entropy on purchase labels, with position bias correction
- Output: ranked top-10 products with relevance scores, in < 100ms

**Stage 3: LLM Response Generation**
- Input: user query + top-5 ranked products with scores and metadata
- Output: natural language response explaining why these products are recommended
- Model: fine-tuned LLM (8B parameter, quantized for fast inference)
- Key: ground the LLM's response in the product data — no hallucinated product names or prices

### Serving Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│ Query Encoder     │ (encode query to 256-dim, 20ms)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ANN Search        │ (search product index, top-100, 30ms)
│ (FAISS/ScaNN)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature Lookup    │ (fetch user + product features, 20ms)
│ (Feature Store)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Re-ranker Model   │ (score 100 candidates, return top-10, 80ms)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Generation    │ (generate response with top-5 products, 800ms)
└────────┬─────────┘
         │
         ▼
Response to User (total: ~1 second)
```

**Latency budget:** 20 + 30 + 20 + 80 + 800 = ~950ms. Under 3 seconds with streaming.

**Scaling:**
- Product embeddings: pre-computed nightly, served from FAISS index (horizontal sharding)
- Feature store: Redis cluster for low-latency feature lookup
- Re-ranker: GPU-backed model server, horizontally scaled
- LLM: quantized model on GPU, autoscaled based on concurrent conversations

### Monitoring

**Model metrics:**
- Retrieval recall@100 (does the right product appear in top-100 candidates?)
- Re-ranker NDCG@10 (are the top-10 products ranked correctly?)
- LLM response quality (LLM-as-judge score)

**Business metrics:**
- Conversion rate (% of conversations leading to purchase)
- Average order value
- Click-through rate on recommended products
- User satisfaction (thumbs up/down on responses)

**System metrics:**
- P50/P90/P99 latency per component
- Error rate
- Cost per conversation

**Drift detection:**
- Monitor product embedding distribution (new products should be indexed)
- Monitor query distribution (new query patterns may need model retraining)
- Compare online vs offline metrics weekly

### Check Your Understanding

<details>
<summary>1. In the AI Personal Shopper design, why is a two-stage pipeline (retrieval + re-ranking) used instead of a single model that ranks all products?</summary>

With hundreds of millions of products, scoring every product with a complex model is computationally infeasible within the latency budget. The two-stage pipeline solves this: the retrieval stage uses a lightweight two-tower model with ANN search to find the top-100 candidates in under 50ms (sublinear in the number of products). The re-ranking stage then applies an expensive cross-encoder with rich features to score only those 100 candidates in under 100ms. This is the standard pattern for large-scale recommendation systems because it balances coverage (retrieval) with precision (re-ranking) within tight latency constraints.
</details>

<details>
<summary>2. What are the four types of training labels used for the recommendation model, and why are weak positive signals important?</summary>

(1) Positive: user purchased the recommended product. (2) Weak positive: user clicked or viewed the recommended product. (3) Negative: user saw the recommendation but did not engage. Weak positive signals (clicks, views) are important because purchases are rare events -- relying only on purchases would give too few positive examples to train effectively. Click and view signals provide much more training data, though they are noisier (a click does not guarantee intent to buy). This is a common tradeoff in recommendation systems: use weaker but more abundant signals to supplement sparse strong signals.
</details>

---

## Worked Example 2: Design a Fraud Detection System

### Requirements Clarification

- **Goal:** Detect fraudulent transactions in real-time for an e-commerce platform
- **Scale:** 100,000 transactions per day, 0.5% fraud rate
- **Latency:** Must respond within 100ms (blocks payment authorization)
- **Constraints:** False positive rate must be < 2% (blocking legitimate customers is costly)

### Data and Features

**Transaction features:** amount, currency, payment method, billing/shipping address match, time of day, device fingerprint, IP geolocation

**User history features:** account age, previous order count, previous fraud flags, average order value, shipping address change frequency

**Velocity features (real-time):** orders in last hour, distinct cards in last day, distinct addresses in last week, total spend in last 24 hours

**Network features:** is this IP associated with other flagged transactions? Is this device fingerprint shared with known fraudulent accounts?

### Model Architecture

**Gradient-boosted trees (XGBoost)** — not a neural network. Why:
- Tabular data with engineered features → tree-based models consistently outperform neural networks
- Extremely fast inference (< 5ms for XGBoost vs 50-100ms for a neural network)
- Interpretable feature importances (important for explaining fraud decisions to merchants)
- Easy to update incrementally as new fraud patterns emerge

**Two-model cascade:**
1. **Fast filter (rule-based):** Instantly approve low-risk transactions (returning customer, small amount, same address) — handles 70% of transactions in < 1ms
2. **ML model:** Score remaining 30% of transactions with XGBoost — handles in < 10ms

**Class imbalance handling:**
- 0.5% positive rate (fraud) → extreme imbalance
- Use SMOTE for training data augmentation
- Use focal loss or weighted cross-entropy
- Optimize for precision-recall tradeoff, not accuracy

### Serving

- **Feature store:** Redis for real-time velocity features, updated on every transaction
- **Model serving:** XGBoost model loaded in memory, inference < 5ms
- **Decision logic:** score > 0.7 → block, 0.3-0.7 → manual review, < 0.3 → approve
- **Feedback loop:** manual review decisions feed back into training data

### Monitoring

- **Precision and recall** updated daily on labeled data
- **False positive rate** tracked in real-time (merchants complain about blocked legitimate orders)
- **Alert:** if fraud rate spikes > 2x normal, trigger model review
- **Retraining:** monthly with latest fraud patterns, or on-demand if new fraud vector detected

### Check Your Understanding

<details>
<summary>1. Why does the fraud detection design use XGBoost instead of a neural network?</summary>

Three reasons: (1) The data is tabular with engineered features, and tree-based models consistently outperform neural networks on tabular data. (2) Inference must be under 100ms (it blocks payment authorization), and XGBoost inference is under 5ms vs. 50-100ms for a neural network. (3) Interpretability matters -- XGBoost provides feature importances that explain fraud decisions to merchants, which is required for compliance and trust. The two-model cascade further improves efficiency: a rule-based fast filter handles 70% of transactions in under 1ms, and only the remaining 30% need the ML model.
</details>

<details>
<summary>2. How does the fraud detection system handle the 0.5% positive rate (extreme class imbalance)?</summary>

Three approaches combined: (1) SMOTE for synthetic oversampling of the minority class in training data. (2) Focal loss or weighted cross-entropy to increase the loss contribution of minority class examples. (3) Optimize for precision-recall tradeoff rather than accuracy -- with 0.5% fraud rate, a model predicting "not fraud" for everything achieves 99.5% accuracy but is useless. The decision thresholds (score > 0.7 block, 0.3-0.7 manual review, < 0.3 approve) are tuned based on the precision-recall curve and business cost of false positives vs. false negatives.
</details>

---

## Worked Example 3: Design a Search Ranking System

### Requirements Clarification

- **Goal:** Rank search results for a product search engine (e-commerce store search)
- **Scale:** 50 million queries per day across all stores
- **Latency:** < 200ms end-to-end
- **Key challenge:** Search must work for millions of different stores, each with different products

### Architecture

**Three-stage pipeline:**

1. **Query understanding (20ms):** Parse query, extract intent, expand synonyms
   - "red nike running shoes size 10" → {color: red, brand: nike, category: running shoes, size: 10}
   - Spell correction, synonym expansion ("sneakers" → "shoes")

2. **Candidate retrieval (50ms):** Fetch matching products from the index
   - Elasticsearch/Solr with BM25 scoring
   - Semantic search (query embedding vs product embedding) for conceptual matches
   - Return top-500 candidates

3. **Learning-to-rank re-ranker (100ms):** Score and re-rank candidates
   - Features: BM25 score, semantic similarity, product popularity, price, availability, click-through rate, personalization signals
   - Model: LambdaMART (gradient-boosted trees optimized for ranking)
   - Loss: pairwise ranking loss (LambdaRank)
   - Training data: click logs with position bias correction

### Key Design Decisions

**Per-store vs global model:** Train one global model, with store-level features (store category, product count, traffic level). A per-store model would overfit — most stores don't have enough click data.

**Position bias correction:** Users click the first result more, regardless of relevance. Correct for this in training by including position as a feature and marginalizing it out at serving time.

**Real-time personalization:** For logged-in users, inject user preference features into the re-ranker. For anonymous users, use session-level signals (products viewed so far).

### Check Your Understanding

<details>
<summary>1. Why does the search ranking design use a global model instead of per-store models, and what design decision makes this work?</summary>

A per-store model would overfit because most stores do not have enough click data to train a reliable model. A global model is trained on data from all stores, giving it enough volume to learn robust ranking patterns. The key design decision that makes this work is including store-level features (store category, product count, traffic level) so the model can learn store-specific patterns without needing a separate model per store. This is a common pattern: one model with entity-level features instead of many per-entity models.
</details>

<details>
<summary>2. What is position bias in search ranking, and how is it corrected?</summary>

Position bias is the tendency for users to click the first search result more frequently regardless of its relevance. This biases click-based training data: the first result gets more clicks not because it is better, but because it is first. Correction: include the display position as a feature during training so the model learns to separate true relevance from position effects. At serving time, marginalize out the position feature (set it to a neutral value) so ranking decisions are based on relevance alone.
</details>

---

## Common Mistakes in ML System Design Interviews

### 1. Jumping to the Model

The biggest mistake. Candidates say "I'd use a transformer" in the first 30 seconds without understanding the data, features, or constraints. The model is important but it's 20% of the system.

**Fix:** Follow the framework. Spend the first 15 minutes on clarification, data, and features before mentioning any model architecture.

### 2. Over-Engineering

Proposing a complex architecture when a simple one would work. "I'd use a multi-head attention-based retrieval model with contrastive pre-training" when BM25 + a simple re-ranker would solve 90% of the problem.

**Fix:** Always start with a baseline. "My first version would be [simple approach]. Once that's working, I'd improve with [advanced approach]." Interviewers love this — it shows you understand iterative development.

### 3. Ignoring Scale and Latency

Proposing a system that works on a laptop but not at production scale. "I'd compute all pairwise similarities between users and products" — that's O(users × products), which is billions of operations.

**Fix:** Always ask about scale. Always compute back-of-the-envelope numbers. "With 1M users and 1M products, pairwise similarity would be 10^12 operations — that's not feasible. I'd use an ANN index to reduce this to O(log n) per query."

### 4. No Monitoring Plan

Designing a system with no plan for knowing if it works in production.

**Fix:** Always include monitoring. At minimum: model metrics (accuracy, latency), business metrics (conversion, revenue), data quality checks, and drift detection.

### 5. Not Acknowledging Tradeoffs

Every design decision has tradeoffs. Candidates who present one option without discussing alternatives seem inexperienced.

**Fix:** For every major decision, briefly mention the alternative and why you chose your approach. "I'd use XGBoost over a neural network here because the data is tabular, inference needs to be < 10ms, and interpretability matters for fraud decisions."

---

## How to Handle Questions You Don't Know

This happens to everyone. The interviewer asks about a technique or system you haven't encountered.

### Strategy 1: Reason from First Principles

"I'm not familiar with that specific algorithm, but based on the name and context, I'd guess it [reason about what it might do]. Here's how I'd approach the same problem: [describe your approach]."

### Strategy 2: Redirect to What You Know

"I haven't worked with [specific thing], but I've solved similar problems using [related technique]. Let me walk you through how that would work here."

### Strategy 3: Be Honest and Ask

"I'm not familiar with that. Could you give me a brief overview, and I'll work with it in my design?"

Interviewers respect honesty far more than bluffing. They've seen hundreds of candidates — they can tell when you're making things up.

---

## Common Pitfalls

**1. Spending 80% of the time on the model architecture.**
The model is approximately 20% of the system. Interviewers evaluate your ability to think about data, features, serving, and monitoring just as much. If you spend 40 minutes on model architecture and 5 minutes on everything else, you will not pass. Follow the time allocation in the framework: 5 min clarify, 5-8 min data, 5-8 min features, 10-12 min model, 8-10 min serving, 5 min monitoring.

**2. Not computing back-of-the-envelope numbers.**
Proposing "compute all pairwise similarities between users and products" without checking feasibility is a red flag. With 1M users and 1M products, that is 10^12 operations -- clearly infeasible. Always estimate: data volume, storage requirements, QPS, latency per component, and total cost. Interviewers expect you to sanity-check your own design.

**3. Proposing a complex architecture from the start.**
Saying "I'd use a multi-head attention-based retrieval model" as your first proposal signals inexperience. Always start with a baseline ("My v1 would be BM25 + a logistic regression re-ranker") and then propose improvements ("For v2, I'd add a two-tower retrieval model with learned embeddings"). This shows iterative thinking and practical judgment.

**4. Forgetting the feedback loop.**
Many ML systems have feedback loops: the model's predictions influence user behavior, which becomes the next round of training data. In fraud detection, if the model blocks certain transactions, you never learn whether they were truly fraudulent. In recommendations, users can only click on products the model shows them. Mentioning feedback loops and how to mitigate their effects (exploration, counterfactual evaluation) signals senior-level thinking.

---

## Hands-On Exercises

### Exercise 1: Timed System Design Practice

Set a timer for 45 minutes and design one of the following systems from scratch, using the six-step framework. Write out your full design (clarification questions and assumed answers, data sources, feature engineering, model architecture, serving architecture with latency budget, and monitoring plan):

1. A content moderation system for a social media platform (100M posts/day, must flag harmful content in under 500ms)
2. A dynamic pricing system for a ride-sharing service (must adjust prices in real-time based on supply/demand)
3. A job-candidate matching system for a hiring platform (rank candidates for each job posting)

After completing the design, review it against the "Common Mistakes" section and check: did you start with a baseline? Did you compute back-of-the-envelope numbers? Did you include monitoring and retraining?

### Exercise 2: Design Review

Take one of the worked examples from this lesson (AI Personal Shopper, Fraud Detection, or Search Ranking) and identify:

1. One component that could be simplified for a v1 launch
2. One component that would need to change if scale increased 100x
3. One missing consideration (e.g., privacy, fairness, internationalization) and how you would address it
4. What the feedback loop is in this system and how it could bias the model over time

---

## Key Takeaways

1. Use the framework: Clarify → Data → Features → Model → Serving → Monitoring
2. Spend 60% of your time on data, features, serving, and monitoring — not just the model
3. Start with a simple baseline, then propose improvements
4. Always discuss tradeoffs for major decisions
5. Compute back-of-the-envelope numbers for scale and latency
6. Include monitoring and retraining in every design
7. Be honest about what you don't know — reason from first principles

---

## Summary

This lesson provided a structured framework for ML system design interviews: Clarify requirements, define Data sources, engineer Features, choose and justify the Model, design Serving infrastructure, and plan Monitoring. Three worked examples -- an AI personal shopper (two-tower retrieval + re-ranking + LLM generation), a fraud detection system (XGBoost with rule-based cascade), and a search ranking system (query understanding + BM25 retrieval + LambdaMART re-ranking) -- demonstrate the framework in practice. Key principles include always starting with a simple baseline, computing back-of-the-envelope numbers, discussing tradeoffs for every major decision, and including monitoring and retraining in every design.

## What's Next

Complete the interview prep with [Pair Programming Interview Prep](../pair-programming/COURSE.md), which covers timed coding tasks you will encounter in ML engineering interviews -- data pipeline construction, model training and evaluation, fine-tuning scripts, debugging non-converging models, and building model serving endpoints.
