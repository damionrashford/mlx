# ML Evaluation Metrics: Measuring What Matters

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- Why accuracy is misleading for imbalanced problems and how precision, recall, F1, AUC-ROC, and AUC-PR each tell a different story
- How ranking metrics (NDCG, MAP, MRR, Recall@K) evaluate recommendation and search systems at different stages of the pipeline
- Why offline metric improvements do not always translate to production wins, and how A/B testing bridges this gap

**Apply:**
- Select the right evaluation metric for a given business problem by analyzing the relative cost of false positives versus false negatives
- Design cross-validation strategies (stratified, time-series, group) that produce reliable performance estimates without data leakage

**Analyze:**
- Diagnose model failure modes by connecting confusion matrix patterns, calibration plots, and online/offline metric discrepancies to root causes

## Prerequisites

- **Supervised Learning** — understanding classification and regression algorithms is necessary to interpret what the metrics are measuring and why different models produce different error profiles (see [Supervised Learning](../supervised/COURSE.md))
- **Probability and Statistics** — concepts like distributions, expected value, and statistical significance underpin log loss, calibration, and A/B testing methodology (see [Probability & Statistics](../../01-foundations/probability-statistics/COURSE.md))

## Why This Matters

Choosing the right metric is as important as choosing the right model. A model optimized for the wrong metric can score perfectly on paper and be useless in production. This is one of the most common interview topics — given a business problem, pick the right metric and explain why.

The critical skill: connecting metrics to business outcomes. "We improved recall from 85% to 92%" means nothing. "We now catch 7 more fraudulent transactions per day, saving approximately $35K monthly in chargebacks" gets you hired.

---

## 1. Why Accuracy is Misleading

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**The classic trap:** e-commerce fraud detection. 99.5% of transactions are legitimate, 0.5% are fraudulent. A model that predicts "not fraud" for everything achieves 99.5% accuracy. It catches zero fraud. This model is completely useless despite near-perfect accuracy.

**When accuracy works:** Balanced classes where all errors are equally costly. This is rare in practice. Even in balanced settings, accuracy hides which types of errors you're making.

**Rule:** Never report accuracy alone for imbalanced problems. This is an interview red flag — if you mention "accuracy" for fraud detection or anomaly detection, the interviewer is already skeptical.

---

## 2. Confusion Matrix: The Foundation

```
                    Predicted
                  Positive    Negative
Actual Positive  [ TP=85   |  FN=15  ]   <- 100 actual positives
Actual Negative  [ FP=30   |  TN=870 ]   <- 900 actual negatives
                   ^115 pred+   ^885 pred-
```

**Reading it:**
- **TP (True Positive)**: Correctly predicted positive. The model said fraud, and it was fraud.
- **FP (False Positive)**: Predicted positive, actually negative. False alarm. Type I error.
- **FN (False Negative)**: Predicted negative, actually positive. Missed case. Type II error.
- **TN (True Negative)**: Correctly predicted negative.

Every classification metric is derived from these four numbers. If you can draw and populate a confusion matrix for any problem, you can derive any metric on the spot.

**Multi-class extension:** For K classes, the confusion matrix is K x K. Each row = actual class, each column = predicted class. Diagonal = correct. Off-diagonal cells reveal specific confusion patterns ("the model confuses category A with category B 23% of the time").

**Derived metrics from the example above:**
```
Accuracy    = (85 + 870) / 1000 = 95.5%
Precision   = 85 / (85 + 30) = 73.9%
Recall      = 85 / (85 + 15) = 85.0%
Specificity = 870 / (870 + 30) = 96.7%
F1          = 2 * (0.739 * 0.85) / (0.739 + 0.85) = 79.1%
```

---

## 3. Precision

```
Precision = TP / (TP + FP)
```

"Of everything I predicted positive, how many were actually positive?"

Precision measures how much you can **trust** a positive prediction. High precision = when the model says "yes," it's almost always right.

**Optimize precision when false positives are costly:**

| Scenario | Why FP is costly |
|----------|-----------------|
| Spam filter | Legitimate email sent to spam = missed business opportunity |
| Content moderation | Wrongly removing a product listing = revenue loss + trust damage |
| Product recommendations | Showing irrelevant products erodes user trust and engagement |
| Automated account suspension | Suspending a legitimate account = catastrophic |

**E-commerce example:** Product recommendations. You show 10 products and 8 are relevant. Precision@10 = 80%. Users trust the recommendations and click. If precision drops to 20%, users stop engaging and the feature becomes dead weight.

---

## 4. Recall (Sensitivity / True Positive Rate)

```
Recall = TP / (TP + FN)
```

"Of all actual positives, how many did I catch?"

Recall measures **coverage** — are you finding all the positives? High recall = you catch most real cases, even if it means some false alarms.

**Optimize recall when false negatives are costly:**

| Scenario | Why FN is costly |
|----------|-----------------|
| Fraud detection | Missed fraud = financial loss, chargebacks |
| Disease screening | Missed cancer = delayed treatment, death |
| Security threats | Missed intrusion = data breach |
| Platform abuse detection | Missed policy violation = platform risk, legal exposure |
| Churn prediction | Missed churning user = lost recurring revenue |

**Production example:** Fraud detection. Out of 100 actual fraud cases, your model catches 92. Recall = 92%. The 8 missed cases result in chargebacks averaging $500 each = $4,000 in direct losses. You want recall as high as possible here, even at the cost of flagging some legitimate orders for manual review (lower precision).

---

## 5. The Precision-Recall Tradeoff

You can't maximize both simultaneously. Every classifier that outputs scores (probabilities) has a threshold — adjusting it trades precision for recall:

- **Lower threshold** (more permissive): predict more positives -> higher recall, lower precision
- **Higher threshold** (more conservative): predict fewer positives -> higher precision, lower recall

The right threshold depends on business costs:
```
If cost_of_FN = $1000 (missing a fraud case)
and cost_of_FP = $10 (manually reviewing a legit order)
then the optimal threshold heavily favors recall.
```

**This is the most important concept in applied ML evaluation.** Every interview question about metrics comes down to: "What's more costly — a false positive or a false negative? And by how much?"

---

## 6. F1 Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The **harmonic mean** of precision and recall. Not the arithmetic mean — the harmonic mean penalizes extreme imbalances. If precision = 100% and recall = 1%, arithmetic mean = 50.5%, but F1 = 1.98%. The harmonic mean correctly reflects that this is a terrible model.

**When to use:** When you need a single number that balances precision and recall and you don't have a clear cost preference between FP and FN. Common in NLP tasks (NER, information extraction).

**F-beta score for asymmetric costs:**
```
F_beta = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
```
- beta = 1: standard F1 (equal weight)
- beta = 2: weights recall 2x more (use for fraud detection — missing fraud is worse than false alarms)
- beta = 0.5: weights precision 2x more (use for spam filtering — blocking legit email is worse than letting some spam through)

**Practical note:** In real production systems, you rarely optimize F1 directly. You pick a threshold that optimizes the business-relevant cost function. F1 is useful for comparing models and for benchmarks.

---

### Check Your Understanding

1. Your fraud detection model has precision = 90% and recall = 20%. What does this mean in business terms, and is this likely a good operating point for a payment platform?
2. Why does the F1 score use the harmonic mean rather than the arithmetic mean of precision and recall?
3. You are building a content moderation system to flag policy-violating product listings. False positives remove legitimate listings (lost revenue, merchant trust damage). False negatives leave policy violations visible (legal risk, brand damage). Would you optimize for F0.5 or F2, and why?

<details>
<summary>Answers</summary>

1. Precision = 90% means when the model flags a transaction as fraud, it is correct 90% of the time (low false alarm rate). Recall = 20% means the model only catches 20% of actual fraud cases, missing 80%. For a payment platform, this is likely a poor operating point -- 80% of fraud goes undetected, resulting in massive chargeback losses. You would lower the classification threshold to increase recall (catching more fraud), accepting lower precision (more false alarms sent to manual review), since the cost of missed fraud far exceeds the cost of reviewing a legitimate transaction.

2. The harmonic mean penalizes extreme imbalances between precision and recall. If one is very low (e.g., recall = 1%), the harmonic mean will also be very low (F1 = 1.98%) even if the other is perfect (precision = 100%). The arithmetic mean would give 50.5%, falsely suggesting acceptable performance. The harmonic mean reflects the reality that a model with near-zero recall (or near-zero precision) is essentially useless, regardless of how good the other metric is.

3. Optimize for F0.5, which weights precision twice as much as recall. In this scenario, false positives (removing legitimate listings) cause direct revenue loss and merchant trust damage, which are immediately visible and damaging. While false negatives (missed violations) are also costly, the content moderation system likely has other mechanisms (user reports, periodic audits) to catch what the automated system misses. Prioritizing precision ensures that automated removals are trustworthy.

</details>

---

## 7. AUC-ROC

**The ROC curve plots:**
- X-axis: False Positive Rate = FP / (FP + TN) — what fraction of negatives are falsely flagged
- Y-axis: True Positive Rate (Recall) = TP / (TP + FN) — what fraction of positives are caught

Each point on the curve = one classification threshold. Sweeping the threshold from 0 to 1 traces the curve.

**AUC (Area Under the Curve):**
- AUC = 1.0: perfect classifier (there exists a threshold that perfectly separates classes)
- AUC = 0.5: random guessing (the diagonal line)
- AUC < 0.5: worse than random (flip your predictions)

**Probabilistic interpretation:** AUC = P(model scores a random positive higher than a random negative). This makes it a measure of ranking quality — does the model correctly order positives above negatives?

**When to use:**
- Comparing models before choosing a threshold
- When you care about ranking quality across all operating points
- Binary classification with reasonable class balance
- Model development and selection (not production monitoring)

**Limitations (critical for interviews):**
- **Misleading for imbalanced data.** When negatives vastly outnumber positives, even a small FPR corresponds to many false positives in absolute terms. AUC-ROC can be 0.98 while precision is terrible.
- Doesn't tell you about performance at a specific operating point.
- The FPR axis includes true negatives, which can inflate AUC when negatives dominate.
- Not actionable — you don't deploy a model at "all thresholds simultaneously."

---

## 8. AUC-PR (Precision-Recall Curve)

**Better than ROC for imbalanced data.** The PR curve plots precision (y-axis) vs recall (x-axis) at different thresholds.

**Why it's better for imbalanced data:** The PR curve focuses entirely on the positive class. It doesn't benefit from a large number of true negatives (which inflate AUC-ROC).

**Example revealing the gap:** Fraud detection at 0.5% prevalence:
- AUC-ROC = 0.98 (looks great)
- AUC-PR = 0.35 (reveals the model struggles to identify fraud without many false positives)

The ROC curve is inflated by the massive TN pool. The PR curve tells the real story.

**Average Precision (AP):** Approximates the area under the PR curve. Computed as the weighted mean of precisions at each threshold, with the weight being the increase in recall.

**Baseline comparison:**
- Random classifier AUC-ROC = 0.5 (always)
- Random classifier AUC-PR = prevalence of positive class (e.g., 0.005 for 0.5% fraud)
- Any model with AUC-PR significantly above prevalence is adding value

**When to use:** Any classification task with class imbalance — fraud, anomaly detection, medical diagnosis, abuse detection. This is most real-world problems.

---

## 9. Log Loss (Cross-Entropy)

```
Log Loss = -(1/n) * sum[y*log(p) + (1-y)*log(1-p)]
```

**What it measures:** How well the model's predicted probabilities match reality. It heavily penalizes confident wrong predictions — predicting 0.99 for a true negative incurs enormous loss.

**Why it matters:**
- Used as the training loss for logistic regression and neural networks
- Evaluates probability calibration, not just ranking
- Differentiable — can be optimized directly with gradient descent
- Used in ML competitions (Kaggle) as the primary metric

**When to use:** When you need well-calibrated probabilities (not just correct rankings). For example, if you're using predicted probabilities to make downstream business decisions ("send a retention email if churn probability > 0.3"), log loss ensures those probabilities are meaningful.

---

## 10. Regression Metrics

### RMSE (Root Mean Squared Error)
```
RMSE = sqrt((1/n) * sum((y - y_hat)^2))
```
- Same units as the target — directly interpretable ("predictions are off by $X on average")
- Penalizes large errors heavily (squared before averaging)
- Sensitive to outliers
- **Default regression metric.** A sports prediction model would typically use this or MSE.

### MAE (Mean Absolute Error)
```
MAE = (1/n) * sum(|y - y_hat|)
```
- Treats all errors equally (no squaring)
- Robust to outliers — a single massive error doesn't dominate
- Corresponds to the median prediction (minimizing MAE = predicting the median)
- Use when you want robustness to outliers or when all errors are equally costly

### RMSE vs MAE Decision
- Predicting delivery times where being 2 hours late is MORE than twice as bad as 1 hour late -> RMSE
- Predicting house prices where $100K error on a mansion matters as much as $100K on a starter home -> MAE
- In practice: report both. If RMSE >> MAE, you have a few very large errors driving RMSE up.

### MAPE (Mean Absolute Percentage Error)
```
MAPE = (1/n) * sum(|y - y_hat| / |y|) * 100%
```
- Scale-independent — compare across different targets
- Business-friendly: "predictions are off by 8% on average"
- **Problems:** Undefined when y = 0. Asymmetric — penalizes underprediction more than overprediction.
- Use sMAPE (symmetric MAPE) to fix the asymmetry

### R-squared (Coefficient of Determination)
```
R^2 = 1 - sum((y - y_hat)^2) / sum((y - y_mean)^2)
```
- Fraction of variance explained by the model
- R^2 = 1: perfect. R^2 = 0: no better than predicting the mean. R^2 < 0: worse than the mean.
- **Limitation:** R^2 always increases with more features (even useless ones). Use adjusted R^2 or cross-validated R^2 instead.
- Best for communicating results: "the model explains 78% of variance in merchant revenue."

---

## 11. Recommendation and Ranking Metrics

These are critical for ML engineering roles. Recommendation systems and search don't just predict relevance — they must **order** results correctly.

### Recall@K
```
Recall@K = |relevant items in top K| / |total relevant items|
```
"Of all relevant items, how many did we surface in the top K?"

This is the primary metric for the retrieval stage of a recommendation system. If the user would be interested in 20 products and your top-50 candidates include 15 of them, Recall@50 = 75%.

**E-commerce application:** "Of all products this merchant would actually purchase, how many appear in their top-10 recommendations?"

### Precision@K
```
Precision@K = |relevant items in top K| / K
```
"Of the K items we showed, how many were relevant?"

Measures the quality of the displayed results. Lower K = harder to achieve high precision.

### NDCG (Normalized Discounted Cumulative Gain)

The most important ranking metric. Accounts for both relevance level AND position.

```
DCG@K = sum(i=1 to K) [relevance_i / log2(i + 1)]

NDCG@K = DCG@K / IDCG@K   (IDCG = DCG of the ideal ranking)
```

- The log2 denominator **discounts** relevance at lower positions — position 1 matters most
- Handles graded relevance (not just binary relevant/not-relevant)
- NDCG = 1.0 when your ranking matches the ideal order
- NDCG = 0.0 when no relevant items appear in the top K

**E-commerce application:** "Are the most relevant products appearing first in search results?" A search that puts the perfect product at position 1 scores much higher than one that puts it at position 5, even though both "found" the product.

### MAP (Mean Average Precision)

Average Precision for one query:
```
AP = (1 / |relevant items|) * sum_k [Precision@k * rel(k)]
```
where rel(k) = 1 if item at position k is relevant, 0 otherwise.

MAP = average AP across all queries/users.

- Emphasizes relevant items appearing early in the ranking
- Binary relevance (relevant or not) — no graded relevance like NDCG
- Standard metric for information retrieval

### MRR (Mean Reciprocal Rank)
```
MRR = (1/Q) * sum(1 / rank_of_first_relevant_item)
```
Only cares about the first relevant result. Use for navigational queries where the user wants one specific thing.

### When to Use Each

| Metric | Use When | Example |
|--------|----------|---------|
| NDCG@K | Graded relevance, position matters | Product recommendations, search ranking |
| MAP | Binary relevance, position matters | Information retrieval, document search |
| Recall@K | Coverage matters (retrieval stage) | Candidate generation for recs |
| Precision@K | Result quality matters (top of funnel) | Displayed recommendations |
| MRR | User wants one correct answer | Navigational search, autocomplete |

---

### Check Your Understanding

1. Your fraud detection model has AUC-ROC = 0.97 but AUC-PR = 0.30. Which metric should you trust and why?
2. A recommendation system shows 10 products. The relevant product is at position 8 in model A and position 2 in model B. Both have the same Recall@10 (1.0). Why is NDCG@10 a better metric here, and which model scores higher?
3. What is the probabilistic interpretation of AUC-ROC, and why does this make it a measure of ranking quality?

<details>
<summary>Answers</summary>

1. Trust AUC-PR. With fraud detection, the positive class (fraud) is rare (typically < 1%). AUC-ROC is inflated by the massive pool of true negatives -- even a small false positive rate corresponds to many false positives in absolute terms, but the ROC curve does not reveal this. AUC-PR = 0.30 reveals the real story: the model struggles to identify fraud without generating many false positives. For any imbalanced classification problem, AUC-PR is the more honest metric.

2. NDCG accounts for position through logarithmic discounting: `relevance / log2(position + 1)`. Items at higher positions get more credit. Model B scores much higher on NDCG@10 because the relevant product at position 2 has a discount factor of 1/log2(3) = 0.63, while model A's relevant product at position 8 has a discount factor of 1/log2(9) = 0.32. Recall@10 treats all positions equally (did the item appear in the top 10?), missing the critical insight that users primarily engage with the first few results.

3. AUC-ROC equals the probability that the model assigns a higher score to a randomly chosen positive example than to a randomly chosen negative example. This is a ranking interpretation: it measures how well the model separates positives from negatives across all possible thresholds. It does not measure whether the absolute probability values are correct (that is calibration), only whether the ordering is right. This is why a model can have excellent AUC but poorly calibrated probabilities.

</details>

---

## 12. Calibration

**The question:** When your model says "70% probability of churn," do 70% of those merchants actually churn?

**Why it matters:** If you use predicted probabilities for downstream decisions (send retention emails, adjust fraud thresholds, price risk), those probabilities must be meaningful. A model can have perfect AUC (ranking) but terrible calibration (absolute probabilities).

**How to check:** Plot predicted probability (binned) vs actual frequency. A well-calibrated model follows the diagonal. Use `sklearn.calibration.calibration_curve`.

**How to fix:**
- **Platt scaling**: fit a logistic regression on the model's raw outputs vs true labels. Works well for models that produce uncalibrated scores.
- **Isotonic regression**: fit a non-parametric monotonic function. More flexible but requires more data.
- Always calibrate on a held-out set, not the training set.

**Which models need calibration?**
- Logistic regression: usually well-calibrated out of the box
- Random Forest: tends to push probabilities toward 0.5 (underconfident)
- XGBoost/Neural Networks: often poorly calibrated, always check
- SVM: doesn't output probabilities natively (Platt scaling required)

---

## 13. Cross-Validation

### Why Holdout Isn't Enough
A single train/test split is noisy. You might get lucky or unlucky with which examples end up in test. Cross-validation gives a more reliable performance estimate and crucially also gives you a variance estimate.

### K-Fold Cross-Validation
```
1. Split data into K equal folds (K=5 or K=10)
2. For each fold i:
   - Train on all folds except i
   - Evaluate on fold i
3. Average the K scores
4. Report: mean +/- standard deviation
```
K=5 for large datasets (> 100K), K=10 for smaller ones. Larger K = less bias in estimate but more variance and more computation.

### Stratified K-Fold
Same as K-fold but preserves class distribution in each fold. If data is 95% negative and 5% positive, each fold maintains that ratio. **Always use this for classification**, especially with imbalanced data.

### Time-Series Split
```
Fold 1: Train [months 1-6],   Test [month 7]
Fold 2: Train [months 1-7],   Test [month 8]
Fold 3: Train [months 1-8],   Test [month 9]
```
Training always uses past data, testing always uses future data. Never leak future information into training. **Sports prediction models must use this** — you can't use 2025 games to train a model that predicts 2024 outcomes.

### Group K-Fold
Ensures all samples from the same group (user, merchant, session) stay in the same fold. If merchant A has 50 orders, all 50 are either in train or test, never split. Prevents information leakage from repeated measurements on the same entity.

### Nested Cross-Validation
For hyperparameter tuning + unbiased evaluation:
- Outer loop: K-fold for evaluating final performance
- Inner loop: K-fold for hyperparameter selection
- Prevents overfitting the hyperparameters to the test set

---

## 14. Data Leakage: The Silent Killer

Data leakage occurs when information that wouldn't be available at prediction time contaminates your training features or evaluation. Your model learns shortcuts that don't exist in production. Results look amazing in development, then crash in production.

### Types of Leakage

**Temporal leakage** — using future information to predict the past:
```
BAD:  Rolling mean that includes future values
BAD:  Using tomorrow's stock price as a feature for today's prediction
GOOD: All features computed using only data available at prediction time
```

**Target leakage** — features derived from the target variable:
```
BAD:  Using "fraud_flag" as a feature to predict "is_fraud" (same event, different column)
BAD:  Using "loan_default_date" to predict "will_default"
GOOD: Only features that exist before the prediction moment
```

**Train/test leakage** — test set information contaminating training:
```
BAD:  Fitting StandardScaler on full dataset, then splitting
BAD:  Computing target encoding using all data
BAD:  Feature selection using full dataset
GOOD: Fit all preprocessing on training data only, transform test data
```

### How to Detect Leakage
- **Suspiciously high performance**: If your model is 99.9% accurate, you probably have leakage
- **Feature importance**: If one feature dominates with implausibly high importance, investigate it
- **Train-test gap**: If training performance is 99% but test is 75%, you have a different problem (overfitting), but if both are 99% and production is 60%, that's leakage
- **Temporal validation**: Performance drops significantly with time-series split vs random split? You had temporal leakage in the random split.

### The Golden Rule
For every feature, ask: **"Would I have this information at the exact moment I need to make this prediction in production?"** If the answer is no or "maybe," investigate.

---

## 15. Train/Validation/Test Split

**Why three sets, not two:**
- **Training set** (~70%): model learns parameters
- **Validation set** (~15%): you tune hyperparameters, select features, compare models
- **Test set** (~15%): touched ONCE at the very end for final, unbiased performance estimate

If you tune hyperparameters on the test set, it becomes a second validation set and your reported performance is optimistically biased. The test set must remain untouched until the final evaluation.

**Common mistake:** Repeatedly evaluating on the "test set" during development and selecting the model that scores highest. This is just hyperparameter tuning on the test set with extra steps.

**In practice with cross-validation:** Use CV on training data for model selection and hyperparameter tuning. Report final performance on the held-out test set. This is the gold standard.

---

### Check Your Understanding

1. You are building a churn prediction model for merchants. The dataset spans 24 months. A colleague uses random 5-fold cross-validation and reports AUC = 0.92. You then run time-series cross-validation and get AUC = 0.81. What explains the difference, and which result should you trust?
2. Why is it important to use stratified K-fold (rather than standard K-fold) for a classification problem where the positive class is 3% of the data?
3. A team repeatedly evaluates their model on the test set during development, each time tweaking features to improve the test score. Why is this problematic even if they never explicitly train on the test set?

<details>
<summary>Answers</summary>

1. The difference is caused by temporal leakage in the random split. Random cross-validation lets the model train on future data and predict the past -- for example, using December 2024 merchant behavior to predict June 2024 churn. The model learns temporal patterns (seasonal trends, platform changes) that would not be available at prediction time. Time-series CV always trains on past data and tests on future data, which matches the production setting. Trust the time-series result (AUC = 0.81). The 0.92 is optimistically biased.

2. With 3% positive class and standard K-fold, some folds might receive very few (or even zero) positive examples by random chance, making the evaluation on that fold unreliable. Stratified K-fold preserves the 3%/97% class ratio in every fold, ensuring each fold has a representative sample of both classes. This produces more stable and reliable performance estimates, especially for metrics like precision and recall that depend on the positive class distribution.

3. This is a form of indirect test set leakage. Each time they evaluate and then adjust based on the test score, they are using the test set to make modeling decisions (feature selection, hyperparameter choices). Over many iterations, they effectively overfit to the test set -- the model becomes tuned to the specific quirks of that test data rather than generalizing broadly. The reported test score becomes optimistically biased. The test set should be used exactly once, at the very end, to report a final unbiased performance estimate.

</details>

---

## 16. Online vs Offline Metrics

**The hard truth:** Offline metric improvements don't always translate to production wins. This is one of the most important lessons in applied ML.

**Why the gap exists:**
- Offline metrics measure prediction quality on static historical data
- Online metrics measure actual user behavior in a live system
- User behavior is dynamic — it responds to what you show them
- Historical data may not represent future distribution

**Examples of the disconnect:**
- Higher NDCG (offline) but no improvement in click-through rate (online) — the offline test set doesn't capture the diversity users want
- Better AUC (offline) but worse conversion (online) — the model is too aggressive with thresholds
- Improved MAE (offline) but users complain more (online) — users care about worst-case errors, not average

**Bridging the gap:**
- Use online evaluation (A/B tests) as the ultimate source of truth
- Design offline metrics that correlate with online metrics (validate this empirically)
- Use counterfactual evaluation techniques to estimate online impact from offline data

---

## 17. A/B Testing ML Models

**Why you can't just deploy a new model:** A model that looks better on offline metrics might hurt user experience, revenue, or trust in ways that offline data doesn't capture.

**A/B test setup:**
1. **Control group**: current production model (Model A)
2. **Treatment group**: new candidate model (Model B)
3. **Random assignment**: users randomly assigned to control or treatment
4. **Run long enough**: collect enough data for statistical significance

**Statistical significance:**
- Define success metric upfront (CTR, conversion, revenue per user)
- Choose significance level (alpha = 0.05 typical)
- Compute required sample size beforehand (power analysis)
- Don't peek and stop early (multiple testing problem)
- Use sequential testing if you must monitor continuously

**Guardrail metrics:** Metrics that must NOT degrade, even if the primary metric improves:
- Revenue per user (don't sacrifice revenue for engagement)
- Customer satisfaction scores
- Latency (new model must meet SLA)
- Error rates

**Sample size and duration:**
- Minimum detectable effect: how small an improvement do you need to detect?
- Smaller effect = more samples needed
- Rule of thumb: 2 weeks minimum to capture weekly seasonality
- Use a power calculator: input MDE, baseline rate, significance level -> required sample size

---

## Common Pitfalls

1. **Using AUC-ROC as the primary metric for imbalanced classification.** AUC-ROC is inflated by the large true negative pool in imbalanced problems (fraud, anomaly detection, rare disease). A model can achieve AUC-ROC = 0.98 while having terrible precision. Always use AUC-PR for imbalanced data, which focuses on positive class performance and is not inflated by true negatives.

2. **Optimizing a threshold on the test set.** The classification threshold should be tuned on the validation set using business cost analysis (cost of FP vs cost of FN), not on the test set. Tuning the threshold on the test set is a form of leakage -- your reported performance at that threshold is optimistically biased. The test set should only be used once for final reporting.

3. **Ignoring calibration when using predicted probabilities for decisions.** A model with excellent AUC (ranking) can have terrible calibration (absolute probability values). If you use predicted probabilities to make downstream decisions ("send retention email if churn probability > 0.3"), uncalibrated probabilities lead to incorrect decision thresholds. Always check calibration with a reliability diagram and apply Platt scaling or isotonic regression if needed.

4. **Reporting only offline metrics without A/B testing.** Offline metrics measure performance on static historical data. User behavior is dynamic -- higher NDCG does not guarantee higher click-through rate. Position bias, feedback loops, and distribution shift can all cause offline/online discrepancies. The A/B test is the ultimate source of truth for production impact.

---

## Hands-On Exercises

### Exercise 1: Metric Selection Under Imbalance (20 min)

Using scikit-learn's `make_classification` with class imbalance:

1. Generate a dataset with 10,000 samples, 20 features, and 2% positive class: `make_classification(n_samples=10000, weights=[0.98, 0.02], n_informative=5)`.
2. Train a logistic regression and compute: accuracy, precision, recall, F1, AUC-ROC, and AUC-PR.
3. Plot the ROC curve and PR curve side by side. Which curve reveals the model's true difficulty with the rare class?
4. Vary the classification threshold from 0.1 to 0.9 (in steps of 0.1) and plot precision vs recall at each threshold. Identify the threshold that maximizes F2 (recall-weighted).

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             f1_score, fbeta_score, classification_report)
import matplotlib.pyplot as plt
```

### Exercise 2: Cross-Validation Strategy Comparison (15 min)

Using a time-stamped dataset (create synthetic data with a time trend):

1. Generate 1,000 samples with a feature that trends upward over time and a target correlated with the trend.
2. Evaluate a model using: (a) random 5-fold CV, (b) time-series split (5 folds), and (c) stratified 5-fold CV.
3. Compare the mean and standard deviation of scores across strategies.
4. Why does random CV produce a higher (and misleading) score? Document the temporal leakage.

```python
import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold, TimeSeriesSplit,
                                      cross_val_score)
from sklearn.ensemble import GradientBoostingClassifier
```

---

## 18. Interview Questions with Detailed Answers

**Q: You're building a fraud detection model for a payment platform. What metric do you optimize?**
A: I'd optimize for recall (catching as much fraud as possible) while maintaining a minimum acceptable precision (so the review team isn't overwhelmed). Specifically, I'd use the PR curve to find the threshold where recall is > 95% and precision is still workable (maybe > 30%). I'd report AUC-PR as the model comparison metric, not AUC-ROC, because fraud is rare (< 1%) and ROC would be misleadingly optimistic. I'd also monitor the F2 score (recall-weighted F-score) as a single summary metric.

**Q: Your recommendation model improved NDCG@10 by 5% offline but showed no lift in A/B test. What happened?**
A: Several possibilities. (1) The offline test set doesn't capture how users actually interact — maybe users want diversity and the new model is more accurate but less diverse. (2) Position bias in the offline data — users click what's shown first, so historical data is biased toward the old model's rankings. (3) The improvement is within A/B test noise — need to check statistical power and run longer. (4) Latency regression — the new model might be slower, causing users to disengage before seeing results. I'd investigate by checking diversity metrics, running the A/B test longer with more power, and monitoring guardrail metrics.

**Q: Explain the difference between AUC-ROC and AUC-PR.**
A: Both measure the model's ranking ability across thresholds, but they differ in what they're sensitive to. AUC-ROC plots TPR vs FPR — it benefits from large numbers of true negatives, which inflate the score for imbalanced data. AUC-PR plots precision vs recall — it focuses entirely on positive class performance. For a fraud detection model at 0.5% prevalence, AUC-ROC might be 0.98 while AUC-PR is 0.35, revealing that the model actually struggles to find fraud without many false positives. Use AUC-PR for any imbalanced classification problem.

**Q: When would you use RMSE vs MAE?**
A: RMSE when large errors are disproportionately costly — being 4 hours late on a delivery is more than twice as bad as being 2 hours late. MAE when all errors are equally bad per unit — a $10K pricing error on a $50K item and a $500K item are equally costly in absolute terms. Report both. If RMSE is much larger than MAE, you have outlier predictions that need investigation.

**Q: How do you detect data leakage?**
A: Three signals. (1) Implausibly high performance — if your model is 99.5% accurate on a hard problem, investigate before celebrating. (2) Feature importance anomalies — if one feature dominates importance and it's derived from the target or from future data, that's leakage. (3) Performance gap between random split and temporal split — if the model scores much higher with random CV than time-series CV, temporal information is leaking. Prevention: for every feature, ask "would I have this at prediction time in production?" Fit all preprocessing on training data only.

**Q: How do you evaluate a recommendation system end-to-end?**
A: Offline: measure Recall@K for the retrieval stage (are we surfacing the right candidates?) and NDCG@K for the ranking stage (are we ordering them correctly?). Check diversity metrics (intra-list diversity, coverage of the catalog). Online: A/B test against the current system measuring CTR, add-to-cart rate, revenue per user, and user retention. Set guardrail metrics: latency p99, coverage (% of users receiving recommendations), and novelty (are we showing items users haven't seen?). Monitor for feedback loops — if the model only recommends popular items, they become more popular, creating a winner-take-all dynamic that hurts catalog coverage.

---

## Summary

This lesson covered the evaluation metrics and methodology that determine whether an ML model actually works:

- **Classification metrics:** Accuracy is misleading for imbalanced data. Precision measures trust in positive predictions; recall measures coverage of actual positives. The F1 score balances both, and F-beta generalizes for asymmetric costs.
- **Threshold-independent metrics:** AUC-ROC measures ranking quality across all thresholds but is inflated for imbalanced data. AUC-PR focuses on positive class performance and is the honest metric for rare-event problems.
- **Regression metrics:** RMSE penalizes large errors; MAE is robust to outliers. MAPE provides scale-independent business-friendly reporting. R-squared communicates explained variance.
- **Ranking metrics:** NDCG accounts for position and graded relevance (the standard for search and recommendations). MAP, Recall@K, Precision@K, and MRR each serve specific evaluation needs at different pipeline stages.
- **Calibration** ensures predicted probabilities are meaningful for downstream decisions, not just correctly ranked.
- **Cross-validation** (stratified, time-series, group) produces reliable performance estimates. The train/validation/test split protocol prevents overfitting to the evaluation data.
- **Data leakage** (temporal, target, train/test) is the silent killer of ML projects -- always ask "would I have this at prediction time?"
- **Online vs offline evaluation:** A/B testing is the ultimate source of truth. Offline metrics are necessary but not sufficient.

## What's Next

- **Supervised Learning** — revisit algorithm details with a deeper understanding of which metrics to use for each model type and problem setting (see [Supervised Learning](../supervised/COURSE.md))
- **Feature Engineering** — the features you build directly determine the metrics you achieve; understanding evaluation helps you prioritize which features to invest in (see [Feature Engineering](../feature-engineering/COURSE.md))
