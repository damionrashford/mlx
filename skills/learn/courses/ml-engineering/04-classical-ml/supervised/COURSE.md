# Supervised Learning: The Complete Toolkit

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How each major supervised algorithm (linear regression, logistic regression, decision trees, random forests, gradient boosting, SVMs, KNN, Naive Bayes) works and what assumptions it makes
- The bias-variance tradeoff and how bagging and boosting address it differently
- Regularization techniques (L1, L2, ElasticNet) and their effects on model behavior

**Apply:**
- Select the right algorithm for a given problem based on data size, interpretability needs, latency requirements, and feature characteristics
- Tune key hyperparameters for tree-based ensembles (Random Forest, XGBoost/LightGBM) and interpret feature importance using SHAP

**Analyze:**
- Evaluate tradeoffs between model complexity, interpretability, and production constraints when designing an ML system for a real business problem

## Prerequisites

- **Probability and Statistics** — understanding distributions, Bayes' theorem, and maximum likelihood is essential for grasping logistic regression, Naive Bayes, and evaluation of model performance (see [Probability & Statistics](../01-foundations/probability-statistics/COURSE.md))
- **Optimization** — gradient descent and loss function minimization underpin how most supervised models learn their parameters (see [Optimization](../01-foundations/optimization/COURSE.md))

## Why This Matters

Every production ML system starts here. Recommendation engines, churn models, fraud detection, search ranking — they're all supervised learning. XGBoost is commonly used for tabular prediction tasks like sports outcomes or churn modeling. This lesson gives you the full landscape so you can pick the right tool for any problem and defend that choice in an interview.

Supervised learning = you have labeled data (input X, known output y). The model learns the mapping f(X) -> y. Everything else is details about how it learns that mapping and what assumptions it makes.

---

## 1. Linear Regression

**What it is:** Find the best-fit hyperplane through your data by minimizing the sum of squared residuals.

```
y = w0 + w1*x1 + w2*x2 + ... + wn*xn

Loss = (1/n) * sum((y_actual - y_predicted)^2)
```

This has a closed-form solution (Normal Equation: w = (X^T X)^-1 X^T y) or can be solved with gradient descent. For small datasets, closed-form is instant. For large datasets or online learning, gradient descent scales better.

**When to use it:**
- Continuous target variable (price, revenue, time-on-site)
- You need interpretability ("each additional product view increases conversion by 0.3%")
- Baseline model before trying anything complex
- Features have roughly linear relationship with target

**Assumptions (and what breaks them):**
1. **Linearity** — y is a linear combination of features. Violated? Add polynomial features or switch to tree-based models.
2. **Independence** — observations don't influence each other. Violated in time series data — use autoregressive models.
3. **Homoscedasticity** — constant variance of errors across predictions. Violated? Use weighted least squares or log-transform y.
4. **No multicollinearity** — features shouldn't be highly correlated with each other. If they are, coefficients become unstable and uninterpretable. Fix: drop correlated features, use PCA, or add regularization.
5. **Normal residuals** — errors should be normally distributed. Less critical for prediction accuracy, critical for statistical inference (confidence intervals, p-values).

**Limitations:**
- Can't capture non-linear relationships without manual feature engineering
- Sensitive to outliers (squared loss amplifies them — consider Huber loss for robustness)
- Assumes all features contribute additively (no interactions unless you create them)

**Sports prediction context:** Linear regression would be a poor choice for game outcome prediction because the relationships between stats and wins are non-linear and interactive. A team's win probability doesn't increase linearly with each additional point of net rating — there are diminishing returns and threshold effects that XGBoost captures automatically.

---

## 2. Logistic Regression

**What it is:** Linear regression's output pushed through a sigmoid function to produce a probability between 0 and 1.

```
P(y=1|X) = sigmoid(w*X + b) = 1 / (1 + exp(-(w*X + b)))
```

The model learns weights by minimizing log loss (binary cross-entropy), not squared error:
```
Loss = -sum[y*log(p) + (1-y)*log(1-p)]
```

This penalizes confident wrong predictions exponentially. Predicting 0.01 for a true positive is punished far more than predicting 0.4.

**Why it's a strong baseline:**
- Outputs calibrated probabilities (not just classes) — you can threshold at any point
- Fast to train (seconds on millions of rows), fast to predict (microseconds)
- Coefficients are directly interpretable as log-odds
- Regularization is built in (C parameter in sklearn controls inverse regularization strength)
- Surprisingly hard to beat on well-engineered features — many production systems at top companies run logistic regression

**Decision boundary:** Logistic regression draws a linear boundary in feature space. The sigmoid determines confidence as you move away from that boundary. In 2D it's a line, in nD it's a hyperplane.

**Multi-class extensions:**
- **One-vs-Rest (OvR)**: train K binary classifiers, pick the most confident. Simple, parallelizable.
- **Softmax (Multinomial)**: generalize sigmoid to K classes. Outputs K probabilities that sum to 1. More principled but slower.

**When to use:**
- Binary classification baseline (churn/no-churn, click/no-click, fraud/legit)
- You need probability estimates, not just class predictions
- You need to explain individual predictions to stakeholders
- Real-time serving with sub-millisecond latency requirements

**Production insight:** For merchant churn prediction, logistic regression is always your first model. It trains in seconds, gives you interpretable coefficients ("merchants who haven't logged in for 14+ days are 3x more likely to churn"), and establishes the baseline that any complex model must beat. If logistic regression gets AUC 0.85, the business case for a complex model needs to justify the added infrastructure cost.

---

## 3. Decision Trees

**How splits work:**

A decision tree recursively partitions the feature space by asking binary questions. At each node, it picks the feature and threshold that best separates the target.

**Information Gain (Entropy-based):**
```
Entropy(S) = -sum(p_i * log2(p_i))  for each class i
Information Gain = Entropy(parent) - weighted_avg(Entropy(children))
```
Pure nodes (all one class) have entropy 0. Maximum uncertainty (50/50) has entropy 1. Pick the split that maximizes information gain.

**Gini Impurity:**
```
Gini(S) = 1 - sum(p_i^2)  for each class i
```
Measures the probability of misclassifying a randomly chosen element. Slightly faster to compute than entropy, nearly identical results in practice. sklearn uses Gini by default. For regression trees, minimize variance (MSE) in child nodes.

**Interpretability vs Performance:**
- Single decision trees are the most interpretable ML model. You can draw the decision process and show a non-technical stakeholder exactly why a prediction was made.
- But they overfit badly. A deep tree memorizes the training data — one leaf per sample, 100% training accuracy, terrible generalization.
- They have high variance — small changes in data produce very different trees.
- This is exactly why ensembles were invented.

**Controls against overfitting:**
- `max_depth`: limit tree depth (try 3-10)
- `min_samples_leaf`: require at least N samples in each leaf
- `min_samples_split`: require at least N samples to attempt a split
- Post-pruning: grow the full tree, then remove branches that don't improve validation performance

**When to use a single tree:**
- You need a fully explainable model (regulatory requirements)
- Quick data exploration to understand feature interactions
- Teaching or debugging — never in production for accuracy-critical tasks

---

### Check Your Understanding

1. Why does logistic regression use log loss (binary cross-entropy) instead of mean squared error as its loss function?
2. You are building a model to predict continuous revenue for merchants. Your features include several highly correlated advertising metrics. Which supervised algorithm would you start with and why? What specific technique would you use to handle the correlated features?
3. A colleague claims that a single decision tree with max_depth=50 on 10,000 samples is a strong production model because it achieves 99% training accuracy. What is wrong with this reasoning?

<details>
<summary>Answers</summary>

1. Log loss is convex with respect to the logistic regression parameters, ensuring a single global minimum that gradient descent can reliably find. MSE applied to sigmoid outputs creates a non-convex surface with many local minima. Additionally, log loss penalizes confident wrong predictions exponentially (predicting 0.01 for a true positive is punished far more than predicting 0.4), which produces well-calibrated probability estimates.

2. Start with linear regression with L2 (Ridge) regularization. Ridge handles multicollinearity well by distributing weight among correlated features rather than picking one arbitrarily (which OLS or Lasso might do). If you suspect many of the correlated features are irrelevant, use ElasticNet to get both the sparsity of L1 and the stability of L2.

3. A tree with depth 50 on 10,000 samples has massively overfit. It has essentially memorized the training data (one leaf per sample), creating high variance. Small changes in the data would produce a very different tree. The 99% training accuracy is meaningless without validation performance. This is exactly the problem ensembles (Random Forest, XGBoost) solve by combining many trees to reduce variance.

</details>

---

## 4. Random Forests

**The core idea: Bagging (Bootstrap Aggregating)**

1. Create N bootstrap samples (random samples with replacement) from training data — each uses ~63.2% of unique observations
2. Train a decision tree on each sample, growing it deep (low bias)
3. At each split, only consider a random subset of features (sqrt(n_features) for classification, n_features/3 for regression)
4. Average predictions (regression) or majority vote (classification)

**Why ensembles beat single trees:**
- Each individual tree overfits to its bootstrap sample in a different way
- Averaging many high-variance, low-bias trees reduces overall variance without increasing bias
- The random feature selection decorrelates the trees — this is critical. If all trees use the same dominant feature for the first split, they make correlated errors. Random feature subsets force diversity.
- Mathematically: `Var(average) = Var(single)/n` when errors are uncorrelated

**Out-of-Bag (OOB) Error:**
Each bootstrap sample uses ~63.2% of data. The remaining ~36.8% (out-of-bag) is a free validation set. Each observation's OOB prediction uses only trees that didn't train on it. OOB error closely approximates cross-validation error — without the computational cost of retraining.

**Hyperparameters that matter:**
- `n_estimators`: More trees = better, with diminishing returns. 100-500 typical. No overfitting from too many trees (unlike boosting).
- `max_depth`: None (grow fully) often works well when you have enough trees. More trees compensate for individual tree overfitting.
- `max_features`: sqrt(n_features) for classification, n_features/3 for regression. Controls decorrelation between trees.
- `min_samples_leaf`: Higher = more regularization per tree. Useful for noisy data.

**When to use:**
- Tabular data where you want good performance with minimal tuning
- You need feature importance rankings quickly
- Data has non-linear relationships and interactions
- You want a model that's hard to badly misconfigure — the "safe choice"

---

## 5. Gradient Boosting: XGBoost, LightGBM, CatBoost

XGBoost is commonly used for tabular prediction tasks like sports outcomes. Here's the deeper understanding you need for interviews.

**Sequential ensemble (vs Random Forest's parallel ensemble):**

```
F_0(x) = initial prediction (e.g., mean of target)
F_1(x) = F_0(x) + eta * h_1(x)    where h_1 fits the residuals of F_0
F_2(x) = F_1(x) + eta * h_2(x)    where h_2 fits the residuals of F_1
...
F_M(x) = F_{M-1}(x) + eta * h_M(x)
```

Each h_t is a shallow tree (depth 3-8). eta (learning rate) shrinks each tree's contribution — smaller eta = more trees needed but better generalization. This is gradient descent in function space: you're minimizing a loss function by iteratively adding functions (trees) that point in the direction of steepest descent.

**Why it dominates tabular data:**
- Boosting reduces both bias AND variance (bagging only reduces variance)
- Shallow trees as weak learners limit individual tree complexity
- Rich regularization: learning rate, tree depth, L1/L2 on leaf weights, row/column subsampling
- Handles missing values natively (learns optimal split direction for NaN)
- Feature interactions are learned automatically, not manually engineered
- Has won the vast majority of Kaggle tabular competitions

**XGBoost vs LightGBM vs CatBoost:**

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Tree growth | Level-wise (balanced) | Leaf-wise (faster, deeper) | Symmetric (oblivious) trees |
| Training speed | Fast | Fastest (often 2-5x XGBoost) | Slower training, faster inference |
| Categorical handling | Manual encoding required | Built-in (via binning) | Best native handling (ordered target encoding) |
| Overfitting risk | Good regularization | Can overfit faster (leaf-wise) | Strong default regularization |
| GPU support | Yes | Yes | Yes (best GPU impl) |
| Best for | General purpose, competitions | Large datasets, speed-critical | High-cardinality categoricals, least tuning |

**Key hyperparameters:**
- `learning_rate` (eta): 0.01-0.3. Lower = more trees needed but better generalization.
- `max_depth`: 3-8 for boosting. Shallow trees! This differs from random forests.
- `n_estimators`: Set high (5000-10000) and use early stopping. Never hardcode.
- `subsample`: Row sampling per tree (0.7-0.9). Adds stochasticity like dropout.
- `colsample_bytree`: Column sampling per tree (0.7-0.9).
- `reg_alpha` (L1) and `reg_lambda` (L2): Regularization on leaf weights.
- `min_child_weight`: Minimum sum of instance weight in a leaf. Higher = more conservative.

**Key practical insight:** Early stopping on a validation set is the single most important technique. Set n_estimators=10000 and let early_stopping_rounds=50-100 find the right number of trees automatically.

---

## 6. Support Vector Machines (SVMs)

**Maximum margin classifier:**
Find the hyperplane that separates classes with the largest possible margin. The margin is the distance between the hyperplane and the nearest data points from each class. Those nearest points are the "support vectors" — they alone determine the boundary.

**Why maximum margin?**
Maximizing margin = maximizing generalization. A wider margin means the model is less sensitive to small perturbations. Only support vectors matter — adding more data far from the boundary changes nothing. This gives SVMs strong theoretical guarantees (structural risk minimization).

**Soft margin (C parameter):**
Real data isn't perfectly separable. C controls the tradeoff between a wide margin and correctly classifying training points. High C = narrow margin, fewer misclassifications (risk overfitting). Low C = wide margin, more misclassifications (risk underfitting).

**The kernel trick:**
Data not linearly separable in original space might be separable in a higher-dimensional space. The kernel trick computes dot products in that higher-dimensional space without explicitly transforming the data.

- **Linear**: `K(x,z) = x.z` — use when n_features >> n_samples or data is linearly separable
- **RBF (Gaussian)**: `K(x,z) = exp(-gamma * ||x-z||^2)` — maps to infinite dimensions, handles arbitrary non-linear boundaries. The default.
- **Polynomial**: `K(x,z) = (gamma*x.z + r)^d` — rarely used in practice

**When to use SVMs:**
- Small to medium datasets (< 100K samples) — training is O(n^2) to O(n^3)
- High-dimensional data with few samples (text classification with TF-IDF, genomics)
- When you need strong theoretical guarantees

**When NOT to use:**
- Large datasets (doesn't scale)
- When you need probability estimates (Platt scaling is a hack)
- When interpretability matters (kernel SVMs are black boxes)
- When you have lots of data (gradient boosting or neural networks will be better)

---

### Check Your Understanding

1. What is the key difference between how Random Forest and XGBoost build their ensemble of trees? How does this difference affect what each method reduces (bias, variance, or both)?
2. You have a dataset with 50,000 samples, 300 features (many from TF-IDF text features), and need to classify documents into two categories. Which algorithm would you consider and why?
3. A colleague sets `n_estimators=500` for XGBoost without early stopping and reports strong validation performance. Why is this approach risky compared to using early stopping?

<details>
<summary>Answers</summary>

1. Random Forest builds trees independently in parallel on bootstrap samples, using deep trees (low bias, high variance). Averaging these uncorrelated trees reduces variance without increasing bias. XGBoost builds trees sequentially, where each shallow tree corrects the residual errors of the ensemble so far. This sequential correction reduces bias, while the shallow tree depth and regularization (learning rate, subsampling) control variance. Boosting reduces both bias and variance; bagging (Random Forest) primarily reduces variance.

2. SVM with a linear kernel would be a strong choice. With 300 features (many from TF-IDF, creating high-dimensional sparse data) and 50,000 samples, linear SVM excels because: it handles high-dimensional data well, text features are often linearly separable in high dimensions, and it has strong theoretical guarantees via maximum margin. Logistic regression is also an excellent baseline for the same reasons. If you need non-linear boundaries, XGBoost would be the next step.

3. Without early stopping, the fixed 500 trees might be too many (overfitting) or too few (underfitting) for the specific dataset. The correct approach is to set a high n_estimators (e.g., 10,000) and use early_stopping_rounds (e.g., 50-100) on a validation set. This automatically finds the optimal number of trees where validation loss stops improving, preventing overfitting and removing the need to guess the right number.

</details>

---

## 7. K-Nearest Neighbors (KNN)

**How it works:** No training — the model IS the data. To predict for a new point, find the K closest training points and average (regression) or majority vote (classification). It's a "lazy learner."

**Distance metrics:**
- Euclidean: `sqrt(sum((x_i - y_i)^2))` — most common, assumes isotropic feature space
- Manhattan: `sum(|x_i - y_i|)` — more robust in high dimensions, less affected by outliers
- Cosine similarity: `(x.y) / (||x|| * ||y||)` — measures angle, ignores magnitude. Great for text and embeddings.
- Minkowski: generalizes Euclidean and Manhattan

**Curse of dimensionality:**
The fundamental limitation. In high dimensions, all points become roughly equidistant. In 1D with 100 points, your 10 nearest neighbors cover 10% of the feature space. In 100D, covering 10% of the space requires 10^100 points. With a model that has 200 features, KNN would be essentially random without dimensionality reduction first.

**When to use:**
- Small, low-dimensional datasets where you want a quick baseline
- Recommendation systems (item-based collaborative filtering is KNN on item vectors)
- Anomaly detection (points with distant neighbors are outliers)
- After dimensionality reduction (PCA -> KNN can work well)

**When NOT to use:**
- Large datasets (O(n*d) prediction time without approximate nearest neighbor structures)
- High-dimensional data (curse of dimensionality)
- When you need a compact deployable model (KNN stores all training data)

---

## 8. Naive Bayes

**The probabilistic approach:**
```
P(class | features) = P(features | class) * P(class) / P(features)
                    proportional to P(class) * product(P(feature_i | class))
```

The "naive" assumption: all features are conditionally independent given the class. This is almost never true, yet Naive Bayes works because the ranking of probabilities is often correct even when absolute values are wrong.

**Variants:**
- **Gaussian NB**: Assumes features follow normal distributions. Use for continuous features.
- **Multinomial NB**: Assumes features are counts. The standard for text classification (word frequencies).
- **Bernoulli NB**: Assumes binary features. Use for binary occurrence vectors.

**Why it's surprisingly good for text:**
- High-dimensional sparse data benefits from low-variance models
- The independence assumption is wrong (words co-occur) but the class-conditional rankings are preserved
- Extremely fast to train (single pass through data) and predict
- Works well even with small training sets
- Competitive with deep learning for short-text classification tasks

**When to use:**
- Text classification baseline (spam, sentiment, topic classification)
- Very little training data available
- Real-time prediction with minimal latency
- Multi-class problems where you need quick probabilistic predictions

---

## 9. Algorithm Comparison Table

| Algorithm | Best Data Size | Interpretability | Train Speed | Predict Speed | Non-linear | Scaling Needed | Missing Values |
|-----------|---------------|-----------------|-------------|--------------|------------|---------------|----------------|
| Linear Regression | Any | High | Very Fast | Very Fast | No | Yes | No (impute) |
| Logistic Regression | Any | High | Very Fast | Very Fast | No | Yes | No (impute) |
| Decision Tree | Small-Med | Very High | Fast | Very Fast | Yes | No | Some impls |
| Random Forest | Med-Large | Medium | Medium | Medium | Yes | No | Some impls |
| XGBoost/LightGBM | Med-Large | Low-Med | Medium | Fast | Yes | No | Yes (native) |
| SVM (kernel) | Small-Med | Low | Slow | Medium | Yes (kernel) | Yes | No (impute) |
| KNN | Small | Medium | None | Slow | Yes | Yes | No (impute) |
| Naive Bayes | Any | Medium | Very Fast | Very Fast | No | No | Handles zeros |

---

## 10. Decision Framework: When to Use What

**Step 1 — What's your target?**
- Continuous value -> Regression (Linear, RF, XGBoost)
- Binary class -> Classification (Logistic, RF, XGBoost)
- Multiple classes -> Multi-class classification (same algorithms, check if multi-label)
- Ranked list -> Learning to Rank (LambdaMART = XGBoost under the hood)

**Step 2 — How much labeled data do you have?**
- < 1K samples -> Logistic Regression, Naive Bayes, SVM (simpler models generalize better)
- 1K-100K samples -> Random Forest, XGBoost, SVM
- 100K-10M samples -> XGBoost, LightGBM, Neural Networks
- > 10M samples -> LightGBM, Neural Networks, Linear models (scale linearly)

**Step 3 — Do you need interpretability?**
- Full transparency required -> Logistic Regression, Decision Tree, GAMs
- Feature importance sufficient -> RF, XGBoost with SHAP
- Black box acceptable -> Deep ensembles, Neural Networks

**Step 4 — What's your latency budget?**
- < 1ms (real-time serving) -> Logistic Regression, small tree ensemble, distilled model
- 1-10ms -> XGBoost with < 500 trees
- 10-100ms -> Large ensembles, medium neural networks
- > 100ms (batch scoring) -> Anything goes

**E-commerce platform scenarios:**
- Product recommendations: Collaborative filtering retrieval -> XGBoost or two-tower neural net ranker
- Merchant churn: XGBoost with SHAP explanations for account managers
- Fraud detection: XGBoost (handles imbalance, tabular data, fast inference, missing values)
- Search ranking: LambdaMART (learning to rank using gradient boosting)
- Email send-time optimization: Logistic regression (simple, fast, interpretable, easy to A/B test)

---

## 11. Feature Importance: How Different Models Expose It

**Linear models (Linear/Logistic Regression):**
- Coefficient magnitude indicates importance (after feature scaling!)
- Sign indicates direction of relationship
- exp(coefficient) gives odds ratio in logistic regression
- Caveat: correlated features split importance unpredictably

**Tree-based models (Random Forest, XGBoost):**
- **Split-based / Frequency**: how often a feature is used for splitting across all trees
- **Gain importance**: total loss reduction from splits on that feature
- **Cover**: average number of samples passing through splits on that feature
- Caveat: biased toward high-cardinality and continuous features (more possible split points)

**Model-agnostic methods (use these in interviews):**
- **Permutation importance**: randomly shuffle one feature's values, measure accuracy drop. Unbiased but slow. Works for any model.
- **SHAP values**: game-theoretic approach. Assigns each feature a contribution to each individual prediction. Shows direction + magnitude + interactions. The gold standard. Use `shap.TreeExplainer` for XGBoost — it's exact and fast.
- **LIME**: local interpretable model-agnostic explanations. Fits a simple model around each prediction. Good for explaining individual predictions.

**Practical example:** For a sports prediction model, `model.feature_importances_` from XGBoost provides a quick view. In interviews, talk about SHAP instead. SHAP shows not just which features matter globally, but how they affect each individual prediction — "for this specific game, the home team's 5-game rolling net rating contributed +0.12 to the win probability."

---

## 12. Regularization

Regularization penalizes model complexity to prevent overfitting. It adds a penalty term to the loss function.

**L1 Regularization (Lasso):**
```
Loss = MSE + alpha * sum(|w_i|)
```
- Drives some weights exactly to zero -> automatic feature selection
- Produces sparse models (interpretable, fast inference)
- The absolute value creates a diamond-shaped constraint region where corners touch axes -> zeroes
- Use when you suspect many features are irrelevant

**L2 Regularization (Ridge):**
```
Loss = MSE + alpha * sum(w_i^2)
```
- Shrinks all weights toward zero but never exactly to zero
- Handles multicollinearity well (distributes weight among correlated features rather than picking one arbitrarily)
- Almost always improves generalization
- The default choice. Always start here.

**ElasticNet:**
```
Loss = MSE + alpha * (l1_ratio * sum(|w_i|) + (1-l1_ratio) * sum(w_i^2))
```
- Combines L1 and L2
- Gets the sparsity of Lasso with the stability of Ridge
- l1_ratio controls the mix (0 = pure Ridge, 1 = pure Lasso)
- Use when you have groups of correlated features and want selection

**Regularization in tree-based models:**
- `max_depth`: limits tree complexity directly
- `min_samples_leaf`: prevents learning from tiny sample groups
- `learning_rate` in boosting: shrinks each tree's contribution (most important)
- `reg_alpha` (L1) and `reg_lambda` (L2) in XGBoost: regularize leaf weights
- `subsample` and `colsample_bytree`: inject randomness, reduce overfitting

**Practical rule:** Always use some regularization. The question is never "should I regularize?" but "how much?" Use cross-validation to find the optimal strength.

---

### Check Your Understanding

1. You have a dataset with 500 features but suspect only 20-30 are truly relevant. Which regularization approach would you use and why?
2. In the algorithm comparison table, why do linear models and SVMs require feature scaling but tree-based models do not?
3. A model achieves excellent AUC on the training set but poor AUC on the validation set. Name two specific techniques (one from regularization, one from ensemble methods) you would try to close this gap and explain why each helps.

<details>
<summary>Answers</summary>

1. Use ElasticNet (or Lasso/L1 regularization). L1 regularization drives irrelevant feature coefficients exactly to zero, performing automatic feature selection. ElasticNet is preferred over pure Lasso when features may be correlated, because it combines L1's sparsity with L2's stability for correlated feature groups. You could also use L1 as a feature selection step and then train any model on the surviving features.

2. Linear models compute weighted sums of features, so a feature with range 0-1,000,000 dominates one with range 0-1 in the gradient updates and final prediction. SVMs compute distances, so magnitude differences distort the distance metric. KNN has the same issue. Tree-based models split on feature thresholds by testing "is feature X > value?" — they only care about the ordering of values, not their magnitude, so scaling has no effect on splits.

3. The large train-validation gap indicates overfitting (high variance). (1) Regularization: increase regularization strength (e.g., increase alpha for Ridge/Lasso, decrease C for logistic regression, increase reg_lambda for XGBoost). This trades slightly higher bias for substantially lower variance. (2) Ensemble methods: use Random Forest or bagging, which averages many high-variance models to reduce overall variance. If already using boosting, reduce learning rate and max_depth to constrain individual tree complexity.

</details>

---

## 13. Practical: Merchant Churn Prediction

**Problem:** Predict which merchants will cancel their platform subscription in the next 30 days.

**Features you'd engineer:**
- Days since last login (engagement signal)
- Revenue trend (MoM growth rate over last 3 months)
- Number of products listed (investment signal)
- Number of orders in last 30 days (business health)
- Support ticket count (frustration signal)
- App installs/uninstalls (platform engagement)
- Theme changes (customization effort)
- Payment failures (billing friction)
- Plan type and tenure (commitment level)
- Feature adoption rate (how many platform features they use)

**Model selection walkthrough:**

1. **Start with Logistic Regression.** Trains in seconds. Gives interpretable coefficients. Establishes baseline — if this gets AUC 0.82, anything fancier needs to demonstrably beat it.

2. **Try XGBoost/LightGBM.** Handles non-linear relationships (churn risk spikes after 3 missed logins, not linearly). Handles missing values. Likely gets AUC 0.87-0.92.

3. **Add SHAP explanations.** For each merchant predicted to churn, surface the top 3 reasons to the account manager: "This merchant hasn't logged in for 21 days, revenue dropped 40% MoM, and they uninstalled 2 apps last week."

4. **Calibrate probabilities.** XGBoost raw probabilities are often poorly calibrated. Use isotonic regression or Platt scaling so that "70% churn probability" actually means 70% of those merchants churn. This matters for downstream decision-making.

5. **Set threshold based on business costs.** If losing a merchant costs $10K/year in recurring revenue and a retention intervention costs $50, set a low threshold — intervene aggressively. The optimal threshold is where: `P(churn) * cost_of_losing_merchant > cost_of_intervention`.

---

## Common Pitfalls

1. **Fitting preprocessing on the full dataset before splitting.** If you fit a StandardScaler or compute target encodings on the entire dataset (including test data), you leak information from the test set into training. Always fit transformers on the training fold only, then apply to validation/test. This applies to scaling, imputation, encoding, and feature selection.

2. **Using accuracy as the primary metric for imbalanced problems.** A model that predicts the majority class for everything can score 99%+ accuracy on a fraud or churn dataset. Always use metrics appropriate to the class distribution: AUC-PR, F1/F2, or recall at a fixed precision. See the [Evaluation Metrics](../evaluation-metrics/COURSE.md) lesson for details.

3. **Hardcoding n_estimators for gradient boosting instead of using early stopping.** Picking a fixed number of trees is guesswork. Set n_estimators high (5,000-10,000) and let early_stopping_rounds find the optimal number automatically on a validation set. This is the single most important tuning technique for XGBoost/LightGBM.

4. **Ignoring multicollinearity in linear models.** When features are highly correlated, linear regression coefficients become unstable and uninterpretable — small data changes produce wildly different coefficients. The model may still predict well, but you cannot trust individual coefficient values. Use Ridge regularization, drop redundant features, or switch to tree-based models if interpretability of individual features matters.

---

## Hands-On Exercises

### Exercise 1: Algorithm Showdown on Tabular Data (20 min)

Using the scikit-learn `breast_cancer` dataset, compare Logistic Regression, Random Forest, and XGBoost:

1. Load the data and split into train/test (80/20, stratified).
2. Train all three models with default hyperparameters.
3. Report accuracy, precision, recall, and F1 for each.
4. For XGBoost, retrain with early stopping (set `n_estimators=1000`, `early_stopping_rounds=20`, use 20% of training data as eval set). Compare the number of trees used vs the default.
5. Use `sklearn.inspection.permutation_importance` on the test set for Random Forest. Which top 3 features drive predictions?

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
```

### Exercise 2: Regularization Effects on Linear Models (15 min)

Using the scikit-learn `diabetes` dataset (regression):

1. Fit Linear Regression, Ridge (alpha=1.0), Lasso (alpha=1.0), and ElasticNet (alpha=1.0, l1_ratio=0.5).
2. Compare the number of non-zero coefficients for each model.
3. Plot the coefficient values across all four models as a bar chart.
4. Which features does Lasso eliminate? Do they make sense given their correlation to the target?

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import numpy as np
```

---

## 14. Interview Questions with Answers

**Q: What's the bias-variance tradeoff?**
A: Bias is error from wrong assumptions (model too simple, underfits — misses the signal). Variance is error from sensitivity to training data (model too complex, overfits — fits the noise). Total error = bias^2 + variance + irreducible noise. Simple models have high bias, low variance. Complex models have low bias, high variance. The sweet spot minimizes total error. Bagging (Random Forest) reduces variance. Boosting reduces both bias and variance. Regularization trades slightly higher bias for substantially lower variance.

**Q: Why does XGBoost dominate tabular data benchmarks?**
A: Three reasons. (1) Boosting reduces both bias and variance, unlike bagging which only reduces variance. (2) Rich regularization prevents overfitting: learning rate, tree depth, L1/L2 on leaf weights, row and column subsampling. (3) It handles the messy realities of tabular data natively — missing values, mixed feature types, non-linear relationships, feature interactions — without the careful preprocessing that linear models or neural networks require. Neural nets struggle on tabular data because they need more data and careful architecture design to match what boosted trees do by default.

**Q: When would you use logistic regression over XGBoost?**
A: When interpretability matters more than marginal accuracy (regulatory settings, stakeholder communication). When you need sub-millisecond prediction latency at scale. When you have very little training data (simpler models generalize better with few samples). When you need a fast baseline before investing in complex model development. In practice, many production ML systems at top companies are logistic regression with strong feature engineering.

**Q: Explain bagging vs boosting.**
A: Bagging (Random Forest) trains many models independently in parallel on different bootstrap samples, then averages. Each model is a full-depth tree (low bias, high variance). Averaging reduces variance. Boosting (XGBoost) trains models sequentially, where each model corrects the errors of previous ones. Each model is a shallow tree (higher bias, lower variance individually). The sequential correction reduces bias. Boosting typically achieves better accuracy but is harder to tune and can overfit.

**Q: How do you handle class imbalance?**
A: In priority order: (1) Use appropriate metrics — AUC-PR, not accuracy. (2) Adjust class weights (`scale_pos_weight` in XGBoost, `class_weight='balanced'` in sklearn). (3) Threshold tuning — train normally, optimize the classification threshold on validation data for your business objective. (4) Oversampling (SMOTE) or undersampling — use with caution, can introduce artifacts and doesn't always help. (5) For extreme imbalance (< 0.1%), reframe as anomaly detection with Isolation Forest or One-Class SVM.

**Q: Walk me through building a product recommendation system for an e-commerce platform.**
A: I'd use a retrieval-ranking architecture. Retrieval stage: collaborative filtering (matrix factorization or item-based nearest neighbors) generates 100-500 candidates from millions of products. This must be fast (< 50ms). Ranking stage: XGBoost or a two-tower neural network scores those candidates using rich features — user browsing history, product metadata, contextual signals (time, device, session behavior), and collaborative signals (users-who-bought-this-also-bought). Evaluate with NDCG@K and recall@K offline, then A/B test click-through rate online. Handle cold-start for new products using content-based features (category, description embeddings, price range). Start simple (popularity -> collaborative filtering -> learned ranker) and validate each step lifts metrics before adding complexity.

**Q: What's the kernel trick in SVMs?**
A: SVMs only need dot products between data points, not the coordinates themselves. The kernel function K(x,z) computes the dot product in a (potentially infinite-dimensional) transformed space without computing the transformation explicitly. The RBF kernel maps to infinite dimensions, letting the SVM find arbitrarily complex non-linear boundaries while keeping computation O(n^2) in the number of support vectors. This matters because explicit transformation to high dimensions would be computationally impossible.

**Q: When would you NOT use a neural network for tabular data?**
A: Most of the time. Recent benchmarks (Grinsztajn et al., 2022) show tree-based methods outperform neural networks on typical tabular datasets. Trees handle heterogeneous features (mix of categorical and numerical), missing values, and irregular target functions better. Neural nets need more data, more tuning, more engineering (feature normalization, architecture search, learning rate schedules), and provide less interpretability. I'd reach for neural nets on tabular data only when I have millions of rows, complex feature interactions that benefit from learned representations, or when I need to jointly learn from tabular + unstructured data (text, images).

---

## Summary

This lesson covered the core supervised learning algorithms that form the backbone of production ML:

- **Linear and logistic regression** provide fast, interpretable baselines with strong regularization options (L1/L2/ElasticNet).
- **Decision trees** are the most interpretable model but overfit badly; they serve as the foundation for ensemble methods.
- **Random Forests** (bagging) reduce variance by averaging many decorrelated deep trees, providing reliable performance with minimal tuning.
- **Gradient boosting** (XGBoost, LightGBM, CatBoost) reduces both bias and variance through sequential correction, dominating tabular data benchmarks.
- **SVMs** offer strong theoretical guarantees for small-to-medium datasets, especially in high-dimensional spaces.
- **KNN** and **Naive Bayes** serve as effective baselines for specific problem types (low-dimensional data and text classification, respectively).

The decision framework for algorithm selection considers target type, data size, interpretability needs, and latency budget. In practice, most tabular ML problems are best served by starting with logistic regression (baseline) and XGBoost (performance ceiling), using SHAP for feature importance and early stopping for regularization.

## What's Next

- **Unsupervised Learning** — clustering, dimensionality reduction, and anomaly detection techniques that complement supervised methods (see [Unsupervised Learning](../unsupervised/COURSE.md))
- **Feature Engineering** — the techniques for transforming raw data into the features that make these algorithms perform well (see [Feature Engineering](../feature-engineering/COURSE.md))
- **Evaluation Metrics** — how to measure model performance correctly, including the precision-recall tradeoff, ranking metrics, and calibration (see [Evaluation Metrics](../evaluation-metrics/COURSE.md))
