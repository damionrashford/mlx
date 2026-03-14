# Feature Engineering: The Biggest Lever in ML

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- Why feature engineering typically yields 5-20% model improvement compared to 1-3% from algorithm selection
- How to handle numerical, categorical, text, and time-series features with appropriate transformations
- The types of data leakage (target, temporal, train/test) and how feature stores prevent training/serving skew

**Apply:**
- Engineer effective features for tabular ML problems including scaling, encoding, interaction features, and rolling window statistics
- Perform feature selection using filter methods, SHAP values, and permutation importance to reduce a large feature set for production

**Analyze:**
- Decide when to hand-engineer features versus let the model learn representations, based on data type, dataset size, and production constraints

## Prerequisites

- **Supervised Learning** — understanding how different model types (linear, tree-based, neural) consume features is essential for choosing the right transformation (e.g., scaling for linear models, no scaling for trees) (see [Supervised Learning](../supervised/COURSE.md))
- **Probability and Statistics** — distributional knowledge (skewness, normality, variance) guides when to apply log transforms, power transforms, and statistical imputation methods (see [Probability & Statistics](../../01-foundations/probability-statistics/COURSE.md))

## Why This Matters

Feature engineering is transforming raw data into inputs that make ML models work better. It is consistently the single largest lever for model performance in applied ML.

> "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering." — Andrew Ng

In practice, a well-engineered sports prediction model might have 200 engineered features feeding XGBoost, and that feature work is why the model performs well. This module systematizes that knowledge, covers production concerns like feature stores and leakage, and prepares you for interview questions about building features for recommendation systems, churn prediction, and search ranking.

---

## 1. What Features Are

A feature is a measurable property of the phenomenon you're modeling. Features are the columns in your dataset, the input signals to your model. The quality of your features determines the ceiling of your model's performance — no algorithm can extract signal that isn't in the data.

### Why Features Matter More Than Models

Consider two scenarios:
- **Scenario A**: Logistic regression with brilliant features -> 92% AUC
- **Scenario B**: Complex neural network with raw data -> 85% AUC

Scenario A wins, and it's cheaper to deploy, faster to serve, easier to debug. Practitioners consistently find:
1. A simple model with great features beats a complex model with raw features
2. Feature engineering yields 5-20% improvement; model selection yields 1-3%
3. Features encode domain knowledge that models can't learn from limited data alone

### The Feature Engineering Mindset
For any prediction problem, ask: "What information would a domain expert use to make this prediction?"

For a sports game prediction model, a basketball analyst would consider:
- Recent team performance (rolling averages of stats)
- Home/away advantage
- Rest days between games (fatigue effects)
- Head-to-head history (matchup dynamics)
- Injury impact on team performance
- Schedule strength and travel distance

Each insight becomes an engineered feature. XGBoost can't discover "days since last game" from box score data alone — you compute and provide it. The model handles the complex interactions between these features.

---

## 2. Numerical Features

### Scaling

**Why scale?** Many algorithms (linear regression, logistic regression, SVM, KNN, neural networks) are sensitive to feature magnitude. A feature ranging 0-1,000,000 will dominate a feature ranging 0-1 in distance calculations and gradient updates.

**StandardScaler (Z-score normalization):**
```
x_scaled = (x - mean) / std_dev
```
- Result: mean=0, std=1
- Use when features are roughly normally distributed
- Most common default choice for linear models and neural networks

**MinMaxScaler:**
```
x_scaled = (x - x_min) / (x_max - x_min)
```
- Result: values in [0, 1]
- Use when you need bounded values (neural networks with sigmoid/tanh output)
- Sensitive to outliers — a single extreme value stretches the range

**RobustScaler:**
```
x_scaled = (x - median) / IQR
```
- Uses median and interquartile range instead of mean and std
- Resistant to outliers
- Use when your data has outliers that would distort StandardScaler

**Critical rule:** Tree-based models (XGBoost, Random Forest, Decision Trees) don't need scaling. They only care about feature ordering (which value is bigger), not magnitude. This is one reason they're so practical — no preprocessing step to leak information or maintain in production.

**Critical rule:** Always fit scalers on training data only, then transform test data. Fitting on the full dataset before splitting is data leakage.

### Log Transforms

For right-skewed distributions (revenue, prices, counts, populations):
```
x_log = log(x + 1)    # +1 to handle zeros
```

**Why:** Compresses the range, makes the distribution more symmetric, reduces the influence of extreme values. Many real-world quantities are log-normally distributed.

**E-commerce example:** Merchant GMV ranges from $0 to $100M+. Without log transform, a linear model's coefficients are dominated by the handful of massive merchants. Log-transform makes the distribution roughly normal and gives the model signal from the full range.

**When to use:** Revenue, prices, counts, population, anything that spans multiple orders of magnitude. When in doubt, check the distribution — if it's heavily right-skewed, log-transform helps most models.

### Binning (Discretization)

Convert continuous values into categorical bins:
```
account_age_days: [0-30] -> "new", [31-180] -> "growing", [181-365] -> "established", [365+] -> "mature"
```

**When to use:**
- The relationship between feature and target has step-wise jumps (not smooth)
- You want to capture non-linear effects in linear models
- Domain knowledge suggests meaningful thresholds ("merchants under 30 days have 3x churn rate")
- You want the model to treat similar values identically (noise reduction)

**When NOT to use:** Tree-based models handle non-linearity natively — binning discards information that XGBoost can use. Don't bin for tree models unless you have a strong domain reason.

### Power Transforms
- **Box-Cox**: `(x^lambda - 1) / lambda` — finds optimal lambda to make data approximately normal. Requires x > 0.
- **Yeo-Johnson**: extends Box-Cox to handle zero and negative values.
- Use when your model assumes normally distributed features (linear regression, LDA). sklearn provides `PowerTransformer`.

---

## 3. Categorical Features

### One-Hot Encoding
Convert each category into a binary column:
```
color: [red, blue, green] ->
  color_red:   [1, 0, 0]
  color_blue:  [0, 1, 0]
  color_green: [0, 0, 1]
```

**When to use:** Nominal categories (no natural ordering), relatively few unique values (< 50). Required for linear models and neural networks.

**Problems:**
- High cardinality (10,000 product categories) creates 10,000 sparse columns
- Curse of dimensionality — too many features, too few samples per feature
- Doesn't capture similarity between categories ("running_shoes" and "hiking_boots" are unrelated in one-hot)

### Label Encoding
Assign each category an integer:
```
color: red -> 0, blue -> 1, green -> 2
```

**When to use:** Ordinal categories (low/medium/high, small/medium/large) or tree-based models that can split on arbitrary integer thresholds.

**Warning for linear models:** Label encoding implies ordering (green > blue > red). Linear models treat this as a real numerical relationship. Only use for genuinely ordinal features. Tree-based models handle it fine because they can split anywhere along the integer range.

### Target Encoding (Mean Encoding)
Replace each category with the mean of the target variable for that category:
```
city: "Toronto" -> mean(target where city="Toronto") = 0.73
city: "Vancouver" -> mean(target where city="Vancouver") = 0.58
```

**Powerful but dangerous:**
- Directly captures the relationship between category and target
- Works for arbitrarily high cardinality
- **Major leakage risk**: you're using the target to create a feature. Solutions:
  - Leave-one-out encoding: for each row, use the mean computed without that row
  - Fold-based encoding: split data into folds, compute encoding for each fold using only other folds
  - CatBoost's ordered target encoding: handles this correctly by design
  - Add smoothing: blend category mean with global mean based on sample size
  ```
  encoded = (n * category_mean + m * global_mean) / (n + m)
  ```
  where n = category count, m = smoothing parameter

### Embeddings (Learned Representations)
For very high-cardinality features (product IDs, user IDs, merchant IDs):
- Learn a dense vector (e.g., 64 dimensions) for each category during model training
- Similar categories get similar vectors (learned from co-occurrence or supervised signal)
- Used in neural network recommender systems (the entire concept of collaborative filtering at scale)
- Can pre-train embeddings and feed them as features to tree models

**E-commerce application:** Each product gets a 128-dim embedding learned from purchase co-occurrence. Products frequently bought together get similar embeddings. Use these embeddings for similarity search, recommendation, and clustering.

### Frequency Encoding
Replace category with its count or frequency:
```
city: "Toronto" -> count("Toronto") / total_rows = 0.12
```
Simple, no leakage risk, captures the prevalence signal. Often surprisingly effective.

### Hashing Trick
Hash categories to a fixed number of buckets:
```
feature = hash(category) % n_buckets
```
Handles arbitrary cardinality with fixed memory. Some hash collisions, but works in practice for large-scale systems. Used in Vowpal Wabbit, online learning, and any system that can't maintain a category vocabulary.

---

### Check Your Understanding

1. You are training a logistic regression model with a "city" feature that has 50 unique values. You use label encoding (Toronto=0, Vancouver=1, Montreal=2, ...). What problem does this create, and what encoding should you use instead?
2. Why is target encoding (mean encoding) a leakage risk, and name two techniques to mitigate it?
3. Your dataset has a "merchant_revenue" feature that ranges from $0 to $50M with a heavy right skew. You plan to use it in a linear regression. What transformation would you apply and why? Would you apply the same transformation for XGBoost?

<details>
<summary>Answers</summary>

1. Label encoding imposes a false ordinal relationship on nominal categories. The logistic regression will interpret the integer values as a numerical scale, treating Montreal (2) as "greater than" Vancouver (1). This creates meaningless linear relationships. Use one-hot encoding instead -- it creates a separate binary column for each city, letting the model learn an independent coefficient for each without implying order. With 50 cities this is still feasible (50 columns).

2. Target encoding replaces each category with the mean of the target for that category, which uses the target variable to create a feature. This is leakage because the feature directly encodes the very thing you are trying to predict. If computed naively, categories with few samples get extreme encoded values that memorize noise. Mitigation techniques: (a) fold-based encoding -- compute the encoding for each fold using only data from other folds (similar to cross-validation), and (b) smoothing -- blend the category mean with the global mean weighted by sample size, so rare categories are pulled toward the global average.

3. Apply a log transform: `log(revenue + 1)`. This compresses the extreme right tail, making the distribution more symmetric and reducing the disproportionate influence of a few very-high-revenue merchants on the linear regression coefficients. For XGBoost, the log transform is not necessary -- tree-based models split on feature ordering (rank), not magnitude, so skewness does not affect them. However, it can still sometimes help by making the splits more evenly distributed along the feature range.

</details>

---

## 4. Text Features

### Bag of Words (BoW)
Count word occurrences in each document. Result: a sparse vector with one dimension per unique word. Loses word order but captures word frequency.

### TF-IDF (Term Frequency-Inverse Document Frequency)
```
TF(term, doc) = count(term in doc) / total_words(doc)
IDF(term) = log(total_docs / docs_containing(term))
TF-IDF = TF * IDF
```

- High for words that are frequent in a document but rare across the corpus
- Low for common words ("the", "is") — they have low IDF
- Produces sparse vectors. Excellent with linear models (logistic regression, SVM).

**E-commerce application:** Representing product descriptions for similarity search, category classification, or duplicate detection.

### Word Embeddings (Word2Vec, GloVe, FastText)
- Dense vectors (100-300 dimensions) where semantically similar words are nearby
- Pre-trained on large corpora (Wikipedia, Common Crawl), transfer to your domain
- Simple document representation: average word embeddings across all words
- FastText handles out-of-vocabulary words using subword information

### Sentence/Document Embeddings (Sentence-BERT, OpenAI Embeddings)
- Dense vector (384-1536 dimensions) for an entire sentence or paragraph
- Captures semantic meaning, not just word overlap ("comfortable shoes" and "cozy footwear" are similar)
- State-of-the-art for text similarity, semantic search, classification
- Pre-compute once, store in a vector database (Pinecone, Weaviate, pgvector), query at serving time

### Practical Approach
```
Quick baseline:     TF-IDF + Logistic Regression (minutes to build, surprisingly strong)
Better:             Pre-trained sentence embeddings + any classifier
Best:               Fine-tuned transformer on your labeled data
Production:         Pre-compute embeddings offline, store and retrieve at serving time
```

---

## 5. Time Features

### Cyclical Encoding
Hours, days of week, and months are cyclical — hour 23 is close to hour 0, but integer encoding treats them as maximally far apart.

**Solution:** Encode as sin/cos pairs:
```
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
```

Now hour 23 and hour 0 are adjacent in the sin/cos space. Apply to: hour of day, day of week (0-6), month (1-12), day of month (1-31).

### Lag Features
Use previous values as predictors — fundamental for time series:
```
sales_lag_1 = sales at time t-1
sales_lag_7 = sales at time t-7 (same day last week)
sales_lag_30 = sales at time t-30
sales_lag_365 = sales at time t-365 (same day last year)
```

**Sports prediction models use these** — team stats from previous N games as features for the next game.

### Rolling Window Statistics
```
rolling_7d_mean = mean(last 7 days)
rolling_7d_std = std(last 7 days)
rolling_30d_mean = mean(last 30 days)
momentum = rolling_7d_mean / rolling_30d_mean  # > 1 means increasing trend
```

Captures trends, volatility, and momentum. The ratio of short-window to long-window averages is especially powerful.

### Date-Derived Features
Extract from timestamps:
- `is_weekend`, `is_holiday`, `is_month_end`, `is_quarter_end`
- `days_since_last_purchase`, `days_until_subscription_renewal`
- `part_of_day`: morning (6-12), afternoon (12-18), evening (18-22), night (22-6)
- `day_of_year`: captures annual seasonality

### Time-Based Aggregations for E-Commerce
- Orders per day/week/month (growth trend)
- Average order value over rolling windows (spending pattern changes)
- Time between orders (purchase frequency, recency)
- Session count and duration over time (engagement trend)
- Feature adoption timeline (when they first used X feature)

**Leakage warning:** All time features must only use data from before the prediction point. Rolling windows must not include the current or future data points. This is where most time-series leakage happens.

---

## 6. Interaction Features (Feature Crosses)

### What They Are
Combine two or more features to capture relationships the model might not find on its own:
```
price_per_unit = total_price / quantity
conversion_rate = orders / visitors
revenue_per_employee = revenue / headcount
bmi = weight / height^2
```

### When They Matter
- Linear models can't learn interactions without explicit crosses. If churn depends on the COMBINATION of low revenue AND high support tickets, you need: `low_revenue_high_tickets = revenue < threshold AND tickets > threshold`
- Tree-based models can find interactions automatically through multi-level splits, but explicit crosses make the interaction immediately available — the model doesn't have to "discover" it through sequential splits
- Ratios are often more predictive than raw values: revenue_per_order is more informative than revenue and orders separately

### Common Patterns
- **Ratios**: feature_A / feature_B (efficiency, intensity metrics)
- **Products**: feature_A * feature_B (interaction strength, combined effect)
- **Differences**: feature_A - feature_B (relative comparison, gap)
- **Categorical crosses**: city + device_type -> "Toronto_mobile" (segment-specific patterns)
- **Polynomial**: feature_A^2, feature_A^3 (non-linear effects for linear models)

### E-Commerce Feature Crosses
```
average_order_value = total_revenue / num_orders
conversion_rate = num_orders / num_sessions
revenue_per_visitor = total_revenue / unique_visitors
days_since_last_order = today - last_order_date
order_frequency = num_orders / account_age_days
app_adoption_rate = num_apps_used / num_apps_available
revenue_growth_rate = (revenue_this_month - revenue_last_month) / revenue_last_month
```

Each derived feature captures a business concept that raw features alone don't express. An experienced ML engineer proposes these features immediately when hearing the problem description.

---

## 7. Missing Data

### Types of Missingness (Know These for Interviews)
- **MCAR (Missing Completely at Random)**: missingness is unrelated to any variable, observed or unobserved. Coin flip. Safe to drop or impute with simple methods. Rare in practice.
- **MAR (Missing at Random)**: missingness depends on observed variables but not on the missing value itself. Example: income missing more often for young people (age is observed). Can be handled with model-based imputation using observed variables.
- **MNAR (Missing Not at Random)**: missingness depends on the missing value itself. Example: high earners are less likely to report income. Hardest to handle — requires domain-specific solutions or modeling the missingness mechanism.

### Imputation Strategies

**Simple imputation:**
- **Mean/Median**: fast, reasonable for MCAR. Median is robust to outliers. Use median by default.
- **Mode**: for categorical features.
- **Constant**: fill with a sentinel value (0, -1, "Unknown"). The model can learn to treat this specially.

**Model-based imputation:**
- **KNN imputation**: use K nearest complete samples to estimate missing values. Captures local patterns.
- **Iterative imputer (MICE)**: model each missing feature as a function of all other features. Iterate until convergence. Most sophisticated but slowest.

**When missingness is informative:**
Sometimes the FACT that data is missing is itself a signal:
```python
df['income_is_missing'] = df['income'].isna().astype(int)
df['income'] = df['income'].fillna(df['income'].median())
```
Create a binary "is_missing" indicator AND impute the value. The model can learn from both the imputed value and the missingness pattern. This is almost always a good idea.

**XGBoost handles missing values natively.** During training, it learns the optimal split direction for missing values at each node. This is one of the reasons XGBoost works so well on real-world data — you don't need to impute first, and it can learn informative missingness patterns automatically.

### Don't Just Drop Rows
- Reduces dataset size, sometimes drastically (if many features have some missing values, most rows get dropped)
- Introduces bias if data isn't MCAR (the remaining data isn't representative)
- Wastes the information in complete columns for those rows
- Only drop if very few rows are affected (< 1%) and you're confident the data is MCAR

---

### Check Your Understanding

1. You are building a churn model and notice that 30% of merchants have a missing "last_login_date" field. A colleague suggests dropping all rows with missing values. Why is this a bad idea, and what should you do instead?
2. What is the difference between MCAR and MNAR missingness? Give a concrete e-commerce example of each.
3. Why is it recommended to create a binary "is_missing" indicator column in addition to imputing the missing value?

<details>
<summary>Answers</summary>

1. Dropping 30% of rows is problematic for multiple reasons: (a) it significantly reduces the training data, (b) if missingness is not MCAR (e.g., churned merchants are more likely to have never logged in), dropping creates a biased dataset that underrepresents the very population you want to predict, and (c) it wastes the information in other complete columns for those rows. Instead: create a binary indicator `login_date_is_missing`, impute the value (e.g., with median or a sentinel like a very old date), and let the model learn from both the imputed value and the missingness pattern. If using XGBoost, it handles missing values natively -- you may not need to impute at all.

2. MCAR (Missing Completely at Random): missingness has no relationship to any variable. Example -- a random server error causes 2% of order records to lose their shipping address, unrelated to the order value, customer, or product. MNAR (Missing Not at Random): missingness depends on the missing value itself. Example -- merchants with very low revenue are less likely to fill in their revenue field on a survey because they are embarrassed about low numbers. The missingness pattern is informative about the missing value, making it the hardest type to handle.

3. The fact that a value is missing often carries predictive signal beyond the value itself. For example, a missing "last_login_date" might indicate a merchant who never engaged with the platform at all -- this is a stronger churn signal than any imputed date. By creating the binary indicator, you give the model two pieces of information: the imputed value (a reasonable estimate of what the value might be) and whether the original value was observed. The model can learn different relationships for each case.

</details>

---

## 8. Feature Selection

### Why Select Features?
- Reduces overfitting (fewer features = less chance of fitting noise)
- Faster training and inference (production serving latency matters)
- Better interpretability (which features actually drive predictions?)
- With 200 engineered features — are all 200 contributing? Probably not. Some are redundant, some are noise.

### Filter Methods (fast, model-independent)

**Correlation analysis:**
```python
# Drop features with > 0.95 correlation with another feature
corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
```
If two features are 0.99 correlated, one is redundant. Drop the one with lower correlation to the target.

**Mutual information:** Measures how much knowing one variable tells you about another. Unlike correlation, it captures non-linear relationships.
```python
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y)
```

**Variance threshold:** Remove features with near-zero variance (they're essentially constant and carry no information).

### Wrapper Methods (thorough, expensive)

**Recursive Feature Elimination (RFE):**
1. Train model with all features
2. Rank features by importance
3. Remove the least important feature
4. Repeat until desired number of features
5. Pick the set that maximizes CV performance

Computationally expensive but thorough. Use `RFECV` in sklearn for automatic selection with cross-validation.

### Embedded Methods (built into training)

**L1 Regularization (Lasso):** Drives irrelevant feature coefficients exactly to zero during training. Train Lasso at various regularization strengths, keep features with non-zero coefficients. Automatic feature selection as a byproduct of training.

**Tree-based importance:**
- **Gain**: total reduction in loss from splits using each feature
- **Cover**: number of samples affected by splits on each feature
- **Frequency**: how often each feature is used in splits
- Caveat: biased toward high-cardinality and continuous features

### SHAP Values (The Gold Standard)
More than just importance — shows the direction and magnitude of each feature's contribution for each individual prediction.

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)  # Global importance + direction
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])  # Single prediction
```

**Permutation importance** (model-agnostic): shuffle one feature's values, measure accuracy drop. If performance doesn't change, the feature isn't useful. More reliable than built-in importance but slower.

---

## 9. Feature Stores

### What They Are
A feature store is a centralized system for defining, computing, storing, and serving ML features. It bridges the gap between feature engineering in notebooks and feature serving in production.

### The Problem They Solve: Training/Serving Skew
You engineer features in Jupyter with pandas. In production, those same features must be computed in real-time using a completely different system (Spark, Flink, SQL). Any discrepancy between training features and serving features silently degrades model performance.

Common symptoms of training/serving skew:
- Model performs well in development, degrades in production
- A/B test results don't match offline evaluation
- Features have subtly different distributions in training vs serving

### Key Components

**Feature definitions** — code that computes features from raw data, defined once:
```python
@feature
def avg_order_value_30d(merchant_id, timestamp):
    orders = get_orders(merchant_id, window=last_30_days(timestamp))
    return orders.total.mean() if len(orders) > 0 else 0.0
```

**Offline store:** Batch-computed features stored in a data warehouse (BigQuery, Snowflake). Used for training data generation. Supports historical point-in-time queries.

**Online store:** Low-latency feature serving for real-time inference (Redis, DynamoDB). Latest feature values. Sub-millisecond reads. Updated by streaming or periodic batch jobs.

**Feature registry:** Catalog of all features with metadata, owners, descriptions, data lineage, freshness, and quality metrics.

### Point-in-Time Correctness
When generating training data, you need features as they existed at prediction time, not as they exist now.

If training a model to predict fraud at the time of a transaction on March 15, you need the merchant's average order value as of March 15, not today's value. Using today's value would be temporal leakage.

Feature stores handle this with **time-travel queries**: "Give me merchant X's features as of March 15, 2025, 2:30 PM."

### Popular Feature Stores
- **Feast** (open source): Simple, integrates with existing infra, good starting point
- **Tecton** (managed): Enterprise-grade, built by the Uber Michelangelo team
- **Hopsworks**: Open source with managed option, strong integration with Python ML ecosystem
- **Custom**: Many large companies build their own to fit specific architecture

### When You Need One
- Multiple models share the same features (avoid recomputing)
- Real-time serving requires low-latency feature lookup
- Training/serving skew is causing production model degradation
- Team is large enough that feature discoverability matters
- You need point-in-time correctness for temporal data

---

### Check Your Understanding

1. What is training/serving skew, and why does it cause models to perform worse in production than in development?
2. Explain what "point-in-time correctness" means in the context of generating training data. Give an example of what goes wrong without it.
3. You are choosing between SHAP-based feature selection and L1 (Lasso) regularization to reduce 200 features to the most important 50. What are the tradeoffs?

<details>
<summary>Answers</summary>

1. Training/serving skew occurs when the features computed during model training differ from those computed during production serving. This typically happens because training features are computed in Python/pandas in a notebook while serving features are computed in a different system (SQL, Spark, Flink). Subtle differences in logic (e.g., different handling of nulls, different time zone conversions, or different aggregation windows) mean the model sees slightly different input distributions in production, degrading its predictions. Feature stores solve this by defining feature computation once and serving consistently for both training and inference.

2. Point-in-time correctness means that when generating training data for a historical event, you use only the feature values that existed at the time of that event, not current values. Example: if training a fraud model for a transaction on March 15, you need the merchant's 30-day average order value as of March 15, not today's value. Without this, you introduce temporal leakage -- the model trains on "future" information it would not have at prediction time, leading to inflated offline metrics that do not hold in production.

3. SHAP-based selection: model-agnostic, captures non-linear feature contributions, provides per-feature importance with direction, and works with any model (especially tree-based). However, it requires training a full model first and is computationally expensive for large datasets. L1/Lasso: performs selection during training (embedded method), is fast, and naturally handles correlated features by picking one from each group. However, it only captures linear relationships, so features that are important for non-linear models like XGBoost might be incorrectly eliminated. For tree-based production models, SHAP is generally more reliable.

</details>

---

## 10. Putting It Together: Sports Prediction Example

A well-engineered sports prediction model might include 200 features. Here is how to discuss that kind of feature work in interviews:

**Typical engineered features:**
- Rolling window statistics (team performance over N games)
- Lag features (last game stats, last 5 games)
- Interaction features (net rating, offensive/defensive efficiency ratios)
- Situational features (home/away, rest days, schedule context)

**How to improve (discussion points for interviews):**
- Feature selection: Are all 200 features contributing? Use SHAP or permutation importance to find the top 50 that drive 95% of predictions. Remove the rest — they're adding noise and computation cost.
- Target encoding: For categorical features like opponent, target encoding (with proper fold-based computation) captures matchup effects.
- Time-aware features: Ensure all rolling windows are strictly past-looking. Validate with time-series CV.
- Feature store thinking: If this were production, how would you serve these features for real-time predictions? What would the batch vs streaming split look like?

---

## 11. E-Commerce Feature Engineering

### Merchant Features (for churn, segmentation, health scoring)
```
# Engagement
days_since_last_login
logins_per_week_rolling_30d
admin_pages_viewed_per_session
features_used_count / features_available_count  # adoption rate

# Business health
gmv_rolling_30d / gmv_rolling_90d  # revenue momentum
order_count_rolling_7d
avg_order_value_rolling_30d
unique_customers_rolling_30d

# Platform investment
num_apps_installed
num_theme_changes_last_90d
custom_domain_set (binary)
num_products_listed

# Risk signals
payment_failure_count_30d
support_tickets_30d
days_on_current_plan
plan_downgrades_count
```

### Product Features (for recommendations, search ranking, clustering)
```
# Content
title_embedding (sentence-BERT, 384-dim)
description_embedding
category (target-encoded or embedded)
price_normalized_by_category  # relative pricing within category

# Performance
views_30d
purchases_30d
conversion_rate = purchases / views
add_to_cart_rate
return_rate

# Quality signals
num_images
description_length
has_size_chart (binary)
avg_review_score
num_reviews
```

### User Behavior Features (for personalization, recommendations)
```
# Session-level
pages_viewed_this_session
time_on_site_this_session
search_queries_this_session
categories_browsed  # diversity of interest

# Historical
purchase_history_embedding  # aggregate of purchased product embeddings
avg_purchase_price
purchase_frequency
days_since_last_purchase
favorite_categories (top 3 by purchase count)
price_sensitivity = discount_purchases / total_purchases
```

---

## 12. When to Engineer Features vs Let the Model Learn

**Engineer features (tabular data, tree-based models):**
- When you have domain knowledge the model can't discover from raw data
- When the feature is a known business metric (conversion rate, ARPU, churn rate)
- When the dataset is small-to-medium (< 10M rows) — models need help
- When features span different tables/sources that need joining
- When temporal logic is involved (rolling windows, lag features, seasonality)

**Let the model learn (deep learning, unstructured data):**
- Images: CNNs learn features from raw pixels. Hand-crafted image features (SIFT, HOG) are obsolete.
- Text: Transformers learn contextual representations from raw text. TF-IDF is a baseline, not the ceiling.
- Audio: spectrogram -> CNN or raw waveform -> WaveNet
- Very large tabular datasets (> 100M rows): neural networks can learn interactions you'd never think to engineer

**The hybrid approach (production recommendation systems):**
- Pre-trained embeddings (learned features) for items and users
- Hand-engineered features for business metrics, context, and freshness
- Combine both in a ranking model (XGBoost or a neural ranker)
- This is what Netflix, YouTube, Amazon, and other large-scale platforms actually do

---

## Common Pitfalls

1. **Fitting scalers or encoders on the full dataset before train/test split.** This is one of the most common forms of data leakage. If you fit StandardScaler on all data, the test set's mean and standard deviation influence the training features. Always fit on training data only, then transform test data using the training-fitted transformer. This applies to all preprocessing: scaling, imputation, target encoding, and feature selection.

2. **Applying one-hot encoding to high-cardinality categorical features.** A "product_category" with 10,000 unique values creates 10,000 sparse columns, leading to curse of dimensionality and extreme sparsity. Use target encoding (with fold-based regularization), frequency encoding, embedding layers, or the hashing trick instead.

3. **Including future information in time-based features.** When computing rolling window statistics, the window must only look backward from the prediction point. A rolling 7-day average that includes today's value or tomorrow's value is temporal leakage. This is especially insidious because it is hard to detect -- the model trains successfully and looks great offline, but fails in production where future values are unavailable.

4. **Binning continuous features for tree-based models.** Trees already find optimal split points on continuous features. Binning discards fine-grained information that the tree could use (e.g., binning revenue into "low/medium/high" prevents the tree from splitting at revenue=$47,500 if that is the optimal threshold). Only bin for tree models if you have a strong domain reason.

---

## Hands-On Exercises

### Exercise 1: End-to-End Feature Engineering Pipeline (25 min)

Using the Kaggle "Titanic" dataset (or `sns.load_dataset('titanic')` from seaborn):

1. Handle missing values: impute "age" with median, "embarked" with mode, and create `age_is_missing` and `embarked_is_missing` indicator columns.
2. Encode categorical features: one-hot encode "embarked" and "sex", and create an ordinal encoding for "pclass".
3. Engineer interaction features: `family_size = sibsp + parch + 1`, `is_alone = family_size == 1`, `fare_per_person = fare / family_size`.
4. Apply log transform to "fare" (it is heavily right-skewed).
5. Train a logistic regression and XGBoost model on the engineered features. Compare performance (AUC) to models trained on only the raw numerical columns.

```python
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
```

### Exercise 2: Feature Selection with SHAP (20 min)

Using the scikit-learn `california_housing` dataset:

1. Train an XGBoost regressor on all 8 features.
2. Compute SHAP values using `shap.TreeExplainer` and generate a summary plot.
3. Identify the top 4 features by mean absolute SHAP value.
4. Retrain on only those 4 features. Compare RMSE (5-fold CV) to the full-feature model.
5. How much accuracy did you lose? How much did inference speed improve?

```python
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
import shap
```

---

## 13. Interview Questions with Answers

**Q: You're building a merchant churn model. What features would you engineer?**
A: I'd organize features into four groups. (1) Engagement: days since last login, login frequency trend, feature adoption rate, admin page views. (2) Business health: GMV trend (30d vs 90d ratio), order count trend, average order value, customer count growth. (3) Platform investment: number of apps installed, theme customizations, whether they've set up a custom domain, product count. (4) Risk signals: payment failures, support ticket count, plan downgrades, days on current plan. I'd compute rolling windows (7d, 30d, 90d) and trends (ratios of short to long windows) for each metric. All features would be point-in-time correct — computed using only data available at the prediction moment.

**Q: How do you handle a categorical feature with 100,000 unique values?**
A: One-hot is impossible (100K sparse columns). Options in order of preference: (1) Learned embeddings if using a neural network — the standard approach for entity IDs. (2) Target encoding with proper regularization (fold-based or smoothed) for tree models — powerful but requires careful handling to avoid leakage. (3) Frequency encoding as a simple baseline — replace each category with its count or frequency. (4) Hashing trick to a fixed number of buckets (e.g., 1000) — handles open vocabularies with bounded memory. (5) Clustering categories first (group rare categories into "other" or use hierarchical categories), then encode the clusters.

**Q: What's data leakage and how do you prevent it?**
A: Leakage is when information unavailable at prediction time contaminates your training features. Three types: (1) Target leakage — feature derived from the target ("fraud_reported_date" to predict "is_fraud"). (2) Temporal leakage — using future data (rolling mean including future values). (3) Train/test leakage — fitting preprocessing on full data before splitting. Prevention: for every feature, ask "would I have this at prediction time in production?" Fit all transformations (scaling, encoding, imputation) on training data only. Use time-series splits for temporal data. If performance is suspiciously high, check for leakage before celebrating.

**Q: When would you use feature selection vs letting the model use all features?**
A: Feature selection when: (1) you have more features than samples (risk of overfitting), (2) serving latency matters (fewer features = faster inference), (3) you need interpretability (explain top 10 drivers of churn), (4) some features are expensive to compute in production. Let the model use all features when: (1) you have ample data relative to features, (2) you're using a well-regularized model (XGBoost with early stopping), (3) inference speed isn't a constraint. In practice, I'd start with all features, use SHAP to understand contributions, then prune features that contribute < 1% of total importance and validate that performance holds.

**Q: Explain the difference between TF-IDF and embeddings for text.**
A: TF-IDF produces sparse vectors based on word frequency — it captures what words appear and how distinctive they are, but loses word order and doesn't understand meaning. "Happy dog" and "joyful puppy" are completely different vectors. Embeddings produce dense vectors that capture semantic meaning — similar concepts get similar vectors regardless of exact words used. TF-IDF is faster, simpler, and works well with linear models for classification. Embeddings are better for semantic similarity, search, and as features for downstream models. In production, I'd start with TF-IDF + logistic regression as a baseline, then move to pre-trained embeddings if I need semantic understanding.

**Q: What's a feature store and when would you use one?**
A: A feature store centralizes feature computation, storage, and serving. It defines features once and serves them consistently for both offline training and online inference, solving training/serving skew. Key components: offline store (data warehouse for historical features), online store (Redis/DynamoDB for low-latency serving), and a feature registry for discoverability. I'd introduce one when: multiple models share features (avoid recomputation), real-time serving needs low-latency feature lookup, or the team is large enough that feature discoverability and consistency across models matters. The critical capability is point-in-time correctness — getting features as they existed at a historical timestamp for training data generation.

**Q: An XGBoost model uses 200 features. How would you reduce that for production?**
A: Step 1: Run SHAP analysis to rank all 200 features by absolute mean SHAP value. Step 2: Identify the top 50 that account for 95%+ of total prediction impact. Step 3: Check for highly correlated feature pairs in the top 50 — drop one from each pair (the one with lower SHAP value). Step 4: Retrain with the reduced feature set and validate that cross-validated performance drops by less than 1%. Step 5: If any of the dropped features are expensive to compute in production, confirm they're truly unnecessary with an ablation study. The result is a faster, cheaper, more maintainable model with nearly identical performance.

---

## Summary

This lesson covered feature engineering as the single largest lever for model performance in applied ML:

- **Numerical features** require appropriate scaling (StandardScaler, MinMaxScaler, RobustScaler) for linear models and neural networks, but not for tree-based models. Log transforms compress skewed distributions.
- **Categorical features** need encoding strategies matched to cardinality and model type: one-hot for low cardinality with linear models, target encoding (with fold-based regularization) or embeddings for high cardinality.
- **Text features** range from simple (TF-IDF + logistic regression) to sophisticated (pre-trained sentence embeddings), with the practical approach being to start simple and upgrade only when needed.
- **Time features** require cyclical encoding, lag features, and rolling window statistics, with strict attention to temporal ordering to avoid leakage.
- **Interaction features** encode domain knowledge (ratios, products, differences) that models may not discover on their own, especially linear models.
- **Missing data** should be imputed thoughtfully (with missingness indicators), not dropped. XGBoost handles it natively.
- **Feature selection** via SHAP, permutation importance, or L1 regularization reduces feature sets for production deployment.
- **Feature stores** solve training/serving skew by centralizing feature computation and ensuring point-in-time correctness.

## What's Next

- **Evaluation Metrics** — how to measure whether your engineered features actually improve model performance, including the precision-recall tradeoff, ranking metrics for recommendations, and calibration (see [Evaluation Metrics](../evaluation-metrics/COURSE.md))
- **Supervised Learning** — revisit the algorithm details to understand how different models consume the features you engineer (see [Supervised Learning](../supervised/COURSE.md))
