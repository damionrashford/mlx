# ML Data Pipelines

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The difference between ETL and ELT paradigms and when to use each
- Core concepts of batch pipelines (idempotency, backfilling) and streaming pipelines (Kafka, Pub/Sub)
- The data quality framework: completeness, uniqueness, validity, timeliness, consistency, and accuracy

**Apply:**
- Design Airflow DAGs with proper validation, idempotency, and backfill support for ML feature computation
- Implement data validation using schema checks, distribution checks, and quality gates at pipeline boundaries

**Analyze:**
- Evaluate feature pipeline architectures (batch vs. streaming vs. hybrid) and identify training-serving skew risks in a given data pipeline design

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- **ML system design fundamentals** -- the six-step design framework, serving patterns, and where data pipelines fit in the overall architecture (see [System Design](./system-design/COURSE.md))

---

## Why This Matters

Applied ML engineering roles explicitly list data pipelines and ETL. In practice, ML engineers spend 60-80% of their time on data work. Building reliable, testable, idempotent data pipelines is the core skill that separates ML engineers who ship from those who only prototype.

This lesson covers the architecture and tooling of ML data pipelines end to end.

---

## ETL vs ELT

### ETL (Extract, Transform, Load)

The traditional approach. Transform data before loading it into the warehouse.

```
Source Systems → Extract → Transform (external compute) → Load → Data Warehouse
```

**Characteristics:**
- Transformation happens outside the warehouse (Spark, Python scripts)
- Data is cleaned and structured before it enters the warehouse
- Common when warehouses were expensive (storage was the bottleneck)

### ELT (Extract, Load, Transform)

The modern approach. Load raw data first, transform inside the warehouse.

```
Source Systems → Extract → Load (raw) → Data Warehouse → Transform (SQL/DBT)
```

**Characteristics:**
- Raw data lands in the warehouse first
- Transformation uses the warehouse's compute engine (BigQuery, Snowflake)
- Cheaper storage makes this viable — store everything, transform what you need
- **DBT is the dominant tool for the "T" in ELT**

### Which to Use

| Factor | ETL | ELT |
|--------|-----|-----|
| Warehouse cost | Expensive (legacy) | Cheap (cloud-native) |
| Transform complexity | Complex (ML, APIs) | SQL-expressible |
| Data volume | Moderate | Large (petabytes) |
| Tooling | Spark, custom scripts | DBT, warehouse SQL |
| Industry context | Legacy systems | **Modern stack (preferred)** |

**Many large e-commerce companies use ELT with BigQuery + DBT.** Raw events land in BigQuery, DBT models transform them into feature tables.

---

## Batch Pipelines

Batch pipelines process data in discrete chunks on a schedule.

### Core Concepts

**Idempotency:** Running the same pipeline twice with the same input produces the same output. This is non-negotiable.

```python
# BAD: Not idempotent — appends duplicates on re-run
INSERT INTO features SELECT * FROM raw_events WHERE date = '2024-01-15'

# GOOD: Idempotent — replaces the partition
DELETE FROM features WHERE date = '2024-01-15';
INSERT INTO features SELECT * FROM raw_events WHERE date = '2024-01-15';

# BETTER: Use MERGE or partition overwrite
MERGE INTO features USING (
    SELECT * FROM raw_events WHERE date = '2024-01-15'
) AS source ON features.id = source.id
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...
```

**Backfilling:** Reprocessing historical data when you change pipeline logic. Your pipeline must support running for arbitrary date ranges.

```python
# Pipeline should accept date parameters
def compute_features(execution_date: str):
    """Compute features for a specific date. Idempotent."""
    query = f"""
    CREATE OR REPLACE TABLE features.daily_{execution_date} AS
    SELECT
        merchant_id,
        COUNT(*) as order_count,
        SUM(total_price) as revenue,
        AVG(total_price) as avg_order_value
    FROM orders
    WHERE DATE(created_at) = '{execution_date}'
    GROUP BY merchant_id
    """
    run_query(query)
```

### Apache Airflow

Airflow is the industry standard for orchestrating batch pipelines. Many large companies use it.

**Core concepts:**
- **DAG** (Directed Acyclic Graph): defines task dependencies
- **Operator**: a single task (run SQL, execute Python, call API)
- **Schedule**: cron expression or preset (`@daily`, `@hourly`)
- **Sensor**: wait for a condition (file exists, partition ready)

```python
# Example: Daily feature computation DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'merchant_features_daily',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't backfill on deploy
    tags=['ml', 'features'],
) as dag:

    validate_source = BigQueryInsertJobOperator(
        task_id='validate_source_data',
        configuration={
            'query': {
                'query': """
                    SELECT COUNT(*) as row_count
                    FROM `project.raw.orders`
                    WHERE DATE(created_at) = '{{ ds }}'
                    HAVING COUNT(*) > 0
                """,
                'useLegacySql': False,
            }
        },
    )

    compute_features = BigQueryInsertJobOperator(
        task_id='compute_merchant_features',
        configuration={
            'query': {
                'query': """
                    CREATE OR REPLACE TABLE `project.features.merchant_daily`
                    PARTITION BY date AS
                    SELECT
                        DATE('{{ ds }}') as date,
                        merchant_id,
                        COUNT(*) as order_count_1d,
                        SUM(total_price) as revenue_1d,
                        COUNT(DISTINCT customer_id) as unique_customers_1d
                    FROM `project.raw.orders`
                    WHERE DATE(created_at) = '{{ ds }}'
                    GROUP BY merchant_id
                """,
                'useLegacySql': False,
            }
        },
    )

    validate_output = PythonOperator(
        task_id='validate_features',
        python_callable=validate_feature_table,
        op_kwargs={'date': '{{ ds }}'},
    )

    validate_source >> compute_features >> validate_output
```

### DAG Design Patterns

```
Pattern 1: Linear Pipeline
  extract → transform → validate → load

Pattern 2: Fan-out/Fan-in
  extract → [transform_A, transform_B, transform_C] → combine → load

Pattern 3: Sensor-Triggered
  wait_for_upstream_dag → extract → transform → load

Pattern 4: Branching
  extract → check_data_quality → (pass) → transform → load
                                → (fail) → alert → skip
```

---

### Check Your Understanding: ETL/ELT and Batch Pipelines

**1. Why is idempotency non-negotiable for production data pipelines?**

<details>
<summary>Answer</summary>

Pipelines fail and must be re-run. Without idempotency, re-running a pipeline creates duplicate data, which corrupts downstream features and model training. Idempotent pipelines produce the same result regardless of how many times they run with the same input, making retries and backfills safe. Techniques include partition overwriting (DELETE + INSERT) and MERGE statements.
</details>

**2. In the Airflow DAG example, what does `catchup=False` do and when would you set it to `True`?**

<details>
<summary>Answer</summary>

`catchup=False` prevents Airflow from executing the DAG for all missed schedule intervals between `start_date` and the current date when the DAG is first deployed. You would set `catchup=True` when you need to backfill historical data -- for example, when deploying a new feature pipeline and you need to compute features for the past 90 days to build a training dataset.
</details>

**3. In the ELT approach, why is DBT the dominant tool for the "T" (Transform) step?**

<details>
<summary>Answer</summary>

DBT allows data transformations to be written as SQL SELECT statements, version-controlled in git, tested with built-in assertions, and documented. It leverages the warehouse's own compute engine (BigQuery, Snowflake), so there is no need to move data out of the warehouse for transformation. It also supports dependency management between models, incremental builds, and environment separation (dev/staging/production).
</details>

---

## Streaming Pipelines

Streaming pipelines process data continuously as it arrives.

### Apache Kafka

Distributed event streaming platform. The backbone of real-time data infrastructure.

**Core concepts:**
- **Topic**: a named stream of events (e.g., `platform.orders.created`)
- **Producer**: writes events to a topic
- **Consumer**: reads events from a topic
- **Consumer group**: multiple consumers sharing the work of reading a topic
- **Partition**: a topic is split into partitions for parallelism
- **Offset**: position of a consumer in a partition

```
Producer → Topic (partitioned) → Consumer Group
                                    ├── Consumer 1 (partition 0, 1)
                                    ├── Consumer 2 (partition 2, 3)
                                    └── Consumer 3 (partition 4, 5)
```

### Google Cloud Pub/Sub

Managed alternative to Kafka. Serverless — no cluster to manage.

```
Publisher → Topic → Subscription → Subscriber
                  → Subscription → Subscriber (multiple subscriptions per topic)
```

**Kafka vs Pub/Sub:**

| Factor | Kafka | Pub/Sub |
|--------|-------|---------|
| Managed | Self-hosted or Confluent Cloud | Fully managed |
| Ordering | Per-partition guaranteed | Best-effort (ordering key available) |
| Replay | Consumer seeks to offset | Seek to timestamp |
| Cost | Cluster cost (fixed) | Per-message (variable) |
| Industry context | Common in large orgs | Natural for GCP stack |

### Real-Time Feature Computation

Streaming pipelines compute features that need to be fresh for serving.

```python
# Example: Compute real-time velocity features with Apache Flink (pseudocode)
# "Number of transactions from this credit card in the last hour"

class FraudVelocityFeature(ProcessWindowFunction):
    def process(self, key, context, elements):
        card_id = key
        window_start = context.window().start
        window_end = context.window().end
        txn_count = len(elements)
        total_amount = sum(e.amount for e in elements)

        yield VelocityFeature(
            card_id=card_id,
            txn_count_1h=txn_count,
            total_amount_1h=total_amount,
            window_end=window_end,
        )

# Write to online feature store (Redis) for real-time serving
```

---

## Data Validation

Bad data silently destroys model performance. Validate at every boundary.

### Schema Validation

Ensure data conforms to expected structure.

```python
# Using Great Expectations
import great_expectations as gx

context = gx.get_context()

# Define expectations for order data
validator = context.sources.pandas_default.read_csv("orders.csv")
validator.expect_column_to_exist("order_id")
validator.expect_column_to_exist("merchant_id")
validator.expect_column_values_to_not_be_null("order_id")
validator.expect_column_values_to_be_between("total_price", min_value=0, max_value=1_000_000)
validator.expect_column_values_to_be_in_set("currency", ["USD", "CAD", "GBP", "EUR"])

results = validator.validate()
if not results.success:
    raise DataValidationError(results)
```

### Distribution Validation

Detect when data distributions shift from expected ranges.

```python
def validate_distributions(current_df, reference_df, columns, threshold=0.05):
    """Compare distributions using Kolmogorov-Smirnov test."""
    from scipy import stats
    alerts = []
    for col in columns:
        statistic, p_value = stats.ks_2samp(
            reference_df[col].dropna(),
            current_df[col].dropna()
        )
        if p_value < threshold:
            alerts.append({
                'column': col,
                'ks_statistic': statistic,
                'p_value': p_value,
                'status': 'DRIFT_DETECTED'
            })
    return alerts
```

### Validation Checkpoints in a Pipeline

```
Raw Data → [Schema Check] → Staging → [Distribution Check] → Feature Store → [Completeness Check]
              ↓ fail              ↓ fail                        ↓ fail
            Alert + halt       Alert + investigate            Alert + fallback
```

**What to check at each stage:**

| Stage | Checks |
|-------|--------|
| Ingestion | Schema matches, no null PKs, row count within expected range |
| Transformation | No NaN explosion, feature ranges valid, join completeness |
| Feature Store | Feature coverage (% of entities with features), staleness |
| Model Input | No missing features, input shape matches model expectation |

---

### Check Your Understanding: Streaming and Data Validation

**1. What is the key trade-off between Kafka and Google Cloud Pub/Sub for ML streaming pipelines?**

<details>
<summary>Answer</summary>

Kafka provides per-partition ordering guarantees and consumer offset control (allowing precise replay), but requires managing a cluster (or paying for Confluent Cloud). Pub/Sub is fully managed (serverless) with per-message pricing, but only offers best-effort ordering (with ordering keys available for stricter guarantees). Choose Kafka when you need strict ordering and replay control; choose Pub/Sub when you want operational simplicity and are on GCP.
</details>

**2. In the data validation pipeline, why should you have different checks at each stage (ingestion, transformation, feature store, model input)?**

<details>
<summary>Answer</summary>

Each stage introduces different failure modes. At ingestion, the raw data may have schema violations or missing primary keys. At transformation, SQL bugs may produce NaN explosions or broken joins. At the feature store level, features may be stale or have incomplete coverage. At model input, the assembled feature vector may have missing features or incorrect shapes. Catching errors early (at ingestion) is cheaper than catching them late (at model input), but each stage needs its own specific checks because upstream validation cannot catch downstream-specific issues.
</details>

---

## Training Data Pipelines

### The Training Data Lifecycle

```
Raw Events → Labeling → Cleaning → Splitting → Feature Engineering → Training Dataset
```

### Labeling Strategies

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| Explicit labels | Merchant marks order as fraud | High quality | Expensive, slow, sparse |
| Implicit signals | Purchase = positive, no purchase = negative | Cheap, abundant | Noisy, biased |
| Delayed labels | Chargeback arrives 30-90 days later | Ground truth | Can't train on recent data |
| Human annotation | Labelers classify product categories | Flexible | Expensive, inter-annotator disagreement |
| Weak supervision | Heuristic rules + Snorkel | Scalable | Lower quality |

### Data Splitting

**Never split randomly for time-series or event data.** Use temporal splits.

```python
# WRONG: Random split leaks future information
train, test = train_test_split(data, test_size=0.2, random_state=42)

# RIGHT: Temporal split
train = data[data['date'] < '2024-01-01']
val   = data[(data['date'] >= '2024-01-01') & (data['date'] < '2024-02-01')]
test  = data[data['date'] >= '2024-02-01']
```

**Why temporal splits matter:**
- Random splits let the model "peek" at future patterns
- Production model will only ever see past data
- Temporal splits simulate real deployment conditions

### Data Cleaning

```python
def clean_training_data(df):
    """Standard cleaning pipeline for ML training data."""
    # Remove duplicates
    df = df.drop_duplicates(subset=['order_id'])

    # Handle missing values
    df['shipping_address_country'] = df['shipping_address_country'].fillna('UNKNOWN')
    df['total_price'] = df['total_price'].fillna(df['total_price'].median())

    # Remove outliers (domain-specific)
    df = df[df['total_price'] > 0]
    df = df[df['total_price'] < df['total_price'].quantile(0.999)]

    # Type conversion
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Log transform skewed features
    df['log_total_price'] = np.log1p(df['total_price'])

    return df
```

---

## Feature Pipelines

Feature pipelines compute features for both training and serving. This is the most important pipeline to get right.

### Batch Feature Pipeline

```python
# BigQuery SQL for merchant features (runs daily via Airflow)
CREATE OR REPLACE TABLE `project.features.merchant_features_v2`
PARTITION BY snapshot_date
CLUSTER BY merchant_id
AS
WITH order_stats AS (
    SELECT
        merchant_id,
        DATE('{{ ds }}') as snapshot_date,
        -- 7-day features
        COUNTIF(DATE(created_at) >= DATE_SUB('{{ ds }}', INTERVAL 7 DAY)) as orders_7d,
        SUM(IF(DATE(created_at) >= DATE_SUB('{{ ds }}', INTERVAL 7 DAY), total_price, 0)) as revenue_7d,
        -- 30-day features
        COUNTIF(DATE(created_at) >= DATE_SUB('{{ ds }}', INTERVAL 30 DAY)) as orders_30d,
        SUM(IF(DATE(created_at) >= DATE_SUB('{{ ds }}', INTERVAL 30 DAY), total_price, 0)) as revenue_30d,
        -- Lifetime features
        COUNT(*) as orders_lifetime,
        SUM(total_price) as revenue_lifetime,
        MIN(created_at) as first_order_at,
        MAX(created_at) as last_order_at,
    FROM `project.raw.orders`
    WHERE DATE(created_at) <= '{{ ds }}'
    GROUP BY merchant_id
),
product_stats AS (
    SELECT
        merchant_id,
        COUNT(*) as active_products,
        AVG(price) as avg_product_price,
        COUNT(DISTINCT product_type) as product_type_count,
    FROM `project.raw.products`
    WHERE status = 'active'
    GROUP BY merchant_id
)
SELECT
    o.*,
    p.active_products,
    p.avg_product_price,
    p.product_type_count,
    SAFE_DIVIDE(o.revenue_7d, o.revenue_30d) as revenue_7d_30d_ratio,
    DATE_DIFF('{{ ds }}', DATE(o.last_order_at), DAY) as days_since_last_order,
FROM order_stats o
LEFT JOIN product_stats p USING (merchant_id);
```

### Feature Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Batch Feature Pipeline (Airflow, daily)                        │
│                                                                │
│ Raw tables → SQL transforms → Feature table (BigQuery)         │
│                                    ↓                           │
│                              Sync to Redis (online store)      │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Streaming Feature Pipeline (Flink/Dataflow)                    │
│                                                                │
│ Event stream → Window aggregation → Online store (Redis)       │
│ (Kafka/PubSub)   (count, sum, avg)                            │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Training reads from: BigQuery (batch features) + Redis snap    │
│ Serving reads from:  Redis (all features, low latency)         │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Versioning

### Why Version Data

Models are deterministic given code + data + hyperparameters. If you change data, you need to track what changed.

### Tools

**DVC (Data Version Control):**
```bash
# Track a training dataset
dvc add data/training_data_v3.parquet
git add data/training_data_v3.parquet.dvc
git commit -m "Update training data: added Q4 2024 orders"

# Reproduce an old experiment
git checkout abc123  # old commit
dvc checkout          # fetches the data version from that commit
```

**Delta Lake / Apache Iceberg:**
- Table formats that support versioning, time travel, and schema evolution
- Built into Databricks, Snowflake (Iceberg)
- Query data as of any timestamp: `SELECT * FROM orders TIMESTAMP AS OF '2024-01-01'`

**BigQuery snapshots:**
```sql
-- Query a table as it existed 7 days ago
SELECT * FROM `project.dataset.table`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
```

---

## Data Quality: Systematic Checks

### The Data Quality Framework

```
Dimension        Check                           Example
─────────────────────────────────────────────────────────────────
Completeness     No unexpected NULLs             merchant_id is never null
Uniqueness       No duplicate primary keys       order_id is unique
Validity         Values in expected range         price > 0
Timeliness       Data arrives on schedule         Partition for today exists by 6am
Consistency      Cross-table relationships hold   Every order references a valid merchant
Accuracy         Data matches reality             Spot-check against source systems
```

### Implementing Quality Gates

```python
# Quality gate: block pipeline if checks fail
class QualityGate:
    def __init__(self, table_name, date):
        self.table = table_name
        self.date = date
        self.checks = []

    def add_check(self, name, query, threshold):
        self.checks.append({'name': name, 'query': query, 'threshold': threshold})

    def run(self):
        failures = []
        for check in self.checks:
            result = run_query(check['query'])
            if result < check['threshold']:
                failures.append(f"{check['name']}: {result} < {check['threshold']}")
        if failures:
            raise QualityGateFailure(failures)

# Usage
gate = QualityGate('merchant_features', '2024-01-15')
gate.add_check(
    'row_count',
    "SELECT COUNT(*) FROM features WHERE date = '2024-01-15'",
    threshold=100_000  # expect at least 100K merchants
)
gate.add_check(
    'null_rate',
    "SELECT 1 - COUNTIF(revenue_30d IS NULL) / COUNT(*) FROM features WHERE date = '2024-01-15'",
    threshold=0.99  # expect < 1% null rate
)
gate.run()
```

---

### Check Your Understanding: Training Data and Feature Pipelines

**1. Why must you use temporal splits instead of random splits for time-series or event data?**

<details>
<summary>Answer</summary>

Random splits leak future information into the training set. In production, the model will only ever have access to data from the past, so training must simulate this constraint. With a random split, the model might see January 15th data during training and be evaluated on January 10th data -- it has effectively "seen the future." Temporal splits ensure all training data precedes all validation data, which precedes all test data, accurately simulating real deployment conditions.
</details>

**2. Why is it critical that feature pipelines produce identical features for training and serving?**

<details>
<summary>Answer</summary>

If training features are computed differently from serving features (training-serving skew), the model receives inputs in production that are statistically different from what it learned on. This causes silent performance degradation -- the model does not error, it just makes worse predictions. Common causes include using different SQL vs. Python logic, different libraries, or different data sources for the same feature between training and serving.
</details>

---

## Production Context

### Data Sources at E-Commerce Scale

| Source | Volume | Use Cases |
|--------|--------|-----------|
| Order events | Billions/month | Revenue prediction, fraud detection |
| Product catalog | Hundreds of millions | Recommendations, search ranking |
| Merchant profiles | Millions | Churn prediction, segmentation |
| Storefront events | Trillions of events/year | Clickstream analysis, funnel optimization |
| Payment transactions | Billions/month | Fraud detection, risk scoring |

### Pipeline Challenges at Scale
- **Data skew**: Top 1% of merchants generate 50%+ of events. Pipelines must handle this.
- **Late-arriving data**: Events from mobile can arrive hours after occurrence.
- **Schema evolution**: Product catalog schema changes as new features are added.
- **Multi-region**: Data residency requirements (GDPR) affect pipeline design.
- **Cost**: A single bad query on petabyte-scale data can cost thousands of dollars.

---

## Common Pitfalls

**1. Non-idempotent pipelines with INSERT-only logic.** Using `INSERT INTO` without first clearing the target partition means re-runs append duplicate rows. This silently corrupts feature tables and model training data. Always use DELETE+INSERT, MERGE, or partition overwrite patterns.

**2. Not designing for backfill from the start.** Hardcoding `CURRENT_DATE` or using non-parameterized queries means you cannot reprocess historical data when pipeline logic changes. Every pipeline should accept an execution date parameter and process exactly that date's data.

**3. Ignoring late-arriving data.** In mobile and distributed systems, events can arrive hours or even days late. Pipelines that only process data from the exact execution date will miss these events, leading to incomplete features. Design pipelines with a lookback window or a late-arrival reconciliation step.

**4. Skipping data validation because "the data is always clean."** Data sources change without warning -- schemas evolve, upstream pipelines break, third-party APIs return unexpected values. Without validation gates, bad data flows silently through the pipeline and into model training, often causing failures that are only detected weeks later through degraded model performance.

---

## Hands-On Exercises

### Exercise 1: Build an Idempotent Feature Pipeline

Write a SQL-based feature pipeline (or pseudocode) for computing weekly customer engagement features from an orders table. Your pipeline must:

1. Accept an `execution_date` parameter
2. Compute at least 3 features (e.g., order count, total revenue, distinct products purchased) over a 7-day window ending on `execution_date`
3. Be fully idempotent -- running it twice for the same date produces identical results
4. Include at least 2 data validation checks (e.g., minimum row count, null rate threshold)

### Exercise 2: Design a Validation Strategy

Given the following pipeline:

```
Raw order events (Kafka) -> Streaming aggregation (Flink) -> Redis (online store)
                         -> Daily batch job (BigQuery) -> Feature table -> Training
```

For each boundary in this pipeline, list specific validation checks you would implement, what thresholds you would set, and what action to take on failure (halt, alert, fallback). Consider: What happens if the Kafka topic stops producing events? What if the Flink job silently drops records? What if the BigQuery batch job produces features with different distributions than the streaming path?

---

## Practice Interview Questions

1. "Design a data pipeline that computes daily merchant health scores. What happens when the pipeline fails? How do you backfill?"
2. "We have a feature that takes 3 hours to compute. Serving needs it in real-time. How do you solve this?"
3. "How would you detect that a data pipeline is producing incorrect results before the model is affected?"
4. "A feature pipeline ran successfully but the downstream model's accuracy dropped 5%. What do you investigate?"

---

## Key Takeaways

1. ELT (load first, transform in warehouse) is the modern standard. DBT handles the T.
2. Idempotency is non-negotiable. Every pipeline must produce identical results on re-run.
3. Temporal splits for time-series data. Random splits leak future information.
4. Validate data at every pipeline boundary. Schema + distribution + completeness checks.
5. Feature pipelines must produce identical features for training and serving.
6. Version your data. Model reproducibility requires knowing exactly what data was used.
7. Design for backfilling from day one. You will need to reprocess historical data.

---

## Summary and What's Next

This lesson covered the full landscape of ML data pipelines: ETL vs. ELT paradigms, batch pipeline design with Airflow (idempotency, backfilling, DAG patterns), streaming pipelines with Kafka and Pub/Sub, data validation at every boundary, training data lifecycle, feature pipeline architecture, and data versioning. These are the foundational engineering skills that support every other component of a production ML system.

**Where to go from here:**
- **Model Serving** (./model-serving/COURSE.md) -- learn how the features your pipelines produce are consumed by serving infrastructure to deliver predictions at scale
- **Monitoring and Drift** (./monitoring-drift/COURSE.md) -- understand how to detect when the data flowing through your pipelines has shifted, and how that affects model performance
- **Experiment Tracking** (./experiment-tracking/COURSE.md) -- learn how to track the training datasets your pipelines produce and connect them to reproducible experiments
