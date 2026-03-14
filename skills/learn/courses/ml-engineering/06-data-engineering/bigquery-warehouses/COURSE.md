# Cloud Data Warehouses and BigQuery

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How BigQuery's serverless, columnar architecture differs from traditional OLTP databases
- The role of partitioning, clustering, and nested fields (STRUCT/ARRAY) in query performance and cost
- When to use BQML vs Python-based training and how BigQuery fits into the broader warehouse landscape

**Apply:**
- Design partitioned and clustered BigQuery tables optimized for ML feature queries
- Export BigQuery data to GCS for ML training pipelines and write predictions back via streaming inserts

**Analyze:**
- Diagnose and reduce BigQuery costs by evaluating query patterns, column selection, and materialization strategies

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- Advanced SQL concepts covered in [Advanced SQL for ML Engineers](../sql-advanced/COURSE.md), including window functions, CTEs, and aggregation patterns

---

## Why This Matters

Production ML roles commonly require BigQuery experience. BigQuery is where large-scale platforms store and process the data that feeds every ML model — orders, products, merchants, sessions, payments. Understanding how BigQuery works under the hood lets you write faster queries, lower costs, and build efficient data pipelines.

---

## What Is a Data Warehouse

A data warehouse is a database optimized for **analytical queries** (aggregations, joins over large datasets) rather than **transactional operations** (insert one row, read one row).

### Warehouse vs Transactional Database

| Property | OLTP (PostgreSQL, MySQL) | OLAP (BigQuery, Snowflake) |
|----------|--------------------------|----------------------------|
| Workload | Many small reads/writes | Few large analytical queries |
| Schema | Normalized (3NF) | Denormalized (star/snowflake schema) |
| Data volume | GB to low TB | TB to PB |
| Query pattern | Lookup by primary key | Full table scans, aggregations |
| Latency | Milliseconds | Seconds to minutes |
| Concurrency | Thousands of connections | Tens to hundreds of queries |
| Storage format | Row-oriented | Column-oriented |

### Column-Oriented Storage

This is the key insight. Traditional databases store data row by row. Warehouses store data column by column.

```
Row-oriented (PostgreSQL):
  Row 1: [order_id=1, merchant_id=M001, total_price=45.00, created_at=2024-01-01]
  Row 2: [order_id=2, merchant_id=M002, total_price=120.00, created_at=2024-01-01]
  Row 3: [order_id=3, merchant_id=M001, total_price=30.00, created_at=2024-01-02]

Column-oriented (BigQuery):
  order_id:     [1, 2, 3, ...]
  merchant_id:  [M001, M002, M001, ...]
  total_price:  [45.00, 120.00, 30.00, ...]
  created_at:   [2024-01-01, 2024-01-01, 2024-01-02, ...]
```

**Why this matters for ML:**
- `SELECT SUM(total_price) FROM orders` only reads the `total_price` column, not the entire table
- Compression is much better within a column (all values same type)
- Aggregations are extremely fast (columnar data fits in CPU cache)

---

## BigQuery Architecture

### Serverless

No clusters to manage. You submit a query, BigQuery allocates compute, runs it, and releases resources.

```
Your SQL query → BigQuery API → Dremel execution engine → Results
                                 (auto-scaled workers)
```

**Implications:**
- No infrastructure to provision or manage
- Pay per query (on-demand) or flat rate (reserved slots)
- Queries on 1TB and 1PB use the same API — just cost differently

### Storage and Compute Separation

BigQuery separates storage (Colossus, Google's distributed file system) from compute (Dremel engine). This means:
- Storage is cheap ($0.02/GB/month)
- Compute scales independently of data size
- You can store everything and query what you need

### Slots

A "slot" is a unit of compute in BigQuery. Each query is assigned slots.
- On-demand: up to 2,000 slots per project (shared pool)
- Flat-rate: you buy dedicated slots (e.g., 500 slots for $10K/month)

---

## BigQuery SQL Differences

BigQuery uses Standard SQL with some extensions. Key differences from PostgreSQL/MySQL:

### Data Types

```sql
-- BigQuery-specific types
INT64      -- (not INTEGER in standard SQL, though INTEGER works as alias)
FLOAT64    -- (not DOUBLE PRECISION)
BOOL       -- (not BOOLEAN, though both work)
STRING     -- (not VARCHAR)
BYTES
DATE, TIME, DATETIME, TIMESTAMP
STRUCT     -- nested record
ARRAY      -- repeated field

-- No SERIAL/AUTO_INCREMENT — use GENERATE_UUID() or row numbers
```

### Safe Operations

```sql
-- BigQuery has null-safe versions of common operations
SAFE_DIVIDE(a, b)           -- returns NULL instead of error on division by zero
SAFE_CAST(x AS INT64)       -- returns NULL instead of error on invalid cast
IFNULL(x, default_value)    -- coalesce with exactly 2 args
COALESCE(a, b, c, d)        -- first non-null value
```

### Date Functions

```sql
-- Date arithmetic in BigQuery
DATE_ADD(date, INTERVAL 7 DAY)
DATE_SUB(date, INTERVAL 30 DAY)
DATE_DIFF(date1, date2, DAY)      -- returns integer
DATE_TRUNC(date, MONTH)           -- truncate to month start
FORMAT_DATE('%Y-%m', date)        -- format as string
EXTRACT(MONTH FROM date)          -- extract component

-- Generate a date sequence
SELECT date
FROM UNNEST(GENERATE_DATE_ARRAY('2024-01-01', '2024-12-31')) as date;
```

### String Functions

```sql
-- BigQuery string operations used in feature engineering
REGEXP_EXTRACT(email, r'@(.+)')           -- extract domain from email
REGEXP_CONTAINS(url, r'example\.com')     -- regex match
SPLIT(tags, ',')                          -- returns ARRAY
ARRAY_LENGTH(SPLIT(tags, ','))            -- count tags
```

---

### Check Your Understanding: Architecture and SQL

**1. Why does BigQuery scan less data when you SELECT specific columns instead of using SELECT *?**

<details>
<summary>Answer</summary>

BigQuery uses columnar storage, meaning each column is stored separately on disk. When you SELECT specific columns, BigQuery only reads those columns' data. With SELECT *, it reads every column. Since BigQuery charges per byte scanned, selecting fewer columns is both faster and cheaper.
</details>

**2. What is the difference between SAFE_DIVIDE(a, b) and a / b in BigQuery?**

<details>
<summary>Answer</summary>

SAFE_DIVIDE returns NULL when the denominator is zero, while a / b throws a division-by-zero error. SAFE_DIVIDE is preferred in feature engineering pipelines where zero denominators are common (e.g., a merchant with zero orders in a time window).
</details>

---

## Partitioning and Clustering

These are the two most important performance optimizations in BigQuery.

### Partitioning

Divide a table into segments based on a column value. Queries that filter on the partition column only scan relevant partitions.

```sql
-- Create a partitioned table
CREATE TABLE `project.dataset.orders`
PARTITION BY DATE(created_at)
AS
SELECT * FROM `project.raw.orders`;

-- This query scans only 1 day of data (not the entire table)
SELECT * FROM `project.dataset.orders`
WHERE DATE(created_at) = '2024-01-15';
-- Cost: ~1/365th of scanning the full year
```

**Partition types:**
- `DATE`, `DATETIME`, `TIMESTAMP` column (most common)
- Integer range partitioning
- Ingestion time (`_PARTITIONDATE`)

**Limits:**
- Maximum 4,000 partitions per table
- One partition column per table

### Clustering

Sort data within each partition by up to 4 columns. Queries that filter on clustered columns skip irrelevant data blocks.

```sql
-- Partition by date, cluster by merchant_id and product_category
CREATE TABLE `project.dataset.orders`
PARTITION BY DATE(created_at)
CLUSTER BY merchant_id, product_category
AS
SELECT * FROM `project.raw.orders`;

-- This query benefits from both partition pruning AND cluster filtering
SELECT * FROM `project.dataset.orders`
WHERE DATE(created_at) = '2024-01-15'
AND merchant_id = 'M001';
-- Scans a tiny fraction of the table
```

**When to use clustering:**
- Columns frequently used in WHERE, JOIN, or GROUP BY
- High-cardinality columns (merchant_id, product_id)
- Order matters: put the most-filtered column first

### Partitioning + Clustering Decision Matrix

| Scenario | Recommendation |
|----------|---------------|
| Time-series data (events, orders) | Partition by date, cluster by entity_id |
| Entity-centric queries (merchant features) | Partition by date, cluster by merchant_id |
| Large joins | Cluster both tables on the join key |
| Infrequent queries on historical data | Partition by date (only scan recent) |

---

### Check Your Understanding: Partitioning and Clustering

**1. A table has 10 years of data partitioned by date. A query filters on `WHERE merchant_id = 'M001'` but does not filter on date. Does partitioning help?**

<details>
<summary>Answer</summary>

No. Without a filter on the partition column (date), BigQuery scans all partitions. The query reads the entire table. To benefit from partitioning, you must include a filter on the partition column. Clustering on merchant_id would help this query, however.
</details>

**2. You have a table clustered by (merchant_id, product_category). A query filters only on product_category. Does clustering help?**

<details>
<summary>Answer</summary>

Minimally or not at all. Clustering sorts data by columns in order, so the first clustering column (merchant_id) is the primary sort key. Filtering only on the second column (product_category) cannot skip data blocks effectively because product_category values are interleaved across merchant_id groups. Put the most-filtered column first in the clustering definition.
</details>

---

## Nested and Repeated Fields (STRUCT, ARRAY)

BigQuery supports nested data natively. This is common in event data and denormalized schemas.

### STRUCT (Nested Record)

```sql
-- Define a table with nested fields
CREATE TABLE `project.dataset.orders` (
    order_id STRING,
    merchant_id STRING,
    total_price FLOAT64,
    shipping_address STRUCT<
        street STRING,
        city STRING,
        state STRING,
        country STRING,
        zip STRING
    >,
    created_at TIMESTAMP
);

-- Query nested fields with dot notation
SELECT
    order_id,
    shipping_address.city,
    shipping_address.country
FROM `project.dataset.orders`
WHERE shipping_address.country = 'US';
```

### ARRAY (Repeated Field)

```sql
-- Orders with line items as an array
CREATE TABLE `project.dataset.orders_with_items` (
    order_id STRING,
    merchant_id STRING,
    line_items ARRAY<STRUCT<
        product_id STRING,
        quantity INT64,
        price FLOAT64
    >>
);

-- Query arrays with UNNEST
SELECT
    o.order_id,
    o.merchant_id,
    item.product_id,
    item.quantity,
    item.price
FROM `project.dataset.orders_with_items` o,
UNNEST(o.line_items) as item;

-- Aggregate within arrays
SELECT
    order_id,
    ARRAY_LENGTH(line_items) as item_count,
    (SELECT SUM(i.price * i.quantity) FROM UNNEST(line_items) i) as computed_total
FROM `project.dataset.orders_with_items`;
```

### Why Nested Fields Matter for ML

Denormalized tables with nested fields avoid expensive joins:

```sql
-- Instead of joining orders + order_items + products (3 tables, 2 joins):
-- Store denormalized with nested arrays (1 table, 0 joins):

SELECT
    merchant_id,
    COUNT(*) as order_count,
    SUM(ARRAY_LENGTH(line_items)) as total_items,
    AVG((SELECT AVG(i.price) FROM UNNEST(line_items) i)) as avg_item_price
FROM `project.dataset.orders_denormalized`
GROUP BY merchant_id;
-- Much faster and cheaper than joining 3 tables
```

---

## Cost Optimization

### Pricing Models

| Model | How It Works | Best For |
|-------|-------------|----------|
| On-demand | $6.25 per TB scanned | Exploratory queries, low volume |
| Flat-rate | $2,400/month per 100 slots | Predictable workloads, high volume |
| Editions | Autoscaling slots with commitment | Variable workloads |

### Cost Reduction Strategies

```sql
-- 1. SELECT only needed columns (avoid SELECT *)
-- BAD: scans all columns
SELECT * FROM `project.dataset.orders` WHERE merchant_id = 'M001';
-- GOOD: scans only 2 columns
SELECT order_id, total_price FROM `project.dataset.orders` WHERE merchant_id = 'M001';

-- 2. Use partitioned tables and filter on partition column
-- BAD: scans entire table
SELECT * FROM orders WHERE merchant_id = 'M001';
-- GOOD: scans only one partition
SELECT * FROM orders WHERE DATE(created_at) = '2024-01-15' AND merchant_id = 'M001';

-- 3. Preview with LIMIT (still scans full data in BigQuery!)
-- NOTE: LIMIT doesn't reduce data scanned in BigQuery
-- Use table preview or partition filters instead

-- 4. Use approximate functions
SELECT APPROX_COUNT_DISTINCT(customer_id) FROM orders;  -- 10x cheaper than exact

-- 5. Materialize intermediate results
CREATE TABLE `project.dataset.merchant_features` AS
SELECT merchant_id, ... FROM expensive_query;
-- Then query the materialized table (much cheaper than re-running)
```

### Estimating Query Cost

```sql
-- Dry run: estimate bytes scanned without running the query
-- In BigQuery UI: click "Validator" to see estimated bytes
-- In CLI:
-- bq query --dry_run "SELECT * FROM orders WHERE ..."

-- Rule of thumb: $6.25 per TB scanned (on-demand pricing)
-- 1 TB table, SELECT 3 of 20 columns → ~150 GB scanned → ~$0.94
```

---

### Check Your Understanding: Nested Fields and Cost

**1. Why are denormalized tables with nested ARRAY fields often preferred over normalized tables with joins in BigQuery?**

<details>
<summary>Answer</summary>

Joins in BigQuery require shuffling data across nodes, which is expensive at scale. Denormalized tables with nested ARRAY fields keep related data in the same row, eliminating the need for joins entirely. Queries are faster, cheaper, and simpler to write. The UNNEST function provides access to nested data when needed.
</details>

**2. Does adding LIMIT 100 to a BigQuery query reduce the amount of data scanned and the cost?**

<details>
<summary>Answer</summary>

No. In BigQuery, LIMIT does not reduce data scanned. BigQuery scans all matching data first, then returns only the requested number of rows. To reduce cost, use partition filters, column selection, and clustering -- not LIMIT.
</details>

---

## BigQuery ML (BQML)

Train models directly in SQL. Useful for quick prototyping and simple models.

### When to Use BQML

| Use BQML When | Use Python/sklearn When |
|---------------|------------------------|
| Quick prototype | Production model |
| Simple model (logistic regression, boosted trees) | Complex pipeline |
| Data already in BigQuery | Custom preprocessing needed |
| Non-ML-engineer running the query | Full control over training |

### Example: Churn Prediction in BQML

```sql
-- Step 1: Create training data
CREATE OR REPLACE TABLE `project.ml.churn_training` AS
SELECT
    merchant_id,
    orders_30d,
    revenue_30d,
    days_since_last_order,
    product_count,
    CASE WHEN days_since_last_order > 90 THEN 1 ELSE 0 END as churned  -- label
FROM `project.features.merchant_features`
WHERE snapshot_date = '2024-01-01';

-- Step 2: Train model
CREATE OR REPLACE MODEL `project.ml.churn_model`
OPTIONS(
    model_type='BOOSTED_TREE_CLASSIFIER',
    input_label_cols=['churned'],
    max_iterations=50,
    learn_rate=0.1,
    data_split_method='RANDOM',
    data_split_eval_fraction=0.2
) AS
SELECT * FROM `project.ml.churn_training`;

-- Step 3: Evaluate
SELECT * FROM ML.EVALUATE(MODEL `project.ml.churn_model`);
-- Returns: precision, recall, accuracy, f1_score, log_loss, roc_auc

-- Step 4: Predict
SELECT
    merchant_id,
    predicted_churned,
    predicted_churned_probs[OFFSET(1)].prob as churn_probability
FROM ML.PREDICT(
    MODEL `project.ml.churn_model`,
    (SELECT * FROM `project.features.merchant_features` WHERE snapshot_date = CURRENT_DATE())
);

-- Step 5: Feature importance
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `project.ml.churn_model`);
```

### BQML Model Types

| Model Type | SQL Name | Use Case |
|-----------|----------|----------|
| Logistic regression | `LOGISTIC_REG` | Binary classification |
| Linear regression | `LINEAR_REG` | Regression |
| Boosted trees | `BOOSTED_TREE_CLASSIFIER/REGRESSOR` | Tabular classification/regression |
| Random forest | `RANDOM_FOREST_CLASSIFIER/REGRESSOR` | Ensemble |
| K-means | `KMEANS` | Clustering |
| Matrix factorization | `MATRIX_FACTORIZATION` | Recommendations |
| Time series | `ARIMA_PLUS` | Forecasting |
| Deep neural network | `DNN_CLASSIFIER/REGRESSOR` | Complex patterns |
| Imported TensorFlow | `TENSORFLOW` | Custom models |

---

## Alternatives: Brief Comparison

### Snowflake

```
Architecture: Shared-nothing, separate compute clusters ("warehouses")
Strengths: Multi-cloud (AWS, Azure, GCP), data sharing, Snowpark (Python UDFs)
Pricing: Credit-based (compute) + storage
When to choose: Multi-cloud requirements, need data marketplace
```

### Amazon Redshift

```
Architecture: Cluster-based (provisioned), recently added serverless option
Strengths: AWS ecosystem integration, mature
Pricing: Node-based (provisioned) or per-query (serverless)
When to choose: Deep AWS investment
```

### Databricks (Lakehouse)

```
Architecture: Unified analytics platform (Spark-based), Delta Lake storage
Strengths: Combines warehouse + data lake, excellent ML/AI integration
Pricing: DBU-based (compute units)
When to choose: Heavy ML workloads, need Spark, want unified platform
```

### Quick Comparison

| Feature | BigQuery | Snowflake | Redshift | Databricks |
|---------|----------|-----------|----------|------------|
| Serverless | Yes | Yes | Optional | Yes |
| ML integration | BQML | Snowpark ML | SageMaker | Native (MLflow) |
| Streaming | Built-in | Snowpipe | Kinesis | Structured Streaming |
| Cost model | Per-TB scanned | Credits | Per-node/query | DBUs |
| Nested data | Native (STRUCT/ARRAY) | VARIANT (semi-structured) | SUPER type | Native (Spark) |

---

## Connecting BigQuery to ML Pipelines

### Export Patterns

```python
# Python: Read BigQuery into pandas for training
from google.cloud import bigquery

client = bigquery.Client()

query = """
SELECT * FROM `project.features.merchant_features`
WHERE snapshot_date >= '2024-01-01'
"""

# For small datasets (< 1GB): direct to pandas
df = client.query(query).to_dataframe()

# For large datasets: export to GCS, then read
export_query = """
EXPORT DATA OPTIONS(
    uri='gs://bucket/exports/features_*.parquet',
    format='PARQUET',
    overwrite=true
) AS
SELECT * FROM `project.features.merchant_features`
WHERE snapshot_date >= '2024-01-01'
"""
client.query(export_query).result()

# Then read from GCS
import pandas as pd
df = pd.read_parquet('gs://bucket/exports/features_*.parquet')
```

### Streaming Inserts

```python
# Write predictions back to BigQuery in real-time
from google.cloud import bigquery

client = bigquery.Client()
table_id = "project.predictions.merchant_churn"

rows_to_insert = [
    {"merchant_id": "M001", "churn_probability": 0.85, "prediction_date": "2024-01-15"},
    {"merchant_id": "M002", "churn_probability": 0.12, "prediction_date": "2024-01-15"},
]

errors = client.insert_rows_json(table_id, rows_to_insert)
if errors:
    print(f"Insert errors: {errors}")
```

### BigQuery in Airflow

```python
# Airflow DAG that computes features in BigQuery
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

compute_features = BigQueryInsertJobOperator(
    task_id='compute_merchant_features',
    configuration={
        'query': {
            'query': open('sql/merchant_features.sql').read(),
            'useLegacySql': False,
            'destinationTable': {
                'projectId': 'my-project',
                'datasetId': 'features',
                'tableId': 'merchant_features${{ ds_nodash }}',
            },
            'writeDisposition': 'WRITE_TRUNCATE',  # Overwrite partition
        }
    },
)
```

---

## Practice Interview Questions

1. "You have a 10TB table of events. A query that joins it with a 500MB merchants table takes 5 minutes. How do you optimize it?"
2. "Explain the difference between partitioning and clustering in BigQuery. When would you use each?"
3. "Your BigQuery bill doubled this month. How do you investigate and reduce it?"
4. "When would you use BQML vs training a model in Python?"
5. "How would you design a BigQuery schema for e-commerce order data that supports both analytical queries and ML feature computation?"

---

## Key Takeaways

1. BigQuery is serverless and columnar. SELECT only the columns you need.
2. Partition by date, cluster by high-cardinality filter/join columns. This is the single most impactful optimization.
3. Nested fields (STRUCT/ARRAY) avoid expensive joins. Denormalize for analytics.
4. Cost = bytes scanned. Reduce bytes with column selection, partitioning, clustering, and materialization.
5. BQML is great for prototyping. Use Python for production models.
6. BigQuery's SAFE_DIVIDE, APPROX functions, and GENERATE_DATE_ARRAY are tools you will use daily.
7. Export to GCS (Parquet format) for large training datasets. Streaming insert for real-time predictions.

---

## Common Pitfalls

**1. Assuming LIMIT reduces cost.** Unlike traditional databases, BigQuery scans all matching data before applying LIMIT. A `SELECT * FROM large_table LIMIT 10` scans the entire table. Use partition filters and column selection to control cost.

**2. Forgetting that clustering order matters.** Clustering by (A, B, C) is most effective when queries filter on A, then A+B, then A+B+C. Filtering only on C provides little or no benefit from clustering. Choose the column order based on your most common query patterns.

**3. Using on-demand pricing for repetitive queries.** If the same expensive query runs daily (e.g., in an Airflow pipeline), the cost accumulates quickly. Materialize the results as a table or use scheduled queries to avoid re-scanning the same data repeatedly.

**4. Not using dry runs to estimate cost before executing.** In the BigQuery UI, the validator shows estimated bytes scanned. In the CLI, use `bq query --dry_run`. Always check before running queries on large tables in production.

---

## Hands-On Exercises

### Exercise: Design a Partitioned and Clustered Table

Given an `orders` table with columns (order_id, merchant_id, customer_id, product_category, total_price, created_at), write the CREATE TABLE statement that:
1. Partitions by date on created_at
2. Clusters by merchant_id and product_category
3. Includes a STRUCT for shipping_address (street, city, state, country, zip)

Then write two queries: one that benefits from both partitioning and clustering, and one that benefits only from partitioning.

### Exercise: Cost Estimation

You have a 5 TB table with 20 columns, each averaging 250 GB. At on-demand pricing ($6.25/TB), estimate the cost of:
1. `SELECT * FROM table WHERE date = '2024-01-15'` (365 partitions, uniform data)
2. `SELECT merchant_id, total_price FROM table WHERE date = '2024-01-15'`
3. The same queries without partitioning

---

## Summary

This lesson covered BigQuery's architecture (serverless, columnar, storage/compute separation), the critical performance optimizations (partitioning and clustering), nested data types (STRUCT and ARRAY), cost management strategies, BigQuery ML for prototyping, comparisons with Snowflake/Redshift/Databricks, and patterns for connecting BigQuery to ML pipelines.

### What's Next

Continue to [DBT (Data Build Tool)](../dbt/COURSE.md) to learn how to organize and transform your BigQuery data using SQL-based models, automated testing, and dependency management -- the standard approach for building reliable ML feature tables.
