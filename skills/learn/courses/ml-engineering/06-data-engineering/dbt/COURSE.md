# DBT (Data Build Tool)

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How DBT fits into the ELT paradigm as the transformation layer inside the data warehouse
- The role of sources, refs, and the DAG in managing transformation dependencies
- The four materialization strategies (view, table, incremental, ephemeral) and when to use each

**Apply:**
- Build DBT models for ML feature tables using Jinja macros, incremental materialization, and multi-window feature generation
- Write schema tests and custom data quality tests to validate ML feature tables

**Analyze:**
- Evaluate whether a feature table should use incremental vs full-table materialization based on data volume, freshness requirements, and late-arriving data patterns

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- BigQuery concepts covered in [Cloud Data Warehouses and BigQuery](../bigquery-warehouses/COURSE.md), including partitioning, clustering, and nested fields
- Advanced SQL patterns from [Advanced SQL for ML Engineers](../sql-advanced/COURSE.md), particularly CTEs and window functions

---

## Why This Matters

ML engineering roles commonly require DBT experience. DBT is how modern data teams transform raw data into clean, tested, documented tables inside the data warehouse. For ML engineers, DBT builds the feature tables that feed training pipelines and the clean datasets you analyze during exploration.

If BigQuery is where your data lives, DBT is how you organize and transform it.

---

## What DBT Is

DBT is a SQL-based transformation framework. You write SELECT statements, and DBT handles:
- Turning those SELECTs into tables or views in your warehouse
- Managing dependencies between transformations
- Running tests on the output data
- Generating documentation automatically

### DBT is the "T" in ELT

```
Extract  → Load     → Transform
(Fivetran)  (BigQuery)  (DBT)

Raw data lands in BigQuery first (E and L).
DBT transforms it inside BigQuery (T).
```

### What DBT is NOT

- Not an orchestrator (use Airflow for scheduling)
- Not an ETL tool (doesn't extract or load data)
- Not a database (runs against your existing warehouse)
- Not a Python framework (it's SQL + Jinja templating)

---

## Core Concepts

### Models

A DBT model is a SQL SELECT statement in a `.sql` file. DBT materializes it as a table or view.

```sql
-- models/staging/stg_orders.sql
-- This SELECT becomes a table (or view) called stg_orders

SELECT
    id as order_id,
    shop_id as merchant_id,
    CAST(total_price AS FLOAT64) as total_price,
    financial_status,
    fulfillment_status,
    TIMESTAMP(created_at) as created_at,
    TIMESTAMP(updated_at) as updated_at
FROM {{ source('raw', 'orders') }}
WHERE _fivetran_deleted = false
```

When you run `dbt run`, this becomes:
```sql
CREATE OR REPLACE TABLE `project.staging.stg_orders` AS
SELECT ... (the query above)
```

### Project Structure

```
dbt_project/
├── dbt_project.yml          # Project configuration
├── models/
│   ├── staging/             # Clean raw data (1:1 with source tables)
│   │   ├── stg_orders.sql
│   │   ├── stg_products.sql
│   │   ├── stg_merchants.sql
│   │   └── _stg_sources.yml  # Source definitions
│   ├── intermediate/        # Business logic transformations
│   │   ├── int_order_items_enriched.sql
│   │   └── int_merchant_daily_metrics.sql
│   ├── marts/               # Final tables for consumption
│   │   ├── dim_merchants.sql
│   │   ├── fct_orders.sql
│   │   └── ml_merchant_features.sql  # Feature tables for ML
│   └── schema.yml           # Tests and documentation
├── macros/                  # Reusable SQL templates
│   └── generate_schema_name.sql
├── tests/                   # Custom data tests
│   └── assert_positive_revenue.sql
└── seeds/                   # Static CSV data (country codes, etc.)
    └── country_codes.csv
```

---

## Sources and Refs

### Sources

Define where raw data comes from. This creates a dependency graph and enables freshness checks.

```yaml
# models/staging/_stg_sources.yml
version: 2

sources:
  - name: raw
    database: project
    schema: raw_ecommerce
    tables:
      - name: orders
        loaded_at_field: _fivetran_synced
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
      - name: products
      - name: merchants
      - name: order_items
```

Reference sources in SQL:
```sql
SELECT * FROM {{ source('raw', 'orders') }}
-- Compiles to: SELECT * FROM `project.raw_ecommerce.orders`
```

### Refs

Reference other DBT models. This is how DBT builds the dependency graph.

```sql
-- models/marts/fct_orders.sql
SELECT
    o.order_id,
    o.merchant_id,
    o.total_price,
    o.created_at,
    m.merchant_name,
    m.country,
    m.plan_type
FROM {{ ref('stg_orders') }} o
LEFT JOIN {{ ref('dim_merchants') }} m ON o.merchant_id = m.merchant_id
```

**Why refs matter:**
- DBT knows that `fct_orders` depends on `stg_orders` and `dim_merchants`
- It runs them in the correct order automatically
- If you change `stg_orders`, DBT knows to rebuild `fct_orders` too

### Dependency Graph (DAG)

```
source('raw', 'orders')     → stg_orders     → fct_orders     → ml_merchant_features
source('raw', 'products')   → stg_products   → fct_orders
source('raw', 'merchants')  → stg_merchants  → dim_merchants  → fct_orders
source('raw', 'order_items')→ stg_order_items → int_order_items_enriched → fct_orders
```

DBT generates this DAG automatically and provides a visual lineage graph.

---

### Check Your Understanding: Models, Sources, and Refs

**1. What happens if you reference another model using a hardcoded table name instead of `{{ ref('model_name') }}`?**

<details>
<summary>Answer</summary>

DBT will not know about the dependency between the two models. It cannot build the correct DAG, so it may run models in the wrong order, leading to stale or missing data. The ref() function is what enables DBT's automatic dependency management.
</details>

**2. In the staging/intermediate/marts project structure, what is the purpose of the intermediate layer?**

<details>
<summary>Answer</summary>

The intermediate layer contains business logic transformations that combine or enrich staged data but are not yet in their final consumable form. It sits between staging (1:1 cleaning of source tables) and marts (final tables consumed by dashboards, ML pipelines, etc.). This layer keeps the staging models simple and the mart models focused on consumption patterns.
</details>

---

## Tests

DBT tests validate your data at every layer. This is critical for ML — bad data in a feature table silently degrades model performance.

### Schema Tests (Built-in)

```yaml
# models/schema.yml
version: 2

models:
  - name: stg_orders
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
      - name: merchant_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_merchants')
              field: merchant_id
      - name: total_price
        tests:
          - not_null
      - name: financial_status
        tests:
          - accepted_values:
              values: ['paid', 'pending', 'refunded', 'voided', 'partially_refunded']

  - name: ml_merchant_features
    columns:
      - name: merchant_id
        tests:
          - unique
          - not_null
      - name: revenue_30d
        tests:
          - not_null
      - name: order_count_30d
        tests:
          - not_null
```

### Custom Tests

```sql
-- tests/assert_positive_revenue.sql
-- This test fails if any merchant has negative revenue

SELECT merchant_id, revenue_30d
FROM {{ ref('ml_merchant_features') }}
WHERE revenue_30d < 0
```

If this query returns any rows, the test fails.

### Data Quality Tests for ML

```sql
-- tests/assert_feature_coverage.sql
-- Ensure at least 95% of active merchants have features

WITH active_merchants AS (
    SELECT COUNT(DISTINCT merchant_id) as total
    FROM {{ ref('dim_merchants') }}
    WHERE status = 'active'
),
merchants_with_features AS (
    SELECT COUNT(DISTINCT merchant_id) as total
    FROM {{ ref('ml_merchant_features') }}
)
SELECT
    am.total as active_merchants,
    mf.total as merchants_with_features,
    SAFE_DIVIDE(mf.total, am.total) as coverage
FROM active_merchants am, merchants_with_features mf
WHERE SAFE_DIVIDE(mf.total, am.total) < 0.95  -- fail if coverage < 95%
```

```sql
-- tests/assert_no_future_dates.sql
-- Features should not reference future data (training-serving skew)

SELECT *
FROM {{ ref('ml_merchant_features') }}
WHERE snapshot_date > CURRENT_DATE()
```

### Running Tests

```bash
dbt test                          # Run all tests
dbt test --select stg_orders      # Test one model
dbt test --select tag:ml          # Test all models tagged "ml"
```

---

### Check Your Understanding: Tests

**1. A custom DBT test query returns rows. Does that mean the test passed or failed?**

<details>
<summary>Answer</summary>

The test failed. DBT custom tests are "assertion queries" -- they select rows that violate the expected condition. If the query returns any rows, those rows represent violations, and the test fails. An empty result set means the test passes.
</details>

**2. Why is testing ML feature tables more critical than testing regular analytics tables?**

<details>
<summary>Answer</summary>

Bad data in a feature table silently degrades model performance. Unlike a dashboard where a wrong number might be visually noticed, a feature with unexpected NULLs, negative values, or stale data feeds directly into model training and inference. The model will learn from corrupted features without raising errors, producing worse predictions with no obvious signal that something is wrong.
</details>

---

## Materializations

How DBT turns your SELECT into a database object.

### View

```sql
-- Creates a view (no data stored, query runs on access)
{{ config(materialized='view') }}

SELECT * FROM {{ source('raw', 'orders') }}
WHERE _fivetran_deleted = false
```

**Use when:** Data is small, query is fast, you want zero storage cost.

### Table

```sql
-- Creates a table (data stored, fast to query)
{{ config(materialized='table') }}

SELECT merchant_id, COUNT(*) as order_count
FROM {{ ref('stg_orders') }}
GROUP BY merchant_id
```

**Use when:** Query is expensive, table is queried frequently.

### Incremental

```sql
-- Only processes new/changed rows. Dramatically faster for large tables.
{{ config(
    materialized='incremental',
    unique_key='order_id',
    partition_by={
        "field": "created_at",
        "data_type": "timestamp",
        "granularity": "day"
    }
) }}

SELECT
    order_id,
    merchant_id,
    total_price,
    created_at
FROM {{ source('raw', 'orders') }}

{% if is_incremental() %}
    -- Only process rows newer than the latest in the existing table
    WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

**Use when:** Table is large, most data doesn't change, you want fast builds.
**Caution:** Late-arriving data can be missed. Use a lookback window:

```sql
{% if is_incremental() %}
    WHERE created_at > (
        SELECT DATE_SUB(MAX(created_at), INTERVAL 3 DAY) FROM {{ this }}
    )
{% endif %}
```

### Ephemeral

```sql
-- Not materialized at all. Inlined as a CTE in downstream models.
{{ config(materialized='ephemeral') }}

SELECT id as order_id, shop_id as merchant_id
FROM {{ source('raw', 'orders') }}
```

**Use when:** A reusable transformation that doesn't need its own table.

### Decision Framework

```
┌──────────────────────────────────────────────────────────────────┐
│ Is the model queried directly by users/dashboards/ML pipelines? │
│   NO  → ephemeral (just a reusable CTE)                        │
│   YES → Is the source data small (< 1GB)?                      │
│           YES → view (no storage cost, always fresh)            │
│           NO  → Does the data change frequently?               │
│                   YES → incremental (process only new rows)     │
│                   NO  → table (full rebuild is fine)            │
└──────────────────────────────────────────────────────────────────┘
```

---

### Check Your Understanding: Materializations

**1. An incremental model processes only new rows using `WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})`. What happens if an order arrives 2 days late?**

<details>
<summary>Answer</summary>

The late-arriving row is missed entirely. The incremental filter has already advanced past that timestamp. To handle late-arriving data, use a lookback window: `WHERE created_at > (SELECT DATE_SUB(MAX(created_at), INTERVAL 3 DAY) FROM {{ this }})`. This re-processes the last 3 days on each run, catching late arrivals at the cost of some reprocessing.
</details>

**2. When would you choose ephemeral materialization over a view?**

<details>
<summary>Answer</summary>

Use ephemeral when the transformation is only used as an intermediate step by other DBT models and never needs to be queried directly. Ephemeral models are inlined as CTEs into downstream models, so they create no database object. Use a view when the model needs to be queried directly by users, dashboards, or external tools.
</details>

---

## Macros and Jinja

DBT uses Jinja templating to make SQL reusable and dynamic.

### Basic Jinja

```sql
-- Variable substitution
{% set lookback_days = 30 %}

SELECT *
FROM {{ ref('stg_orders') }}
WHERE created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL {{ lookback_days }} DAY)
```

### Macros (Reusable Functions)

```sql
-- macros/rolling_aggregate.sql
{% macro rolling_aggregate(column, partition_col, order_col, window_size) %}
    SUM({{ column }}) OVER (
        PARTITION BY {{ partition_col }}
        ORDER BY {{ order_col }}
        ROWS BETWEEN {{ window_size - 1 }} PRECEDING AND CURRENT ROW
    )
{% endmacro %}
```

```sql
-- Usage in a model
SELECT
    merchant_id,
    order_date,
    revenue,
    {{ rolling_aggregate('revenue', 'merchant_id', 'order_date', 7) }} as revenue_7d,
    {{ rolling_aggregate('revenue', 'merchant_id', 'order_date', 30) }} as revenue_30d,
    {{ rolling_aggregate('order_count', 'merchant_id', 'order_date', 7) }} as orders_7d
FROM {{ ref('int_merchant_daily_metrics') }}
```

### Loops

```sql
-- Generate multiple rolling window features dynamically
{% set windows = [7, 14, 30, 60, 90] %}

SELECT
    merchant_id,
    snapshot_date,
    {% for w in windows %}
    SUM(order_count) OVER (
        PARTITION BY merchant_id ORDER BY snapshot_date
        ROWS BETWEEN {{ w - 1 }} PRECEDING AND CURRENT ROW
    ) as order_count_{{ w }}d,
    SUM(revenue) OVER (
        PARTITION BY merchant_id ORDER BY snapshot_date
        ROWS BETWEEN {{ w - 1 }} PRECEDING AND CURRENT ROW
    ) as revenue_{{ w }}d{{ "," if not loop.last }}
    {% endfor %}
FROM {{ ref('int_merchant_daily_metrics') }}
```

This generates features for all 5 windows without writing repetitive SQL.

---

## Documentation

DBT auto-generates documentation from YAML descriptions.

```yaml
# models/schema.yml
version: 2

models:
  - name: ml_merchant_features
    description: >
      Daily feature table for merchant churn prediction model.
      One row per merchant per snapshot_date. Features include
      rolling order counts, revenue, product metrics, and
      engagement signals. Used by the merchant_churn_v3 model.
    columns:
      - name: merchant_id
        description: "Unique identifier for the merchant"
      - name: snapshot_date
        description: "Date the features were computed for (point-in-time)"
      - name: revenue_30d
        description: "Total revenue in the 30 days prior to snapshot_date"
      - name: order_count_7d
        description: "Number of orders in the 7 days prior to snapshot_date"
      - name: days_since_last_order
        description: "Days between snapshot_date and the merchant's most recent order"
      - name: product_count
        description: "Number of active products as of snapshot_date"
```

```bash
# Generate and serve documentation
dbt docs generate
dbt docs serve  # Opens a browser with interactive documentation + lineage graph
```

---

## DBT for ML Feature Engineering

This is where DBT becomes critical for ML engineers.

### Feature Table Pattern

```sql
-- models/marts/ml_merchant_features.sql
{{ config(
    materialized='incremental',
    unique_key=['merchant_id', 'snapshot_date'],
    partition_by={
        "field": "snapshot_date",
        "data_type": "date",
        "granularity": "day"
    },
    cluster_by=['merchant_id'],
    tags=['ml', 'features']
) }}

WITH daily_orders AS (
    SELECT
        merchant_id,
        DATE(created_at) as order_date,
        COUNT(*) as order_count,
        SUM(total_price) as revenue,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM {{ ref('stg_orders') }}
    GROUP BY merchant_id, DATE(created_at)
),

date_spine AS (
    SELECT date as snapshot_date
    FROM UNNEST(GENERATE_DATE_ARRAY(
        {% if is_incremental() %}
            (SELECT DATE_SUB(MAX(snapshot_date), INTERVAL 3 DAY) FROM {{ this }})
        {% else %}
            '2023-01-01'
        {% endif %},
        CURRENT_DATE()
    )) as date
),

merchant_list AS (
    SELECT DISTINCT merchant_id FROM {{ ref('dim_merchants') }}
    WHERE status = 'active'
),

base AS (
    SELECT ds.snapshot_date, ml.merchant_id
    FROM date_spine ds
    CROSS JOIN merchant_list ml
),

features AS (
    SELECT
        b.merchant_id,
        b.snapshot_date,

        -- Order features (multiple windows)
        {% for w in [7, 14, 30, 60, 90] %}
        COALESCE(SUM(CASE
            WHEN d.order_date BETWEEN DATE_SUB(b.snapshot_date, INTERVAL {{ w }} DAY)
                                  AND b.snapshot_date
            THEN d.order_count END), 0) as order_count_{{ w }}d,
        COALESCE(SUM(CASE
            WHEN d.order_date BETWEEN DATE_SUB(b.snapshot_date, INTERVAL {{ w }} DAY)
                                  AND b.snapshot_date
            THEN d.revenue END), 0) as revenue_{{ w }}d,
        {% endfor %}

        -- Recency
        DATE_DIFF(b.snapshot_date, MAX(d.order_date), DAY) as days_since_last_order,

        -- Product features
        p.active_products,
        p.avg_product_price

    FROM base b
    LEFT JOIN daily_orders d ON b.merchant_id = d.merchant_id
                            AND d.order_date <= b.snapshot_date
    LEFT JOIN {{ ref('int_merchant_product_stats') }} p
        ON b.merchant_id = p.merchant_id
    GROUP BY b.merchant_id, b.snapshot_date, p.active_products, p.avg_product_price
)

SELECT
    *,
    -- Derived ratio features
    SAFE_DIVIDE(revenue_7d, revenue_30d) as revenue_7d_30d_ratio,
    SAFE_DIVIDE(order_count_7d, order_count_30d) as order_velocity_ratio
FROM features
```

### Testing Feature Tables

```yaml
# models/schema.yml
models:
  - name: ml_merchant_features
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - merchant_id
            - snapshot_date
    columns:
      - name: order_count_30d
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 100000
      - name: revenue_30d
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
      - name: revenue_7d_30d_ratio
        tests:
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 10  # ratio shouldn't exceed 10x
```

---

## DBT + BigQuery: Common Patterns

### Partition Management

```sql
-- BigQuery-specific materialization config
{{ config(
    materialized='incremental',
    partition_by={
        "field": "snapshot_date",
        "data_type": "date",
        "granularity": "day"
    },
    cluster_by=['merchant_id'],
    require_partition_filter=true  -- Force callers to filter on partition
) }}
```

### Using BigQuery-Specific Functions

```sql
-- BigQuery STRUCT and ARRAY in DBT
SELECT
    merchant_id,
    STRUCT(
        order_count_7d,
        order_count_30d,
        revenue_7d,
        revenue_30d
    ) as order_features,
    ARRAY_AGG(STRUCT(product_id, product_name) ORDER BY revenue DESC LIMIT 10)
        as top_products
FROM {{ ref('int_merchant_features_base') }}
GROUP BY merchant_id, order_count_7d, order_count_30d, revenue_7d, revenue_30d
```

### Scheduled Runs with Airflow

```python
# Airflow DAG to run DBT daily
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG('dbt_daily', schedule_interval='@daily', start_date=datetime(2024, 1, 1)) as dag:

    dbt_run = BashOperator(
        task_id='dbt_run',
        bash_command='cd /opt/dbt && dbt run --select tag:daily',
    )

    dbt_test = BashOperator(
        task_id='dbt_test',
        bash_command='cd /opt/dbt && dbt test --select tag:daily',
    )

    dbt_run >> dbt_test
```

---

## DBT Commands Reference

```bash
# Run all models
dbt run

# Run specific models
dbt run --select stg_orders          # One model
dbt run --select staging.*           # All models in staging/
dbt run --select tag:ml              # All models tagged "ml"
dbt run --select ml_merchant_features+  # Model and all downstream

# Test
dbt test                              # All tests
dbt test --select stg_orders          # Tests for one model

# Documentation
dbt docs generate                     # Generate docs
dbt docs serve                        # Serve docs locally

# Freshness
dbt source freshness                  # Check source data freshness

# Full rebuild (ignore incremental)
dbt run --full-refresh --select ml_merchant_features

# Compile (see generated SQL without running)
dbt compile --select ml_merchant_features
```

---

## Practice Interview Questions

1. "Explain the difference between a DBT model materialized as a table vs incremental. When would you use each for ML feature tables?"
2. "How would you use DBT to build a feature table that feeds both training (historical) and serving (current) for an ML model?"
3. "A DBT test fails on your feature table. The test checks that revenue_30d is never negative. How do you debug this?"
4. "How does DBT fit into the ML pipeline alongside Airflow and BigQuery?"
5. "Write a DBT model that computes 7-day and 30-day rolling revenue per merchant."

---

## Key Takeaways

1. DBT is the "T" in ELT. It transforms data inside the warehouse using SQL.
2. Models are just SELECT statements. DBT handles materialization, dependencies, and testing.
3. Use `ref()` to reference other models and `source()` to reference raw tables. This builds the DAG.
4. Incremental materialization is essential for large feature tables — only process new data.
5. Tests are mandatory for ML feature tables. Schema tests + custom data quality tests.
6. Jinja macros eliminate repetitive SQL. Use loops for multi-window features.
7. DBT + BigQuery + Airflow is the modern ML data stack. DBT transforms, BigQuery stores, Airflow orchestrates.

---

## Common Pitfalls

**1. Incremental models without a lookback window.** Late-arriving data is common in production (events delayed by network issues, timezone mismatches, etc.). Always include a lookback window (e.g., 3 days) in your incremental filter, and ensure you use `unique_key` so re-processed rows are upserted rather than duplicated.

**2. Not running `dbt test` after `dbt run`.** Models can build successfully but contain bad data. Always pair `dbt run` with `dbt test` in your pipeline. In Airflow, chain them: `dbt_run >> dbt_test`.

**3. Using hardcoded project/dataset names in SQL.** This breaks when deploying to different environments (dev, staging, production). Use `{{ source() }}` and `{{ ref() }}` exclusively, and configure target schemas in `dbt_project.yml` or profiles.

**4. Over-using table materialization for large datasets.** Full table rebuilds rewrite all data on every run. For tables with billions of rows, this is slow and expensive. Use incremental materialization for any table where only recent data changes.

---

## Hands-On Exercises

### Exercise: Build a Feature Model with Jinja Loops

Write a DBT model called `ml_merchant_features` that computes rolling features for each merchant. Use a Jinja loop to generate features for 7, 14, 30, and 90-day windows. Each window should include order_count, revenue, and unique_customers. Configure the model as incremental with partition by snapshot_date and clustering by merchant_id.

### Exercise: Write Custom Data Quality Tests

Write three custom DBT tests for the feature model above:
1. A test that fails if any merchant has negative revenue in any window
2. A test that fails if the 7-day revenue exceeds the 30-day revenue for any merchant (logically impossible)
3. A test that fails if feature coverage drops below 90% of active merchants

---

## Summary

This lesson covered DBT's role as the transformation layer in the modern data stack: models as SELECT statements, the DAG built from refs and sources, four materialization strategies, schema and custom testing, Jinja macros for reusable SQL, documentation generation, and the specific patterns for building ML feature tables with incremental materialization and multi-window features.

### What's Next

Continue to [Pipeline Orchestration and Streaming](../streaming-batch/COURSE.md) to learn how Airflow and Kafka orchestrate the batch and streaming pipelines that run your DBT models, train your ML models, and serve predictions in production.
