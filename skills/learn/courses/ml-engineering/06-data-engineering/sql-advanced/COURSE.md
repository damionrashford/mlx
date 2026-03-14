# Advanced SQL for ML Engineers

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How window functions (ROW_NUMBER, LAG, LEAD, running aggregates) enable feature engineering in SQL
- The differences between correlated subqueries, CTEs, and window-function-based approaches for performance
- Query optimization strategies including partition pruning, clustering, and approximate functions

**Apply:**
- Write sessionization, funnel analysis, and cohort analysis queries for e-commerce data
- Build rolling time-series features (7-day, 30-day windows) for ML training datasets

**Analyze:**
- Evaluate query execution plans to identify performance bottlenecks and choose the optimal approach for a given data pattern

---

## Prerequisites

This lesson is standalone and does not have strict prerequisites. You should be comfortable with basic SQL (SELECT, FROM, WHERE, JOIN, GROUP BY). Familiarity with a data warehouse environment like BigQuery is helpful but not required.

---

## Why This Matters

SQL is the universal language of data. In a large-scale e-commerce platform, the data warehouse (BigQuery) contains the raw material for every ML model: orders, products, merchants, sessions, payments. You will write SQL daily — to explore data, build features, validate pipelines, and debug models.

This lesson goes beyond SELECT-FROM-WHERE. It covers the patterns you'll actually use as an ML engineer working with e-commerce data at scale.

---

## Window Functions

Window functions compute a value for each row based on a "window" of related rows — without collapsing the result set like GROUP BY does.

### Syntax

```sql
function_name(...) OVER (
    PARTITION BY column    -- define the window groups
    ORDER BY column        -- define the order within each group
    ROWS/RANGE BETWEEN ... -- define the frame (optional)
)
```

### ROW_NUMBER, RANK, DENSE_RANK

Assign a position to each row within a partition.

```sql
-- For each merchant, rank their orders by total price (highest first)
SELECT
    merchant_id,
    order_id,
    total_price,
    ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY total_price DESC) as row_num,
    RANK()       OVER (PARTITION BY merchant_id ORDER BY total_price DESC) as rank,
    DENSE_RANK() OVER (PARTITION BY merchant_id ORDER BY total_price DESC) as dense_rank
FROM orders;
```

```
merchant_id | order_id | total_price | row_num | rank | dense_rank
------------|----------|-------------|---------|------|-----------
M001        | O100     | 500         | 1       | 1    | 1
M001        | O101     | 500         | 2       | 1    | 1          ← tie
M001        | O102     | 300         | 3       | 3    | 2          ← RANK skips, DENSE_RANK doesn't
M001        | O103     | 100         | 4       | 4    | 3
```

**ML use case:** Get the most recent order for each merchant (deduplication).

```sql
-- Get each merchant's most recent order
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY created_at DESC) as rn
    FROM orders
)
SELECT * FROM ranked WHERE rn = 1;
```

### LAG and LEAD

Access values from previous or next rows.

```sql
-- For each order, compute days since the merchant's previous order
SELECT
    merchant_id,
    order_id,
    created_at,
    LAG(created_at) OVER (PARTITION BY merchant_id ORDER BY created_at) as prev_order_at,
    DATE_DIFF(
        created_at,
        LAG(created_at) OVER (PARTITION BY merchant_id ORDER BY created_at),
        DAY
    ) as days_since_prev_order
FROM orders
ORDER BY merchant_id, created_at;
```

```
merchant_id | order_id | created_at | prev_order_at | days_since_prev_order
------------|----------|------------|---------------|----------------------
M001        | O100     | 2024-01-01 | NULL          | NULL                    ← first order
M001        | O101     | 2024-01-08 | 2024-01-01    | 7
M001        | O102     | 2024-01-20 | 2024-01-08    | 12
M001        | O103     | 2024-02-15 | 2024-01-20    | 26
```

**ML use case:** Feature engineering — inter-order interval is a strong churn predictor.

### Running Aggregates

```sql
-- Cumulative revenue per merchant over time
SELECT
    merchant_id,
    created_at,
    total_price,
    SUM(total_price) OVER (
        PARTITION BY merchant_id
        ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_revenue,
    AVG(total_price) OVER (
        PARTITION BY merchant_id
        ORDER BY created_at
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_7_order_avg,
    COUNT(*) OVER (
        PARTITION BY merchant_id
        ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_order_count
FROM orders;
```

### FIRST_VALUE, LAST_VALUE, NTH_VALUE

```sql
-- Compare each order to the merchant's first and most recent order
SELECT
    merchant_id,
    order_id,
    total_price,
    FIRST_VALUE(total_price) OVER (
        PARTITION BY merchant_id ORDER BY created_at
    ) as first_order_value,
    total_price - FIRST_VALUE(total_price) OVER (
        PARTITION BY merchant_id ORDER BY created_at
    ) as diff_from_first_order
FROM orders;
```

### PERCENT_RANK and NTILE

```sql
-- Assign merchants to deciles by revenue
SELECT
    merchant_id,
    total_revenue,
    NTILE(10) OVER (ORDER BY total_revenue) as revenue_decile,
    PERCENT_RANK() OVER (ORDER BY total_revenue) as revenue_percentile
FROM merchant_revenue;
```

**ML use case:** Creating bucketed features (merchant size tier = revenue decile).

---

### Check Your Understanding: Window Functions

**1. What is the difference between RANK() and DENSE_RANK() when there are tied values?**

<details>
<summary>Answer</summary>

RANK() skips positions after ties (e.g., 1, 1, 3), while DENSE_RANK() does not skip (e.g., 1, 1, 2). ROW_NUMBER() assigns unique sequential numbers regardless of ties.
</details>

**2. You need to compute the difference between each order's total and the previous order's total for the same merchant. Which window function would you use?**

<details>
<summary>Answer</summary>

Use LAG() to access the previous row's value, then subtract:
`total_price - LAG(total_price) OVER (PARTITION BY merchant_id ORDER BY created_at)`
</details>

**3. What does the frame clause `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` mean in a rolling aggregate?**

<details>
<summary>Answer</summary>

It defines a window of 7 rows total: the current row plus the 6 rows immediately before it. When combined with ORDER BY on a date column with one row per day, this computes a 7-day rolling aggregate.
</details>

---

## CTEs (Common Table Expressions)

CTEs make complex queries readable by breaking them into named steps.

### Basic CTE

```sql
WITH daily_orders AS (
    SELECT
        merchant_id,
        DATE(created_at) as order_date,
        COUNT(*) as order_count,
        SUM(total_price) as daily_revenue
    FROM orders
    GROUP BY merchant_id, DATE(created_at)
),
merchant_stats AS (
    SELECT
        merchant_id,
        AVG(order_count) as avg_daily_orders,
        AVG(daily_revenue) as avg_daily_revenue,
        STDDEV(daily_revenue) as std_daily_revenue,
        MAX(order_date) as last_order_date
    FROM daily_orders
    GROUP BY merchant_id
)
SELECT
    m.merchant_id,
    m.avg_daily_orders,
    m.avg_daily_revenue,
    m.std_daily_revenue,
    DATE_DIFF(CURRENT_DATE(), m.last_order_date, DAY) as days_since_last_order,
    CASE
        WHEN DATE_DIFF(CURRENT_DATE(), m.last_order_date, DAY) > 30 THEN 'at_risk'
        WHEN DATE_DIFF(CURRENT_DATE(), m.last_order_date, DAY) > 90 THEN 'churned'
        ELSE 'active'
    END as merchant_status
FROM merchant_stats m;
```

### Recursive CTEs

Traverse hierarchical data (product categories, referral chains).

```sql
-- Traverse a product category tree
WITH RECURSIVE category_tree AS (
    -- Base case: top-level categories
    SELECT id, name, parent_id, 0 as depth, CAST(name AS STRING) as path
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    -- Recursive case: children
    SELECT c.id, c.name, c.parent_id, ct.depth + 1,
           CONCAT(ct.path, ' > ', c.name) as path
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree ORDER BY path;
```

```
id  | name          | depth | path
----|---------------|-------|---------------------------
1   | Clothing      | 0     | Clothing
11  | Men's         | 1     | Clothing > Men's
111 | Shirts        | 2     | Clothing > Men's > Shirts
12  | Women's       | 1     | Clothing > Women's
```

---

## Subqueries

### Correlated Subqueries

A subquery that references the outer query. Runs once per row (can be slow).

```sql
-- Find merchants whose last order value was above their average
SELECT
    merchant_id,
    order_id,
    total_price
FROM orders o
WHERE created_at = (
    SELECT MAX(created_at)
    FROM orders
    WHERE merchant_id = o.merchant_id
)
AND total_price > (
    SELECT AVG(total_price)
    FROM orders
    WHERE merchant_id = o.merchant_id
);
```

**Performance note:** Correlated subqueries are often slow. Rewrite with window functions when possible:

```sql
-- Same query, faster (window function version)
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY created_at DESC) as rn,
        AVG(total_price) OVER (PARTITION BY merchant_id) as avg_price
    FROM orders
)
SELECT merchant_id, order_id, total_price
FROM ranked
WHERE rn = 1 AND total_price > avg_price;
```

### EXISTS vs IN

```sql
-- Find merchants who have at least one order over $1000
-- EXISTS: stops at first match (faster for large tables)
SELECT m.merchant_id, m.name
FROM merchants m
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.merchant_id = m.merchant_id
    AND o.total_price > 1000
);

-- IN: materializes the full subquery result (can be slower)
SELECT merchant_id, name
FROM merchants
WHERE merchant_id IN (
    SELECT merchant_id FROM orders WHERE total_price > 1000
);
```

**Rule of thumb:** Use EXISTS when the subquery table is large. Use IN when the subquery returns a small, distinct list.

---

## Aggregation Patterns

### GROUP BY with HAVING

```sql
-- Find merchants with declining revenue (this month vs last month)
SELECT
    merchant_id,
    SUM(CASE WHEN DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
             THEN total_price ELSE 0 END) as revenue_30d,
    SUM(CASE WHEN DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
             AND DATE(created_at) < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
             THEN total_price ELSE 0 END) as revenue_prev_30d
FROM orders
WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
GROUP BY merchant_id
HAVING revenue_30d < revenue_prev_30d * 0.8  -- Revenue dropped > 20%
ORDER BY revenue_30d - revenue_prev_30d;
```

### ROLLUP and CUBE

Multi-level aggregation in a single query.

```sql
-- Revenue breakdown with subtotals
SELECT
    COALESCE(country, 'ALL COUNTRIES') as country,
    COALESCE(product_category, 'ALL CATEGORIES') as product_category,
    SUM(total_price) as total_revenue,
    COUNT(*) as order_count
FROM orders o
JOIN merchants m ON o.merchant_id = m.merchant_id
JOIN products p ON o.product_id = p.product_id
GROUP BY ROLLUP(country, product_category)
ORDER BY country, product_category;
```

```
country     | product_category | total_revenue | order_count
------------|-----------------|---------------|------------
Canada      | Clothing        | 1,200,000     | 5,000
Canada      | Electronics     | 800,000       | 2,000
Canada      | ALL CATEGORIES  | 2,000,000     | 7,000        ← subtotal
US          | Clothing        | 3,500,000     | 15,000
US          | Electronics     | 2,100,000     | 8,000
US          | ALL CATEGORIES  | 5,600,000     | 23,000       ← subtotal
ALL COUNTRIES| ALL CATEGORIES | 7,600,000     | 30,000       ← grand total
```

---

## Joins at Scale

### Join Types Quick Reference

| Join | Returns | NULL behavior |
|------|---------|---------------|
| INNER JOIN | Rows matching in both tables | Excludes non-matching rows |
| LEFT JOIN | All rows from left + matching right | NULLs for non-matching right |
| RIGHT JOIN | All rows from right + matching left | NULLs for non-matching left |
| FULL OUTER JOIN | All rows from both tables | NULLs for non-matching rows |
| CROSS JOIN | Every combination (cartesian product) | No NULLs, but huge result |

### Broadcast vs Shuffle Joins

In distributed databases (BigQuery, Spark), join performance depends on how data is distributed.

```
Shuffle Join (both tables large):
  Table A (partitioned) ←→ Table B (partitioned)
  Both tables shuffled across nodes by join key
  Expensive: network transfer + disk I/O

Broadcast Join (one table small):
  Table A (large) + Table B (small, broadcast to all nodes)
  Small table sent to every node
  Fast: no shuffling of large table
```

**BigQuery optimization:**

```sql
-- If merchants table is small (< 500MB), BigQuery broadcasts it automatically
-- For explicit control, you can hint:
SELECT o.*, m.name, m.country
FROM orders o
JOIN merchants m ON o.merchant_id = m.merchant_id;

-- For very large joins, ensure both tables are clustered on the join key
-- BigQuery handles this automatically with its columnar storage
```

### Common Join Pitfalls

```sql
-- PITFALL 1: Accidental cross join (missing join condition)
-- This creates rows * rows results
SELECT * FROM orders, products;  -- DANGER: cartesian product

-- PITFALL 2: Join explosion (many-to-many)
-- If an order can have multiple line items, and a product can have multiple tags:
SELECT * FROM order_items oi
JOIN product_tags pt ON oi.product_id = pt.product_id;
-- One order item + 5 tags = 5 rows. Aggregations will be wrong.

-- SOLUTION: Aggregate before joining
WITH product_tag_list AS (
    SELECT product_id, ARRAY_AGG(tag) as tags
    FROM product_tags
    GROUP BY product_id
)
SELECT oi.*, pt.tags
FROM order_items oi
LEFT JOIN product_tag_list pt ON oi.product_id = pt.product_id;
```

---

### Check Your Understanding: Subqueries and Joins

**1. Why is EXISTS generally preferred over IN when the subquery table is large?**

<details>
<summary>Answer</summary>

EXISTS stops evaluating as soon as it finds the first matching row, while IN materializes the entire subquery result set first. For large subquery tables, EXISTS avoids processing unnecessary rows.
</details>

**2. What causes a "join explosion" and how do you prevent it?**

<details>
<summary>Answer</summary>

A join explosion occurs in many-to-many joins where one row on each side can match multiple rows on the other side, creating a multiplicative increase in output rows. The solution is to aggregate one side before joining (e.g., use ARRAY_AGG or GROUP BY to collapse the many-side into one row per key).
</details>

---

## Query Optimization

### Reading EXPLAIN Plans

```sql
-- BigQuery: use EXPLAIN to understand query execution
-- (BigQuery shows estimated bytes processed)

-- Before optimization: full table scan
SELECT * FROM orders WHERE merchant_id = 'M001';
-- Scans entire orders table (could be TB of data)

-- After optimization: partitioned + clustered
-- Table partitioned by DATE(created_at), clustered by merchant_id
SELECT * FROM orders
WHERE DATE(created_at) = '2024-01-15'
AND merchant_id = 'M001';
-- Scans only the 2024-01-15 partition, further filtered by cluster
```

### Optimization Strategies

| Strategy | When to Use | Impact |
|----------|------------|--------|
| Partition pruning | Query filters on partition column | 10-100x faster |
| Clustering | Frequent filters/joins on specific columns | 2-10x faster |
| Materialized views | Repeated expensive aggregations | Avoids recomputation |
| Approximate functions | Exact count not needed | APPROX_COUNT_DISTINCT is 10x faster |
| Avoid SELECT * | Only need specific columns | Less data scanned |
| Pre-aggregate | Joining aggregated data, not raw rows | Fewer rows to process |

```sql
-- Use APPROX_COUNT_DISTINCT for large-scale analytics
-- Exact: expensive
SELECT merchant_id, COUNT(DISTINCT customer_id) as unique_customers
FROM orders GROUP BY merchant_id;

-- Approximate (< 1% error, much faster):
SELECT merchant_id, APPROX_COUNT_DISTINCT(customer_id) as approx_unique_customers
FROM orders GROUP BY merchant_id;
```

---

### Check Your Understanding: Optimization

**1. A query scans 5 TB of data but only needs one day of results for a single merchant. What two optimizations would reduce this dramatically?**

<details>
<summary>Answer</summary>

(1) Partition the table by date (DATE(created_at)) so the query only scans the relevant date partition. (2) Cluster the table by merchant_id so the query further filters within the partition. Together, these can reduce the scan by 100x or more.
</details>

**2. When should you use APPROX_COUNT_DISTINCT instead of COUNT(DISTINCT)?**

<details>
<summary>Answer</summary>

When exact counts are not required (e.g., analytics dashboards, feature engineering at scale). APPROX_COUNT_DISTINCT uses HyperLogLog and is roughly 10x faster with less than 1% error. Use exact COUNT(DISTINCT) only when precision matters (e.g., billing, compliance reporting).
</details>

---

## Common ML Data Patterns

### Sessionization

Group events into sessions based on inactivity gaps.

```sql
-- Define sessions: a gap of > 30 minutes starts a new session
WITH events_with_prev AS (
    SELECT *,
        LAG(event_timestamp) OVER (
            PARTITION BY user_id ORDER BY event_timestamp
        ) as prev_event_timestamp
    FROM storefront_events
),
session_boundaries AS (
    SELECT *,
        CASE
            WHEN prev_event_timestamp IS NULL THEN 1  -- first event
            WHEN TIMESTAMP_DIFF(event_timestamp, prev_event_timestamp, MINUTE) > 30 THEN 1
            ELSE 0
        END as is_session_start
    FROM events_with_prev
),
sessions AS (
    SELECT *,
        SUM(is_session_start) OVER (
            PARTITION BY user_id ORDER BY event_timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as session_id
    FROM session_boundaries
)
SELECT
    user_id,
    session_id,
    MIN(event_timestamp) as session_start,
    MAX(event_timestamp) as session_end,
    COUNT(*) as event_count,
    TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), SECOND) as session_duration_sec,
    COUNTIF(event_type = 'purchase') as purchases_in_session
FROM sessions
GROUP BY user_id, session_id;
```

### Funnel Analysis

Track conversion through a series of steps.

```sql
-- E-commerce funnel: visit → product_view → add_to_cart → checkout → purchase
WITH funnel AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) as visited,
        MAX(CASE WHEN event_type = 'product_view' THEN 1 ELSE 0 END) as viewed_product,
        MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) as added_to_cart,
        MAX(CASE WHEN event_type = 'checkout_start' THEN 1 ELSE 0 END) as started_checkout,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchased
    FROM storefront_events
    WHERE DATE(event_timestamp) = '2024-01-15'
    GROUP BY user_id
)
SELECT
    COUNT(*) as total_visitors,
    SUM(viewed_product) as viewed_product,
    SUM(added_to_cart) as added_to_cart,
    SUM(started_checkout) as started_checkout,
    SUM(purchased) as purchased,
    ROUND(SUM(viewed_product) / COUNT(*) * 100, 1) as view_rate,
    ROUND(SUM(added_to_cart) / NULLIF(SUM(viewed_product), 0) * 100, 1) as atc_rate,
    ROUND(SUM(started_checkout) / NULLIF(SUM(added_to_cart), 0) * 100, 1) as checkout_rate,
    ROUND(SUM(purchased) / NULLIF(SUM(started_checkout), 0) * 100, 1) as purchase_rate
FROM funnel;
```

### Cohort Analysis

Track behavior of groups defined by their start date.

```sql
-- Monthly merchant retention cohort
WITH merchant_cohorts AS (
    SELECT
        merchant_id,
        DATE_TRUNC(MIN(DATE(created_at)), MONTH) as cohort_month
    FROM orders
    GROUP BY merchant_id
),
monthly_activity AS (
    SELECT DISTINCT
        merchant_id,
        DATE_TRUNC(DATE(created_at), MONTH) as activity_month
    FROM orders
)
SELECT
    c.cohort_month,
    DATE_DIFF(a.activity_month, c.cohort_month, MONTH) as months_since_signup,
    COUNT(DISTINCT a.merchant_id) as active_merchants,
    COUNT(DISTINCT a.merchant_id) / MAX(cohort_size.cnt) as retention_rate
FROM merchant_cohorts c
JOIN monthly_activity a ON c.merchant_id = a.merchant_id
JOIN (
    SELECT cohort_month, COUNT(*) as cnt
    FROM merchant_cohorts
    GROUP BY cohort_month
) cohort_size ON c.cohort_month = cohort_size.cohort_month
GROUP BY c.cohort_month, months_since_signup
ORDER BY c.cohort_month, months_since_signup;
```

### Time-Series Aggregation for Features

```sql
-- Compute rolling features for each merchant at each date
-- Used as training data for ML models
WITH date_spine AS (
    -- Generate one row per merchant per date
    SELECT merchant_id, date
    FROM UNNEST(GENERATE_DATE_ARRAY('2023-01-01', CURRENT_DATE())) as date
    CROSS JOIN (SELECT DISTINCT merchant_id FROM orders)
),
daily_orders AS (
    SELECT
        merchant_id,
        DATE(created_at) as order_date,
        COUNT(*) as order_count,
        SUM(total_price) as revenue,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM orders
    GROUP BY merchant_id, DATE(created_at)
)
SELECT
    ds.merchant_id,
    ds.date,
    -- 7-day rolling features
    SUM(COALESCE(d.order_count, 0)) OVER (
        PARTITION BY ds.merchant_id ORDER BY ds.date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as orders_7d,
    SUM(COALESCE(d.revenue, 0)) OVER (
        PARTITION BY ds.merchant_id ORDER BY ds.date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as revenue_7d,
    -- 30-day rolling features
    SUM(COALESCE(d.order_count, 0)) OVER (
        PARTITION BY ds.merchant_id ORDER BY ds.date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as orders_30d,
    SUM(COALESCE(d.revenue, 0)) OVER (
        PARTITION BY ds.merchant_id ORDER BY ds.date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as revenue_30d,
    -- Ratio features
    SAFE_DIVIDE(
        SUM(COALESCE(d.revenue, 0)) OVER (
            PARTITION BY ds.merchant_id ORDER BY ds.date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        SUM(COALESCE(d.revenue, 0)) OVER (
            PARTITION BY ds.merchant_id ORDER BY ds.date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )
    ) as revenue_7d_30d_ratio
FROM date_spine ds
LEFT JOIN daily_orders d ON ds.merchant_id = d.merchant_id AND ds.date = d.order_date;
```

---

## Practical Exercises with E-commerce Data

### Exercise 1: Find High-Growth Merchants

```sql
-- Merchants whose order count more than doubled month-over-month
WITH monthly AS (
    SELECT
        merchant_id,
        DATE_TRUNC(DATE(created_at), MONTH) as month,
        COUNT(*) as order_count
    FROM orders
    GROUP BY merchant_id, DATE_TRUNC(DATE(created_at), MONTH)
),
growth AS (
    SELECT
        merchant_id,
        month,
        order_count,
        LAG(order_count) OVER (PARTITION BY merchant_id ORDER BY month) as prev_month_orders,
        SAFE_DIVIDE(order_count,
            LAG(order_count) OVER (PARTITION BY merchant_id ORDER BY month)
        ) as growth_rate
    FROM monthly
)
SELECT * FROM growth
WHERE growth_rate > 2.0 AND prev_month_orders >= 10
ORDER BY growth_rate DESC;
```

### Exercise 2: Product Affinity (Co-Purchase Analysis)

```sql
-- Find products frequently purchased together
WITH order_products AS (
    SELECT DISTINCT order_id, product_id
    FROM order_items
),
product_pairs AS (
    SELECT
        a.product_id as product_a,
        b.product_id as product_b,
        COUNT(DISTINCT a.order_id) as co_purchase_count
    FROM order_products a
    JOIN order_products b ON a.order_id = b.order_id AND a.product_id < b.product_id
    GROUP BY a.product_id, b.product_id
    HAVING co_purchase_count >= 10
)
SELECT
    pp.*,
    pa.name as product_a_name,
    pb.name as product_b_name,
    SAFE_DIVIDE(pp.co_purchase_count, pa_orders.cnt) as support_a,
    SAFE_DIVIDE(pp.co_purchase_count, pb_orders.cnt) as support_b
FROM product_pairs pp
JOIN products pa ON pp.product_a = pa.product_id
JOIN products pb ON pp.product_b = pb.product_id
JOIN (SELECT product_id, COUNT(DISTINCT order_id) as cnt FROM order_items GROUP BY product_id) pa_orders
    ON pp.product_a = pa_orders.product_id
JOIN (SELECT product_id, COUNT(DISTINCT order_id) as cnt FROM order_items GROUP BY product_id) pb_orders
    ON pp.product_b = pb_orders.product_id
ORDER BY co_purchase_count DESC
LIMIT 100;
```

---

## Practice Interview Questions

1. "Write a query to find the top 10 merchants by revenue growth rate, comparing this month to last month."
2. "Given a table of storefront events, sessionize the data with a 30-minute inactivity timeout."
3. "Write a query to compute 7-day and 30-day rolling average order values per merchant."
4. "Find all merchants who had their first order in January 2024 and are still active (ordered in the last 30 days)."
5. "Optimize this query that's scanning 5TB of data but only needs results for one merchant on one day."

---

## Key Takeaways

1. Window functions are the most important SQL concept for ML feature engineering. Master PARTITION BY + ORDER BY + frame clauses.
2. CTEs make complex queries readable. Use them liberally.
3. Temporal patterns (LAG, LEAD, rolling windows) are the most common ML feature patterns in SQL.
4. Always think about query cost at scale. Partition pruning and clustering can reduce costs 100x.
5. Sessionization, funnel analysis, and cohort analysis are the three patterns you'll use most often.
6. Prefer window functions over correlated subqueries for performance.
7. SAFE_DIVIDE and COALESCE are your friends in BigQuery — null-safe operations prevent silent errors.

---

## Common Pitfalls

**1. Confusing ROWS BETWEEN and RANGE BETWEEN in window frames.** `ROWS BETWEEN` counts physical rows. `RANGE BETWEEN` considers the logical value of the ORDER BY column. If there are gaps in your date sequence (no row for days with zero orders), `ROWS BETWEEN 6 PRECEDING` does not mean "last 7 days" -- it means "last 7 rows," which could span weeks. Always use a date spine to fill gaps when computing rolling features.

**2. Using LAST_VALUE without specifying the frame.** The default window frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which means LAST_VALUE returns the current row's value, not the last row in the partition. You must explicitly specify `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` to get the actual last value.

**3. NULL handling in aggregations and joins.** NULLs are silently excluded from COUNT, SUM, and AVG. A LEFT JOIN that produces NULLs will cause aggregation results to differ from what you expect. Always use COALESCE to provide default values and SAFE_DIVIDE to avoid division-by-zero errors.

**4. Correlated subqueries running once per row.** A correlated subquery references the outer query and re-executes for every row. On a table with millions of rows, this can be catastrophically slow. Rewrite using window functions or CTEs with joins whenever possible.

---

## Hands-On Exercises

### Exercise: Build a Churn Feature Set

Using the patterns from this lesson, write a query that produces one row per merchant with the following features:
- `orders_7d`: order count in the last 7 days
- `orders_30d`: order count in the last 30 days
- `revenue_7d`: total revenue in the last 7 days
- `avg_order_value_30d`: average order value in the last 30 days
- `days_since_last_order`: days between today and the merchant's most recent order
- `order_velocity_ratio`: ratio of 7-day orders to 30-day orders (using SAFE_DIVIDE)

Hint: Use a CTE to compute daily aggregates first, then use window functions or conditional aggregation over the daily data.

### Exercise: Funnel Drop-Off Analysis

Using the funnel analysis pattern from this lesson, extend the query to identify the step with the largest absolute drop-off (most users lost) and the step with the largest percentage drop-off. Output should include step name, users entering the step, users completing the step, and drop-off rate.

---

## Summary

This lesson covered the advanced SQL patterns most relevant to ML engineering: window functions for feature engineering, CTEs for readable multi-step queries, subquery optimization, aggregation patterns, join strategies at scale, query optimization techniques, and common ML data patterns including sessionization, funnel analysis, cohort analysis, and time-series feature generation.

### What's Next

Continue to [Cloud Data Warehouses and BigQuery](../bigquery-warehouses/COURSE.md) to learn how these SQL patterns execute inside BigQuery's distributed architecture, how partitioning and clustering optimize performance, and how to connect warehouse data to ML pipelines.
