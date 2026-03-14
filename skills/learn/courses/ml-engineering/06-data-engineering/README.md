# 06 — Data Engineering

> Building the pipelines that feed ML models. A core requirement for applied ML engineering roles.

## Why This Matters

From the job posting: "Experience in building data pipelines and driving ETL design decisions using disparate data sources. Proficiency in streaming and batch data pipelines, DBT, BigQuery, BigTable, or equivalent, and orchestration tools."

## Subdirectories

```
06-data-engineering/
├── sql-advanced/           # Window functions, CTEs, query optimization
├── bigquery-warehouses/    # Cloud data warehouses, BigQuery specifics
├── dbt/                    # Data transformation, modeling, testing
└── streaming-batch/        # Kafka, Pub/Sub, Airflow, Prefect
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| DeepLearning.AI: Data Engineering | Pipelines, orchestration | ~4 weeks |
| Google BigQuery Sandbox (free) | Hands-on warehouse experience | ~1 week |
| DBT docs + tutorials (docs.getdbt.com) | Data transformation | ~1 week |
| Mode SQL Tutorial (free) | Advanced SQL practice | ~1 week |

## Key Concepts

### ETL vs ELT
- **ETL:** Extract, Transform, Load — transform before loading into warehouse (traditional)
- **ELT:** Extract, Load, Transform — load raw data, transform in warehouse (modern, BigQuery/DBT approach)

### Tools to Know
| Tool | What it does | Industry relevance |
|---|---|---|
| BigQuery | Google's data warehouse — SQL at massive scale | Explicitly required |
| DBT | SQL-based data transformation framework | Explicitly required |
| Airflow / Prefect | Workflow orchestration (DAGs) | Required ("orchestration tools") |
| Kafka / Pub/Sub | Real-time data streaming | "Streaming pipelines" |

### SQL Skills to Build
- [ ] Window functions (ROW_NUMBER, RANK, LAG, LEAD, running aggregates)
- [ ] CTEs (WITH clauses for readable complex queries)
- [ ] Subqueries and correlated subqueries
- [ ] Query optimization (EXPLAIN, indexing, partitioning)
- [ ] Joins at scale (broadcast vs shuffle)
