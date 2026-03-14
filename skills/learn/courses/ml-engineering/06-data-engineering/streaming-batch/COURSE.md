# Pipeline Orchestration and Streaming

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The tradeoffs between batch and stream processing in terms of latency, cost, complexity, and correctness
- Core Airflow concepts (DAGs, operators, sensors, XComs) and how they orchestrate ML pipelines
- Kafka's architecture (topics, partitions, consumer groups, offsets) and how it enables real-time ML features

**Apply:**
- Design Airflow DAGs for ML training, feature computation, batch prediction, and monitoring workflows
- Choose between Lambda and Kappa architectures based on feature freshness and system complexity requirements

**Analyze:**
- Evaluate when to use batch vs streaming vs hybrid architectures for a given ML system, considering data freshness, cost, and operational complexity

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- Data pipeline concepts from [Data Pipelines](../../05-production-ml/data-pipelines/COURSE.md), including pipeline design patterns and data validation

---

## Why This Matters

Production ML roles require experience with orchestration tools. ML models don't exist in isolation — they depend on data pipelines that extract, transform, validate, train, evaluate, and deploy. Orchestration tools manage this complexity by defining dependencies, scheduling runs, handling retries, and alerting on failures.

This lesson covers batch orchestration (Airflow, Prefect), streaming (Kafka, Pub/Sub), and the architectural patterns that connect them to ML systems.

---

## Batch vs Streaming: When to Use Each

### Batch Processing

Process data in discrete chunks on a schedule.

```
Trigger: Schedule (every hour, daily, weekly)
Input:   All data accumulated since last run
Output:  Updated tables, models, predictions
Latency: Minutes to hours
```

**Examples:**
- Daily feature computation for training
- Weekly model retraining
- Nightly batch predictions for all merchants
- Monthly reporting aggregations

### Stream Processing

Process data continuously as it arrives.

```
Trigger: Each event as it arrives
Input:   Individual events or micro-batches
Output:  Updated state, real-time features, alerts
Latency: Milliseconds to seconds
```

**Examples:**
- Real-time fraud scoring (velocity features)
- Live dashboard updates
- Real-time personalization signals
- Anomaly detection on metrics

### Decision Framework

```
┌──────────────────────────────────────────────────────────────┐
│ Does the consumer need data freshness < 1 minute?            │
│   YES → Streaming                                            │
│   NO  → Does the consumer need data freshness < 1 hour?     │
│           YES → Micro-batch (5-15 min intervals)             │
│           NO  → Batch (hourly or daily)                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Is the computation stateless (no aggregation over time)?     │
│   YES → Streaming is straightforward                         │
│   NO  → Does the window fit in memory?                      │
│           YES → Streaming with windowing                     │
│           NO  → Batch (need full dataset)                    │
└──────────────────────────────────────────────────────────────┘
```

### Cost Comparison

| Factor | Batch | Streaming |
|--------|-------|-----------|
| Infrastructure | Ephemeral (spin up, process, shut down) | Always-on (constant compute) |
| Complexity | Lower (SQL, Python scripts) | Higher (state management, exactly-once) |
| Cost at low volume | Cheaper (pay only when running) | More expensive (idle resources) |
| Cost at high volume | Can be expensive (large batch jobs) | More efficient (amortized) |
| Debugging | Easier (replay, inspect) | Harder (ephemeral state, timing issues) |
| Exactly-once | Natural (idempotent rewrites) | Hard (requires careful design) |

---

### Check Your Understanding: Batch vs Streaming

**1. A product recommendation system needs to update recommendations every 24 hours. Should you use batch or streaming?**

<details>
<summary>Answer</summary>

Batch. A 24-hour refresh cycle is well within batch processing territory. The computation can run as a scheduled job (e.g., daily Airflow DAG) that reads from the warehouse, computes recommendations, and writes results. Streaming would add unnecessary complexity and infrastructure cost for this latency requirement.
</details>

**2. Why is exactly-once processing hard to achieve in streaming systems but natural in batch systems?**

<details>
<summary>Answer</summary>

In batch systems, processing is idempotent by design: you read a fixed dataset, process it, and write the output (overwriting any previous result). Re-running produces the same result. In streaming, events arrive continuously and state must be maintained across messages. A consumer crash after processing a message but before committing the offset leads to reprocessing (at-least-once) or loss (at-most-once). Exactly-once requires transactional coordination between processing and offset commits, which is complex.
</details>

---

## Apache Airflow

The industry standard for orchestrating batch pipelines. Many large companies use Airflow.

### Core Concepts

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| DAG | Directed Acyclic Graph of tasks | A recipe with ordered steps |
| Task | A single unit of work | One step in the recipe |
| Operator | Template for a type of task | The type of cooking action |
| Sensor | Task that waits for a condition | "Wait until the oven is preheated" |
| XCom | Data passed between tasks | Passing ingredients between steps |
| Connection | Credentials for external systems | Your login to BigQuery, AWS, etc. |
| Variable | Global configuration values | Environment-specific settings |
| Pool | Limit concurrent task execution | "Only 4 burners on the stove" |

### DAG Example: ML Training Pipeline

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'email_on_failure': True,
    'email': ['ml-team@company.com'],
}

with DAG(
    'merchant_churn_training',
    default_args=default_args,
    schedule_interval='@weekly',  # Retrain every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'churn'],
    max_active_runs=1,  # Only one training run at a time
) as dag:

    # Wait for feature pipeline to complete
    wait_for_features = ExternalTaskSensor(
        task_id='wait_for_features',
        external_dag_id='merchant_features_daily',
        external_task_id='validate_features',
        execution_delta=timedelta(days=1),  # Yesterday's features
        timeout=3600,  # Wait up to 1 hour
        poke_interval=300,  # Check every 5 minutes
    )

    # Extract training data
    extract_training_data = BigQueryInsertJobOperator(
        task_id='extract_training_data',
        configuration={
            'query': {
                'query': """
                    EXPORT DATA OPTIONS(
                        uri='gs://ml-bucket/training/{{ ds }}/data_*.parquet',
                        format='PARQUET',
                        overwrite=true
                    ) AS
                    SELECT f.*, l.churned
                    FROM `project.features.merchant_features` f
                    JOIN `project.labels.churn_labels` l
                        ON f.merchant_id = l.merchant_id
                        AND f.snapshot_date = l.label_date
                    WHERE f.snapshot_date BETWEEN
                        DATE_SUB('{{ ds }}', INTERVAL 90 DAY) AND '{{ ds }}'
                """,
                'useLegacySql': False,
            }
        },
    )

    # Validate training data
    validate_data = PythonOperator(
        task_id='validate_training_data',
        python_callable=validate_training_data,
        op_kwargs={
            'data_path': 'gs://ml-bucket/training/{{ ds }}/',
            'min_rows': 100000,
            'max_null_rate': 0.01,
        },
    )

    # Train model
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_churn_model,
        op_kwargs={
            'data_path': 'gs://ml-bucket/training/{{ ds }}/',
            'model_output': 'gs://ml-bucket/models/churn/{{ ds }}/',
            'experiment_name': 'merchant_churn_weekly',
        },
    )

    # Evaluate model
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_and_compare,
        op_kwargs={
            'model_path': 'gs://ml-bucket/models/churn/{{ ds }}/',
            'min_auc': 0.85,  # Must beat this threshold
        },
    )

    # Deploy (only if evaluation passes)
    deploy_model = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_to_serving,
        op_kwargs={
            'model_path': 'gs://ml-bucket/models/churn/{{ ds }}/',
            'canary_percentage': 10,
        },
    )

    # Define task dependencies
    (wait_for_features
     >> extract_training_data
     >> validate_data
     >> train_model
     >> evaluate_model
     >> deploy_model)
```

### Airflow Best Practices

**1. Keep DAGs simple.** Each DAG should do one thing well. Don't cram feature computation and model training into one DAG.

**2. Use sensors for cross-DAG dependencies.** Don't couple DAGs by calling them directly.

**3. Make tasks idempotent.** Re-running a task with the same parameters must produce the same result.

**4. Avoid storing large data in XComs.** XComs are stored in the Airflow database. Pass file paths, not datasets.

**5. Set retries and timeouts.** Network calls fail. BigQuery queries can hang.

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}
```

**6. Use task groups for readability.**

```python
from airflow.utils.task_group import TaskGroup

with TaskGroup('data_validation') as validation:
    check_schema = PythonOperator(task_id='check_schema', ...)
    check_distributions = PythonOperator(task_id='check_distributions', ...)
    check_completeness = PythonOperator(task_id='check_completeness', ...)

extract >> validation >> transform
```

### Airflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Airflow Components                                          │
│                                                             │
│  Scheduler    → Reads DAG files, schedules task instances   │
│  Webserver    → UI for monitoring, triggering, debugging    │
│  Workers      → Execute tasks (Celery, Kubernetes, Local)   │
│  Metadata DB  → Stores DAG runs, task states, XComs         │
│  DAG Files    → Python files defining pipelines             │
└─────────────────────────────────────────────────────────────┘

Execution flow:
  Scheduler → Reads DAGs → Determines tasks to run → Sends to Workers
  Workers → Execute tasks → Update Metadata DB → Scheduler sees completion
```

---

### Check Your Understanding: Airflow

**1. Why should Airflow tasks be idempotent?**

<details>
<summary>Answer</summary>

Tasks can fail and be retried automatically. If a task is not idempotent, re-running it with the same parameters could produce different results, corrupt data, or create duplicates. Idempotent tasks produce the same output regardless of how many times they run, making retries safe and debugging straightforward.
</details>

**2. Why should you avoid storing large data in XComs?**

<details>
<summary>Answer</summary>

XComs are stored in Airflow's metadata database (typically PostgreSQL or MySQL), which is not designed for large data. Storing datasets in XComs bloats the database, slows down the Airflow UI, and can cause out-of-memory errors. Instead, pass file paths (e.g., GCS URIs) via XComs and read the actual data from cloud storage.
</details>

---

## Prefect

A modern alternative to Airflow with a simpler API.

### Key Differences from Airflow

| Feature | Airflow | Prefect |
|---------|---------|---------|
| DAG definition | Python (verbose, operators) | Python (decorators, natural) |
| Scheduling | Built-in scheduler | Prefect Cloud / Server |
| Dynamic DAGs | Limited | First-class support |
| Local testing | Requires Airflow running | Run as regular Python |
| Error handling | Task-level retries | Rich retry/caching/state |
| Deployment | Complex (workers, scheduler) | `prefect deploy` |

### Prefect Example

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(retries=3, retry_delay_seconds=60, cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract_data(date: str) -> str:
    """Extract training data from BigQuery."""
    query = f"SELECT * FROM features WHERE date = '{date}'"
    path = f"gs://bucket/training/{date}/"
    run_bq_export(query, path)
    return path

@task
def validate_data(data_path: str) -> bool:
    """Validate training data quality."""
    df = read_parquet(data_path)
    assert len(df) > 100_000, "Too few rows"
    assert df.isnull().mean().max() < 0.01, "Too many nulls"
    return True

@task
def train_model(data_path: str) -> str:
    """Train and log model."""
    df = read_parquet(data_path)
    model = train(df)
    model_path = save_model(model)
    return model_path

@task
def evaluate_model(model_path: str) -> dict:
    """Evaluate model against production baseline."""
    metrics = evaluate(model_path)
    if metrics['auc'] < 0.85:
        raise ValueError(f"AUC {metrics['auc']} below threshold")
    return metrics

@flow(name="merchant-churn-training")
def training_pipeline(date: str):
    """Weekly training pipeline."""
    data_path = extract_data(date)
    validate_data(data_path)
    model_path = train_model(data_path)
    metrics = evaluate_model(model_path)
    return metrics

# Run locally for testing — just call the function!
if __name__ == "__main__":
    training_pipeline("2024-01-15")
```

**Key advantage:** Prefect flows are regular Python functions. You can test them locally without any infrastructure.

### When to Use Prefect vs Airflow

```
Use Airflow when:
  - Your company already uses it (many large companies do)
  - You need battle-tested reliability
  - You need the rich ecosystem of providers (GCP, AWS, etc.)
  - You have a platform team managing Airflow

Use Prefect when:
  - Starting fresh with no existing orchestrator
  - You want simpler Python-native syntax
  - You need dynamic pipelines (tasks that spawn tasks)
  - You want easier local development
```

---

## Apache Kafka

Distributed event streaming platform. The backbone of real-time data infrastructure.

### Core Architecture

```
┌─────────┐     ┌──────────────────────────────────┐     ┌──────────┐
│ Producer │────→│           Kafka Cluster           │────→│ Consumer │
│ (app)   │     │                                    │     │ (app)    │
└─────────┘     │  Topic: orders.created             │     └──────────┘
                │  ┌───────┐ ┌───────┐ ┌───────┐    │
┌─────────┐     │  │ P0    │ │ P1    │ │ P2    │    │     ┌──────────┐
│ Producer │────→│  │ msg1  │ │ msg2  │ │ msg3  │    │────→│ Consumer │
│ (app)   │     │  │ msg4  │ │ msg5  │ │ msg6  │    │     │ (app)    │
└─────────┘     │  └───────┘ └───────┘ └───────┘    │     └──────────┘
                └──────────────────────────────────────┘
```

### Key Concepts

**Topic:** A named stream of events. Like a table in a database, but append-only.
```
Topic: platform.orders.created
Topic: platform.products.updated
Topic: platform.payments.processed
```

**Partition:** A topic is split into partitions for parallelism. Each partition is an ordered, immutable sequence.

**Consumer Group:** Multiple consumers that share the work of reading a topic. Each partition is read by exactly one consumer in the group.

**Offset:** The position of a consumer in a partition. Consumers track their offset to know where they left off.

### Kafka for ML: Real-Time Features

```python
# Python consumer that computes real-time velocity features
from kafka import KafkaConsumer
import json
import redis
from collections import defaultdict
from datetime import datetime, timedelta

consumer = KafkaConsumer(
    'platform.payments.processed',
    bootstrap_servers=['kafka:9092'],
    group_id='fraud-feature-computer',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
)

redis_client = redis.Redis()

for message in consumer:
    event = message.value
    card_id = event['card_id']
    amount = event['amount']
    timestamp = event['timestamp']

    # Update velocity features in Redis
    # Key: "velocity:{card_id}", Value: sorted set of (timestamp, amount)
    redis_client.zadd(
        f"velocity:{card_id}",
        {f"{timestamp}:{amount}": timestamp}
    )

    # Remove events older than 24 hours
    cutoff = datetime.utcnow() - timedelta(hours=24)
    redis_client.zremrangebyscore(
        f"velocity:{card_id}", 0, cutoff.timestamp()
    )

    # Compute and store velocity features
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).timestamp()
    txn_count_1h = redis_client.zcount(f"velocity:{card_id}", one_hour_ago, '+inf')

    redis_client.hset(f"features:{card_id}", mapping={
        'txn_count_1h': txn_count_1h,
        'txn_count_24h': redis_client.zcard(f"velocity:{card_id}"),
        'last_updated': datetime.utcnow().isoformat(),
    })
```

### Kafka Guarantees

| Guarantee | Meaning | Config |
|-----------|---------|--------|
| At-most-once | Messages may be lost, never duplicated | `enable.auto.commit=true` (default) |
| At-least-once | Messages never lost, may be duplicated | Manual commit after processing |
| Exactly-once | Messages never lost, never duplicated | Kafka transactions (complex) |

**For ML features:** At-least-once is usually fine. Feature computations are idempotent (reprocessing an event doesn't corrupt the result).

---

## Google Cloud Pub/Sub

Managed messaging service. Serverless alternative to Kafka.

### Architecture

```
Publisher → Topic → Subscription 1 → Subscriber A
                  → Subscription 2 → Subscriber B
                  → Subscription 3 → Subscriber C
```

**Key difference from Kafka:** Multiple subscriptions per topic (like Kafka consumer groups), but fully managed with no clusters to provision.

### Pub/Sub Example

```python
from google.cloud import pubsub_v1
import json

# Publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('my-project', 'order-events')

def publish_order_event(order):
    data = json.dumps(order).encode('utf-8')
    future = publisher.publish(topic_path, data, event_type='order.created')
    return future.result()

# Subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('my-project', 'feature-computer-sub')

def callback(message):
    event = json.loads(message.data.decode('utf-8'))
    compute_features(event)
    message.ack()

streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
streaming_pull_future.result()  # Block and process messages
```

### Kafka vs Pub/Sub

| Factor | Kafka | Pub/Sub |
|--------|-------|---------|
| Management | Self-hosted or Confluent | Fully managed by Google |
| Ordering | Per-partition guaranteed | Per-ordering-key (optional) |
| Retention | Configurable (days to forever) | 31 days max |
| Replay | Seek to any offset | Seek to timestamp |
| Throughput | Very high (millions/sec) | High (auto-scaled) |
| Cost | Cluster cost (fixed) | Per-message (variable) |
| Ecosystem | Rich (Kafka Streams, ksqlDB, Connect) | GCP-native |
| Best for | Large-scale, multi-consumer | GCP-native apps, simpler ops |

---

### Check Your Understanding: Kafka and Pub/Sub

**1. In Kafka, what happens if two consumers in the same consumer group try to read from the same partition?**

<details>
<summary>Answer</summary>

Only one consumer in a consumer group is assigned to each partition. Kafka enforces this -- a partition is read by exactly one consumer within a group. If you have more consumers than partitions, some consumers will be idle. To increase parallelism, you need more partitions.
</details>

**2. A Kafka consumer crashes after processing a message but before committing the offset. What happens when it restarts?**

<details>
<summary>Answer</summary>

The consumer restarts from the last committed offset and re-processes the message. This is at-least-once delivery -- the message is processed again. Your processing logic should be idempotent to handle this safely (e.g., writing features to a key-value store where rewriting the same key is harmless).
</details>

---

## Lambda Architecture

Combine batch and streaming for both accuracy and freshness.

```
                    ┌─────────────────────────────┐
Raw Data ──────────→│      Batch Layer             │──→ Batch View
    │               │ (accurate, complete, slow)    │    (BigQuery)
    │               └─────────────────────────────┘
    │
    └──────────────→┌─────────────────────────────┐
                    │      Speed Layer             │──→ Real-time View
                    │ (fast, approximate, recent)   │    (Redis)
                    └─────────────────────────────┘

Serving Layer: merge(Batch View, Real-time View)
```

**Example at a large-scale platform:**
- Batch layer: daily aggregation of merchant revenue (accurate, covers all history)
- Speed layer: streaming aggregation of today's revenue (approximate, real-time)
- Serving: combine yesterday's batch + today's stream

**Pros:** Accurate historical data + real-time updates
**Cons:** Maintaining two code paths (batch SQL + streaming code) that must produce consistent results. This is the main drawback.

---

## Kappa Architecture

Everything through streaming. No separate batch layer.

```
Raw Data → Stream Processing → Serving Layer

Reprocessing: replay events from the beginning of the stream
```

**Key idea:** If you can replay all events, you don't need a separate batch layer. Just replay through the streaming pipeline.

**Pros:** One code path, simpler to maintain
**Cons:** Requires a durable event log (Kafka with long retention), reprocessing can be slow

### Lambda vs Kappa

```
Use Lambda when:
  - You already have batch pipelines
  - Historical data is in a warehouse (BigQuery)
  - Some features require full-history aggregation
  - Team has SQL skills (batch) + streaming skills

Use Kappa when:
  - Starting from scratch
  - Events are already in Kafka with long retention
  - All features can be computed from event streams
  - Team prefers a single programming model
```

**Reality:** Most production ML systems use a hybrid. Batch features from BigQuery (via DBT/Airflow) for historical patterns, streaming features from Kafka for real-time signals. Both land in a feature store for unified access.

---

## Orchestration Patterns for ML

### Pattern 1: Training DAG

```
┌─────────────────────────────────────────────────────────────┐
│ Weekly Training DAG                                          │
│                                                              │
│ wait_for_features → extract_data → validate → train → eval  │
│                                                          ↓   │
│                                              [passes gate?]  │
│                                              YES → deploy    │
│                                              NO  → alert     │
└─────────────────────────────────────────────────────────────┘
```

### Pattern 2: Feature DAG (Fan-Out)

```
┌──────────────────────────────────────────────────────────────┐
│ Daily Feature DAG                                             │
│                                                               │
│ validate_sources → ┌─ compute_order_features ──┐              │
│                    ├─ compute_product_features ─┤→ combine     │
│                    ├─ compute_session_features ─┤   → validate │
│                    └─ compute_payment_features ─┘   → publish  │
└──────────────────────────────────────────────────────────────┘
```

### Pattern 3: Inference DAG (Batch Prediction)

```
┌──────────────────────────────────────────────────────────────┐
│ Daily Inference DAG                                           │
│                                                               │
│ load_model → load_features → predict_all → validate_preds    │
│                                                → write_to_db  │
│                                                → update_cache │
└──────────────────────────────────────────────────────────────┘
```

### Pattern 4: Monitoring DAG

```
┌──────────────────────────────────────────────────────────────┐
│ Hourly Monitoring DAG                                         │
│                                                               │
│ check_data_drift → check_prediction_dist → check_latency     │
│        ↓                    ↓                    ↓            │
│ [drift detected?]   [shift detected?]    [SLA breached?]     │
│   YES → alert          YES → alert         YES → page        │
│   YES + severe →       YES + severe →                        │
│   trigger_retrain      trigger_retrain                       │
└──────────────────────────────────────────────────────────────┘
```

---

## When to Use What: Decision Framework

### Orchestration Tool Selection

```
┌──────────────────────────────────────────────────────────────┐
│ Do you need to schedule and coordinate batch jobs?            │
│   YES → Do you already have Airflow?                         │
│           YES → Use Airflow                                   │
│           NO  → Is your team Python-first?                   │
│                   YES → Consider Prefect                     │
│                   NO  → Airflow (more documentation/support) │
│   NO  → Do you need event-driven pipelines?                  │
│           YES → Kafka / Pub/Sub + compute (Flink, Dataflow)  │
│           NO  → Maybe you don't need orchestration yet       │
└──────────────────────────────────────────────────────────────┘
```

### Streaming Platform Selection

```
┌──────────────────────────────────────────────────────────────┐
│ Are you on GCP?                                               │
│   YES → Is simplicity more important than control?           │
│           YES → Pub/Sub (fully managed, zero ops)            │
│           NO  → Confluent Kafka on GCP (more features)       │
│   NO  → Are you on AWS?                                      │
│           YES → MSK (managed Kafka) or Kinesis               │
│           NO  → Self-hosted Kafka or Confluent Cloud          │
└──────────────────────────────────────────────────────────────┘
```

### ML Pipeline Architecture Selection

```
┌──────────────────────────────────────────────────────────────┐
│ Does your ML model need real-time features?                   │
│   NO  → Batch only (Airflow + DBT + BigQuery)                │
│   YES → Does it also need historical features?               │
│           NO  → Streaming only (Kafka + Flink + Redis)       │
│           YES → Hybrid (batch features + streaming features) │
│                 Batch: Airflow + DBT + BigQuery → offline     │
│                 Stream: Kafka + Flink → online (Redis)       │
│                 Serving: read from both stores                │
└──────────────────────────────────────────────────────────────┘
```

---

## Practice Interview Questions

1. "Design the data pipeline for a fraud detection system that needs both historical features (30-day averages) and real-time features (transactions in last hour)."
2. "Your Airflow DAG takes 6 hours to run and frequently fails at step 4 of 7. How do you improve reliability and reduce runtime?"
3. "Compare Airflow and Prefect. When would you choose each?"
4. "Explain the difference between Lambda and Kappa architecture. Which would you use for a product recommendation system?"
5. "A feature pipeline in Kafka is producing incorrect velocity counts. How do you debug it?"
6. "How do you ensure exactly-once processing in a streaming pipeline?"

---

## Key Takeaways

1. Airflow is the industry standard for batch orchestration. Know DAGs, operators, sensors, and XComs.
2. Prefect is simpler but less mature. Use Airflow unless starting fresh.
3. Kafka is the standard for event streaming. Understand topics, partitions, consumer groups, and offsets.
4. Pub/Sub is the GCP-managed alternative. Simpler ops, less control.
5. Most production ML systems are hybrid: batch features (Airflow + DBT) + streaming features (Kafka) + feature store (unified access).
6. Lambda architecture (batch + speed layers) is common but expensive to maintain. Kappa (streaming-only) is simpler but requires event replay capability.
7. Make every pipeline step idempotent. This is the most important engineering principle for data pipelines.

---

## Common Pitfalls

**1. Building streaming when batch would suffice.** Streaming infrastructure (Kafka, Flink, always-on consumers) is significantly more complex and expensive to operate than batch jobs. If your use case tolerates hourly or daily latency, use batch. Over-engineering for real-time when it is not needed is a common and costly mistake.

**2. Not setting retries and timeouts on Airflow tasks.** Network calls to BigQuery, GCS, or external APIs can fail intermittently. Without retries, a transient failure causes the entire DAG to fail. Without timeouts, a hanging query can block the DAG indefinitely. Always set both.

**3. Tightly coupling DAGs by calling one from another.** Instead of having DAG A trigger DAG B directly, use sensors (ExternalTaskSensor) to wait for upstream DAGs to complete. This keeps DAGs independent, testable, and easier to debug.

**4. Ignoring consumer lag in Kafka.** If your consumer falls behind producers, the lag grows and real-time features become stale. Monitor consumer lag as a key metric and alert when it exceeds acceptable thresholds. Common causes include slow processing logic, insufficient consumer instances, or too few partitions.

---

## Hands-On Exercises

### Exercise: Design an ML Feature Pipeline DAG

Design an Airflow DAG (describe the tasks and dependencies, not full code) for a daily feature pipeline that:
1. Checks source freshness for orders and products tables
2. Runs DBT models to compute merchant features (fan-out: order features, product features, session features run in parallel)
3. Validates the output feature table (row count, null rate, value ranges)
4. Publishes features to a feature store
5. Notifies the team on failure

Draw the dependency graph and identify which tasks can run in parallel.

### Exercise: Choose the Architecture

For each scenario, decide whether to use batch-only, streaming-only, or hybrid architecture. Justify your choice:
1. A merchant churn prediction model that runs weekly
2. A fraud detection system that must score transactions within 100ms
3. A product recommendation system that updates hourly but also incorporates real-time browsing signals

---

## Summary

This lesson covered the orchestration and data movement layer of ML systems: batch processing with Airflow and Prefect, stream processing with Kafka and Pub/Sub, architectural patterns (Lambda and Kappa), and the specific DAG patterns used in ML pipelines for training, feature computation, inference, and monitoring.

### What's Next

With the data engineering foundation complete (SQL, BigQuery, DBT, and orchestration), you are ready to move into [GPU Computing Fundamentals](../../07-distributed-gpu/gpu-fundamentals/COURSE.md) to understand the hardware and memory constraints that shape how ML models are trained and served.
