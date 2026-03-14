# Gap Analysis — ML Engineer Readiness

> Last updated: 2026-03-08
> Target: Applied ML Engineering - GenAI, AI Agents

## Current Skill Assessment

```
CATEGORY                    CURRENT     TARGET      GAP (Agent-Adjusted)
──────────────────────────────────────────────────────────────────────────
Mathematics                 ██░░░░░░░░  ████████░░  ████████  LARGE
Deep Learning concepts      █░░░░░░░░░  ████████░░  ██████░░  MEDIUM-LARGE
Classical ML                ███░░░░░░░  ████████░░  ████░░░░  MEDIUM-SMALL
NLP / LLM Internals         ██░░░░░░░░  ████████░░  ██████░░  MEDIUM-LARGE
MLOps / Production ML       █░░░░░░░░░  ████████░░  ████░░░░  MEDIUM
Distributed / GPU           ░░░░░░░░░░  ███████░░░  ██████░░  MEDIUM-LARGE
Data Engineering            ██░░░░░░░░  ████████░░  ████░░░░  MEDIUM
Software Engineering        ████████░░  ████████░░  ░░░░░░░░  AHEAD
AI Agents / Tools           █████████░  ██████░░░░  ░░░░░░░░  AHEAD
Python                      ███████░░░  ████████░░  █░░░░░░░  SMALL
```

> **Instructions:** Fill in the CURRENT column based on an honest self-assessment. Adjust the GAP column accordingly. Use this as a living document to track your progress.

## Role Requirements vs Current Skills

### Direct Requirements (from job posting)

| Requirement | Current Level | Gap | Priority |
|---|---|---|---|
| Generative AI, NLP, ML model development & deployment | _[e.g., "Use LLMs daily, have built tabular ML projects"]_ | Need deployment at scale | HIGH |
| Scalable AI/ML system architectures | _[e.g., "Built multi-component systems, not yet at production ML scale"]_ | Need production ML scale | HIGH |
| Data pipelines for fine-tuning LLMs | _[e.g., "Built data pipelines, but no LLM fine-tuning yet"]_ | **KEY GAP** | CRITICAL |
| Python, shell scripting | _[e.g., "Strong"]_ | — | DONE |
| Vector databases | _[e.g., "Used in projects"]_ | Small gap | LOW |
| BigQuery, BigTable, or equivalent | _[e.g., "Not demonstrated"]_ | **KEY GAP** | CRITICAL |
| DBT | _[e.g., "Not demonstrated"]_ | **KEY GAP** | HIGH |
| Orchestration tools | _[e.g., "Not demonstrated"]_ | Need Airflow/Prefect | HIGH |
| ML in parallel environments (GPU, distributed) | _[e.g., "Not demonstrated"]_ | **KEY GAP** | CRITICAL |
| Pair programming interview | _[e.g., "Code daily in IDE"]_ | Need ML-specific practice | MEDIUM |

### What Sets You Apart (Competitive Advantages)

> Fill in your own differentiators. Consider: domain expertise, prior projects, unique skill combinations, and workflow strengths.

1. **Domain expertise** — _[e.g., "N years in [industry] — understand the data, users, and business problems"]_
2. **AI agent experience** — _[e.g., "Built autonomous agents, tool-use systems, or multi-step workflows"]_
3. **Infrastructure/tooling** — _[e.g., "Built developer tools, APIs, or internal platforms"]_
4. **Applied ML experience** — _[e.g., "Trained and evaluated models on real-world datasets"]_
5. **Full-stack engineering** — _[e.g., "Can build end-to-end systems, not just models"]_
6. **AI-native workflow** — _[e.g., "Use AI tools and agents to accelerate development"]_

## Gap Categories

### CRITICAL (Must close before applying)
- [ ] LLM fine-tuning pipelines (LoRA, QLoRA, full fine-tune)
- [ ] Distributed/GPU training fundamentals
- [ ] Data warehouse tools (BigQuery/DBT)
- [ ] Transformer internals (attention, positional encoding, architecture)

### HIGH (Should close before applying)
- [ ] Neural network fundamentals (backprop, loss functions, optimization)
- [ ] Production ML (model serving, monitoring, drift detection)
- [ ] Data pipeline orchestration (Airflow/Prefect)
- [ ] Math foundations (enough for interviews)

### MEDIUM (Strengthen for interviews)
- [ ] Classical ML breadth (beyond XGBoost)
- [ ] ML evaluation beyond accuracy (precision/recall, AUC, confusion matrices)
- [ ] System design practice (whiteboard ML architecture)

### LOW (Nice to have)
- [ ] Computer vision depth
- [ ] Advanced NLP (pre-LLM techniques)
- [ ] Research paper reading
