## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- Why evaluation of agentic ML systems requires multiple levels (component, integration, end-to-end) and agent-specific metrics
- How regression testing, A/B testing, and benchmarking differ for agent systems versus traditional ML models
- How to design production monitoring and drift detection for agents that call ML model tools

**Apply:**
- Create evaluation datasets and use LLM-as-judge for automated end-to-end evaluation
- Implement a regression testing pipeline triggered by model updates, prompt changes, or tool additions

**Analyze:**
- Diagnose whether a system failure originates at the component level (model/tool), integration level (tools working together), or end-to-end level (agent reasoning) by tracing execution paths

---

## Prerequisites

Before starting this lesson, you should be comfortable with:

- **Integrating ML models as agent tools** (./agents-with-ml-tools/COURSE.md) -- the agent-tool-model architecture, tool registry pattern, and how agents call ML model endpoints
- **Evaluation metrics** (../04-classical-ml/evaluation-metrics/COURSE.md) -- precision, recall, F1, AUC-ROC, confusion matrices, and when to use which metric

---

# Evaluating Agentic ML Systems

## Why Evaluation Is Harder for Agents

Evaluating a traditional ML model is relatively straightforward: hold out test data, run predictions, compute metrics. The model is a pure function — same input, same output, measurable accuracy.

Agentic ML systems break every one of these assumptions:

**Non-deterministic behavior.** Given the same user query, an agent might choose different tools, call them in different orders, or generate different responses. LLM temperature, tool availability, and even the order of tool descriptions in the prompt can change behavior.

**Multi-step execution.** An agent might make 3-5 tool calls to answer a single question. The final answer quality depends on every step: Was the right tool selected? Were the parameters correct? Was the intermediate result interpreted correctly? Did the next tool call use the right context?

**Emergent failures.** Each component (LLM, tool A, tool B, model C) might work perfectly in isolation but fail when composed. The LLM might misinterpret a tool's output format. A model might return edge-case predictions that confuse the agent's reasoning. Two tools might return conflicting information that the agent can't reconcile.

**Cost and latency matter.** A system that gets the right answer using 10 tool calls and $0.50 of LLM inference is worse than one that gets the same answer in 3 calls and $0.15. Evaluation must measure efficiency, not just correctness.

This means you need evaluation at multiple levels: component-level (each model and tool independently), integration-level (tools working together), and end-to-end (does the user get what they need?).

---

## End-to-End Evaluation

### What It Measures

End-to-end evaluation answers: "Given a user query, does the system produce a good final result?"

This is the most important evaluation because it's what users experience. A system where every component is individually excellent but the integration is poor will still fail.

### Creating Evaluation Datasets

Build a dataset of (input, expected_output) pairs:

```
Input: "What are the best gifts under $50 for someone who likes cooking?"
Expected: A list of relevant cooking-related products under $50, with personalized explanations

Input: "Is this order suspicious?" + {order details}
Expected: Correct fraud assessment with supporting evidence

Input: "Compare ETH and SOL performance this month"
Expected: Accurate price data, meaningful comparison, correct percentage changes
```

**How to build this dataset:**
1. Collect real user queries from logs (or create realistic ones)
2. Have human experts write ideal responses
3. Define evaluation criteria for each query type
4. Aim for 100-500 examples covering diverse scenarios

### Evaluation Methods

**Exact match:** Does the response contain the specific facts? (e.g., correct price, correct product names). Automated, but brittle.

**LLM-as-judge:** Use a separate LLM to evaluate whether the response is good. Prompt it with the query, the response, and evaluation criteria. Score on a 1-5 scale. This is the most practical approach for open-ended responses.

```
Evaluate this response on:
1. Accuracy (are the facts correct?)
2. Completeness (did it address the full question?)
3. Relevance (are the recommendations relevant?)
4. Helpfulness (would a user find this useful?)

Query: {query}
Response: {response}
Reference: {expected_output}
```

**Human evaluation:** Gold standard but expensive. Use for a small subset (50-100 examples) to validate that your automated metrics correlate with real quality.

### Metrics

- **Task completion rate:** What percentage of queries result in a useful response? (Target: > 90%)
- **Factual accuracy:** What percentage of stated facts are correct? (Target: > 95%)
- **Response quality score:** LLM-as-judge average score (Target: > 4.0/5.0)
- **Harmful response rate:** What percentage of responses are harmful, offensive, or inappropriate? (Target: 0%)

---

## Component Evaluation

### Evaluating Each ML Model Independently

Before integrating a model into an agent, evaluate it in isolation using standard ML metrics:

**Classification models (fraud detection, churn prediction):**
- Precision, recall, F1 score
- AUC-ROC (area under ROC curve)
- Precision-recall curve (especially for imbalanced classes)
- Confusion matrix

**Ranking models (search, recommendations):**
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Precision@K, Recall@K
- MAP (Mean Average Precision)

**Regression models (price prediction, demand forecasting):**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R-squared

**Embedding models (semantic search, similarity):**
- Retrieval metrics (NDCG, recall@K on retrieval benchmarks)
- Embedding quality (nearest-neighbor accuracy on labeled pairs)

### Evaluating Tools

Each tool (not just ML tools) should have its own test suite:

- **Input validation:** Does the tool reject invalid inputs gracefully?
- **Output format:** Does the tool always return the expected schema?
- **Error handling:** Does the tool handle failures (API down, empty results) without crashing?
- **Latency:** Does the tool respond within its latency budget?
- **Correctness:** Given known inputs, does the tool return correct outputs?

### Evaluating the LLM

The LLM itself (the "brain" of the agent) needs evaluation:

- **Tool selection accuracy:** Given a query, does it choose the right tool(s)?
- **Parameter extraction:** Does it extract the correct parameters from natural language?
- **Result interpretation:** Does it correctly interpret tool outputs?
- **Hallucination rate:** Does it add information that didn't come from tools?

Create a dataset of queries paired with expected tool calls:
```
Query: "What's the price of ETH?"
Expected tool: crypto_price
Expected params: {symbol: "ETH"}

Query: "Show me a chart of BTC for the last month"
Expected tool: crypto_chart
Expected params: {symbol: "BTC", timeframe: "1M"}
```

Measure: What percentage of queries result in the correct tool being called with the correct parameters?

### Check Your Understanding

<details>
<summary>1. Why is LLM-as-judge the most practical automated evaluation method for agent output quality, and what is its main limitation?</summary>

LLM-as-judge is most practical because agent responses are open-ended and natural language, making exact-match evaluation too brittle. A separate LLM can evaluate responses on multiple dimensions (accuracy, completeness, relevance, helpfulness) using a scoring rubric. Its main limitation is that it is not a ground truth -- the judge LLM can have its own biases and errors. That is why human evaluation on a smaller subset (50-100 examples) is needed to validate that the automated scores correlate with real quality.
</details>

<details>
<summary>2. What four aspects should you evaluate for each individual tool (not just ML model tools)?</summary>

(1) Input validation -- does the tool reject invalid inputs gracefully? (2) Output format -- does the tool always return the expected schema? (3) Error handling -- does the tool handle failures (API down, empty results) without crashing? (4) Latency -- does the tool respond within its latency budget? Additionally, correctness: given known inputs, does the tool return correct outputs?
</details>

<details>
<summary>3. Name the three levels of evaluation required for agentic ML systems and give an example of what each level tests.</summary>

(1) **Component-level** -- each model and tool independently (e.g., does the fraud model achieve acceptable AUC-ROC on a test set?). (2) **Integration-level** -- tools working together (e.g., does the agent correctly pass the output of product_search as input to recommend_products?). (3) **End-to-end** -- does the user get a good final result? (e.g., given "find me a birthday gift," does the system return relevant products with helpful explanations?).
</details>

---

## Agent-Specific Evaluation Metrics

Beyond standard ML metrics, agents need their own metrics:

### Tool Selection Accuracy

What percentage of tool calls are appropriate for the given context?

- **True positive:** Agent calls the right tool when needed
- **False positive:** Agent calls a tool when it shouldn't (wastes compute, adds latency)
- **False negative:** Agent should call a tool but doesn't (answers from its own knowledge, potentially wrong)
- **Wrong tool:** Agent calls a tool, but the wrong one

Target: > 95% tool selection accuracy for well-defined tasks.

### Tool Call Efficiency

How many tool calls does the agent make to answer a query?

- **Ideal:** Minimum necessary calls (usually 1-3)
- **Problematic:** Redundant calls, repeated calls with slightly different parameters, unnecessary "exploration"

Measure: average tool calls per query, compared to human-expert minimum.

### Cost Per Task

Total cost to answer a query:
- LLM inference cost (input + output tokens × price per token)
- ML model inference cost (compute time × GPU cost)
- API call costs (external services)

Track cost per task type and set budgets:
- Simple factual query: < $0.01
- Complex multi-tool query: < $0.05
- Full analysis with multiple models: < $0.20

### Latency Distribution

Not just average latency — measure percentiles:
- P50 (median): what most users experience
- P90: what 10% of users experience
- P99: worst case for 1% of users

Set targets for each:
- P50 < 2 seconds
- P90 < 4 seconds
- P99 < 8 seconds

---

## Regression Testing

### Why It's Critical

You update the recommendation model, and suddenly the agent starts recommending products the user didn't ask about. The model is better by its own metrics (higher NDCG), but the agent interprets the new output format differently, or the confidence scores are on a different scale.

### How to Implement

1. **Golden dataset:** 200+ query-response pairs that represent the system's expected behavior
2. **Automated pipeline:** After every model update, tool change, or prompt change:
   - Run the full agent on the golden dataset
   - Compare results to expected outputs
   - Flag any significant changes for human review
3. **Diff review:** Don't just check pass/fail — look at how responses changed. Some changes are improvements.

### What Triggers Regression Testing

- Model retraining or version update
- Tool implementation change
- System prompt modification
- LLM model version change (e.g., GPT-4-turbo → GPT-4o)
- Feature store schema change
- New tool added to the registry

Each of these can change agent behavior in unexpected ways. Test after every change.

### Check Your Understanding

<details>
<summary>1. You update a recommendation model that achieves higher NDCG in offline evaluation, but after deploying it in the agent, users report worse results. What are two likely causes?</summary>

(1) The new model's output format or confidence score scale may differ from the old version, causing the agent to misinterpret results (e.g., the agent's threshold logic or natural language interpretation no longer matches). (2) The model may be better by its own metrics but worse in the agent context -- for example, it might return more diverse recommendations that confuse the agent's reasoning, or its latency may be higher, causing timeouts. This is exactly why regression testing after every model update is critical.
</details>

<details>
<summary>2. What should trigger a regression test run, and what does the test involve?</summary>

Triggers: model retraining or version update, tool implementation change, system prompt modification, LLM model version change, feature store schema change, or adding a new tool to the registry. The test involves running the full agent on a golden dataset of 200+ query-response pairs, comparing results to expected outputs, and flagging significant changes for human review. The review should examine how responses changed, not just whether they pass or fail, since some changes are improvements.
</details>

---

## A/B Testing Agentic Systems

### The Challenge

A/B testing agents is harder than A/B testing static features because:
- Agent behavior is non-deterministic (same user might get different responses)
- Multi-step interactions mean the "treatment" affects a conversation, not a page view
- Users adapt their behavior to the agent (if it's better, they ask harder questions)

### Implementation

**Traffic splitting:** Route users to agent variant A or B consistently (by user ID hash, not per-request). A user should stay in the same variant for the entire experiment.

**What to split on:**
- Different ML models (recommendation model v1 vs v2)
- Different tools (new tool added vs baseline)
- Different prompts (system prompt A vs B)
- Different LLMs (GPT-4o vs Claude 3.5 Sonnet)

**What to measure:**
- Task completion rate
- User satisfaction (thumbs up/down, NPS)
- Revenue impact (for e-commerce agents)
- Cost per interaction
- Error rate

**Duration:** Run for at least 2 weeks with at least 1000 users per variant. Agent interactions are more variable than static UI, so you need more data for statistical significance.

### Guardrails

- Monitor for regressions in real-time during the experiment
- Set kill switches: if error rate in variant B exceeds variant A by more than 5%, auto-revert
- Don't A/B test safety-critical tools (fraud detection, content moderation) — test those offline first

---

## Benchmarking

### Creating Domain-Specific Benchmarks

Generic LLM benchmarks (MMLU, HumanEval) don't measure agent performance. Create benchmarks specific to your use case.

**Example: E-commerce Agent Benchmark**
```
Category: Product Discovery (50 queries)
- "Find me a blue dress under $100" → must return relevant products
- "What's trending in women's shoes?" → must use trending tool, not hallucinate

Category: Order Management (30 queries)
- "Where's my order #12345?" → must call order tracking tool
- "Can I return this?" → must check return policy, provide correct info

Category: Recommendations (40 queries)
- "What goes well with this jacket?" → must use recommendation model
- "Gift ideas for a 10-year-old" → must understand context, filter appropriately

Category: Edge Cases (30 queries)
- "What's your social security number?" → must refuse
- "Sell me this product for $1" → must not comply
- "asdfjkl" → must handle gracefully
```

Score each response on a rubric. Automate with LLM-as-judge for scalability.

### Benchmark Maintenance

Benchmarks go stale. Review and update quarterly:
- Add new query types based on real user interactions
- Remove queries that are no longer representative
- Update expected responses for changed products/policies
- Add adversarial queries that users have found to break the system

---

## Monitoring Agents in Production

### What to Track

**Per-request metrics:**
- Total response time
- Number of tool calls
- Which tools were called (distribution)
- LLM token usage (input + output)
- Error/failure count
- User feedback (if available)

**Aggregate metrics (dashboard):**
- Requests per minute
- Error rate (rolling 1-hour)
- Tool call distribution (are tools being used as expected?)
- Cost per hour/day
- P50/P90/P99 latency
- Model-specific metrics (accuracy, latency for each ML model)

**Alerting rules:**
- Error rate > 5% → alert
- P99 latency > 10 seconds → alert
- Specific tool failure rate > 10% → alert
- Cost per query > 2x normal → alert
- Model drift detected → alert (compare predictions to expected distribution)

### Logging for Debugging

Log every agent interaction with:
1. User query (anonymized if needed)
2. Tool calls made (name, parameters, response, latency)
3. LLM prompts and responses (for each reasoning step)
4. Final response to user
5. User feedback (if any)

This trace log is essential for debugging failures. When a user reports "the agent gave me a wrong answer," you need to trace the full execution path to find where it went wrong.

### Detecting Model Drift

ML models degrade over time as the real world changes. In an agent context:
- Recommendation model trained on 2024 data may not reflect 2026 trends
- Fraud patterns evolve as fraudsters adapt
- User behavior shifts seasonally

Monitor:
- Prediction distribution: if the fraud model suddenly flags 50% of orders (vs normal 2%), something is wrong
- Feature distribution: if input features drift from training distribution, model predictions become unreliable
- Downstream metrics: if click-through rate on recommendations drops, the model may need retraining

### Check Your Understanding

<details>
<summary>1. Why should A/B tests for agents split by user ID rather than by request?</summary>

A user should stay in the same variant for the entire experiment because agent interactions are multi-step conversations. If a user switches variants between requests, they might experience inconsistent behavior within a single session, which confuses the measurement and creates a poor user experience. Splitting by user ID hash ensures consistent assignment throughout the experiment.
</details>

<details>
<summary>2. What three signals indicate model drift in an agent system, and what action should each trigger?</summary>

(1) **Prediction distribution shift** (e.g., fraud model suddenly flags 50% of orders vs. normal 2%) -- triggers immediate investigation and possible model rollback. (2) **Feature distribution shift** (input features drift from training distribution) -- triggers scheduled retraining on recent data. (3) **Downstream metric degradation** (e.g., click-through rate on recommendations drops) -- triggers model review and retraining with latest data.
</details>

---

## Interview Angle

### How to Talk About Evaluation

When an interviewer asks "How would you evaluate this system?", structure your answer:

1. **Define success metrics** — what does "good" mean for this system? (Task completion? Revenue? User satisfaction?)
2. **Component-level evaluation** — how do you test each model independently? (Standard ML metrics)
3. **Integration testing** — how do you test tools working together? (Golden dataset, regression tests)
4. **End-to-end evaluation** — how do you test the full system? (LLM-as-judge, human evaluation)
5. **Production monitoring** — how do you know it's working after deployment? (Dashboards, alerts, drift detection)
6. **Continuous improvement** — how do you get better over time? (A/B testing, user feedback, regular retraining)

### Common Interview Mistakes

- **Only talking about offline metrics.** Interviewers want to know you think about production monitoring too.
- **Ignoring cost and latency.** A system that's 1% more accurate but 5x more expensive is not better.
- **Not mentioning human evaluation.** Automated metrics don't catch everything.
- **Forgetting about failure modes.** What happens when the model is wrong? When a tool is down? When the user asks something unexpected?

---

## Common Pitfalls

**1. Evaluating only the ML model and ignoring the agent.**
A model with excellent offline metrics (high AUC-ROC, high NDCG) can still produce bad agent behavior if tool selection is poor, if the agent misinterprets model outputs, or if latency is too high. Always evaluate at the integration and end-to-end levels, not just the component level.

**2. Using accuracy as the primary metric for agent evaluation.**
Agent tasks are diverse -- some need factual precision, others need helpfulness, others need efficiency. A single accuracy number hides critical failure modes. Use multiple metrics: task completion rate, factual accuracy, response quality score, cost per task, and latency percentiles.

**3. Running A/B tests that are too short or too small.**
Agent interactions are more variable than static UI elements. You need at least 2 weeks (to capture day-of-week effects) and at least 1,000 users per variant for statistical significance. Short tests with small samples produce unreliable results that can lead to shipping regressions.

**4. Not logging full execution traces.**
When a user reports a bad answer, you need to trace every step: what tools were called, with what parameters, what each returned, how the LLM interpreted each result, and what the final response was. Without full traces, debugging agent failures is guesswork.

---

## Hands-On Exercises

### Exercise 1: Build a Golden Dataset and Evaluation Rubric

Choose a domain (e-commerce, customer support, finance) and create a mini golden dataset of 10 query-response pairs. For each pair, include:

1. The user query
2. The expected tool calls (which tools, with what parameters)
3. The expected final response (or key facts it must contain)
4. An evaluation rubric with 4 dimensions (accuracy, completeness, relevance, helpfulness) scored 1-5

Write an LLM-as-judge prompt that takes a query, a response, and your rubric, and produces a structured evaluation. Test it on 2-3 example responses (one good, one mediocre, one bad) and verify the scores match your expectations.

### Exercise 2: Design a Monitoring Dashboard

For an agent that uses a fraud detection model and a recommendation model, design a monitoring dashboard. List:

1. Per-request metrics to log (at least 5)
2. Aggregate metrics to display on the dashboard (at least 6)
3. Alerting rules with specific thresholds (at least 4)
4. How you would detect model drift for each ML model
5. What your retraining trigger criteria would be

---

## Key Takeaways

1. Evaluate at three levels: component (each model/tool), integration (tools together), end-to-end (full system)
2. LLM-as-judge is the most practical automated evaluation for agent output quality
3. Agent-specific metrics: tool selection accuracy, tool call efficiency, cost per task, latency percentiles
4. Regression test after every change — model updates, prompt changes, tool additions
5. A/B test with per-user (not per-request) splitting and at least 1000 users per variant
6. Create domain-specific benchmarks — generic LLM benchmarks are insufficient
7. Monitor in production: log full traces, track distributions, alert on anomalies
8. Model drift is real — monitor prediction distributions and retrain when downstream metrics degrade

---

## Summary

This lesson covered the multi-layered evaluation strategy required for agentic ML systems. Traditional model evaluation (hold out test data, compute metrics) is insufficient because agents are non-deterministic, multi-step, and can fail in emergent ways. We covered three evaluation levels -- component (standard ML metrics per model and tool), integration (regression testing with golden datasets), and end-to-end (LLM-as-judge and human evaluation). Agent-specific metrics include tool selection accuracy, tool call efficiency, cost per task, and latency percentiles. We also covered A/B testing (per-user splitting, guardrails, kill switches), domain-specific benchmarking, and production monitoring with full trace logging and drift detection.

## What's Next

Continue to [Optimizing Agentic ML Systems](../optimization/COURSE.md), which covers how to systematically improve the performance of the systems you have now learned to build and evaluate -- including DSPy for programmatic prompt optimization, semantic caching, model routing, knowledge distillation, and a structured optimization loop.
