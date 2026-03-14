## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How ML models are integrated as tools within agent architectures using the tool registry pattern
- The latency, caching, and error-handling considerations unique to serving ML models through agents
- The difference between real-time, batch, and hybrid inference patterns for agent-called models

**Apply:**
- Design an MCP tool definition and implementation flow for an ML model (e.g., recommendation, fraud detection)
- Implement caching strategies (Redis, TTL-based) for ML model outputs within agent systems

**Analyze:**
- Evaluate architectural tradeoffs when deciding where feature engineering, model inference, and output interpretation occur in an agent pipeline

---

## Prerequisites

Before starting this lesson, you should be comfortable with:

- **Model serving** (../05-production-ml/model-serving/COURSE.md) -- deploying models behind APIs, managing inference endpoints, and understanding latency budgets
- **System design** (../05-production-ml/system-design/COURSE.md) -- architecting production ML systems including feature stores, serving infrastructure, and monitoring

---

# Integrating ML Models as Agent Tools

## The Paradigm Shift

The traditional ML deployment model is straightforward: train a model, deploy it behind an API, and build an application that calls it. The input is structured, the output is structured, and the application code decides when and how to use the model.

Agents invert this relationship. An LLM agent decides when to call an ML model, what inputs to provide, and how to interpret the results. The ML model becomes a tool — one of potentially dozens — that the agent selects based on the user's request and the current context.

This is not a trivial distinction. When a human developer writes `response = fraud_model.predict(transaction)`, the call is deterministic — same code path every time. When an agent decides "I should check this transaction for fraud," it is making a probabilistic tool-selection decision. The ML model behind that tool must be robust to unexpected inputs, fast enough for interactive use, and its outputs must be interpretable enough for the agent to reason about.

Engineers who build agents that call tools dynamically are learning one half of this equation. Engineers who build ML models are learning the other. The combination — building agents that intelligently leverage ML models as tools — is a capability very few people have.

---

## Architecture: Agent → Tool Registry → Model API → Response

### The Tool Registry Pattern

Every agent framework (LangChain, CrewAI, MCP, or custom implementations) follows the same pattern:

1. **Tool definition:** A tool has a name, description, parameter schema, and implementation
2. **Tool registry:** The agent has access to a list of available tools
3. **Tool selection:** The LLM reads the user query and tool descriptions, then decides which tool(s) to call
4. **Tool execution:** The selected tool runs with the LLM-chosen parameters
5. **Result integration:** The tool's output is fed back to the LLM, which generates a response

For ML models as tools, step 4 involves calling a model inference endpoint.

### Concrete Architecture

```
User: "Will this customer churn?"
         │
         ▼
    ┌─────────────┐
    │  LLM Agent   │  ← Decides to call churn_prediction tool
    └──────┬──────┘
           │  tool_call: churn_prediction(customer_id="C-1234")
           ▼
    ┌─────────────┐
    │ Tool Handler │  ← Fetches customer features from feature store
    └──────┬──────┘
           │  features = {tenure: 24, monthly_spend: 89, ...}
           ▼
    ┌─────────────┐
    │  ML Model    │  ← model.predict(features) → 0.73
    │  (Endpoint)  │
    └──────┬──────┘
           │  prediction = {churn_prob: 0.73, risk: "high"}
           ▼
    ┌─────────────┐
    │  LLM Agent   │  ← Interprets: "This customer has a 73% churn risk..."
    └─────────────┘
```

### Key Design Decisions

**Where does feature engineering happen?** The tool handler is responsible for transforming raw data into model features. The LLM provides high-level inputs (customer_id, product_name); the tool handler fetches and engineers the features the model needs.

**Where does the model run?** Options:
- Same process (small models like scikit-learn classifiers)
- Separate microservice (GPU-based models, ML frameworks)
- Managed endpoint (SageMaker, Vertex AI, Replicate)
- Third-party API (OpenAI embeddings, Cohere rerank)

**Who interprets the output?** The LLM agent. The tool returns structured data (probabilities, embeddings, rankings), and the agent synthesizes this into a natural language response.

### Check Your Understanding

<details>
<summary>1. In the agent-tool-model architecture, why does the tool handler perform feature engineering rather than the LLM agent?</summary>

The LLM agent provides high-level inputs (e.g., customer_id, product_name) because it reasons in natural language. The tool handler is responsible for fetching raw data and transforming it into the exact feature vectors the ML model expects (e.g., numeric arrays with specific encodings). The LLM does not have access to feature stores or knowledge of the model's input schema, so delegating feature engineering to the tool handler ensures correctness and consistency.
</details>

<details>
<summary>2. What are the five steps of the tool registry pattern?</summary>

1. **Tool definition** -- a tool has a name, description, parameter schema, and implementation.
2. **Tool registry** -- the agent has access to a list of available tools.
3. **Tool selection** -- the LLM reads the user query and tool descriptions, then decides which tool(s) to call.
4. **Tool execution** -- the selected tool runs with the LLM-chosen parameters.
5. **Result integration** -- the tool's output is fed back to the LLM, which generates a response.
</details>

<details>
<summary>3. Name three deployment options for where the ML model runs when called as a tool.</summary>

Any three of: (1) same process as the agent (small models like scikit-learn classifiers), (2) separate microservice (GPU-based models), (3) managed endpoint (SageMaker, Vertex AI, Replicate), (4) third-party API (OpenAI embeddings, Cohere rerank).
</details>

---

## Deploying a Model as an MCP Tool

MCP (Model Context Protocol) is the emerging standard for tool integration. If you've built MCP servers before, the pattern will be familiar. Here's how an ML model becomes an MCP tool.

### Conceptual Walkthrough

An MCP server exposing a product recommendation model:

**Tool definition:**
```json
{
  "name": "recommend_products",
  "description": "Given a customer profile and browsing context, returns ranked product recommendations with relevance scores",
  "parameters": {
    "customer_id": "string (optional - for personalized recs)",
    "current_product_id": "string (the product being viewed)",
    "context": "string (what the customer asked about)",
    "num_results": "integer (default 5)"
  }
}
```

**Implementation flow:**
1. MCP server receives tool call with parameters
2. Fetch customer embedding from feature store (if customer_id provided)
3. Fetch product embedding for current_product_id
4. Encode context string using the query encoder
5. Call the two-tower retrieval model: find top-K candidate products by embedding similarity
6. Call the re-ranking model: score candidates using customer features + product features + context
7. Return ranked products with scores and explanations

**Response to agent:**
```json
{
  "recommendations": [
    {"product_id": "P-5678", "name": "Wool Blend Scarf", "score": 0.92, "reason": "frequently bought with current product"},
    {"product_id": "P-9012", "name": "Leather Gloves", "score": 0.87, "reason": "matches customer style profile"}
  ]
}
```

The agent then uses this structured data to generate a natural response like: "Based on your style preferences, you might love our Wool Blend Scarf — it pairs beautifully with the jacket you're looking at."

---

## Function Calling: How LLMs Select Tools

### The Mechanism

Modern LLMs (GPT-4, Claude, Grok) support function calling: the model is trained to recognize when a user's request requires an external tool and to output structured tool invocations rather than plain text.

The process:
1. System prompt includes tool definitions (name, description, parameters)
2. User sends a message
3. LLM decides: respond directly OR call a tool
4. If calling a tool: LLM outputs a structured tool call (function name + arguments as JSON)
5. Application executes the tool and sends the result back to the LLM
6. LLM generates a final response incorporating the tool result

### What Makes Good Tool Descriptions

The LLM's tool selection accuracy depends heavily on tool descriptions. Bad descriptions cause the agent to:
- Call the wrong tool
- Skip a tool when it should use it
- Provide incorrect parameters

**Good tool description principles:**
- State what the tool does in one sentence
- State when to use it (and when NOT to use it)
- Describe each parameter clearly with examples
- Specify the return format

**Bad:** `"Get product data"`
**Good:** `"Retrieve detailed product information including title, price, inventory count, and variants. Use this when the user asks about a specific product. Do NOT use for searching across products — use product_search instead."`

### Multi-Tool Orchestration

Advanced agents call multiple tools in sequence or parallel:
- **Sequential:** "Check inventory" → if low → "Create reorder alert"
- **Parallel:** "Get product recommendations" + "Check user's purchase history" simultaneously
- **Conditional:** "Run fraud check" → if suspicious → "Escalate to human review"

Many production agents implement this — parallel tool execution via `Promise.all` in the agent loop. The same pattern applies to ML model tools.

---

## Real Example: Recommendation Model for an AI Shopping Assistant

This is directly relevant to applied ML engineering roles at top companies.

### The System

An AI personal shopper helps customers find products through conversational AI. It combines:
1. **LLM agent** — understands natural language, manages conversation
2. **Product search tool** — retrieves products from the catalog
3. **Recommendation model** — ranks products for the specific user
4. **Personalization model** — adapts language and suggestions to user preferences

### How the Recommendation Model Serves as a Tool

**Agent receives:** "I'm looking for a birthday gift for my mom, she likes gardening"

**Agent's reasoning:**
1. Call `product_search(query="gardening gifts")` → returns 50 candidates
2. Call `recommend_products(customer_context="birthday gift for mother who likes gardening", candidates=[...50 product IDs...])` → returns ranked top 5
3. Synthesize results into a personalized response

**The recommendation model:**
- Input: user embedding (if logged in), query embedding, candidate product embeddings, context features
- Architecture: two-tower for retrieval, cross-attention re-ranker for scoring
- Output: ranked product list with relevance scores
- Latency requirement: < 100ms for retrieval, < 200ms for re-ranking (agent adds its own latency)

**Why the model is a tool, not a standalone system:**
- The LLM handles the conversational aspect (understanding "birthday gift for mom who likes gardening")
- The model handles the ranking aspect (which products best match this specific need)
- The LLM interprets the model's rankings for the user (explaining why each product is a good fit)

Neither the LLM nor the model alone can do this well. The LLM doesn't have real-time product inventory data. The model doesn't understand free-form natural language requests.

### Check Your Understanding

<details>
<summary>1. Why is a good tool description critical for agent-ML integration, and what are two consequences of a bad one?</summary>

The LLM's tool selection accuracy depends heavily on tool descriptions. A bad description causes the agent to: (1) call the wrong tool or skip a tool when it should use it, and (2) provide incorrect parameters to the tool. Good descriptions state what the tool does in one sentence, when to use it (and when NOT to), describe each parameter with examples, and specify the return format.
</details>

<details>
<summary>2. In the AI shopping assistant example, why does the agent call product_search before recommend_products?</summary>

The product_search tool retrieves a broad set of candidate products from the catalog (50 candidates). The recommend_products tool then takes those candidates and re-ranks them using the recommendation model (two-tower retrieval + cross-attention re-ranker) with the user's specific context. This two-stage approach separates retrieval (finding relevant items) from ranking (ordering them for this specific user and context), which is the standard pattern for large-scale recommendation systems.
</details>

---

## Real Example: Fraud Detection Agent

### The System

A payment processing agent that uses a fraud detection ML model:

**Agent receives:** A webhook for a new $2,340 order from a first-time customer, shipping to a different country than the billing address.

**Agent's reasoning:**
1. Call `fraud_score(order_id="ORD-789", features={amount: 2340, new_customer: true, ship_bill_mismatch: true})` → returns `{score: 0.82, risk: "high", top_factors: ["amount", "new_customer", "country_mismatch"]}`
2. Score > 0.7 threshold → Call `hold_order(order_id="ORD-789", reason="high fraud risk")`
3. Call `notify_merchant(message="Order ORD-789 held for review: high fraud risk (82%) due to high amount from new customer with billing/shipping mismatch")`

**The fraud model:**
- Input: 50+ features (transaction amount, customer history, device fingerprint, address matching, velocity checks)
- Architecture: gradient-boosted trees (XGBoost) — fast inference, interpretable feature importances
- Output: fraud probability (0-1) + top contributing features
- Latency requirement: < 50ms (payment authorization is time-sensitive)
- Feature store: pre-computed features updated in real-time

**Why the agent wraps the model:**
- The model outputs a number. The agent makes a decision (hold, approve, flag for review).
- The agent can apply business rules on top of the model (always hold orders > $5000, always approve returning customers with > 10 orders).
- The agent can explain the decision to the merchant in natural language.
- The agent can take follow-up actions (notify, escalate, request additional verification).

---

## Latency Considerations

Agents add overhead. Every tool call requires:
1. LLM inference to decide which tool to call (~200-1000ms)
2. Tool execution (variable — your ML model inference)
3. LLM inference to interpret the result (~200-1000ms)

For a multi-tool interaction, total latency = LLM_calls × LLM_latency + sum(tool_latencies).

### Latency Budget

For a conversational agent, total response time should be < 3 seconds (ideally < 2).

If you have 2 LLM calls at 500ms each, that leaves 1-2 seconds for all tool execution combined.

**Implications for ML model tools:**
- Model inference should be < 200ms (ideally < 100ms)
- Use batch inference for multiple items (one call, many predictions)
- Pre-compute embeddings (don't compute them at request time)
- Cache model outputs aggressively (see caching section)
- Use quantized models for faster inference

### Optimization Strategies

**Parallel tool calls:** If the agent needs recommendations AND inventory data, call both tools simultaneously. This is a common pattern in production agents.

**Streaming:** Start returning results while waiting for slower tools. Show product descriptions immediately, add recommendations when they arrive.

**Prefetching:** If you can predict what the agent will need next (user is browsing products → they'll likely want recommendations), start computing before the request.

---

## Error Handling

ML models fail in ways that traditional software doesn't:

### Model Returns Bad Predictions

- **Out-of-distribution inputs:** Customer asks about a product category the model was never trained on
- **Data drift:** Model was trained on 2024 data, now it's 2026 and shopping patterns have changed
- **Stale features:** Feature store hasn't updated, model is using old data

**Agent-level mitigation:**
- Always return confidence scores with predictions
- Agent should distrust low-confidence predictions and fall back to simpler heuristics
- "I'm not confident in this recommendation — here are some popular items in this category instead"

### Model Service Is Down

- **Timeout:** Model inference takes too long
- **Error:** Model service returns 500

**Agent-level mitigation:**
- Tool handler should have timeout (e.g., 2 seconds) and retry logic (1 retry with backoff)
- Fallback behavior: if recommendation model is down, use rule-based recommendations (most popular, recently added)
- Never let a model failure crash the entire agent interaction

### Model Returns Unexpected Format

- Model version was updated and the response schema changed
- Model returns null or empty results

**Agent-level mitigation:**
- Validate model responses against expected schema
- Handle empty results gracefully ("I couldn't find specific recommendations, but here's what's popular right now")

---

## Caching Model Outputs

ML model inference is expensive (compute cost) and slow (latency). Cache when possible.

### When Caching Works

- **Identical inputs:** Same customer viewing same product → same recommendations. Cache with (customer_id, product_id) as key.
- **Slowly changing features:** Customer preferences don't change minute-to-minute. Cache with a 15-minute TTL.
- **Embeddings:** Product embeddings don't change unless the product changes. Cache permanently, invalidate on product update.

### When Caching Doesn't Work

- **Real-time features:** Fraud detection depends on the last 5 minutes of transactions. Can't cache.
- **Highly personalized context:** Every conversation is different. The query embedding changes every turn.

### Implementation

Use Redis or Memcached with thoughtful cache keys:
- `recommend:{customer_id}:{product_id}:{context_hash}` → TTL 15 minutes
- `embedding:product:{product_id}` → TTL until product updated
- `fraud_score:{order_id}` → no cache (real-time features)

Many production agents already use caching patterns (e.g., snapshot caches with short TTLs). The same principle applies to ML model outputs.

### Check Your Understanding

<details>
<summary>1. An agent wraps a fraud detection model. The model returns a score of 0.82 with top contributing factors. Why is the agent layer valuable here beyond just calling the model?</summary>

The agent adds multiple layers of value: (1) it makes decisions based on the score (hold, approve, or flag for review) rather than just returning a number, (2) it can apply business rules on top of the model (e.g., always hold orders over $5,000), (3) it explains the decision to the merchant in natural language, and (4) it can take follow-up actions like notifying stakeholders, escalating, or requesting additional verification. The model alone only outputs a probability; the agent turns that probability into an actionable workflow.
</details>

<details>
<summary>2. Given an agent latency budget of 3 seconds with two LLM calls at 500ms each, what is the maximum acceptable latency for ML model tool execution, and what are two strategies to meet it?</summary>

With two LLM calls consuming 1000ms of the 3-second budget, tool execution has 1-2 seconds total. ML model inference should target under 200ms (ideally under 100ms). Strategies include: (1) batch inference -- making one call for many predictions instead of multiple calls, (2) pre-computing embeddings rather than computing them at request time, (3) caching model outputs aggressively, and (4) using quantized models for faster inference.
</details>

---

## Batch vs Real-Time

### Real-Time (Online Inference)

The agent calls the model for a single prediction during the conversation.

- **Use when:** Response depends on current context (what the user just said)
- **Latency:** Must be fast (< 200ms)
- **Example:** "Is this transaction fraudulent?" → needs answer NOW

### Batch (Pre-computed)

Pre-compute predictions for all likely inputs and store them. The agent looks up the pre-computed result.

- **Use when:** Predictions don't depend on real-time context
- **Latency:** Lookup is < 10ms (just a database read)
- **Example:** "What are the top 10 products for this customer segment?" → pre-computed daily

### Hybrid Pattern

Pre-compute a coarse set of candidates, then re-rank in real-time:
1. **Batch:** Nightly, compute top-100 product candidates for each customer segment
2. **Real-time:** When agent needs recommendations, fetch the pre-computed candidates and re-rank using the current conversation context

This is the standard pattern for large-scale recommendation systems. Large-scale e-commerce recommendation systems typically use this approach.

---

## Your Unique Position

Engineers who build agents that call tools dynamically are learning one discipline. Engineers who build ML models (feature engineering, training, evaluation) are learning another. The intersection — building the models AND the agents that call them — is rare.

Most ML engineers build models but don't build agents. They deploy a model endpoint and hand it to the application team. They don't understand how agents select tools, how latency compounds across multiple tool calls, or how to design model outputs that LLMs can interpret.

Most agent builders use off-the-shelf APIs (OpenAI embeddings, third-party classifiers) but can't build custom models. They can't fine-tune a model for their specific use case, can't optimize model inference for agent latency requirements, and can't evaluate whether the model is actually helping.

Doing both is a differentiator. In applied ML interviews, you can speak to:
- How the recommendation model should be architected (two-tower retrieval + re-ranker)
- How to expose it as an MCP tool with the right schema
- How the agent should select between search, recommendation, and personalization tools
- How to cache embeddings and pre-compute candidates for latency
- How to evaluate the system end-to-end (not just the model in isolation)

This is the complete picture. Most candidates can only speak to part of it.

---

## Common Pitfalls

**1. Ignoring latency compounding across tool calls.**
Agents add overhead on every tool call (LLM inference to select the tool, tool execution, LLM inference to interpret results). Engineers who optimize model inference in isolation often forget that agent latency is the sum of multiple LLM calls plus all tool calls. A model that is fast in isolation can still make the agent feel slow if the agent makes several sequential tool calls.

**2. Writing vague tool descriptions.**
The LLM selects tools based on their descriptions. Descriptions like "Get product data" cause the agent to confuse similar tools or skip the tool entirely. Every tool description should state what the tool does, when to use it, when NOT to use it, and include parameter examples. This is one of the highest-leverage investments in agent-ML integration.

**3. Not handling model failures at the agent level.**
ML models fail in ways traditional software does not -- out-of-distribution inputs, stale features, data drift, service outages. If the tool handler does not include timeouts, retry logic, and fallback behavior, a single model failure crashes the entire agent interaction. Always return confidence scores and have the agent distrust low-confidence predictions.

**4. Caching real-time-dependent model outputs.**
Not all model outputs are cacheable. Fraud detection depends on the last few minutes of transaction velocity; caching those outputs would serve stale and dangerous results. Before caching, ask whether the model's inputs include real-time features that change between requests.

---

## Hands-On Exercises

### Exercise 1: Design a Tool Definition for an ML Model

Pick an ML model you are familiar with (e.g., a sentiment classifier, a churn prediction model, a product recommendation model). Write a complete tool definition for it, including:

1. Tool name and description (following the good description principles from this lesson)
2. Parameter schema with types and examples
3. Expected response format (JSON)
4. A brief implementation flow (what the tool handler does step by step)
5. Where feature engineering happens and where the model runs
6. A caching strategy: what to cache, cache key structure, and TTL

Evaluate your tool description by asking: would an LLM know when to call this tool and when NOT to? Would it know what parameters to provide?

### Exercise 2: Latency Budget Analysis

Design a latency budget for an agent that uses two ML model tools (a retrieval model and a re-ranking model) to answer product recommendation queries. Your constraints:

- Total response time must be under 3 seconds
- You have two LLM calls (tool selection and response generation)
- The retrieval model takes 50ms and the re-ranker takes 100ms
- Feature store lookup takes 20ms

Draw out the full pipeline with timing for each step. Identify where you could apply parallel execution, prefetching, or caching to reduce total latency. Calculate the maximum acceptable LLM latency per call.

---

## Key Takeaways

1. ML models as agent tools = model inference wrapped in tool handlers with proper schemas
2. Tool descriptions determine whether the agent selects the right tool — invest heavily in them
3. Latency budgets are tight: model inference should be < 200ms when called by an agent
4. Cache aggressively: embeddings, pre-computed candidates, recent predictions
5. Handle model failures gracefully: timeouts, fallbacks, confidence-based routing
6. The hybrid batch + real-time pattern is standard for recommendations at scale
7. Building both agents AND models is a rare, high-value skill combination

---

## Summary

This lesson covered how ML models are integrated as tools within agent architectures. The tool registry pattern (definition, registry, selection, execution, result integration) is the universal structure across agent frameworks. We walked through two detailed real-world examples -- a recommendation model for an AI shopping assistant and a fraud detection agent -- showing how the agent handles conversational understanding while the ML model handles prediction and ranking. Key operational concerns include tight latency budgets (model inference should be under 200ms), strategic caching (with awareness of what can and cannot be cached), graceful error handling for model failures, and the hybrid batch + real-time pattern for recommendations at scale.

## What's Next

Continue to [Evaluating Agentic ML Systems](../evaluation/COURSE.md), which covers how to evaluate the systems you build in this lesson -- from component-level ML metrics to end-to-end agent evaluation, regression testing, A/B testing, and production monitoring. After that, [Optimizing Agentic ML Systems](../optimization/COURSE.md) addresses how to systematically improve latency, cost, and quality in the agent-ML pipeline.
