## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The measure-identify-optimize-measure loop and why optimizing without measuring is the most common mistake
- How DSPy replaces manual prompt iteration with systematic, data-driven optimization
- The tradeoffs between optimization techniques (streaming, semantic caching, model routing, knowledge distillation) in terms of impact, effort, and applicability

**Apply:**
- Apply semantic caching with embedding similarity to reduce redundant LLM and model inference calls
- Use the optimization decision framework to select the right technique for a given bottleneck

**Analyze:**
- Given instrumentation data (latency breakdown, cost breakdown), identify the actual bottleneck in an agentic ML pipeline and select the highest-impact optimization

---

## Prerequisites

Before starting this lesson, you should be comfortable with:

- **Integrating ML models as agent tools** (./agents-with-ml-tools/COURSE.md) -- the agent-tool-model architecture, tool selection, latency budgets, and caching patterns
- **Inference optimization** (../03-llm-internals/inference-optimization/COURSE.md) -- model quantization, batching, KV-cache, and serving infrastructure for LLMs

---

# Optimizing Agentic ML Systems

## The Optimization Mindset

Every agentic ML system has a performance ceiling determined by its weakest component. An agent with a perfect recommendation model but a 3-second LLM latency is slow. An agent with fast inference but poor tool selection wastes money on wrong tool calls. An agent that's fast and accurate but costs $0.50 per query is uneconomical at scale.

Optimization follows a simple loop:

1. **Measure** — instrument everything, establish baselines
2. **Identify the bottleneck** — where is time/money/quality being lost?
3. **Optimize the bottleneck** — apply the right technique
4. **Measure again** — verify improvement, check for regressions
5. **Repeat** — the next bottleneck is now something else

The biggest mistake is optimizing without measuring. Engineers spend days optimizing model inference from 80ms to 40ms when the real bottleneck is a 1.5-second LLM call. Always measure first.

---

## DSPy: Programmatic Optimization of LLM Programs

### What DSPy Is

DSPy (Declarative Self-improving Language Programs in Python) is a framework from Stanford that treats LLM interactions as optimizable programs rather than manual prompt engineering.

The core insight: instead of manually writing prompts and hoping they work, define what you want (input/output signatures) and let DSPy optimize how the LLM achieves it — including prompt wording, few-shot examples, and even which LLM to use.

### Why It Matters for Agents

Agent developers typically iterate on prompts manually:
1. Write a system prompt
2. Test on a few examples
3. Find a failure case
4. Modify the prompt to handle that case
5. Retest — now a previously working case breaks
6. Repeat forever

This is unsystematic, unscalable, and fragile. DSPy replaces this with:
1. Define a metric (task completion rate, tool selection accuracy)
2. Provide a training set of examples
3. DSPy automatically optimizes the prompt to maximize the metric
4. The optimized prompt generalizes better than manual iteration

### Key Concepts

**Signatures:** Define what a module does (input → output):
```
"question -> tool_name, tool_params"
"tool_output, context -> final_answer"
```

**Modules:** Composable building blocks (like `dspy.ChainOfThought`, `dspy.ReAct`). Each module generates a prompt for the LLM and processes the response.

**Teleprompters (Optimizers):** Algorithms that optimize modules:
- `BootstrapFewShot`: finds good few-shot examples from training data
- `COPRO`: optimizes the instruction text
- `MIPRO`: combines instruction + few-shot optimization
- `BootstrapFinetune`: generates training data and fine-tunes a smaller model

**Metrics:** Functions that score the quality of a module's output. DSPy optimizes these.

### Practical Application

For an agent that selects tools:

1. Define the signature: `"user_query, available_tools -> selected_tool, parameters"`
2. Create a training set: 200 (query, correct_tool, correct_params) examples
3. Define a metric: `tool_selection_accuracy(predicted_tool, expected_tool)`
4. Run DSPy optimization: `teleprompter.compile(module, trainset=train, metric=metric)`
5. Result: an optimized prompt with few-shot examples that achieves higher tool selection accuracy than your manual prompt

### When to Use DSPy

- You have a measurable metric for your agent's behavior
- You have at least 100 labeled examples
- Manual prompt iteration has plateaued
- You need reproducible, systematic prompt optimization

DSPy is not magic — it requires labeled data and a clear metric. But when you have those, it consistently outperforms manual prompting by 5-20%.

### Check Your Understanding

<details>
<summary>1. What is the biggest mistake engineers make when optimizing agentic ML systems?</summary>

Optimizing without measuring first. Engineers often spend days optimizing one component (e.g., reducing model inference from 80ms to 40ms) when the real bottleneck is elsewhere (e.g., a 1.5-second LLM call). Always instrument the system, establish baselines, and identify the actual bottleneck before applying any optimization technique.
</details>

<details>
<summary>2. What are DSPy's four key concepts, and how do they work together?</summary>

(1) **Signatures** define what a module does (input-output specification). (2) **Modules** are composable building blocks that generate prompts and process responses (e.g., ChainOfThought, ReAct). (3) **Teleprompters (Optimizers)** are algorithms that optimize modules by finding the best prompts, few-shot examples, or fine-tuning data. (4) **Metrics** are scoring functions that evaluate module output quality. Together: you define a signature, wrap it in a module, define a metric, and the teleprompter optimizes the module to maximize the metric on your training data.
</details>

<details>
<summary>3. When should you NOT use DSPy?</summary>

When you do not have a measurable metric for your agent's behavior, when you have fewer than 100 labeled examples, when manual prompt iteration has not yet been tried (DSPy is for when manual iteration has plateaued), or when you do not need reproducible systematic optimization.
</details>

---

## Prompt Optimization

Even without DSPy, there are systematic approaches to prompt optimization.

### Prompt Structure That Works

For agent system prompts, a proven structure:

1. **Role and context** (2-3 sentences): Who is the agent? What domain?
2. **Available tools** (auto-generated): Tool names, descriptions, parameters
3. **Decision rules** (explicit): When to use each tool, when NOT to
4. **Output format** (strict): How to format tool calls and responses
5. **Few-shot examples** (2-5): Real examples of good tool selection and responses
6. **Guardrails** (explicit): What to never do

### Common Prompt Issues and Fixes

**Problem: Agent calls tools when it should answer directly.**
Fix: Add explicit instruction: "Only call a tool when you need information you don't have. For greetings, clarifications, and general knowledge questions, respond directly."

**Problem: Agent provides wrong parameters to tools.**
Fix: Add parameter examples in tool descriptions. Instead of `"query: string"`, use `"query: string (e.g., 'blue running shoes', 'gifts under $50')"`.

**Problem: Agent hallucinates information not returned by tools.**
Fix: Add instruction: "Base your response ONLY on information returned by tools. If tools don't provide enough information, say so. Never invent product names, prices, or details."

**Problem: Agent makes redundant tool calls.**
Fix: Add instruction: "Before calling a tool, check if you already have the needed information from a previous tool call in this conversation."

### Automated Prompt Testing

Create a test suite for your prompt:
```
tests = [
    {"query": "hi", "expected": "no_tool_call"},
    {"query": "ETH price", "expected": "crypto_price"},
    {"query": "compare BTC and ETH", "expected": "crypto_compare"},
    {"query": "my portfolio", "expected": "portfolio_summary"},
]
```

Run these tests after every prompt change. Measure pass rate. Only deploy prompts that pass > 95% of tests.

---

## Tool Selection Optimization

### The Problem

Tool selection accuracy is the single most impactful optimization for agents. If the agent calls the wrong tool, everything downstream is wrong — and you've wasted time and money on a useless tool call.

### Strategies

**1. Better tool descriptions.** The LLM picks tools based on descriptions. Invest time making them precise:
- State what the tool does AND what it doesn't do
- Include example queries that should trigger this tool
- Differentiate similar tools explicitly ("Use product_search for finding products by description. Use product_lookup for fetching a specific product by ID.")

**2. Tool grouping.** If you have 50 tools, the LLM struggles to pick the right one. Group tools and use a two-stage selection:
- Stage 1: LLM picks a category (crypto, DeFi, portfolio, browser)
- Stage 2: LLM picks a specific tool within the category

Production agents with many tools commonly organize them this way (e.g., grouping by domain — market data tools, DeFi tools, chain tools, etc.). The `tool_search` meta-tool is another pattern — let the LLM search for tools by description.

**3. Few-shot examples.** Include 3-5 examples of correct tool selection in the prompt. This is the highest-impact optimization for tool selection accuracy.

**4. Fine-tuning for tool selection.** If you have enough data (1000+ examples), fine-tune a small model specifically for tool selection. This model runs before the main LLM, predicting which tool(s) are needed. It's faster and cheaper than having the main LLM do tool selection.

---

## Semantic Caching

### The Concept

Traditional caching matches exact inputs: same query → cached response. Semantic caching matches similar inputs: "What's ETH trading at?" and "Current Ethereum price?" should return the same cached response.

### Implementation

1. **Embed the query** using a fast embedding model (e.g., all-MiniLM-L6-v2, < 5ms)
2. **Search the cache** for embeddings within a similarity threshold (cosine similarity > 0.92)
3. **If hit:** return cached response (skip LLM + tool calls entirely)
4. **If miss:** run the full agent pipeline, cache the result with its embedding

### Architecture

```
User Query
    │
    ▼
Embed Query (5ms)
    │
    ▼
Search Redis/Vector DB (10ms)
    │
    ├── Cache Hit → Return cached response (total: 15ms)
    │
    └── Cache Miss → Run full agent pipeline (total: 2-5 seconds)
                          │
                          └── Cache result for future queries
```

### When Semantic Caching Works

- **High-repetition queries:** Customer support, FAQ-like questions
- **Stable data:** Product information that doesn't change minute-to-minute
- **Cost-sensitive deployments:** Every cache hit saves $0.01-0.10 in LLM costs

### When It Doesn't Work

- **Personalized queries:** "What should I buy?" depends on the user
- **Real-time data:** "What's the current ETH price?" needs a fresh answer
- **Context-dependent:** Same query might need different answers in different conversations

### Implementation with Redis

Redis supports vector similarity search natively (since Redis 7.0 with RedisSearch). Store query embeddings as vectors, search by cosine similarity. TTL controls cache freshness.

If your agent already uses Redis for caching, adding semantic caching is a natural extension.

### Check Your Understanding

<details>
<summary>1. What is the difference between traditional caching and semantic caching, and what similarity threshold is typically used?</summary>

Traditional caching matches exact inputs (same query string returns the cached response). Semantic caching matches semantically similar inputs by comparing query embeddings -- "What's ETH trading at?" and "Current Ethereum price?" would hit the same cache entry. The typical similarity threshold is cosine similarity > 0.92. A cache hit skips the entire agent pipeline (saving 2-5 seconds), returning in about 15ms (5ms for embedding + 10ms for vector search).
</details>

<details>
<summary>2. Name two scenarios where semantic caching works well and two where it does not.</summary>

Works well: (1) high-repetition queries like customer support or FAQ-like questions, (2) queries about stable data like product information that does not change minute-to-minute. Does not work: (1) personalized queries where the same question needs different answers for different users, (2) real-time data queries where freshness is critical (e.g., "What's the current ETH price?").
</details>

---

## Cost Optimization

### Model Routing

Not every query needs the most powerful (and expensive) LLM. Route queries based on complexity:

| Query Type | Model | Cost per 1K tokens |
|------------|-------|-------------------|
| Simple factual | GPT-4o-mini, Claude 3 Haiku | $0.00015 |
| Standard reasoning | GPT-4o, Claude 3.5 Sonnet | $0.005 |
| Complex analysis | GPT-4, Claude 3 Opus | $0.015 |

**Router implementation:**
1. Small classifier (or rule-based) analyzes the query
2. Simple queries (greetings, lookups, single-tool calls) → cheap model
3. Complex queries (multi-step reasoning, analysis, comparison) → expensive model

**Expected savings:** 60-80% of queries are simple. If 70% of traffic goes to a model that's 10x cheaper, total cost drops by ~63%.

### Token Optimization

LLM cost = input_tokens + output_tokens. Reduce both:

**Reduce input tokens:**
- Trim tool descriptions (remove examples for tools unlikely to be used)
- Summarize long tool outputs before feeding back to the LLM
- Use shorter system prompts (every saved word saves tokens on every request)

**Reduce output tokens:**
- Set max_tokens limits per response type
- Instruct the LLM to be concise
- Use structured output (JSON) instead of verbose prose for intermediate steps

### Caching Cost Impact

At scale, caching has enormous impact:

| Metric | Without Cache | With Semantic Cache (50% hit rate) |
|--------|---------------|-----------------------------------|
| Queries/day | 100,000 | 100,000 |
| LLM calls/day | 100,000 | 50,000 |
| Cost/day | $500 | $250 |
| Monthly savings | - | $7,500 |

---

## Latency Optimization

### Parallel Tool Calls

When an agent needs multiple pieces of information, call tools simultaneously:

```
Sequential (bad):
  crypto_price("ETH") → 200ms
  crypto_price("BTC") → 200ms
  portfolio_summary() → 150ms
  Total: 550ms

Parallel (good):
  Promise.all([
    crypto_price("ETH"),
    crypto_price("BTC"),
    portfolio_summary()
  ]) → 200ms (max of the three)
```

Many production agents already implement parallel tool execution. This is one of the most impactful latency optimizations.

### Prefetching

Predict what the agent will need and start fetching before the request:

- User opens the crypto dashboard → prefetch prices for portfolio holdings
- User asks about a product → prefetch related recommendations
- User starts a conversation about ETH → prefetch ETH price, chart data, recent news

### Streaming

Don't wait for the entire response before showing something to the user:

1. Start streaming the LLM's text response immediately
2. When a tool call is needed, show a loading indicator
3. Resume streaming when the tool returns
4. User sees partial results within 500ms instead of waiting 3 seconds for the complete response

### Connection Pooling and Keep-Alive

For model endpoints called frequently:
- Use HTTP keep-alive to avoid TCP handshake per request
- Use connection pools sized to expected concurrency
- Pre-warm connections on service start

---

## Knowledge Distillation for Agents

### The Concept

Large models are good at many things. Your agent only needs to be good at specific things. Train a smaller model to match the large model's behavior on your specific tasks.

### Application to Agents

**Step 1:** Use a large LLM (GPT-4, Claude Opus) to generate high-quality tool selections and responses for 10,000 queries.

**Step 2:** Fine-tune a small model (Llama-8B, Mistral-7B) on these (query, tool_call, response) pairs.

**Step 3:** Deploy the small model as your agent's LLM.

**Result:** The small model handles 80-90% of queries as well as the large model, at 10-50x lower cost and 3-5x lower latency. Route the remaining 10-20% (complex/unusual queries) to the large model.

### When to Distill

- You have a mature agent with stable tool definitions
- You've collected enough production data to train on (5,000+ examples minimum)
- Cost or latency is a primary concern
- Most queries follow common patterns

### When NOT to Distill

- Your agent is still evolving (tools changing, prompt iterating)
- You need the large model's reasoning for most queries
- Your query distribution is highly diverse (few repeating patterns)

### Check Your Understanding

<details>
<summary>1. How does model routing save costs, and what is the expected savings?</summary>

Model routing directs queries to different LLMs based on complexity. Simple queries (greetings, lookups, single-tool calls) go to cheap models (GPT-4o-mini, Claude 3 Haiku), while complex queries (multi-step reasoning, analysis) go to expensive models. Since 60-80% of queries are typically simple, routing 70% of traffic to a model that is 10x cheaper reduces total cost by approximately 63%.
</details>

<details>
<summary>2. What are the three steps of knowledge distillation for agents, and when should you NOT use it?</summary>

Steps: (1) Use a large LLM (GPT-4, Claude Opus) to generate high-quality tool selections and responses for 10,000+ queries. (2) Fine-tune a small model (Llama-8B, Mistral-7B) on these (query, tool_call, response) pairs. (3) Deploy the small model as your agent's LLM, routing complex queries to the large model. Do NOT use knowledge distillation when your agent is still evolving (tools changing, prompt iterating), when you need the large model's reasoning for most queries, or when your query distribution is highly diverse with few repeating patterns.
</details>

---

## The Optimization Loop in Practice

### Phase 1: Measure Everything

Before optimizing anything, instrument your system:

- Log every LLM call with input/output tokens, latency, and cost
- Log every tool call with parameters, result size, and latency
- Track end-to-end response time per query
- Track task completion rate (does the user get a useful answer?)
- Track total cost per query

### Phase 2: Identify the Bottleneck

Look at the data:

| Component | Avg Latency | % of Total Time | Avg Cost | % of Total Cost |
|-----------|------------|-----------------|----------|-----------------|
| LLM call 1 (tool selection) | 800ms | 32% | $0.003 | 30% |
| Tool execution | 200ms | 8% | $0.001 | 10% |
| LLM call 2 (response) | 1200ms | 48% | $0.005 | 50% |
| Network/overhead | 300ms | 12% | - | 10% |

In this example:
- **Latency bottleneck:** LLM call 2 (response generation) — 48% of time
- **Cost bottleneck:** LLM call 2 — 50% of cost

### Phase 3: Apply the Right Technique

For the example above:
- **Latency:** Stream LLM call 2 (user sees first token in 200ms instead of waiting 1200ms)
- **Cost:** Route simple queries to a cheaper model for call 2. Use semantic caching for repeated queries.
- **Compute:** Quantize the tool's ML model if tool execution is the bottleneck

### Phase 4: Measure Again

After optimization:
- Did latency improve? By how much?
- Did cost decrease? Any quality regression?
- Did anything break? (Regression tests)

### Phase 5: Repeat

The bottleneck has shifted. Maybe now network overhead is the biggest latency component. Or maybe tool selection accuracy dropped after switching to a cheaper model. Optimize the new bottleneck.

---

## Optimization Decision Framework

| Bottleneck | Technique | Expected Impact | Effort |
|------------|-----------|-----------------|--------|
| LLM latency | Streaming | Perceived 3-5x faster | Low |
| LLM latency | Smaller model for simple queries | 2-5x faster | Medium |
| LLM cost | Model routing | 50-70% reduction | Medium |
| LLM cost | Semantic caching | 30-60% reduction | Medium |
| LLM cost | Knowledge distillation | 80-95% reduction | High |
| Tool latency | Parallel execution | 2-5x faster | Low |
| Tool latency | Prefetching | 50-90% reduction | Medium |
| Tool latency | Model quantization | 2-4x faster inference | Medium |
| Tool accuracy | Better descriptions | 5-15% improvement | Low |
| Tool accuracy | Few-shot examples | 10-20% improvement | Low |
| Tool accuracy | DSPy optimization | 5-20% improvement | Medium |
| Tool accuracy | Fine-tuned router | 10-30% improvement | High |
| Quality | Prompt optimization | Variable | Low-Medium |
| Quality | Better ML models | Variable | High |

### Priority Order

1. **Low effort, high impact first:** Streaming, parallel execution, better tool descriptions
2. **Medium effort, high impact next:** Semantic caching, model routing, few-shot examples
3. **High effort for scale:** Knowledge distillation, fine-tuned router, DSPy optimization

---

## Common Pitfalls

**1. Optimizing the wrong bottleneck.**
The most common mistake. Engineers spend days shaving milliseconds off model inference when the real bottleneck is LLM response generation consuming 48% of total latency. Always instrument and measure before optimizing. The optimization decision framework table exists precisely to prevent this.

**2. Setting the semantic cache similarity threshold too low.**
A threshold of 0.85 instead of 0.92 will return cached responses for queries that are similar but not equivalent, leading to incorrect answers. For example, "What's the price of ETH?" and "What's the price of BTC?" might have high cosine similarity but need completely different answers. Start with a conservative threshold (0.92+) and lower it only after validating on real queries.

**3. Distilling too early.**
Knowledge distillation requires a mature agent with stable tool definitions and 5,000+ production examples. Distilling while the agent is still evolving (tools changing, prompts iterating) means the small model learns behaviors that will soon be outdated, wasting the entire effort.

**4. Reducing token count at the expense of tool description quality.**
Token optimization suggests trimming tool descriptions to reduce input tokens. But tool descriptions are the highest-leverage text in the prompt -- vague or truncated descriptions cause tool selection errors that cost far more than the tokens saved. Never sacrifice tool description clarity for token savings.

---

## Hands-On Exercises

### Exercise 1: Bottleneck Analysis

Given the following instrumented data for an agentic ML system, identify the bottleneck and propose the right optimization:

| Component | Avg Latency | % of Total Time | Avg Cost | % of Total Cost |
|-----------|------------|-----------------|----------|-----------------|
| LLM call 1 (tool selection) | 400ms | 16% | $0.002 | 15% |
| Tool A: product search | 150ms | 6% | $0.001 | 8% |
| Tool B: recommendation model | 300ms | 12% | $0.003 | 23% |
| LLM call 2 (response generation) | 1500ms | 60% | $0.006 | 46% |
| Network/overhead | 150ms | 6% | - | 8% |

For each of latency and cost, identify the bottleneck component, propose a specific optimization technique from the decision framework, and estimate the expected impact. Then explain what the new bottleneck would be after your optimization.

### Exercise 2: Semantic Cache Design

Design a semantic caching layer for a customer support agent that handles questions about return policies, shipping times, and order tracking. Specify:

1. Which query types are cacheable and which are not (and why)
2. Your embedding model choice and why
3. Your similarity threshold and reasoning
4. Your TTL strategy for different query types
5. Your cache key structure
6. An estimate of cache hit rate and the resulting cost savings

---

## Key Takeaways

1. Always measure before optimizing — find the actual bottleneck, don't guess
2. DSPy replaces manual prompt iteration with systematic optimization using labeled data
3. Semantic caching can eliminate 30-60% of LLM calls for repetitive queries
4. Model routing (cheap model for simple queries, expensive for complex) saves 50-70% on costs
5. Parallel tool calls are the easiest latency win — many production agents already implement this
6. Knowledge distillation: use a large model to train a small model for your specific agent tasks
7. The optimization loop never ends — fix one bottleneck, find the next
8. Start with low-effort optimizations (streaming, better descriptions) before investing in complex ones

---

## Summary

This lesson covered the systematic optimization of agentic ML systems across latency, cost, and quality dimensions. The core principle is always measure before optimizing -- identify the actual bottleneck rather than guessing. We covered DSPy for programmatic prompt optimization (replacing manual prompt iteration with data-driven optimization), semantic caching (eliminating 30-60% of redundant LLM calls), model routing (directing simple queries to cheap models for 50-70% cost reduction), parallel tool execution (the easiest latency win), and knowledge distillation (training small models on large model outputs for 80-95% cost reduction at scale). The optimization decision framework maps each bottleneck type to the right technique with expected impact and effort level.

## What's Next

With the agent-ML integration module complete (building, evaluating, and optimizing agentic ML systems), continue to [Interview Prep: Core Concepts](../09-interview-prep/concepts/COURSE.md) to consolidate your knowledge of the 20 most-tested ML concepts. From there, [System Design](../09-interview-prep/system-design/COURSE.md) and [Pair Programming](../09-interview-prep/pair-programming/COURSE.md) will prepare you for the full interview loop.
