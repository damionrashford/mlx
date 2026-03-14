# 08 — Agent-ML Integration

> The competitive edge: combining trained ML models with agentic systems. This is what most ML engineers CAN'T do.

## Why This Matters

Modern ML roles involve building systems like an AI Personal Shopper — that's an AGENT that uses ML MODELS as tools. If you already build agents, the next step is learning to build the models those agents call.

## Subdirectories

```
08-agent-ml-integration/
├── agents-with-ml-tools/    # Deploying models as tools agents can call
├── evaluation/              # Evaluating agentic ML systems end-to-end
└── optimization/            # DSPy, programmatic optimization, caching
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| DeepLearning.AI: Evaluating AI Agents | How to evaluate agentic systems | ~2 hrs |
| DeepLearning.AI: DSPy — Build Optimize Agentic Apps | Programmatic agent optimization | ~2 hrs |
| DeepLearning.AI: Building and Evaluating Data Agents | Data agents with evaluation | ~2 hrs |
| DeepLearning.AI: Semantic Caching for AI Agents | Performance optimization | ~2 hrs |

## The Vision

```
Traditional ML Engineer:    Data → Model → API endpoint → done
Traditional AI Engineer:    LLM → Agent → Tools → done
Full-stack AI Engineer:     Data → Model → API → Agent Tool → Autonomous System
```

### Example: AI Personal Shopper

```
Shopper asks: "Find me a cozy sweater under $80"
│
├── AGENT orchestrates the request
│   ├── Calls RECOMMENDATION MODEL (your trained model)
│   │   └── Returns ranked product IDs based on user preferences
│   ├── Calls SEARCH API (retrieval)
│   │   └── Filters by price, availability, attributes
│   ├── Calls PERSONALIZATION MODEL (fine-tuned embeddings)
│   │   └── Re-ranks based on browsing history
│   └── Calls LLM (generation)
│       └── Generates natural language response with recommendations
│
└── Agent returns personalized, conversational product suggestions
```

The goal is to be able to build ALL of this end-to-end.
