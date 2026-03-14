# Context Engineering Patterns

## Memory System Selection

| Pattern | Persistence | Best for |
|---------|-------------|----------|
| System prompt | None (static) | Instructions, persona, constraints |
| Conversation history | Session | Multi-turn chat |
| RAG (vector store) | Persistent | Large document collections |
| Knowledge graph | Persistent | Relational/structured knowledge |
| Filesystem context | Persistent | Code, configs, project state |
| Scratchpad/notes | Session | Working memory, intermediate results |

## Context Window Allocation

| Component | Budget % | Priority |
|-----------|----------|----------|
| System prompt | 5-10% | Highest (always present) |
| Tool definitions | 5-15% | High (needed for tool use) |
| Retrieved documents | 30-50% | Medium (query-dependent) |
| Conversation history | 20-30% | Medium (recency-weighted) |
| Current query | 5-10% | Highest |

## Degradation Strategies

When context exceeds budget:
1. Summarize old conversation turns
2. Reduce retrieved document count
3. Compress tool definitions
4. Truncate long tool outputs
5. Drop least-relevant context

## RAG Pipeline Checklist

- [ ] Chunking strategy matches query granularity
- [ ] Embedding model matches domain
- [ ] Retrieval returns 3-10 chunks (not 50)
- [ ] Re-ranking applied after initial retrieval
- [ ] Retrieved context placed at start or end (not middle)
- [ ] Hallucination guardrails in place
