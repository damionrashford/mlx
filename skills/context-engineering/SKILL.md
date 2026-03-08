---
name: context-engineering
description: >
  Context engineering for building production LLM applications: context window management,
  degradation patterns, optimization strategies, memory system selection, multi-agent
  architecture, filesystem context patterns, and tool design principles.
  Use when building LLM apps, RAG pipelines, AI agents, multi-agent systems, or when
  designing memory, tool APIs, or context strategies for any language model application.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: describe the LLM system you are building (e.g. "RAG pipeline" or "multi-agent orchestrator")
---

# Context Engineering for LLM Applications

Production reference for building reliable, efficient LLM-powered systems. Context engineering is the discipline of curating what enters the model's attention window — not prompt writing, but holistic management of system prompts, tool definitions, retrieved documents, message history, and tool outputs.

---

## The Attention Budget

Context is a **finite resource with diminishing returns**. Every token added depletes an attention budget shared across all content. The engineering goal is the smallest high-signal token set that achieves the desired outcome.

**Critical attention pattern — lost-in-the-middle**: Information buried in the center of context receives 10-40% lower recall accuracy than content at the beginning or end. Always place critical instructions and key facts at the **start or end** of context.

**Context ordering for KV-cache efficiency** (stable → reusable → unique):
```
1. System prompt (never changes)
2. Tool definitions (rarely change)
3. Retrieved documents / examples (reused across requests)
4. Conversation history + current query (unique per request)
```
This ordering maximizes cache hits, dramatically reducing cost and latency.

---

## 5 Degradation Patterns

| Pattern | What happens | Mitigation |
|---------|-------------|------------|
| **Lost-in-middle** | Center content gets less attention | Put critical info at edges |
| **Context poisoning** | Hallucination or error enters context, compounds | Validate tool outputs; restart with clean context |
| **Context distraction** | Irrelevant info competes for attention budget | Filter ruthlessly; exclude what's not needed now |
| **Context confusion** | Model can't tell which context applies to current task | Segment tasks; clear section boundaries |
| **Context clash** | Multiple correct sources contradict each other | Priority rules; version filtering |

**Trigger for compaction**: at 70-80% context utilization. Don't wait for failure.

---

## 4 Optimization Strategies

### 1. Compaction
Summarize old context when approaching limits. Preserve: key decisions, file paths, error messages, current task. Never compress the system prompt. Use structured sections to prevent silent loss:

```markdown
## Session Intent
## Files Modified  
## Decisions Made
## Current State
## Next Steps
```

Correct metric: **tokens-per-task**, not tokens-per-request. A compression saving 0.5% more tokens but causing 20% more re-fetching costs more overall.

### 2. Observation Masking
Tool outputs can be 80%+ of context in agent trajectories. Replace verbose outputs with compact references once their purpose is served:

```python
if len(output) > 2000:
    ref_id = store(output)
    return f"[Result in scratch/{ref_id}.txt — key finding: {summary}]"
```

### 3. KV-Cache Optimization
Reorder context with stable elements first. Avoid dynamic content (timestamps, random IDs) in stable sections. Result: 70%+ cache hit rates on repeated requests.

### 4. Context Partitioning (Sub-Agents)
The most aggressive strategy: split work across isolated context windows. Each sub-agent operates with a clean, focused context — no accumulated history from other tasks. **This is the primary reason to use multi-agent architectures.**

---

## Memory System Selection

Start simple. Add complexity only when retrieval quality fails:

| Stage | Implementation | When to use |
|-------|---------------|-------------|
| Prototype | Plain files + JSON | Always start here — Letta filesystem agents score 74% on LoCoMo, beating Mem0's 68.5% |
| Scale | Vector store / Mem0 | When semantic search is needed or multi-tenant isolation required |
| Complex reasoning | Zep/Graphiti temporal KG | When relationships + time-travel queries matter (facts that change) |
| Full control | Letta, Cognee | When agents need to self-manage memory with deep introspection |

**Key insight**: Tool complexity matters less than reliable retrieval. A filesystem agent with good naming conventions outperforms specialized memory tools for most use cases.

**Memory layers**:
- **Working**: Context window scratchpad — place at attention-favored positions
- **Short-term**: Session files, in-memory cache
- **Long-term**: Key-value store → graph DB for cross-session knowledge
- **Temporal**: Facts with validity periods (prevents context clash from stale data)

**Anti-patterns**: Stuffing everything in context; ignoring temporal validity; no consolidation strategy.

---

## Multi-Agent Patterns

### The 3 Architectures

**Supervisor/Orchestrator** — central agent delegates to specialists
```
User → Supervisor → [Researcher, Coder, Fact-checker] → Synthesis → Output
```
Use when: clear task decomposition, human oversight important.
Weakness: supervisor context becomes bottleneck; "telephone game" fidelity loss.
Fix: `forward_message` tool lets sub-agents respond directly to users, bypassing supervisor synthesis.

**Peer-to-Peer/Swarm** — agents hand off directly
```python
def transfer_to_coder(): return coder_agent  # direct handoff
```
Use when: flexible exploration, emergent requirements.
Benefit: slightly outperforms supervisor when direct response matters.

**Hierarchical** — strategy → planning → execution layers
Use when: large-scale projects with clear organizational structure.

### Context Isolation Rule
Sub-agents exist **primarily to isolate context**, not to simulate organizational roles. Each sub-agent operates in a clean window focused on its subtask.

### Token Economics
| Architecture | Token multiplier |
|---|---|
| Single agent chat | 1× |
| Single agent with tools | ~4× |
| Multi-agent system | ~15× |

**Token budget accounts for 80% of performance variance**. Upgrading the model provides larger gains than doubling token budget. Design for realistic budgets.

### Failure Modes
- **Supervisor bottleneck**: implement output schema constraints so workers return distilled summaries
- **Divergence**: time-to-live limits on agent execution
- **Error propagation**: validate outputs before passing between agents

---

## Filesystem Context Patterns

The filesystem provides unlimited context via just-in-time loading. 6 patterns:

### 1. Scratch Pad (Tool Output Offloading)
Write large tool outputs to files instead of keeping in context:
```python
# Web search returns 10k tokens → write to file, return 100-token reference
"[Results in scratch/search_001.txt. Key finding: rate limit is 1000 req/min]"
# Agent greps file when needing specific details
```

### 2. Plan Persistence
Write plans to structured YAML files. Re-read at each turn to prevent losing track:
```yaml
# scratch/current_plan.yaml
objective: "Build RAG pipeline"
steps:
  - id: 1
    description: "Embed documents"
    status: completed
  - id: 2
    description: "Build retrieval layer"
    status: in_progress
```

### 3. Sub-Agent Communication via Files
Sub-agents write findings directly to files; coordinator reads without message-passing loss:
```
workspace/agents/researcher/findings.md
workspace/agents/coder/changes.md
workspace/coordinator/synthesis.md
```

### 4. Dynamic Skill Loading
Include only skill names in static context. Load full content when relevant:
```
Available skills (load with read_file when needed):
- rag-patterns: Chunking, embedding, retrieval strategies
- prompt-engineering: Few-shot, chain-of-thought, system prompt design
```

### 5. Terminal / Log Persistence
Persist terminal output to files; use grep for targeted extraction rather than loading entire logs.

### 6. Self-Modification (Use with caution)
Agents write learned preferences to their own instruction files, loaded next session.

---

## Tool Design Principles

### The Consolidation Principle
If a human engineer can't definitively say which tool to use in a given situation, an agent can't either. Prefer one comprehensive tool over multiple narrow tools that cover similar ground.

### Description Engineering
Every tool description must answer four questions:
1. **What** does it do? (specific, not vague)
2. **When** to use it? (triggers and contexts)
3. **What inputs** does it accept? (types, formats, examples)
4. **What does it return**? (format, fields, error conditions)

### Architectural Reduction
Sometimes fewer, more primitive tools outperform sophisticated tool collections. Direct filesystem/bash access + good documentation can replace elaborate specialized tools. Ask: does this tool **enable** new capabilities or **constrain** reasoning the model could handle?

### Tool Collection Limits
10-20 tools is the practical ceiling for most applications. Beyond that, use namespacing to create logical groupings. Tool description overlap causes model confusion.

### MCP Naming
Always use fully-qualified names: `ServerName:tool_name`. Without the server prefix, agents fail to locate tools when multiple MCP servers are available.

---

## Rules

- Treat context as a finite resource — exclude anything not needed for the current decision
- Place critical instructions at the beginning or end of context (never buried in middle)
- Trigger compaction at 70-80% context utilization
- Start with filesystem memory; add vector stores or graphs only when retrieval quality demands it
- Use sub-agents for context isolation, not for organizational metaphor
- Optimize for tokens-per-task, not tokens-per-request
- Build minimal tool architectures that benefit from model improvements
