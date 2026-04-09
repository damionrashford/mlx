---
name: claude-code
description: Comprehensive Claude Code knowledge base — plugins, hooks, skills, agents, MCP, channels, headless mode, permissions, settings, and all extensibility features. Use when building, configuring, debugging, or extending Claude Code.
argument-hint: "<Claude Code feature, topic, or question> — <what to configure, understand, or troubleshoot>"
allowed-tools: Read, Glob, Grep, WebFetch, WebSearch
---

The definitive reference skill for Claude Code's entire extensibility platform. This skill covers every feature area — from plugins and hooks to channels, subagents, MCP servers, scheduled tasks, and headless automation. When a user asks how to build, configure, debug, or extend Claude Code, this skill provides authoritative guidance grounded in the official documentation.

> ## Documentation Index
> Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

## Reference Library

27 reference documents covering every Claude Code feature. Read the relevant reference(s) before answering any question.

### Extensibility & Plugins
| Reference | What It Covers |
|-----------|---------------|
| [plugins.md](${CLAUDE_SKILL_DIR}/references/plugins.md) | Creating plugins — manifest, agents, skills, hooks, commands, settings |
| [plugins-reference.md](${CLAUDE_SKILL_DIR}/references/plugins-reference.md) | Plugin spec — plugin.json schema, variable resolution, file structure |
| [plugin-marketplaces.md](${CLAUDE_SKILL_DIR}/references/plugin-marketplaces.md) | Creating and distributing plugin marketplaces |
| [discover-plugins.md](${CLAUDE_SKILL_DIR}/references/discover-plugins.md) | Discovering and installing prebuilt plugins |
| [skills.md](${CLAUDE_SKILL_DIR}/references/skills.md) | Extending Claude with skills — SKILL.md format, frontmatter, scripts |
| [sub-agents.md](${CLAUDE_SKILL_DIR}/references/sub-agents.md) | Creating custom subagents — agent .md files, model routing, tools |

### Hooks & Automation
| Reference | What It Covers |
|-----------|---------------|
| [hooks-guide.md](${CLAUDE_SKILL_DIR}/references/hooks-guide.md) | Hooks walkthrough — setup, common patterns, prompt/agent hooks |
| [hooks-ref.md](${CLAUDE_SKILL_DIR}/references/hooks-ref.md) | Hooks reference — all 22 events, 4 handler types, input schemas, decision control |
| [scheduled-tasks.md](${CLAUDE_SKILL_DIR}/references/scheduled-tasks.md) | Running prompts on a schedule (cron-style) |
| [channels.md](${CLAUDE_SKILL_DIR}/references/channels.md) | Pushing events into a running session |
| [channels-reference.md](${CLAUDE_SKILL_DIR}/references/channels-reference.md) | Channels API reference — event types, protocols |

### Multi-Session & Programmatic
| Reference | What It Covers |
|-----------|---------------|
| [agent-teams.md](${CLAUDE_SKILL_DIR}/references/agent-teams.md) | Orchestrating teams of Claude Code sessions |
| [headless.md](${CLAUDE_SKILL_DIR}/references/headless.md) | Running Claude Code programmatically (CI, scripts, pipelines) |

### Configuration & Security
| Reference | What It Covers |
|-----------|---------------|
| [settings.md](${CLAUDE_SKILL_DIR}/references/settings.md) | All Claude Code settings — global, project, plugin scopes |
| [permissions.md](${CLAUDE_SKILL_DIR}/references/permissions.md) | Permission modes, allow/deny lists, tool-level control |
| [sandboxing.md](${CLAUDE_SKILL_DIR}/references/sandboxing.md) | Sandbox environments — Docker, macOS sandbox, network isolation |
| [model-config.md](${CLAUDE_SKILL_DIR}/references/model-config.md) | Model selection, routing, provider configuration |
| [network-config.md](${CLAUDE_SKILL_DIR}/references/network-config.md) | Enterprise network — proxies, certificates, firewall rules |
| [llm-gateway.md](${CLAUDE_SKILL_DIR}/references/llm-gateway.md) | LLM gateway configuration for enterprise deployments |

### Tools & Interface
| Reference | What It Covers |
|-----------|---------------|
| [tools-reference.md](${CLAUDE_SKILL_DIR}/references/tools-reference.md) | Built-in tools — Read, Write, Edit, Bash, Glob, Grep, Agent, etc. |
| [mcp.md](${CLAUDE_SKILL_DIR}/references/mcp.md) | Connecting Claude Code to external tools via MCP servers |
| [keybindings.md](${CLAUDE_SKILL_DIR}/references/keybindings.md) | Customizing keyboard shortcuts |
| [statusline.md](${CLAUDE_SKILL_DIR}/references/statusline.md) | Customizing the status line display |
| [output-styles.md](${CLAUDE_SKILL_DIR}/references/output-styles.md) | Output formatting and style configuration |
| [terminal-config.md](${CLAUDE_SKILL_DIR}/references/terminal-config.md) | Terminal setup optimization |
| [voice-dictation.md](${CLAUDE_SKILL_DIR}/references/voice-dictation.md) | Voice input configuration |
| [troubleshooting.md](${CLAUDE_SKILL_DIR}/references/troubleshooting.md) | Common issues, diagnostics, and fixes |

## Workflow

1. **Identify the topic** — Parse the user's question to determine which feature area(s) are involved
2. **Read references** — Load the relevant reference file(s) from the table above. For cross-cutting questions, read multiple references.
3. **Synthesize** — Combine reference material with the user's specific context (their project structure, existing config, goals)
4. **Provide guidance** — Give a concrete, actionable answer with code examples. Always cite which reference the guidance comes from.
5. **Verify** — If the user is building something (plugin, hook, skill), validate it against the spec in the reference docs

## Topic Routing

| User Asks About | Read These References |
|-----------------|---------------------|
| Building a plugin | plugins.md, plugins-reference.md |
| Publishing to marketplace | plugin-marketplaces.md, discover-plugins.md |
| Writing a skill | skills.md, plugins.md |
| Creating an agent | sub-agents.md, plugins.md |
| Setting up hooks | hooks-guide.md, hooks-ref.md |
| Hook events / types | hooks-ref.md |
| MCP servers | mcp.md |
| Permissions / security | permissions.md, sandboxing.md |
| Settings config | settings.md |
| Running headless / CI | headless.md |
| Agent teams / multi-session | agent-teams.md, channels.md |
| Channels / events | channels.md, channels-reference.md |
| Scheduled / cron tasks | scheduled-tasks.md |
| Model selection | model-config.md |
| Enterprise / network | network-config.md, llm-gateway.md |
| Keyboard shortcuts | keybindings.md |
| Status line | statusline.md |
| Output formatting | output-styles.md |
| Terminal setup | terminal-config.md |
| Voice input | voice-dictation.md |
| Something broken | troubleshooting.md |
| Built-in tools | tools-reference.md |

## Key Concepts

### The 4 Hook Types
- **command** — Runs a shell command. Exit code 0 = allow, 2 = block.
- **http** — POSTs JSON to a URL. Response body controls decision.
- **prompt** — Single-turn LLM judgment. Sends prompt + `$ARGUMENTS` to Claude for yes/no.
- **agent** — Multi-turn subagent with Read/Grep/Glob tools. Full reasoning for complex checks.

### The 22 Hook Events
SessionStart, UserPromptSubmit, PreToolUse, PermissionRequest, PostToolUse, PostToolUseFailure, Notification, SubagentStart, SubagentStop, Stop, StopFailure, TeammateIdle, TaskCompleted, InstructionsLoaded, ConfigChange, WorktreeCreate, WorktreeRemove, PreCompact, PostCompact, Elicitation, ElicitationResult, SessionEnd

### Plugin Variables
| Variable | Resolves To |
|----------|------------|
| `${CLAUDE_PLUGIN_ROOT}` | Plugin's root directory |
| `${CLAUDE_PLUGIN_DATA}` | Plugin's data/output directory |
| `${CLAUDE_SKILL_DIR}` | Current skill's directory |

### Plugin File Structure
```
plugin-name/
├── .claude-plugin/plugin.json    ← Manifest (required)
├── agents/*.md                   ← Role definitions
├── skills/*/SKILL.md             ← Workflow definitions
├── commands/*.md                 ← Entry points
├── context/*.md                  ← Shared context
├── hooks/hooks.json              ← Hook definitions
└── settings.json                 ← Plugin settings
```

## After this skill

- **Configuration question answered** → apply via **update-config** skill if settings.json change needed
- **Keybinding question answered** → apply via **keybindings-help** skill
- **Hook setup needed** → **update-config** skill handles all hook configuration
