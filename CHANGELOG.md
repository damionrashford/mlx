# Changelog

## [1.1.7] - 2026-04-08
### Fixed (Critical)
- **Agent frontmatter**: removed `hooks` and `mcpServers` from all 4 agents (`ml-engineer`, `dl-engineer`, `data-scientist`, `ml-ops`) — plugin-shipped agents do not support these fields per Claude Code docs; they were silently ignored when installed via marketplace
- **mlx-experiments MCP server**: restored to plugin-level `plugin.json` `mcpServers` (was deleted with .mcp.json, leaving the server unreachable for plugin installs)
- **colab-mcp**: moved from agent frontmatter to plugin-level `mcpServers` so it's accessible to all agents
### Added
- **`WorktreeCreate` hook** → `worktree-setup.sh`: initializes EXPERIMENT.md template, results.tsv header, and `data/` when ml-engineer or dl-engineer creates their git worktree
- **`WorktreeRemove` hook** → `worktree-archive.sh`: archives results.tsv, EXPERIMENT.md, and artifact list to `.claude/experiments/` before worktree teardown
- **`SubagentStart` hook** → `subagent-start.sh`: logs agent kickoff with timestamp to `${CLAUDE_PLUGIN_DATA}/agent-log.txt`
- **`UserPromptSubmit` hook** → `prompt-guard.sh`: fast pattern matching catches train-on-test, full-dataset eval anti-patterns; injects routing hints for paper search, business analysis, DL, and RAG tasks
- **`PermissionRequest` hook**: routes through mlops-safety-check for any permission dialog
- **`PermissionDenied` hook** → `permission-denied.sh`: returns `{retry: true}` for safe read/info-gathering blocks; hard-stops destructive operations
- **`InstructionsLoaded` hook** → `instructions-loaded.sh`: reinjects active experiment hypothesis and run count when CLAUDE.md or rules files load
- **`StopFailure` hook** → `stop-failure.sh`: logs API errors to `${CLAUDE_PLUGIN_DATA}/stop-failures.log`; emits recovery context for next turn
- **`TaskCreated` / `TaskCompleted` hooks** → `task-log.sh`: full task lifecycle audit trail in `${CLAUDE_PLUGIN_DATA}/task-log.txt`
- **`Stop` hook** (plugin-level): `agent-stop-summary.sh` now fires at plugin-level Stop in addition to SubagentStop
- **`prompt` hook type** on `PreToolUse` Write/Edit `*.py`: LLM-based leakage detection as a second validation layer after the regex shell hook
- **`http` hook type** on `FileChanged(results.tsv)`: POSTs experiment events to `webhook_url` userConfig (Slack, Discord, custom endpoint)
- **`dep-check.sh`** SessionStart hook: pre-warms uv dep cache for experiments server using `${CLAUDE_PLUGIN_DATA}` hash-diff pattern — avoids first-invocation delay
- **`webhook_url` userConfig**: optional webhook endpoint for experiment notifications
- **plugin-root `settings.json`**: already present, sets `ml-workbench` as default agent
### Changed
- `PreToolUse` Bash safety check (mlops-safety-check.sh) is now plugin-level (fires for all agents) instead of ml-ops-only — destructive deployment commands blocked everywhere
- `SubagentStop` now calls both `subagent-log.sh` (async) and `agent-stop-summary.sh` (sync emit to parent)
- README hooks table expanded: 11 → 24 rows covering all 20 active hook events with `prompt`/`http` type callouts
- `plugin.json`: `description` "5 free repositories" → "5 repositories"; version bumped to 1.1.7
- `marketplace.json`: version bumped to 1.1.7

## [1.1.6] - 2026-04-08
### Added
- **Colab MCP server** (`colab-mcp`) wired directly into `dl-engineer` and `ml-engineer` frontmatter — connects to active browser Colab session for cloud GPU training
- Non-blocking Colab preflight in both agents: `open "URL" &` opens browser without blocking agent; `## Colab MCP` section added to both system prompts with workflow and usage guidance
- **`mlx-experiments` MCP server** added to `ml-engineer`, `dl-engineer`, and `data-scientist` agent frontmatter for experiment tracking
- **`research` skill** added to `dl-engineer` and `ml-engineer` — closes dataset download gap (was missing from both)
- **`outputStyles` field** added to `plugin.json` pointing to `./output-styles/` — Claude Code now discovers and surfaces output styles
- **Output style frontmatter** added to all 3 style files (`terse.md`, `report.md`, `notebook.md`): `name`, `description`, `keep-coding-instructions: true`
- **Agent Stop hooks** wired for `ml-engineer`, `dl-engineer`, `data-scientist` → `agent-stop-summary.sh` fires at end of each agent session
- **`ml-ops` PreToolUse Bash hook** → `mlops-safety-check.sh` fires before every Bash call to enforce deployment safety guardrails
### Changed
- **Memory scope**: `memory: user` → `memory: project` for all 7 applicable agents (`ml-engineer`, `dl-engineer`, `data-scientist`, `ai-engineer`, `data-analyst`, `data-engineer`, `ml-ops`) — project memory is git-shareable and project-scoped
- **`## Memory` sections** rewritten across all 11 agents with specific save targets, query triggers, and domain-appropriate retention guidance; added Memory sections to `ml-tutor` and `ml-workbench` (were missing)
- **`data-scientist`**: Step 0 cleaned — removed 3 hardcoded `python3 ${CLAUDE_SKILL_DIR}/scripts/datasets.py` calls; replaced with "use the **research skill**"
- **`ml-researcher`**: Protocol sections 2–8 cleaned — removed all `python3 ${CLAUDE_SKILL_DIR}/scripts/` references (search/download/extract/datasets/auth/generate); replaced with skill name references
- **`ai-engineer`**: Phase 2 cleaned — removed hardcoded `datasets.py search` code block; replaced with research skill reference
- **`autoexperiment`** and **`fine-tune`** skills: added `disable-model-invocation: true` — these high-impact skills only run when user explicitly invokes `/mlx:autoexperiment` or `/mlx:fine-tune`
- **`notebook` SKILL.md**: added `## Creating notebooks` section — `.ipynb` JSON structure, cell source array format rule, cell ID naming conventions, ML cell templates (Setup, Config `#@param`, tqdm), programmatic cell operations, Colab collapsible sections, quality checklist
- `plugin.json` + `marketplace.json`: bumped to v1.1.6

## [1.1.5] - 2026-04-08
### Added
- Python LSP via `pyright-langserver` — live type checking and diagnostics on every `.py` edit
- `.vscode/settings.json` + `skill-frontmatter.schema.json` — custom YAML schema covering all valid Claude Code skill frontmatter fields
- `skills/train/references/ml-code-style.md` and `skills/notebook/references/ml-code-style.md` — ML code style guide wired into skills (was orphaned in `references/`)
### Changed
- All 16 `SKILL.md` files: added `model`, `effort`, `compatibility`, `metadata` (category/tags/phase); added `paths` for file-type auto-activation; added script pre-approvals in `allowed-tools`
- `autoexperiment`: `context: fork` + `agent: mlx:ml-engineer` for isolated subagent execution
- `context-engineering`: `user-invocable: false` + `disable-model-invocation: true`
- `ml-docs`: added `WebFetch` to `allowed-tools`; `mcp-builder`: added `allowed-tools` + `argument-hint`
- `train/SKILL.md`: full supervised learning coverage — Naive Bayes, KNN, LDA/QDA, SVM/SVR, Decision Tree, GLM (Poisson/Gamma/Tweedie/NegBin), Gaussian Process templates
- `ml-engineer.md` Phase 3 + `data-scientist.md` Step 6: full algorithm selection tables for all 10 supervised learning families
- `train/evals/evals.json`: 7 new evals for Naive Bayes, KNN, LDA/QDA, SVM, Decision Tree, GLM, Gaussian Process
- `bin/mlx-search`: `python3` → `uv run` for PEP 723 compatibility
- `plugin.json` + `marketplace.json`: bumped to v1.1.5, added `lspServers`, updated description and keywords

## [1.1.4] - 2026-04-08
### Added
- `tests/` directory: 6 test files, 35 tests covering analyze (15), data-prep (6), train (3), drift-detect (2), fine-tune (4), notebook (3), ml-docs (6), autoexperiment (2)
- `tests/fixtures/`: shared CSV/JSONL/TSV fixtures for all skill tests
- `skills/<skill>/evals/evals.json` for all 16 skills following agentskills.io eval framework
- `skills/<skill>/evals/files/` with per-skill eval input fixtures
### Fixed
- `rfm_segmentation.py`: fixed `pd.qcut` label count mismatch when `duplicates="drop"` reduces bin count
- `test_data_prep.py`: corrected `--check-only` assertion to match JSON structured output
### Changed
- All skill scripts with external deps upgraded to **PEP 723 inline dependency declarations** (`# /// script` blocks)
- All `SKILL.md` invocations updated from `python3` → `uv run` (auto-installs deps per PEP 723)
- `plugin.json` and `marketplace.json` bumped to v1.1.3

## [1.1.3] - 2026-04-08
### Changed
- Merged `visualize` into `analyze` — visualization scripts, chart selection guide, and design principles now live in `analyze`
- Merged `compress` into `serve` — compression (quantization, pruning, distillation, ONNX) is now Phase 0 of the serve workflow
- Merged `prototype` into `research` — paper-to-code prototyping is now a section of `research` under `scripts/prototype/`
- Skill count: 19 → 16
- Updated agent skills lists across all affected agents

## [1.1.2] - 2026-04-08
### Added
- `dl-engineer` agent: neural network architecture design, training dynamics, GPU optimization (opus, effort high, worktree isolation)
- `data-engineer` agent: ETL/ELT pipelines, dbt, Spark/DuckDB, data quality, orchestration (sonnet, effort medium)
### Changed
- `ml-workbench`: added dl-engineer and data-engineer to delegation routing

## [1.1.1] - 2026-04-08
### Added
- `ml-docs` skill: on-demand API docs for 19 ML libraries (NumPy, Pandas, PyTorch, sklearn, HuggingFace, etc.)
### Changed
- Added `ml-docs` to agents: ml-workbench, data-scientist, ml-engineer, ai-engineer, data-analyst, ml-ops, ml-tutor, ml-reviewer

## [1.1.0] - 2026-04-08
### Added
- `ml-workbench` main session agent + settings.json
- `ml-reviewer` subagent for ML code review
- `autoexperiment` skill: time-budget autonomous experiment loop
- `fine-tune` skill: LoRA/QLoRA/unsloth LLM fine-tuning
- `explain` skill: SHAP, LIME, integrated gradients
- `compress` skill: quantization, pruning, distillation
- `drift-detect` skill: PSI/KS drift detection, evidently, nannyml
- `.mcp.json` + `servers/experiments.py`: MCP experiment tracking server
- `userConfig` in plugin.json: HF token, W&B key, Kaggle, framework pref
- `bin/mlx-exp`, `bin/mlx-search`, `bin/mlx-status` CLI utilities
- `output-styles/` with report, terse, notebook styles
- `.lsp.json` pyright config
- New hooks: FileChanged, SessionEnd, SubagentStop, CwdChanged, PostCompact
- Advanced training patterns reference (GC freeze, time-budget, MuonAdamW)
- ML code style reference (shape annotations, five-part doc spine)
### Changed
- ml-engineer: isolation worktree, effort high, circuit breaker, +autoexperiment skill
- data-scientist: effort high, +explain skill
- ai-engineer: effort high, +fine-tune skill
- ml-ops: effort medium, +compress and drift-detect skills
- ml-tutor: effort low
- ml-researcher: disallowedTools Write,Edit
- data-analyst: effort medium
- marketplace.json: category data-science, tags added, version 1.1.0

## [1.0.0] - 2026-03-08
- Initial release: 13 skills, 7 agents, 6 hooks
