# Changelog

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
