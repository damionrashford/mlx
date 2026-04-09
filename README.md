<p align="center">
  <strong>MLX</strong>
</p>

<p align="center">
  <em>A full-lifecycle ML workbench for Claude Code — from paper to production in one plugin.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#skills">Skills</a> &middot;
  <a href="#agents">Agents</a> &middot;
  <a href="#experiment-tracking">Experiments</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/version-1.1.6-green.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Claude_Code-Plugin-F97316.svg" alt="Claude Code Plugin">
</p>

---

**MLX** is a Claude Code plugin that gives your agent the complete machine learning toolkit — search papers across 7 academic sources, discover and download datasets from 5 repositories, explore and clean data, engineer features, train models across the full supervised learning spectrum, run autonomous experiments, fine-tune LLMs, build AI applications with LLMs and RAG, deploy models to production, detect data drift, explain predictions with SHAP, generate podcasts from papers, manage and create Jupyter notebooks, extract YouTube content, and learn ML interactively with 3 university-grade courses. **11 agents, 16 skills, 3 CLI tools, 2 MCP servers, 3 output styles, Python LSP.**

## Quick Start

Installing MLX is a two-step process — add the marketplace, then install the plugin.

**Step 1: Add the marketplace**

```bash
/plugin marketplace add damionrashford/mlx
```

**Step 2: Install the plugin**

```bash
/plugin install mlx@damionrashford-mlx
```

Then run `/reload-plugins` to activate. All skills are available as `/mlx:<skill>` and the `ml-workbench` agent starts automatically.

> You can also browse and install interactively: run `/plugin`, go to the **Discover** tab, find MLX, and choose your installation scope (user, project, or local).

**Or clone directly:**

```bash
git clone https://github.com/damionrashford/mlx.git
claude --plugin-dir ./mlx
```

### Prerequisites

| Requirement | Install |
|-------------|---------|
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` — manages Python + all skill script deps via PEP 723 |
| pdftotext (optional, PDF extraction) | `brew install poppler` or `apt install poppler-utils` |
| pyright (optional, Python LSP) | `pnpm add -g pyright` or `npm i -g pyright` |

Python 3.10+ and all Python package dependencies (`yt-dlp`, `notebooklm`, scikit-learn, etc.) are installed automatically by `uv run` on first use — no manual `pip install` needed.

The media skill's content generation requires a Google account with NotebookLM access.

### Recommended Permissions

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run *)",
      "Bash(which *)",
      "Read(*)",
      "Glob(*)"
    ]
  }
}
```

## Skills

MLX ships **16 skills** covering the full ML and data lifecycle. Each is invocable as `/mlx:<skill>` or triggered automatically by context.

| Skill | Command | What it does |
|-------|---------|--------------|
| **research** | `/mlx:research transformer attention` | Search papers (7 sources), find/download datasets (5 sources), structured paper review, paper → code prototyping |
| **data-prep** | `/mlx:data-prep data/train.csv` | EDA, cleaning, feature engineering: distributions, missing values, transforms, encodings |
| **analyze** | `/mlx:analyze data/sales.csv` | Statistical tests, A/B testing, cohort analysis, RFM segmentation, KPIs, trend analysis, pre-delivery QA |
| **train** | `/mlx:train data/features.csv` | Train and iterate: Naive Bayes, KNN, LDA/QDA, SVM, Decision Tree, GLM, Gaussian Process, Ensembles, Neural Nets |
| **evaluate** | `/mlx:evaluate results.tsv` | Multi-dimensional model evaluation, LLM-as-judge, bias detection |
| **autoexperiment** | `/mlx:autoexperiment train.py` | Autonomous time-budget experiment loop — modify, train, eval, record, repeat |
| **fine-tune** | `/mlx:fine-tune mistral data/train.jsonl` | LLM fine-tuning: LoRA, QLoRA, unsloth, DPO, SFTTrainer, instruction tuning |
| **explain** | `/mlx:explain model.joblib` | SHAP, LIME, integrated gradients, feature importance, ICE plots |
| **drift-detect** | `/mlx:drift-detect` | PSI/KS drift detection, evidently, nannyml — production monitoring |
| **serve** | `/mlx:serve model.joblib` | Deploy: inference API (FastAPI), Docker, CI/CD, monitoring, model cards, ONNX/quantization |
| **notebook** | `/mlx:notebook analysis.ipynb` | Create, clean, organize, document, and convert Jupyter notebooks |
| **media** | `/mlx:media paper.pdf` | YouTube extraction + NotebookLM: podcasts, videos, quizzes, reports, slide decks |
| **context-engineering** | (auto) | Context window management, memory systems, multi-agent patterns |
| **mcp-builder** | `/mlx:mcp-builder` | Build MCP servers to connect LLMs with external services |
| **learn** | `/mlx:learn transformers` | Interactive ML education: CS229, Applied ML, ML Engineering (53+ lessons), quizzes, mock interviews |
| **ml-docs** | (auto) | On-demand API docs for 19 ML libraries (NumPy, pandas, PyTorch, sklearn, HuggingFace, etc.) |

### Lifecycle Flow

```
research → data-prep → train → autoexperiment → evaluate → serve
   │            │         │           │               │        │
   │  papers    │  clean  │  baseline │  autonomous   │  bias  │  deploy
   │  datasets  │  feats  │  iterate  │  loop         │  check │  monitor
   └────────────┴─────────┴───────────┴───────────────┴────────┘
   fine-tune ─── LLM fine-tuning on custom data
   explain ───── SHAP/LIME prediction explanations
   drift-detect ─ production data drift monitoring
   media ──────── YouTube + paper → podcast/video/quiz
   learn ──────── study ML interactively
   notebook ───── create and manage Jupyter notebooks
```

## Agents

MLX includes **11 specialized agents** that orchestrate skills for complex workflows.

| Agent | Skills | When to Use |
|-------|--------|-------------|
| **ml-workbench** | all 16 | Main session agent — orchestrates all others, handles simple tasks directly |
| **ml-researcher** | research, media | Find papers, discover datasets, review methodology, generate podcasts, extract YouTube content, prototype algorithms |
| **data-analyst** | data-prep, analyze, evaluate, notebook, ml-docs | Answer business questions: statistics, A/B tests, dashboards, KPIs, segmentation, reports |
| **data-scientist** | research, data-prep, train, evaluate, notebook, explain, ml-docs | Full ML pipeline: find data → explore → clean → engineer → model → evaluate |
| **ml-engineer** | research, data-prep, train, evaluate, notebook, autoexperiment, ml-docs | Focused iteration: feature engineering, hyperparameter sweeps, ablations. Runs in git worktree. |
| **dl-engineer** | research, train, evaluate, autoexperiment, serve, notebook, ml-docs | Neural network architecture design, training dynamics, GPU optimization. Runs in git worktree. |
| **ai-engineer** | research, evaluate, context-engineering, notebook, mcp-builder, fine-tune, ml-docs | Build AI apps: LLM integration, RAG pipelines, prompt engineering, agent architectures |
| **ml-ops** | train, serve, notebook, drift-detect, ml-docs | Deploy models: serialization, serving, Docker, CI/CD, monitoring, model cards |
| **data-engineer** | data-prep, analyze, notebook, drift-detect, ml-docs | ETL/ELT pipelines, dbt, Spark/DuckDB, data quality, orchestration |
| **ml-tutor** | learn, research, evaluate, notebook, ml-docs | Interactive ML education: study concepts, quiz prep, mock interviews, system design |
| **ml-reviewer** | ml-docs | ML code review: data leakage detection, reproducibility audit, train/eval separation (read-only) |

### Agent Routing

```
"Find papers about attention"              → ml-researcher
"Turn this paper into a podcast"           → ml-researcher
"What drove revenue growth last quarter?"  → data-analyst
"Run an A/B test analysis"                 → data-analyst
"I have a CSV, build me a model"           → data-scientist
"Tune the hyperparameters on this model"   → ml-engineer
"Train a transformer from scratch"         → dl-engineer
"Build a RAG chatbot over my docs"         → ai-engineer
"Fine-tune Mistral on my dataset"          → ai-engineer
"Deploy this model with Docker"            → ml-ops
"Build an ETL pipeline for this API"       → data-engineer
"Teach me about transformers"              → ml-tutor
"Review this training script for leakage"  → ml-reviewer
```

## Paper Research

Search across 7 academic sources.

| Source | Search | Fetch | Best for |
|--------|--------|-------|----------|
| arXiv | yes | yes | ML/AI preprints |
| Semantic Scholar | yes | yes | Citations, open-access PDFs |
| Papers with Code | yes | yes | Papers linked to GitHub repos |
| Hugging Face | yes | via arXiv | Trending daily papers |
| JMLR | yes | yes | Peer-reviewed ML journal |
| ACL Anthology | — | by ID | NLP conference papers |
| OpenScholar | — | — | Q&A synthesis over 45M papers |

## Dataset Discovery

Search, inspect, and download ML datasets from 5 sources.

| Source | Search | Info | Download | Format | Best for |
|--------|--------|------|----------|--------|----------|
| HuggingFace | yes | yes | yes | Parquet | NLP, vision, audio (100K+ datasets) |
| OpenML | yes | yes | yes | ARFF/CSV | Tabular benchmarks (5K+ datasets) |
| UCI | yes | yes | yes | CSV/ZIP | Classic ML datasets (600+) |
| Papers with Code | yes | yes | links | — | Datasets linked to papers |
| Kaggle | yes | — | CLI | — | Competition & community (200K+) |

## Experiment Tracking

MLX uses a lightweight TSV-based experiment tracker — no MLflow server, no database.

```
id        metric    val_score  test_score  memory_mb  status   description
exp000    accuracy  0.8523     0.8401      4096       KEEP     baseline logistic
exp001    accuracy  0.8612     0.8498      4096       KEEP     + log features
exp002    accuracy  0.8590     —           4096       DISCARD  lr=0.003 (overfit)
exp003    accuracy  0.8634     0.8521      4352       KEEP     xgboost depth=6
```

Status: `KEEP` | `DISCARD` | `CRASH`

The `autoexperiment` skill runs autonomous loops — modify a training script, train for a fixed wall-clock budget, eval, record, repeat. The ml-experiments MCP server (`servers/experiments.py`) provides structured experiment access to agents.

## Output Styles

Three output styles available via `/config`:

| Style | What it does |
|-------|-------------|
| **Terse** | One-line answers, numbers over prose, zero preamble — fast iteration during experiments |
| **Report** | Stakeholder-ready: executive summary, methodology, results tables, recommendations |
| **Notebook** | Code cell first, then markdown explanation — responses structured as Jupyter cells |

## MCP Servers

| Server | Agents | Purpose |
|--------|--------|---------|
| **mlx-experiments** | ml-engineer, dl-engineer, data-scientist | Structured experiment tracking and results access |
| **colab-mcp** | ml-engineer, dl-engineer | Dispatch training to Google Colab GPUs when local compute is insufficient |

## Architecture

```
mlx/
├── .claude-plugin/
│   └── plugin.json                  # Plugin manifest (v1.1.6)
├── agents/
│   ├── ml-workbench.md              # Main session orchestrator
│   ├── ml-researcher.md             # Papers, datasets, media, prototyping
│   ├── data-analyst.md              # Business analysis, dashboards, KPIs
│   ├── data-scientist.md            # Full pipeline: data → trained model
│   ├── ml-engineer.md               # Model optimization (worktree isolation)
│   ├── dl-engineer.md               # Neural networks, GPU optimization (worktree)
│   ├── ai-engineer.md               # LLM apps, RAG, fine-tuning
│   ├── ml-ops.md                    # Deployment, serving, monitoring
│   ├── data-engineer.md             # ETL, dbt, Spark, data quality
│   ├── ml-tutor.md                  # Interactive ML education
│   └── ml-reviewer.md               # ML code review (read-only)
├── skills/
│   ├── research/                    # Papers (7 sources) + datasets (5 sources) + review + prototype
│   ├── data-prep/                   # EDA, cleaning, feature engineering
│   ├── analyze/                     # Stats, A/B tests, cohort, RFM, visualization, QA
│   ├── train/                       # Full supervised learning spectrum + experiment tracking
│   ├── evaluate/                    # Multi-dimensional evaluation + LLM-as-judge
│   ├── autoexperiment/              # Autonomous time-budget experiment loop
│   ├── fine-tune/                   # LoRA/QLoRA/unsloth LLM fine-tuning
│   ├── explain/                     # SHAP, LIME, integrated gradients
│   ├── drift-detect/                # PSI/KS drift, evidently, nannyml
│   ├── serve/                       # FastAPI, Docker, CI/CD, ONNX, monitoring
│   ├── notebook/                    # Create, clean, organize, convert Jupyter notebooks
│   ├── media/                       # YouTube extraction + NotebookLM content generation
│   ├── context-engineering/         # LLM context window management patterns
│   ├── mcp-builder/                 # Build and evaluate MCP servers
│   ├── learn/                       # CS229, Applied ML, ML Engineering courses (53+ lessons)
│   └── ml-docs/                     # On-demand docs for 19 ML libraries
├── hooks/
│   ├── hooks.json                   # Hook event configuration
│   └── scripts/
│       ├── session-context.sh       # SessionStart: scan ML project state
│       ├── compact-reinject.sh      # SessionStart(compact): restore experiment context
│       ├── post-compact-reinject.sh # PostCompact: reinject ML context
│       ├── validate-ml-code.sh      # PreToolUse(Write|Edit *.py): leakage + seed checks
│       ├── python-syntax-check.sh   # PostToolUse(Write|Edit *.py): syntax validation
│       ├── mlops-safety-check.sh    # PreToolUse(Bash) on ml-ops: block destructive deploys
│       ├── watch-training.sh        # PostToolUse(Bash): capture training metrics
│       ├── ml-error-advisor.sh      # PostToolUseFailure(Bash): diagnose ML errors
│       ├── save-experiment-state.sh # PreCompact: persist experiment state
│       ├── results-changed.sh       # FileChanged(results.tsv): log experiment update
│       ├── experiment-goal-changed.sh # FileChanged(EXPERIMENT.md): reinject hypothesis
│       ├── agent-stop-summary.sh    # Stop: emit experiment state to parent agent
│       ├── session-summary.sh       # SessionEnd: final session summary
│       ├── subagent-log.sh          # SubagentStop: log which agent finished
│       └── cwd-reload.sh            # CwdChanged: reload ML project state
├── output-styles/
│   ├── terse.md                     # Direct answers, numbers over prose
│   ├── report.md                    # Stakeholder report format
│   └── notebook.md                  # Jupyter cell-structured output
├── servers/
│   └── experiments.py               # MCP experiment tracking server
├── bin/
│   ├── mlx-exp                      # CLI: log experiments to results.tsv
│   ├── mlx-search                   # CLI: search papers from terminal
│   └── mlx-status                   # CLI: show current ML project state
├── .mcp.json                        # MCP server configuration
├── .lsp.json                        # Pyright LSP configuration
└── LICENSE
```

### Hooks

MLX includes 11 hook event types running across the ML lifecycle:

| Event | Trigger | What it does |
|-------|---------|--------------|
| `SessionStart` | Session open / after compact | Scans project for ML state; restores experiment context |
| `PreToolUse` | Before Write/Edit on `*.py` | Validates for data leakage, missing seeds, hardcoded paths |
| `PreToolUse` | Before Bash in ml-ops agent | Blocks destructive deployment commands |
| `PostToolUse` | After Write/Edit on `*.py` | Syntax-checks Python before it can cause downstream errors |
| `PostToolUse` | After Bash | Captures training metrics from output |
| `PostToolUseFailure` | Failed Bash | Diagnoses ML errors (missing packages, CUDA, NaN loss) |
| `PreCompact` | Before context compaction | Saves experiment state so it survives the compact |
| `PostCompact` | After context compaction | Rehydrates ML context into new window |
| `FileChanged` | `results.tsv` / `EXPERIMENT.md` | Logs experiment update; reinjects active hypothesis |
| `Stop` | Agent session end | Emits final experiment summary to parent agent |
| `SessionEnd` | Session close | Final session summary |
| `SubagentStop` | Subagent finishes | Logs which agent completed |
| `CwdChanged` | Directory change | Reloads ML project state for new directory |

### Design Principles

- **PEP 723 / uv**: All external-dep scripts use inline dependency declarations — `uv run` auto-installs, no manual pip
- **Progressive complexity**: Slash command → skill → agent → autonomous loop
- **Experiment discipline**: One variable per experiment, validation-only decisions, mandatory results tracking
- **Isolation**: ml-engineer and dl-engineer run in git worktrees — experiments never pollute the working tree
- **No data leakage**: Hooks enforce train/eval separation and random seed hygiene on every Python write

## Supported Frameworks

| Framework | Used in |
|-----------|---------|
| [scikit-learn](https://scikit-learn.org) | train, data-prep, evaluate |
| [XGBoost](https://xgboost.readthedocs.io) / [LightGBM](https://lightgbm.readthedocs.io) | train |
| [PyTorch](https://pytorch.org) | train, dl-engineer |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers) / [PEFT](https://huggingface.co/docs/peft) / [TRL](https://huggingface.co/docs/trl) | fine-tune, train |
| [unsloth](https://github.com/unslothai/unsloth) | fine-tune (4x memory reduction) |
| [pandas](https://pandas.pydata.org) / [polars](https://pola.rs) | data-prep, analyze |
| [scipy](https://scipy.org) | analyze (hypothesis testing) |
| [SHAP](https://shap.readthedocs.io) / [LIME](https://github.com/marcotcr/lime) | explain |
| [evidently](https://www.evidentlyai.com) / [nannyml](https://nannyml.readthedocs.io) | drift-detect |
| [FastAPI](https://fastapi.tiangolo.com) / [uvicorn](https://www.uvicorn.org) | serve |
| [matplotlib](https://matplotlib.org) / [seaborn](https://seaborn.pydata.org) / [plotly](https://plotly.com/python) | analyze |
| [DuckDB](https://duckdb.org) / [Spark](https://spark.apache.org) | data-engineer |

## Rate Limits

All rate limits are enforced automatically.

| Source | Delay | Notes |
|--------|-------|-------|
| arXiv | 3s | Max 200 results per query |
| Semantic Scholar | 4s | ~100 req/5min |
| Papers with Code | 3s | Max 50 results per page |
| JMLR | 3s per volume | Scrapes volume index pages |
| HuggingFace Datasets | none | Be reasonable |
| OpenML | 2s | Returns 412 on no results |
| UCI | 2s | 600+ datasets |
| Kaggle | 2s | Falls back to scraping if API requires auth |

## Contributing

1. Fork the repository
2. Add your skill to `skills/your-skill/SKILL.md`
3. Scripts go in `skills/your-skill/scripts/` — use PEP 723 inline deps (`# /// script` block) for external packages
4. Reference docs go in `skills/your-skill/references/`
5. Update `plugin.json` keywords if relevant
6. Submit a pull request

See the [Claude Code plugin docs](https://code.claude.com/docs/en/plugins) for the directory layout and [plugins reference](https://code.claude.com/docs/en/plugins-reference) for the full manifest schema.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for <a href="https://claude.ai/claude-code">Claude Code</a> by <a href="https://github.com/damionrashford">Damion Rashford</a>
</p>
