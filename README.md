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
  <a href="#dataset-discovery">Datasets</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Claude_Code-Plugin-F97316.svg" alt="Claude Code Plugin">
  <img src="https://img.shields.io/badge/API_keys-none_required-brightgreen.svg" alt="No API Keys">
</p>

---

**MLX** is a Claude Code plugin that gives your agent the complete machine learning toolkit — research papers across 7 academic sources, discover and download datasets from 5 free repositories, explore and clean data, engineer features, train models, run experiments, build AI applications with LLMs and RAG, deploy models to production, and manage notebooks. 6 specialized agents, 12 skills, zero API keys required.

## Quick Start

```bash
# Add the marketplace, then install the plugin
/plugin marketplace add damionrashford/mlx
/plugin install mlx@damionrashford-mlx
```

Or install manually:

```bash
git clone https://github.com/damionrashford/mlx.git
claude --plugin-dir ./mlx
```

### Prerequisites

| Requirement | Install |
|-------------|---------|
| Python 3.10+ | `brew install python` or `apt install python3` |
| pdftotext (optional, for PDF extraction) | `brew install poppler` or `apt install poppler-utils` |

No API keys, no paid services, no external accounts. Every data source is free and public.

### Recommended Permissions

Plugin settings cannot auto-configure permissions. For the smoothest experience, add these to your user or project settings:

```json
{
  "permissions": {
    "allow": [
      "Bash(python3 *)",
      "Bash(pip install *)",
      "Bash(which *)",
      "Read(*)",
      "Glob(*)"
    ]
  }
}
```

## Skills

mlx ships 12 skills that cover the full ML and data lifecycle. Each is invocable as a slash command or triggered automatically by natural language.

| Skill | Command | What it does |
|-------|---------|-------------|
| **research** | `/research transformer attention` | Search papers from 7 sources + find and download datasets from 5 sources |
| **review** | `/review 2401.12345` | Structured paper review: strengths, weaknesses, methodology, reproducibility |
| **prototype** | `/prototype ./paper.pdf` | Convert a research paper into a working code project (Python, TS, Rust, Go) |
| **explore** | `/explore data/train.csv` | Systematic EDA: distributions, correlations, missing values, red flags |
| **prepare** | `/prepare data/raw.csv` | Clean data: duplicates, nulls, outliers, type fixes, validation report |
| **analyze** | `/analyze data/sales.csv` | Statistical tests, A/B testing, cohort analysis, segmentation, KPIs |
| **visualize** | `/visualize data/metrics.csv` | Charts, dashboards, and reports with matplotlib, seaborn, or plotly |
| **engineer** | `/engineer data/clean.csv` | Feature engineering: transforms, encodings, interactions, aggregations |
| **train** | `/train data/features.csv` | Train and evaluate models with sklearn, XGBoost, LightGBM, or PyTorch |
| **experiment** | `/experiment results.tsv` | Systematic hyperparameter search with keep/discard tracking |
| **evaluate** | `/evaluate results.tsv` | Multi-dimensional model evaluation, LLM-as-judge, bias detection |
| **notebook** | `/notebook analysis.ipynb` | Clean, organize, document, and convert Jupyter notebooks |

### Lifecycle Flow

```
research → prototype → explore → prepare → engineer → train → experiment → notebook
   │                      │                    │                    │
   │  find papers &       │  understand        │  build & iterate   │  document
   │  get datasets        │  your data         │  on models         │  results
   └──────────────────────┴────────────────────┴────────────────────┘

Agent coverage:
  ml-researcher ── find papers, datasets, review, prototype
  data-analyst ─── explore, analyze, visualize, report
  data-scientist ─ full pipeline: data → trained model
  ml-engineer ──── optimize: features, tuning, ablations
  ai-engineer ──── LLM apps: RAG, prompts, agents
  mlops ────────── deploy: serialize, serve, Docker, monitor
```

## Paper Research

Search across 7 free academic sources — no API keys, no rate-limit hassle.

| Source | Search | Fetch | Download | Best for |
|--------|--------|-------|----------|----------|
| arXiv | yes | yes | yes | ML/AI preprints |
| Semantic Scholar | yes | yes | — | Citations, open-access PDFs |
| Papers with Code | yes | yes | — | Papers linked to GitHub repos |
| Hugging Face | yes | via arXiv | — | Trending daily papers |
| JMLR | yes | yes | yes | Peer-reviewed ML journal |
| ACL Anthology | — | by ID | yes | NLP conference papers |
| OpenScholar | — | — | — | Q&A synthesis over 45M papers |

```bash
# Search arXiv
/research transformer attention mechanisms

# Multi-source concurrent search
python3 scripts/scientific_search.py "BERT NLP" --max 10

# Download a paper
python3 scripts/download.py 2401.12345 --output ./papers

# Extract text from PDF
python3 scripts/extract.py ./papers/2401.12345.pdf --max-pages 20
```

## Dataset Discovery

Search, inspect, and download ML datasets from 5 free sources — all without API keys.

| Source | Search | Info | Download | Format | Best for |
|--------|--------|------|----------|--------|----------|
| HuggingFace | yes | yes | yes | Parquet | NLP, vision, audio (100K+ datasets) |
| OpenML | yes | yes | yes | ARFF/CSV | Tabular benchmarks (5K+ datasets) |
| UCI | yes | yes | yes | CSV/ZIP | Classic ML datasets (600+) |
| Papers with Code | yes | yes | links | — | Datasets linked to papers |
| Kaggle | yes | — | CLI | — | Competition & community (200K+) |

```bash
# Search for datasets
/research search sentiment analysis datasets

# Or use the datasets script directly
python3 scripts/datasets.py search "image classification" --source huggingface --limit 5

# Inspect a dataset (columns, splits, size)
python3 scripts/datasets.py info imdb --source huggingface

# Download dataset files
python3 scripts/datasets.py download imdb --source huggingface --output ./datasets --split train

# Download from OpenML (auto-converts ARFF to CSV)
python3 scripts/datasets.py download 61 --source openml --output ./datasets
```

## Agents

mlx includes 6 specialized agents that orchestrate skills for complex workflows.

| Agent | Skills Used | When to Use |
|-------|-------------|-------------|
| **ml-researcher** | research, review, prototype | Find papers, discover datasets, review methodology, prototype algorithms |
| **data-analyst** | explore, prepare, analyze, visualize, evaluate, notebook | Answer business questions: statistics, A/B tests, dashboards, KPIs, reports |
| **data-scientist** | research, explore, prepare, engineer, train, experiment, evaluate, notebook | Full ML pipeline: find data, explore, clean, model, evaluate |
| **ml-engineer** | engineer, train, experiment, evaluate, notebook | Focused iteration: feature tuning, hyperparameter sweeps, ablations |
| **ai-engineer** | research, prototype, evaluate, notebook | Build AI apps: LLM integration, RAG pipelines, prompt engineering, agent architectures |
| **mlops** | train, experiment, notebook | Deploy models: serialization, serving code, Docker, CI/CD, monitoring, model cards |

### Agent Routing

```
"Find papers about attention mechanisms"      → ml-researcher
"Review this paper's methodology"             → ml-researcher
"What drove revenue growth last quarter?"      → data-analyst
"Create a dashboard of our KPIs"              → data-analyst
"Run an A/B test analysis on this experiment"  → data-analyst
"I have a CSV, build me a model"              → data-scientist
"Tune the hyperparameters on this model"       → ml-engineer
"Build a RAG chatbot over my docs"             → ai-engineer
"Deploy this model with Docker"                → mlops
```

Each agent follows a strict protocol:

- **ml-researcher**: Scope → Search → Filter → Deep analysis → Review → Dataset discovery → Synthesis → Prototype
- **data-analyst**: Question → Explore → Clean → Analyze → Visualize → Report
- **data-scientist**: Find data → Understand → Explore → Clean → Engineer → Train → Iterate → Report
- **ml-engineer**: Baseline → Features → Model selection → Tuning → Ablation → Final eval → Document
- **ai-engineer**: Requirements → Model selection → Prompt engineering → RAG/embeddings → Eval → Integration → Document
- **mlops**: Model audit → Serialization → Inference API → Containerize → CI/CD → Monitoring → Model card → Reproducibility package

## Architecture

```
mlx/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── skills/
│   ├── research/                # Paper search + dataset discovery
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   │   ├── search.py        # 7-source paper search
│   │   │   ├── fetch.py         # Paper metadata by ID
│   │   │   ├── download.py      # PDF download
│   │   │   ├── extract.py       # PDF text extraction
│   │   │   ├── datasets.py      # 5-source dataset search & download
│   │   │   ├── scientific_search.py  # Concurrent multi-source search
│   │   │   └── analyze_document.py   # Document analysis (PDF, DOCX, TXT)
│   │   └── references/
│   │       ├── sources.md       # API endpoints & rate limits
│   │       └── api-reference.md # Full API documentation
│   ├── prototype/               # Paper → code conversion
│   ├── explore/                 # Exploratory data analysis
│   ├── prepare/                 # Data cleaning & preprocessing
│   ├── engineer/                # Feature engineering
│   ├── analyze/                 # Statistical & business analysis
│   ├── visualize/               # Charts, dashboards, data reports
│   ├── train/                   # Model training & evaluation
│   ├── experiment/              # Experiment tracking & iteration
│   ├── evaluate/                # Multi-dimensional model evaluation
│   ├── review/                  # Structured paper review
│   └── notebook/                # Jupyter notebook management
├── agents/
│   ├── ml-researcher.md         # Research, review & prototyping agent
│   ├── data-analyst.md          # Business analysis & visualization agent
│   ├── data-scientist.md        # Full-pipeline data science agent
│   ├── ml-engineer.md           # Model optimization agent
│   ├── ai-engineer.md           # AI application builder agent
│   └── mlops.md                 # Deployment & operations agent
├── hooks/
│   └── hooks.json               # ML-aware pre/post tool hooks
├── settings.json                # Plugin settings (see Recommended Permissions)
├── LICENSE                      # MIT License
└── .gitignore
```

### Hooks

mlx includes ML-aware hooks that run automatically:

- **PreToolUse** (Write/Edit): Validates training scripts for data leakage, random seed usage, and hardcoded paths
- **PostToolUse** (Bash): Captures training metrics from command output and suggests fixes for common errors (missing packages, CUDA issues)

### Design Principles

- **Zero cost**: Every API and data source is free with no keys required
- **Stdlib first**: Core scripts use Python stdlib (`urllib`, `xml`, `json`) — no pip dependencies for basic functionality
- **Progressive complexity**: Start with a slash command, scale to autonomous agent workflows
- **Experiment discipline**: One variable per experiment, validation-only decisions, mandatory results tracking
- **No data leakage**: Hooks enforce train/eval separation and random seed hygiene

## Supported Frameworks

| Framework | Used in |
|-----------|---------|
| scikit-learn | train, engineer, experiment, analyze |
| XGBoost | train, experiment |
| LightGBM | train, experiment |
| PyTorch | train, experiment |
| pandas | explore, prepare, engineer, analyze |
| scipy | analyze (hypothesis testing) |
| matplotlib | visualize (static charts) |
| seaborn | visualize (statistical plots) |
| plotly | visualize (interactive dashboards) |
| polars | prepare (alternative) |
| PySpark | prepare (distributed) |

## Experiment Tracking

mlx uses a lightweight TSV-based experiment tracker — no MLflow server, no database, just a file.

```
id        metric    val_score  test_score  memory_mb  status   description
exp000    accuracy  0.8523     0.8401      4096       KEEP     baseline
exp001    accuracy  0.8612     0.8498      4096       KEEP     lr=0.001
exp002    accuracy  0.8590     -           4096       DISCARD  lr=0.003 (overfit)
exp003    accuracy  0.8634     0.8521      4352       KEEP     dropout=0.1
```

Status: `KEEP` (improved) | `DISCARD` (same or worse) | `CRASH` (error/OOM/NaN)

The ml-engineer agent runs autonomous experiment loops — 8-10 experiments/hour with automatic keep/discard decisions.

## Rate Limits

All rate limits are enforced automatically in the scripts.

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

## Submit to Official Marketplace

To submit mlx to the official Anthropic plugin marketplace:

- **Claude.ai**: [claude.ai/settings/plugins/submit](https://claude.ai/settings/plugins/submit)
- **Console**: [platform.claude.com/plugins/submit](https://platform.claude.com/plugins/submit)

## Contributing

1. Fork the repository
2. Add your skill to `skills/your-skill/SKILL.md`
3. If your skill needs scripts, add them to `skills/your-skill/scripts/`
4. Update `plugin.json` if adding new keywords
5. Submit a pull request

See the [Claude Code plugin docs](https://code.claude.com/docs/en/plugins) for the expected directory layout and [plugins reference](https://code.claude.com/docs/en/plugins-reference) for the full manifest schema.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for <a href="https://claude.ai/claude-code">Claude Code</a> by <a href="https://github.com/damionrashford">Damion Rashford</a>
</p>
