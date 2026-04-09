---
name: research
description: >
  Search, fetch, download, and extract ML/AI research papers from 7 free academic sources.
  Find and download ML datasets from 5 free sources (HuggingFace, OpenML, UCI, Papers with Code, Kaggle).
  Review a paper, critique methodology, assess reproducibility, evaluate experimental design.
  Convert research papers, articles, or technical documents into working code prototypes.
  Use when the user wants to find papers, search arxiv, get citations, download a PDF,
  extract text from a paper, find/download datasets, review/critique a research paper,
  implement a paper, prototype an algorithm, or convert research to working code.
allowed-tools: >
  Bash(uv run * scripts/scientific_search.py *)
  Bash(uv run * scripts/search.py *)
  Bash(uv run * scripts/fetch.py *)
  Bash(uv run * scripts/download.py *)
  Bash(uv run * scripts/extract.py *)
  Bash(uv run * scripts/analyze_document.py *)
  Bash(uv run * scripts/datasets.py *)
  Read Write WebFetch Glob Grep
argument-hint: search query, paper ID, or path/URL to implement (e.g. "transformer attention", "2401.12345", or "papers/attention.pdf")
model: sonnet
effort: medium
compatibility: ">=1.0"
metadata:
  category: research
  tags:
    [
      arxiv,
      semantic-scholar,
      papers-with-code,
      datasets,
      huggingface,
      openml,
      kaggle,
      paper-review,
      prototype,
    ]
  phase: research
---

# Academic Paper Research & Dataset Discovery

Instructions and tools for searching, fetching, and extracting ML/AI research papers, plus finding and downloading datasets.

## Prerequisites

```bash
which pdftotext || echo "MISSING: brew install poppler (macOS) or apt install poppler-utils (Linux)"
```

## Available sources (7 total)

| Source           | Search | Fetch     | Best for                                 |
| ---------------- | ------ | --------- | ---------------------------------------- |
| arXiv            | yes    | yes       | ML/AI preprints                          |
| Semantic Scholar | yes    | yes       | Citations, open-access PDFs              |
| Papers with Code | yes    | yes       | Papers linked to GitHub repos            |
| Hugging Face     | yes    | via arXiv | Trending daily papers                    |
| JMLR             | yes    | yes       | Peer-reviewed ML journal                 |
| ACL Anthology    | no     | by ID     | NLP conference papers                    |
| OpenScholar      | no     | no        | Q&A synthesis over 45M papers (URL only) |

## Commands

### Search

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/search.py "$ARGUMENTS" --source arxiv --max 5
```

Flags: `--source` (arxiv, semantic_scholar, papers_with_code, huggingface, jmlr, openscholar), `--max N`, `--cat cs.AI,cs.LG`, `--sort relevance|date`

### Fetch metadata

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/fetch.py <paper_id>
```

Auto-detects source: `2401.12345` (arXiv), `2022.acl-long.220` (ACL), `v22/19-920` (JMLR), 40-char hex (Semantic Scholar)

### Download PDF

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/download.py <paper_id> -o ./papers/
```

### Extract text

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/extract.py ./papers/<file>.pdf --max-pages 20
```

## Guidelines

- Search 3-5 papers, not 20. Read abstracts first.
- Only download PDFs for papers the user cares about.
- Use `--max-pages 10` for summaries; go deeper only when asked.
- Summarize key findings, then offer to dive deeper.
- Always cite: title, authors, year, source URL.

## Additional tools

### Multi-source concurrent search (alternative)

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/scientific_search.py "$ARGUMENTS" --max 10
```

Searches arXiv + Semantic Scholar concurrently with deduplication. Add `--datasets` for Kaggle/HuggingFace datasets.

### Document analysis

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/analyze_document.py <url_or_path> [--max-pages 10] [--json]
```

Extracts text from PDFs, Word docs, text files. Supports URLs and local paths.

## Dataset Discovery & Download

### Available dataset sources (5 total, all free, no API keys)

| Source           | Search | Info | Download            | Best for                         |
| ---------------- | ------ | ---- | ------------------- | -------------------------------- |
| HuggingFace      | yes    | yes  | yes (parquet)       | NLP, vision, audio datasets      |
| OpenML           | yes    | yes  | yes (ARFF/CSV)      | Tabular/benchmark datasets       |
| UCI              | yes    | yes  | yes (CSV/ZIP)       | Classic ML datasets              |
| Papers with Code | yes    | yes  | no (links only)     | Datasets linked to papers        |
| Kaggle           | yes    | no   | no (use kaggle CLI) | Competition & community datasets |

### Search datasets

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/datasets.py search "$ARGUMENTS" --source huggingface --limit 10
```

Flags: `--source` (huggingface, openml, uci, paperswithcode, kaggle), `--limit N`

### Get dataset info

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/datasets.py info <dataset_id> --source huggingface
```

Shows description, columns, splits, download URLs. Works with: huggingface, openml, uci, paperswithcode.

### Download dataset

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/datasets.py download <dataset_id> --source huggingface --output ./datasets --split train
```

Downloads to `./datasets/` directory. Flags: `--output DIR`, `--split train|test|validation` (HuggingFace only).

### Dataset guidelines

- Search 3-5 datasets, compare sizes and licenses before downloading.
- Use `info` to check columns/splits before downloading.
- HuggingFace downloads as Parquet (read with `pd.read_parquet()`).
- OpenML downloads as ARFF, auto-converts to CSV when possible.
- For Kaggle downloads, use: `kaggle datasets download -d <owner/dataset>`

## Rate limits (enforced automatically)

arXiv: 3s, Semantic Scholar: 4s, Papers with Code: 3s, JMLR: 3s per volume, OpenML: 2s, UCI: 2s, Kaggle: 2s.

## Reference

- [references/sources.md](references/sources.md) — API details, endpoints, rate limits for all sources

## Paper Review

Structured framework for reviewing ML/AI research papers. Produces fair, constructive, conference-quality reviews.

### Obtain the paper

Use the research scripts to get the paper content:

```bash
# Download by arXiv ID
uv run ${CLAUDE_SKILL_DIR}/scripts/download.py 2401.12345 --output ./papers

# Extract text from PDF
uv run ${CLAUDE_SKILL_DIR}/scripts/extract.py ./papers/2401.12345.pdf --max-pages 30

# Fetch metadata
uv run ${CLAUDE_SKILL_DIR}/scripts/fetch.py 2401.12345
```

If the user provides only a topic, search first, then review the selected paper.

### Review template

#### Summary

2-3 sentences: What is the paper about? What is the key contribution?

#### Strengths

Evaluate each dimension:

- **Novelty**: Is the approach new? Does it advance the field?
- **Experiments**: Well-designed? Sufficient baselines?
- **Clarity**: Well-written? Easy to follow?
- **Significance**: Would this matter if results hold?

#### Weaknesses

Identify specific issues:

- Missing baselines or comparisons
- Claims not supported by evidence
- Methodology gaps or questionable choices
- Limited evaluation scope
- Scalability concerns

### Methodology assessment

| Dimension    | Assessment                      | Notes                     |
| ------------ | ------------------------------- | ------------------------- |
| Splits       | proper / questionable / missing | Train/val/test separation |
| Baselines    | fair / unfair / missing         | SOTA included?            |
| Metrics      | appropriate / limited / wrong   | Multiple metrics?         |
| Significance | reported / missing              | Error bars, CIs, p-values |
| Ablations    | thorough / partial / none       | Component contributions   |

### Reproducibility checklist

- [ ] Code available (or promised)?
- [ ] Dataset available or described sufficiently?
- [ ] Hyperparameters fully specified?
- [ ] Compute requirements stated (GPU type, hours)?
- [ ] Random seeds reported?
- [ ] Training details sufficient to reproduce?
- [ ] Preprocessing steps documented?

### Questions for authors

3-5 specific questions that would strengthen the paper or clarify ambiguities.

### Overall assessment

|                    | Rating                                                   |
| ------------------ | -------------------------------------------------------- |
| **Recommendation** | Accept / Weak Accept / Borderline / Weak Reject / Reject |
| **Confidence**     | High / Medium / Low                                      |
| **Impact**         | What would this enable if results hold?                  |

### Review ethics

- Be constructive — suggest improvements, don't just criticize
- Separate factual issues from opinions
- Acknowledge uncertainty in your assessment
- Evaluate what was claimed, not what wasn't attempted
- Consider the target venue and its standards
- Give credit where due — acknowledge genuine contributions

### Comparative review (multiple papers)

When reviewing multiple papers on the same topic:

| Dimension       | Paper A | Paper B | Paper C |
| --------------- | ------- | ------- | ------- |
| Method          |         |         |         |
| Dataset         |         |         |         |
| Best metric     |         |         |         |
| Reproducibility |         |         |         |
| Novelty         |         |         |         |

Rank by overall contribution, noting complementary strengths.

---

## Prototyping: Paper → Code

Convert a research paper, article, or technical document into a complete working code project.

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/prototype/main.py <source> -o ./prototype [-l python] [-v]
```

| Argument   | Description                                        |
| ---------- | -------------------------------------------------- |
| `<source>` | PDF file, URL, .ipynb, .md, or .txt                |
| `-o`       | Output directory (default: `./prototype`)          |
| `-l`       | Language: python, javascript, typescript, rust, go |
| `-v`       | Verbose output                                     |

### Pipeline

1. **Extract** — parse PDF, web page, notebook, or markdown
2. **Analyze** — detect algorithms, architectures, domain, dependencies
3. **Select language** — explicit flag → code in source → domain default → Python
4. **Generate** — complete project scaffold with algorithm implementations

Every generated project includes: main implementation, dependency manifest, test file, README, .gitignore.

### Language defaults by domain

| Domain                           | Default    |
| -------------------------------- | ---------- |
| ML / data science                | Python     |
| Web / frontend                   | TypeScript |
| Systems / performance            | Rust       |
| Explicit `numpy`/`torch` mention | Python     |
| Explicit `react`/`vue` mention   | JavaScript |

### Code quality requirements

- No TODOs or placeholders — all code must be complete and runnable
- Type hints on all functions (Python, TypeScript)
- Docstrings on all public functions
- Specific error handling (not bare `except`)
- At least one test file
- README includes: title, install, usage, source attribution

See [`references/prototype/generation-rules.md`](references/prototype/generation-rules.md) for full generation rules.
