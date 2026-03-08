---
name: research
description: >
  Search, fetch, download, and extract ML/AI research papers from 7 free academic sources.
  Find and download ML datasets from 5 free sources (HuggingFace, OpenML, UCI, Papers with Code, Kaggle).
  Part of the mlx workbench. Use when the user wants to find papers, look up research,
  search arxiv, get citations, download a PDF, extract text from a paper, or find/download datasets.
allowed-tools: Bash, Read, Write, WebFetch, Glob, Grep
user-invocable: true
argument-hint: search query or paper ID (e.g. "transformer attention" or "2401.12345")
---

# Academic Paper Research & Dataset Discovery

Instructions and tools for searching, fetching, and extracting ML/AI research papers, plus finding and downloading datasets.

## Prerequisites

```bash
python3 --version
which pdftotext || echo "MISSING: brew install poppler (macOS) or apt install poppler-utils (Linux)"
```

## Available sources (7 total, all free, no API keys)

| Source | Search | Fetch | Best for |
|--------|--------|-------|----------|
| arXiv | yes | yes | ML/AI preprints |
| Semantic Scholar | yes | yes | Citations, open-access PDFs |
| Papers with Code | yes | yes | Papers linked to GitHub repos |
| Hugging Face | yes | via arXiv | Trending daily papers |
| JMLR | yes | yes | Peer-reviewed ML journal |
| ACL Anthology | no | by ID | NLP conference papers |
| OpenScholar | no | no | Q&A synthesis over 45M papers (URL only) |

## Commands

### Search
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/search.py "$ARGUMENTS" --source arxiv --max 5
```

Flags: `--source` (arxiv, semantic_scholar, papers_with_code, huggingface, jmlr, openscholar), `--max N`, `--cat cs.AI,cs.LG`, `--sort relevance|date`

### Fetch metadata
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/fetch.py <paper_id>
```

Auto-detects source: `2401.12345` (arXiv), `2022.acl-long.220` (ACL), `v22/19-920` (JMLR), 40-char hex (Semantic Scholar)

### Download PDF
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/download.py <paper_id> -o ./papers/
```

### Extract text
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py ./papers/<file>.pdf --max-pages 20
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
python3 ${CLAUDE_SKILL_DIR}/scripts/scientific_search.py "$ARGUMENTS" --max 10
```
Searches arXiv + Semantic Scholar concurrently with deduplication. Add `--datasets` for Kaggle/HuggingFace datasets.

### Document analysis
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/analyze_document.py <url_or_path> [--max-pages 10] [--json]
```
Extracts text from PDFs, Word docs, text files. Supports URLs and local paths.

## Dataset Discovery & Download

### Available dataset sources (5 total, all free, no API keys)

| Source | Search | Info | Download | Best for |
|--------|--------|------|----------|----------|
| HuggingFace | yes | yes | yes (parquet) | NLP, vision, audio datasets |
| OpenML | yes | yes | yes (ARFF/CSV) | Tabular/benchmark datasets |
| UCI | yes | yes | yes (CSV/ZIP) | Classic ML datasets |
| Papers with Code | yes | yes | no (links only) | Datasets linked to papers |
| Kaggle | yes | no | no (use kaggle CLI) | Competition & community datasets |

### Search datasets
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/datasets.py search "$ARGUMENTS" --source huggingface --limit 10
```

Flags: `--source` (huggingface, openml, uci, paperswithcode, kaggle), `--limit N`

### Get dataset info
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/datasets.py info <dataset_id> --source huggingface
```

Shows description, columns, splits, download URLs. Works with: huggingface, openml, uci, paperswithcode.

### Download dataset
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/datasets.py download <dataset_id> --source huggingface --output ./datasets --split train
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
