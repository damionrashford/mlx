# Paper Source API Reference

All 7 sources are free with no API keys required. All have script support in `scripts/`.

---

## arXiv

- **Script support:** `search.py --source arxiv`, `fetch.py` (auto-detect), `download.py`
- **Query:** `http://export.arxiv.org/api/query`
- **PDF:** `https://arxiv.org/pdf/{id}.pdf`
- **Rate limit:** 1 req/3s (enforced in scripts)

### Search Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `search_query` | Query string (field prefixes: `all`, `ti`, `au`, `abs`, `cat`) | — |
| `id_list` | Comma-separated IDs (fetch by ID without search) | — |
| `max_results` | Results per request (max 2000) | 10 |
| `sortBy` | `relevance`, `lastUpdatedDate`, `submittedDate` | relevance |

### Search Syntax

- `all:large language model` — full-text
- `ti:transformer` — title only
- `au:vaswani` — author
- `cat:cs.AI OR cat:cs.LG` — categories
- `(all:agent) AND (cat:cs.CL)` — combined

### LLM-Relevant Categories

| Code | Domain |
|------|--------|
| cs.AI | Artificial Intelligence |
| cs.LG | Machine Learning |
| cs.CL | Computation and Language (NLP) |
| cs.CV | Computer Vision |
| cs.MA | Multiagent Systems |
| cs.IR | Information Retrieval |

### ID Formats

- **New (post-2007):** `YYMM.NNNNN` e.g. `2309.02427`, `2401.12345v2`
- **Old:** `archive/YYMMNNN` e.g. `hep-ex/0307015`

---

## Semantic Scholar

- **Script support:** `search.py --source semantic-scholar`, `fetch.py` (auto-detect)
- **Base:** `https://api.semanticscholar.org/graph/v1`
- **Rate limit:** ~100 req/5min (4s delay in scripts)

### Endpoints

- **Search:** `GET /paper/search?query=...&limit=10&fields=...`
- **Paper by ID:** `GET /paper/{paperId}?fields=...`
- **Batch:** `POST /paper/batch` with `{"ids": [...]}`

### Fields

`paperId, title, abstract, authors, year, openAccessPdf, citationCount, referenceCount`

### External ID Lookup

Use `arXiv:2309.02427` or `DOI:10.xxxxx` as the paper ID to look up via external identifiers.

### Venue Filter

JMLR search falls back to Semantic Scholar with `venue=JMLR` when no direct matches are found on the JMLR index page.

---

## Papers with Code

- **Script support:** `search.py --source papers-with-code`, `fetch.py --source pwc`
- **Client:** `pip install paperswithcode-client`
- **Rate limit:** 3s between calls
- **Fields:** `arxiv_id, url_pdf, url_abs, title, abstract, authors`

### Usage

```python
from paperswithcode import PapersWithCodeClient
client = PapersWithCodeClient()
client.search(q="transformer", items_per_page=5)
client.paper_list(arxiv_id="2309.02427")
```

---

## ACL Anthology

- **Script support:** `fetch.py` (auto-detect from ID pattern `YYYY.venue-type.NNN`)
- **Module:** `pip install acl-anthology`
- **PDF URL pattern:** `https://aclanthology.org/{id}.pdf`
- **Example:** `https://aclanthology.org/2022.acl-long.220.pdf`
- **Bulk corpus:** `ACL-anthology-corpus` on Hugging Face

---

## Hugging Face Papers

- **Script support:** `search.py --source huggingface`
- **API:** `https://huggingface.co/api/daily_papers` (returns JSON, no key required)
- **Browse:** https://huggingface.co/papers
- **Trending:** https://huggingface.co/papers/trending
- **Email digest:** Subscribe on the papers page (daily/weekly/monthly)
- **Paper-code links dataset:** `pwc-archive/links-between-paper-and-code` (~300K links)

### How it works in scripts

`search.py --source huggingface` fetches the daily papers feed and filters by query terms in title/summary. Each paper includes an arXiv ID, so `fetch.py` and `download.py` resolve through the arXiv pipeline.

---

## JMLR

- **Script support:** `search.py --source jmlr`, `fetch.py` (auto-detect from `v{vol}/{id}`), `download.py`
- **Browse:** https://jmlr.org/papers
- **PDF pattern:** `https://jmlr.org/papers/volume{vol}/{id}/{id}.pdf`
- **Example:** https://jmlr.org/papers/volume22/19-920/19-920.pdf
- **ID format:** `v22/19-920` (volume 22, paper 19-920)

### How it works in scripts

- `search.py --source jmlr` scrapes the JMLR papers index page and filters by query. Falls back to Semantic Scholar with `venue=JMLR` if no matches.
- `fetch.py v22/19-920` auto-detects JMLR from the `v{N}/{id}` pattern, resolves the PDF URL, and scrapes the paper page for title/authors.
- `download.py v22/19-920` resolves to `https://jmlr.org/papers/volume22/19-920/19-920.pdf`.

---

## OpenScholar

- **Script support:** `search.py --source openscholar` (outputs query URL)
- **Demo:** https://openscholar.allen.ai
- **Coverage:** 45M open-access papers
- **Capability:** Q&A with synthesized answers and citations
- **No REST API** — the script constructs the query URL for the agent to use with WebFetch

### How it works in scripts

`search.py --source openscholar` prints the query URL for OpenScholar. The agent should use WebFetch on this URL to get a synthesized literature review with citations. Best for questions like "What methods improve LLM reasoning?" rather than individual paper search.

---

## Dataset Discovery (`datasets.py`)

Unified script for searching, inspecting, and downloading ML datasets. All sources are free, no API keys.

### Commands

| Command | Description |
|---------|-------------|
| `datasets.py search <query> --source SOURCE` | Search datasets |
| `datasets.py info <id> --source SOURCE` | Get detailed info (columns, splits, download URLs) |
| `datasets.py download <id> --source SOURCE --output DIR` | Download dataset files |

### Sources

| Source | Alias | Search | Info | Download | Format |
|--------|-------|--------|------|----------|--------|
| `huggingface` | `hf` | yes | yes | yes | Parquet |
| `openml` | — | yes | yes | yes | ARFF → CSV |
| `uci` | — | yes | yes | yes | CSV/ZIP |
| `paperswithcode` | `pwc` | yes | yes | no | — |
| `kaggle` | — | yes | no | no (use CLI) | — |

### HuggingFace Download

Downloads Parquet files from the datasets server. Use `--split train|test|validation` to select a split.
Falls back to CSV sample (100 rows) if Parquet is unavailable.

### OpenML Download

Downloads ARFF files and auto-converts to CSV using a built-in parser. For complex ARFF, install `scipy` or `liac-arff`.

### UCI Download

Tries CSV first, then ZIP. Extract ZIPs with `unzip`.
