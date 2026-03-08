# Paper Sources Reference

All sources are free, no API keys required.

## arXiv

- **API:** `http://export.arxiv.org/api/query`
- **PDF:** `https://arxiv.org/pdf/{id}.pdf`
- **Rate limit:** 1 req/3s
- **Search params:** `search_query`, `id_list`, `start`, `max_results` (max 2000), `sortBy` (relevance|submittedDate|lastUpdatedDate), `sortOrder`
- **Query syntax:** field prefixes (`ti:`, `au:`, `abs:`, `cat:`), boolean (`AND`, `OR`, `ANDNOT`)
- **ID formats:** `YYMM.NNNNN` (new), `archive/YYMMNNN` (old)
- **Categories:** cs.AI (AI), cs.LG (ML), cs.CL (NLP), cs.CV (Vision), cs.MA (Multiagent), cs.IR (IR)
- **Response:** Atom XML with id, title, summary, author, published, link (PDF)
- **RSS:** `https://rss.arxiv.org/rss/cs.AI` (daily, midnight ET)

## Semantic Scholar

- **API:** `https://api.semanticscholar.org/graph/v1`
- **Rate limit:** ~100 req/5min (scripts enforce 4s delay)
- **Search:** `GET /paper/search?query=...&limit=10&fields=...`
- **Fetch:** `GET /paper/{paperId}?fields=...`
- **Batch:** `POST /paper/batch` with `{"ids": [...]}`
- **Fields:** paperId, title, abstract, authors, year, openAccessPdf, citationCount, referenceCount
- **openAccessPdf:** `{url, status}` when paper has open-access PDF

## Papers with Code

- **Client:** `pip install paperswithcode-client`
- **API:** `https://paperswithcode.com/api/v1/search/`
- **Rate limit:** 3s between calls
- **Paper fields:** arxiv_id, url_pdf, url_abs, title, abstract, authors

## ACL Anthology

- **Module:** `pip install acl-anthology`
- **PDF:** `https://aclanthology.org/{id}.pdf`
- **ID format:** `YYYY.venue-type.NNN` (e.g. `2022.acl-long.220`)
- **Bulk:** `pwc-archive/acl-anthology-corpus` on HuggingFace Datasets

## JMLR

- **No API.** Direct PDF URLs only.
- **Pattern:** `https://jmlr.org/papers/volume{V}/{id}/{id}.pdf`
- **Browse:** https://jmlr.org/papers

## Hugging Face Papers

- **No REST API.** Web only.
- **Browse:** https://huggingface.co/papers
- **Trending:** https://huggingface.co/papers/trending
- **Daily API:** `https://huggingface.co/api/daily_papers` (used by search.py)
- **Paper-code links:** `pwc-archive/links-between-paper-and-code` dataset (~300K links)

## OpenScholar

- **No public API.** Web demo only.
- **Demo:** https://openscholar.allen.ai
- **Use:** Enter query, get synthesized answer with citations over 45M papers

---

# Dataset Sources

All dataset sources are free with no API keys required. Script support in `scripts/datasets.py`.

## HuggingFace Datasets

- **Script support:** `datasets.py search/info/download --source huggingface`
- **Search API:** `GET https://huggingface.co/api/datasets?search=...&limit=N&sort=downloads`
- **Info API:** `GET https://huggingface.co/api/datasets/{id}`
- **Parquet API:** `GET https://datasets-server.huggingface.co/parquet?dataset={id}`
- **First rows:** `GET https://datasets-server.huggingface.co/first-rows?dataset={id}&config=default&split=train`
- **Rate limit:** No strict limit, be reasonable
- **Download format:** Parquet files (read with `pd.read_parquet()`)
- **Coverage:** 100K+ datasets (NLP, vision, audio, tabular)

## OpenML

- **Script support:** `datasets.py search/info/download --source openml`
- **API:** `https://www.openml.org/api/v1/json`
- **Search:** `GET /data/list?data_name=...&limit=N&status=active`
- **Info:** `GET /data/{id}`
- **Features:** `GET /data/features/{id}`
- **Rate limit:** 2s between calls
- **Download format:** ARFF (auto-converts to CSV)
- **Coverage:** 5K+ datasets, mostly tabular/benchmark
- **Note:** Returns 412 when no results found (handled in script)

## UCI ML Repository

- **Script support:** `datasets.py search/info/download --source uci`
- **API:** `https://archive.ics.uci.edu/api/datasets`
- **Search:** `GET /datasets?search=...&skip=0&take=N`
- **Info:** `GET /datasets/{id}`
- **Rate limit:** 2s between calls
- **Download format:** CSV or ZIP
- **Coverage:** 600+ classic ML datasets

## Papers with Code Datasets

- **Script support:** `datasets.py search/info --source paperswithcode`
- **API:** `https://paperswithcode.com/api/v1/datasets/`
- **Search:** `GET /datasets/?q=...&items_per_page=N`
- **Info:** `GET /datasets/{id}/`
- **Rate limit:** 3s between calls
- **Download:** Not directly available — links to external hosting
- **Coverage:** Datasets linked to research papers with benchmarks

## Kaggle

- **Script support:** `datasets.py search --source kaggle`
- **Public API:** `GET https://www.kaggle.com/api/v1/datasets/list?search=...` (often requires auth)
- **Fallback:** HTML scraping of search results page
- **Rate limit:** 2s between calls
- **Download:** Requires Kaggle CLI with API key (`kaggle datasets download -d owner/dataset`)
- **Coverage:** 200K+ community datasets, competition data
