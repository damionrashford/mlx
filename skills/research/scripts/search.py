#!/usr/bin/env python3
"""Unified ML paper search across 7 sources.

Usage:
    search.py <query> [--source SOURCE] [--limit N] [--sort relevance|date] [--cat cs.AI,cs.LG] [--no-cat]

Sources: arxiv (default), semantic-scholar, papers-with-code, huggingface, jmlr, openscholar
All sources are free, no API keys required.
"""

import json
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Standardized result format
# ---------------------------------------------------------------------------

def _print_results(results, source_label):
    """Print results in a consistent format across all sources."""
    print(f"# {source_label}")
    print(f"Found {len(results)} papers")
    print("---")
    for i, p in enumerate(results, 1):
        title = (p.get("title") or "")[:90]
        ellipsis = "..." if len(p.get("title", "")) > 90 else ""
        print(f"\n## [{i}] {p.get('id', '')} — {title}{ellipsis}")
        authors = p.get("authors") or []
        year = f" | {p['year']}" if p.get("year") else ""
        citations = f" | Citations: {p['citations']}" if p.get("citations") is not None else ""
        extra = p.get("extra", "")
        if extra:
            extra = f" | {extra}"
        print(f"Authors: {', '.join(str(a) for a in authors[:5])}{year}{citations}{extra}")
        print(f"Abstract: {(p.get('abstract') or '')[:350]}...")
        pdf = p.get("pdf_url") or ""
        print(f"PDF: {pdf}" if pdf else "PDF: (not available)")
        if p.get("source"):
            print(f"Source: {p['source']}")


# ===========================================================================
# arXiv backend
# ===========================================================================

_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_PDF = "https://arxiv.org/pdf"
_ARXIV_CATS = "cs.AI,cs.LG,cs.CL,cs.CV"


def _arxiv_ns(tag):
    return f"{{http://www.w3.org/2005/Atom}}{tag}"


def _arxiv_ext_ns(tag):
    return f"{{http://arxiv.org/schemas/atom}}{tag}"


def _parse_arxiv_entry(entry):
    """Parse single Atom entry into standardized dict."""
    paper_id = None
    id_elem = entry.find(_arxiv_ns("id"))
    if id_elem is not None and id_elem.text:
        match = re.search(r"arxiv\.org/abs/([^\s\"]+)", id_elem.text)
        if match:
            paper_id = match.group(1).rstrip("/")

    title_elem = entry.find(_arxiv_ns("title"))
    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

    summary_elem = entry.find(_arxiv_ns("summary"))
    abstract = summary_elem.text.strip().replace("\n", " ")[:500] if summary_elem is not None and summary_elem.text else ""

    authors = []
    for author in entry.findall(f".//{_arxiv_ns('author')}"):
        name_elem = author.find(_arxiv_ns("name"))
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    published = ""
    pub_elem = entry.find(_arxiv_ns("published"))
    if pub_elem is not None and pub_elem.text:
        published = pub_elem.text[:10]

    pdf_url = None
    for link in entry.findall(f".//{_arxiv_ns('link')}"):
        if link.get("title") == "pdf" or (link.get("type") == "application/pdf" and "pdf" in link.get("href", "")):
            pdf_url = link.get("href")
            break
    if not pdf_url and paper_id:
        pdf_url = f"{_ARXIV_PDF}/{paper_id}.pdf"

    categories = []
    for cat in entry.findall(f".//{_arxiv_ns('category')}"):
        term = cat.get("term")
        if term:
            categories.append(term)

    comment = ""
    comment_elem = entry.find(_arxiv_ext_ns("comment"))
    if comment_elem is not None and comment_elem.text:
        comment = comment_elem.text.strip()[:80]

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors[:5],
        "year": published[:4] if published else None,
        "pdf_url": pdf_url,
        "citations": None,
        "source": "arXiv",
        "extra": f"Categories: {', '.join(categories[:3])}" + (f" | {comment}" if comment else ""),
    }


def search_arxiv(query, limit=10, sort="relevance", categories=_ARXIV_CATS, use_categories=True):
    """Search arXiv. Rate limit: 1 req/3s (enforced)."""
    time.sleep(3)

    if use_categories and categories:
        cat_list = categories.split(",")
        cat_part = " OR ".join(f"cat:{c.strip()}" for c in cat_list if c.strip())
        search_query = f"({query}) AND ({cat_part})" if cat_part else query
    else:
        search_query = query

    sort_map = {"relevance": "relevance", "date": "submittedDate", "updated": "lastUpdatedDate"}
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(limit, 200),
        "sortBy": sort_map.get(sort, "relevance"),
        "sortOrder": "descending",
    }
    url = f"{_ARXIV_API}?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read().decode("utf-8"))
    except (HTTPError, URLError) as e:
        print(f"arXiv error: {e}", file=sys.stderr)
        return []

    entries = root.findall(f".//{_arxiv_ns('entry')}")
    return [_parse_arxiv_entry(e) for e in entries]


# ===========================================================================
# Semantic Scholar backend
# ===========================================================================

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "paperId,title,abstract,authors,year,openAccessPdf,citationCount"


def search_semantic_scholar(query, limit=10):
    """Search Semantic Scholar. Rate limit: ~100 req/5min (4s delay enforced)."""
    time.sleep(4)
    params = {"query": query, "limit": min(limit, 100), "fields": _S2_FIELDS}
    url = f"{_S2_BASE}/paper/search?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        print(f"Semantic Scholar error: {e}", file=sys.stderr)
        return []

    results = []
    for p in data.get("data", []):
        pdf = (p.get("openAccessPdf") or {}).get("url", "")
        authors = [a.get("name", "") for a in (p.get("authors") or [])[:5] if isinstance(a, dict)]
        results.append({
            "id": p.get("paperId", ""),
            "title": (p.get("title") or "").replace("\n", " "),
            "abstract": ((p.get("abstract") or "")[:500]),
            "authors": authors,
            "year": p.get("year"),
            "pdf_url": pdf,
            "citations": p.get("citationCount"),
            "source": "Semantic Scholar",
            "extra": "",
        })
    return results


# ===========================================================================
# Papers with Code backend
# ===========================================================================

def search_papers_with_code(query, limit=5):
    """Search Papers with Code via REST API. Rate limit: 3s."""
    time.sleep(3)
    params = urlencode({"q": query, "items_per_page": min(limit, 50), "page": 1})
    url = f"https://paperswithcode.com/api/v1/search/?{params}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        print(f"Papers with Code error: {e}", file=sys.stderr)
        return []

    results = []
    for p in data.get("results", []):
        if not isinstance(p, dict):
            continue
        results.append({
            "id": p.get("id", "") or "",
            "title": (p.get("title", "") or ""),
            "abstract": (p.get("abstract", "") or "")[:500],
            "authors": [],
            "year": None,
            "pdf_url": p.get("url_pdf", "") or "",
            "citations": None,
            "source": "Papers with Code",
            "extra": f"Abstract URL: {p.get('url_abs', '') or ''}",
        })
    return results


# ===========================================================================
# Hugging Face Papers backend
# ===========================================================================

_HF_DAILY = "https://huggingface.co/api/daily_papers"


def search_huggingface(query, limit=10):
    """Fetch trending/daily papers from Hugging Face, filtered by query terms.

    HF daily papers link to arXiv — each result includes the arXiv PDF URL.
    """
    try:
        req = Request(_HF_DAILY, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        print(f"Hugging Face error: {e}", file=sys.stderr)
        return []

    if not isinstance(data, list):
        data = data.get("data", data.get("papers", []))

    query_terms = query.lower().split()
    results = []
    for entry in data:
        paper = entry.get("paper", entry) if isinstance(entry, dict) else {}
        if not isinstance(paper, dict):
            continue
        arxiv_id = paper.get("id", "") or ""
        title = (paper.get("title", "") or "").replace("\n", " ")
        summary = (paper.get("summary", paper.get("abstract", "")) or "")[:500]
        upvotes = entry.get("numUpvotes", entry.get("upvotes", ""))

        # Filter by query terms in title or summary
        text = f"{title} {summary}".lower()
        if query_terms and not any(t in text for t in query_terms):
            continue

        authors = []
        for a in (paper.get("authors", []) or []):
            if isinstance(a, dict):
                authors.append(a.get("name", a.get("user", "")))
            elif isinstance(a, str):
                authors.append(a)

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""

        results.append({
            "id": arxiv_id,
            "title": title,
            "abstract": summary,
            "authors": authors[:5],
            "year": None,
            "pdf_url": pdf_url,
            "citations": None,
            "source": "Hugging Face",
            "extra": f"Upvotes: {upvotes}" if upvotes else "",
        })
        if len(results) >= limit:
            break

    return results


# ===========================================================================
# JMLR backend
# ===========================================================================

def search_jmlr(query, limit=10):
    """Search JMLR by scraping recent volume pages and filtering by query.

    JMLR has no search API. This fetches the 3 most recent volume pages
    (structure: <dt>Title</dt> <dd>Authors; pages</dd> [abs][pdf][bib])
    and filters papers by query terms in the title/authors.
    Falls back to Semantic Scholar venue search if no matches.
    """
    query_terms = query.lower().split()
    results = []

    # Scrape the 3 most recent volumes
    for vol in range(26, 23, -1):
        time.sleep(3)
        url = f"https://jmlr.org/papers/v{vol}/"
        try:
            req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
            with urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError):
            continue

        # Parse <dt>Title</dt>\n<dd><b><i>Authors</i></b>; (N):pages, year.\n[abs][pdf][bib]
        for match in re.finditer(
            r"<dt>([^<]+)</dt>\s*<dd><b><i>([^<]+)</i></b>[^[]*"
            r"\[<a[^>]*href='[^']*?/v\d+/([^']+)\.html'>abs</a>\]"
            r"\[<a[^>]*href='[^']*?/volume(\d+)/([^/]+)/[^']*\.pdf'>pdf</a>\]",
            html, re.DOTALL,
        ):
            title = match.group(1).strip()
            authors_str = match.group(2).strip()
            paper_id = match.group(5)
            vol_num = match.group(4)

            text = f"{title} {authors_str}".lower()
            if query_terms and not any(t in text for t in query_terms):
                continue

            authors = [a.strip() for a in authors_str.split(",") if a.strip()]
            pdf_url = f"https://jmlr.org/papers/volume{vol_num}/{paper_id}/{paper_id}.pdf"

            results.append({
                "id": f"v{vol_num}/{paper_id}",
                "title": title,
                "abstract": "",
                "authors": authors[:5],
                "year": None,
                "pdf_url": pdf_url,
                "citations": None,
                "source": "JMLR",
                "extra": f"Volume {vol_num}",
            })
            if len(results) >= limit:
                return results

    if not results:
        # Fallback: search Semantic Scholar with venue filter
        print("(No matches in JMLR index — falling back to Semantic Scholar venue search)", file=sys.stderr)
        return _search_s2_venue(query, "JMLR", limit)

    return results


def _search_s2_venue(query, venue, limit):
    """Search Semantic Scholar filtered to a specific venue."""
    time.sleep(4)
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,title,abstract,authors,year,openAccessPdf,citationCount,venue",
        "venue": venue,
    }
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        print(f"Semantic Scholar (venue) error: {e}", file=sys.stderr)
        return []

    results = []
    for p in data.get("data", []):
        pdf = (p.get("openAccessPdf") or {}).get("url", "")
        authors = [a.get("name", "") for a in (p.get("authors") or [])[:5] if isinstance(a, dict)]
        results.append({
            "id": p.get("paperId", ""),
            "title": (p.get("title") or "").replace("\n", " "),
            "abstract": ((p.get("abstract") or "")[:500]),
            "authors": authors,
            "year": p.get("year"),
            "pdf_url": pdf,
            "citations": p.get("citationCount"),
            "source": f"JMLR (via Semantic Scholar)",
            "extra": "",
        })
    return results


# ===========================================================================
# OpenScholar backend
# ===========================================================================

_OPENSCHOLAR_URL = "https://openscholar.allen.ai"


def search_openscholar(query, limit=10):
    """OpenScholar: AI2's Q&A system over 45M papers.

    No REST API — this constructs the query URL for the agent to use with WebFetch,
    and attempts to fetch results from the web demo directly.
    """
    encoded_q = query.replace(" ", "+")
    demo_url = f"{_OPENSCHOLAR_URL}/?query={encoded_q}"

    print(f"# OpenScholar: {query}")
    print("---")
    print(f"OpenScholar is a Q&A system over 45M open-access papers.")
    print(f"It returns synthesized answers with citations, not individual paper listings.")
    print(f"")
    print(f"Query URL: {demo_url}")
    print(f"")
    print(f"To get results, use WebFetch on the URL above, or direct the user to visit it.")
    print(f"OpenScholar is best for literature synthesis questions like:")
    print(f"  - 'What methods improve LLM reasoning?'")
    print(f"  - 'Compare transformer architectures for code generation'")
    return []


# ===========================================================================
# CLI
# ===========================================================================

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if args else 1)

    query_parts = []
    source = "arxiv"
    limit = 10
    sort = "relevance"
    categories = _ARXIV_CATS
    use_categories = True
    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1].lower()
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--sort" and i + 1 < len(args):
            sort = args[i + 1]
            i += 2
        elif args[i] == "--cat" and i + 1 < len(args):
            categories = args[i + 1]
            i += 2
        elif args[i] == "--no-cat":
            use_categories = False
            i += 1
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    if not query:
        print("Error: no query provided", file=sys.stderr)
        sys.exit(1)

    if source in ("arxiv", "ar"):
        results = search_arxiv(query, limit=limit, sort=sort, categories=categories, use_categories=use_categories)
        _print_results(results, f"arXiv: {query}")
    elif source in ("semantic-scholar", "s2", "semanticscholar"):
        results = search_semantic_scholar(query, limit=limit)
        _print_results(results, f"Semantic Scholar: {query}")
    elif source in ("papers-with-code", "pwc", "paperswithcode"):
        results = search_papers_with_code(query, limit=limit)
        _print_results(results, f"Papers with Code: {query}")
    elif source in ("huggingface", "hf", "hugging-face"):
        results = search_huggingface(query, limit=limit)
        _print_results(results, f"Hugging Face Trending: {query}")
    elif source in ("jmlr",):
        results = search_jmlr(query, limit=limit)
        _print_results(results, f"JMLR: {query}")
    elif source in ("openscholar", "os", "open-scholar"):
        search_openscholar(query, limit=limit)
    else:
        print(f"Unknown source: {source}. Use: arxiv, semantic-scholar, papers-with-code, huggingface, jmlr, openscholar", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
