#!/usr/bin/env python3
"""Unified paper fetch by ID. Auto-detects source from ID format.

Usage:
    fetch.py <paper_id> [--source arxiv|semantic-scholar|papers-with-code|acl|jmlr]

ID auto-detection:
    2401.12345           → arXiv  (YYMM.NNNNN)
    hep-ex/0307015       → arXiv  (old format)
    2022.acl-long.220    → ACL Anthology
    v22/19-920           → JMLR   (volume/paper-id)
    arXiv:2401.12345     → Semantic Scholar (external ID lookup)
    40-char hex string   → Semantic Scholar (native ID)
    anything else        → arXiv (default)
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
# Source auto-detection
# ---------------------------------------------------------------------------

def detect_source(paper_id):
    """Detect which source a paper ID belongs to."""
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", paper_id):
        return "arxiv"
    if re.match(r"^[a-z-]+/\d{7}(v\d+)?$", paper_id):
        return "arxiv"
    if re.match(r"^\d{4}\.[a-zA-Z]+-[a-zA-Z]+\.\d+$", paper_id):
        return "acl"
    # JMLR: v22/19-920 or 22/19-920 or v22/20-1473
    if re.match(r"^v?\d+/\d+-\d+$", paper_id):
        return "jmlr"
    if paper_id.startswith("arXiv:") or paper_id.startswith("DOI:") or paper_id.startswith("PMID:"):
        return "semantic-scholar"
    if re.match(r"^[0-9a-f]{40}$", paper_id):
        return "semantic-scholar"
    return "arxiv"


# ---------------------------------------------------------------------------
# Standardized output
# ---------------------------------------------------------------------------

def _print_paper(p, source_label):
    """Print a single paper in standardized format."""
    print(f"# {source_label}: {p.get('id', '')}")
    print("---")
    title = p.get("title") or ""
    print(f"Title: {title}")
    authors = p.get("authors") or []
    print(f"Authors: {', '.join(str(a) for a in authors[:5])}")
    if p.get("year"):
        print(f"Year: {p['year']}")
    if p.get("abstract"):
        print(f"Abstract: {p['abstract'][:500]}")
    if p.get("pdf_url"):
        print(f"PDF: {p['pdf_url']}")
    if p.get("citations") is not None:
        print(f"Citations: {p['citations']}")
    if p.get("extra"):
        print(p["extra"])


# ---------------------------------------------------------------------------
# arXiv fetch
# ---------------------------------------------------------------------------

def _arxiv_ns(tag):
    return f"{{http://www.w3.org/2005/Atom}}{tag}"

def _arxiv_ext_ns(tag):
    return f"{{http://arxiv.org/schemas/atom}}{tag}"

def fetch_arxiv(paper_id):
    """Fetch paper metadata from arXiv by ID."""
    time.sleep(3)
    url = f"http://export.arxiv.org/api/query?{urlencode({'id_list': paper_id, 'max_results': 5})}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read().decode("utf-8"))
    except (HTTPError, URLError) as e:
        print(f"arXiv error: {e}", file=sys.stderr)
        sys.exit(1)

    entries = root.findall(f".//{_arxiv_ns('entry')}")
    if not entries:
        print(f"Paper not found: {paper_id}", file=sys.stderr)
        sys.exit(1)

    entry = entries[0]
    # Parse fields
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

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    for link in entry.findall(f".//{_arxiv_ns('link')}"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break

    categories = []
    for cat in entry.findall(f".//{_arxiv_ns('category')}"):
        term = cat.get("term")
        if term:
            categories.append(term)

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors[:5],
        "year": published[:4] if published else None,
        "pdf_url": pdf_url,
        "citations": None,
        "extra": f"Categories: {', '.join(categories[:4])}" if categories else "",
    }


# ---------------------------------------------------------------------------
# Semantic Scholar fetch
# ---------------------------------------------------------------------------

def fetch_semantic_scholar(paper_id):
    """Fetch paper from Semantic Scholar by ID (native or external like arXiv:XXXX)."""
    time.sleep(4)
    fields = "paperId,title,abstract,authors,year,openAccessPdf,citationCount,referenceCount"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            p = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        print(f"Semantic Scholar error: {e}", file=sys.stderr)
        sys.exit(1)

    if not p.get("paperId"):
        print(f"Paper not found: {paper_id}", file=sys.stderr)
        sys.exit(1)

    pdf = (p.get("openAccessPdf") or {}).get("url", "")
    authors = [a.get("name", "") for a in (p.get("authors") or [])[:5] if isinstance(a, dict)]

    return {
        "id": p.get("paperId", ""),
        "title": (p.get("title") or "").replace("\n", " "),
        "abstract": (p.get("abstract") or "")[:500],
        "authors": authors,
        "year": p.get("year"),
        "pdf_url": pdf,
        "citations": p.get("citationCount"),
        "extra": f"References: {p.get('referenceCount', 'N/A')}",
    }


# ---------------------------------------------------------------------------
# Papers with Code fetch
# ---------------------------------------------------------------------------

def fetch_papers_with_code(paper_id):
    """Fetch paper from Papers with Code by arXiv ID via REST API."""
    time.sleep(3)
    arxiv_id = paper_id.replace("arXiv:", "")
    params = urlencode({"q": arxiv_id, "items_per_page": 5, "page": 1})
    url = f"https://paperswithcode.com/api/v1/search/?{params}"
    try:
        req = Request(url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        print(f"Papers with Code error: {e}", file=sys.stderr)
        sys.exit(1)

    papers = data.get("results", [])
    if not papers:
        print(f"Paper not found: {paper_id}", file=sys.stderr)
        sys.exit(1)

    p = papers[0]
    return {
        "id": arxiv_id,
        "title": p.get("title", "") or "",
        "abstract": (p.get("abstract", "") or "")[:500],
        "authors": [],
        "year": None,
        "pdf_url": p.get("url_pdf", "") or "",
        "citations": None,
        "extra": f"Abstract URL: {p.get('url_abs', '') or ''}",
    }


# ---------------------------------------------------------------------------
# ACL Anthology fetch
# ---------------------------------------------------------------------------

def fetch_acl(paper_id):
    """Fetch paper from ACL Anthology by ID (e.g. 2022.acl-long.220). Scrapes the web page."""
    time.sleep(3)
    abs_url = f"https://aclanthology.org/{paper_id}/"
    pdf_url = f"https://aclanthology.org/{paper_id}.pdf"

    title = ""
    authors = []
    abstract = ""
    year = None
    try:
        req = Request(abs_url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Title from <title> or <h2>
        title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            title = re.sub(r"\s*[-–|].*ACL Anthology.*$", "", title).strip()
        # Authors from meta tag
        author_match = re.search(r'<meta\s+name="citation_author"\s+content="([^"]+)"', html, re.IGNORECASE)
        if not author_match:
            # Try multiple citation_author tags
            authors = re.findall(r'<meta\s+name="citation_author"\s+content="([^"]+)"', html, re.IGNORECASE)[:5]
        else:
            authors = [author_match.group(1)]
        # Year from ID prefix
        year_match = re.match(r"^(\d{4})\.", paper_id)
        if year_match:
            year = year_match.group(1)
        # Abstract
        abs_match = re.search(r'class="[^"]*abstract[^"]*"[^>]*>(.*?)</(?:div|span|p)', html, re.DOTALL | re.IGNORECASE)
        if abs_match:
            abstract = re.sub(r"<[^>]+>", "", abs_match.group(1)).strip()[:500]
    except (HTTPError, URLError):
        pass  # Metadata is optional — we always have the PDF URL

    return {
        "id": paper_id,
        "title": title or paper_id,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "pdf_url": pdf_url,
        "citations": None,
        "extra": "",
    }


# ---------------------------------------------------------------------------
# JMLR fetch
# ---------------------------------------------------------------------------

def _parse_jmlr_id(paper_id):
    """Parse JMLR ID into (volume, paper_id). Accepts v22/19-920 or 22/19-920."""
    paper_id = paper_id.lstrip("v")
    parts = paper_id.split("/", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def fetch_jmlr(paper_id):
    """Fetch JMLR paper metadata. Resolves ID to PDF URL and scrapes title from the page."""
    vol, pid = _parse_jmlr_id(paper_id)
    if not vol or not pid:
        print(f"Invalid JMLR ID: {paper_id}. Expected format: v22/19-920", file=sys.stderr)
        sys.exit(1)

    pdf_url = f"https://jmlr.org/papers/volume{vol}/{pid}/{pid}.pdf"
    abs_url = f"https://jmlr.org/papers/volume{vol}/{pid}/"

    # Try to scrape the paper page for title/authors
    title = ""
    authors = []
    time.sleep(3)
    try:
        req = Request(abs_url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Extract title from <title> tag or <h2>
        title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            # Remove "JMLR:" prefix if present
            title = re.sub(r"^JMLR:\s*", "", title)
        # Extract author names from common patterns
        author_match = re.search(r"(?:Authors?|By)[:\s]+([^<]+)", html, re.IGNORECASE)
        if author_match:
            authors = [a.strip() for a in author_match.group(1).split(",") if a.strip()][:5]
    except (HTTPError, URLError):
        pass  # Metadata is optional — we always have the PDF URL

    return {
        "id": f"v{vol}/{pid}",
        "title": title or f"JMLR Volume {vol} Paper {pid}",
        "abstract": "",
        "authors": authors,
        "year": None,
        "pdf_url": pdf_url,
        "citations": None,
        "extra": f"Volume: {vol} | Page: {abs_url}",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if args else 1)

    paper_id = None
    source = None
    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1].lower()
            i += 2
        elif not args[i].startswith("-"):
            paper_id = args[i]
            i += 1
        else:
            i += 1

    if not paper_id:
        print("Error: no paper ID provided", file=sys.stderr)
        sys.exit(1)

    if source is None:
        source = detect_source(paper_id)

    if source in ("arxiv", "ar"):
        result = fetch_arxiv(paper_id)
        _print_paper(result, "arXiv")
    elif source in ("semantic-scholar", "s2", "semanticscholar"):
        result = fetch_semantic_scholar(paper_id)
        _print_paper(result, "Semantic Scholar")
    elif source in ("papers-with-code", "pwc", "paperswithcode"):
        result = fetch_papers_with_code(paper_id)
        _print_paper(result, "Papers with Code")
    elif source in ("acl", "acl-anthology"):
        result = fetch_acl(paper_id)
        _print_paper(result, "ACL Anthology")
    elif source in ("jmlr",):
        result = fetch_jmlr(paper_id)
        _print_paper(result, "JMLR")
    else:
        print(f"Unknown source: {source}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
