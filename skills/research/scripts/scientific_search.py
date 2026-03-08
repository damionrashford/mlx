#!/usr/bin/env python3
"""
Multi-source academic paper search and dataset discovery.
Searches arXiv, Semantic Scholar, PubMed concurrently with deduplication.
Discovers datasets from Kaggle and HuggingFace.
Extracted from RivalSearchMCP scientific_research tool.

Usage:
    python3 scientific_search.py "transformer attention" --max 10
    python3 scientific_search.py "BERT NLP" --source arxiv,semantic_scholar
    python3 scientific_search.py "image classification" --datasets
    python3 scientific_search.py "tabular data" --datasets --source kaggle
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlx-research/1.0"})


# ---------------------------------------------------------------------------
# Paper providers
# ---------------------------------------------------------------------------

def search_arxiv(query: str, limit: int = 10) -> list:
    """Search arXiv API."""
    try:
        resp = SESSION.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": query, "start": 0, "max_results": min(limit, 200),
                    "sortBy": "relevance", "sortOrder": "descending"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("a:entry", ns)[:limit]:
            title_el = entry.find("a:title", ns)
            id_el = entry.find("a:id", ns)
            summary_el = entry.find("a:summary", ns)
            published_el = entry.find("a:published", ns)
            authors = [
                a.find("a:name", ns).text.strip()
                for a in entry.findall("a:author", ns)
                if a.find("a:name", ns) is not None
            ]
            url = id_el.text.strip() if id_el is not None else ""
            pub = published_el.text.strip() if published_el is not None else ""
            papers.append({
                "title": title_el.text.strip() if title_el is not None else "",
                "authors": authors,
                "abstract": summary_el.text.strip() if summary_el is not None else "",
                "url": url,
                "year": pub[:4] if pub else None,
                "paperId": url.split("/")[-1] if url else None,
                "source": "arxiv",
            })
        return papers
    except Exception as e:
        print(f"  arXiv error: {e}", file=sys.stderr)
        return []


def search_semantic_scholar(query: str, limit: int = 10) -> list:
    """Search Semantic Scholar API."""
    try:
        fields = "title,abstract,authors,year,venue,citationCount,openAccessPdf,url,paperId"
        resp = SESSION.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": min(limit, 100), "fields": fields},
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        papers = resp.json().get("data", [])
        for p in papers:
            p["source"] = "semantic_scholar"
            if p.get("authors"):
                p["authors"] = [a.get("name", "") for a in p["authors"]]
        return papers
    except Exception as e:
        print(f"  Semantic Scholar error: {e}", file=sys.stderr)
        return []


def search_pubmed(query: str, limit: int = 10) -> list:
    """Search PubMed API."""
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        resp = SESSION.get(
            f"{base}esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": min(limit, 100),
                    "retmode": "json", "sort": "relevance"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_resp = SESSION.get(
            f"{base}efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids[:min(limit, 20)]), "retmode": "xml"},
            timeout=30,
        )
        if fetch_resp.status_code != 200:
            return []

        root = ET.fromstring(fetch_resp.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            pmid_el = article.find(".//PMID")
            year_el = article.find(".//PubDate/Year")
            journal_el = article.find(".//Journal/Title")
            authors = []
            for a in article.findall(".//Author"):
                last = a.find(".//LastName")
                fore = a.find(".//ForeName")
                if last is not None:
                    name = last.text or ""
                    if fore is not None and fore.text:
                        name = f"{fore.text} {name}"
                    authors.append(name.strip())

            if title_el is not None and title_el.text:
                pmid = pmid_el.text.strip() if pmid_el is not None else ""
                papers.append({
                    "title": title_el.text.strip(),
                    "authors": authors,
                    "abstract": abstract_el.text.strip() if abstract_el is not None and abstract_el.text else "",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    "year": year_el.text.strip() if year_el is not None and year_el.text else None,
                    "venue": journal_el.text.strip() if journal_el is not None and journal_el.text else "",
                    "paperId": pmid,
                    "source": "pubmed",
                })
        return papers
    except Exception as e:
        print(f"  PubMed error: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Dataset providers
# ---------------------------------------------------------------------------

def search_kaggle(query: str, limit: int = 10) -> list:
    """Search Kaggle datasets."""
    try:
        resp = SESSION.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": query, "size": min(limit, 100)},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        datasets = []
        for d in resp.json()[:limit]:
            datasets.append({
                "title": d.get("title", d.get("titleNullable", "")),
                "description": d.get("subtitle", ""),
                "url": d.get("url", ""),
                "downloads": d.get("downloadCount", 0),
                "votes": d.get("voteCount", 0),
                "license": d.get("licenseName", ""),
                "last_updated": d.get("lastUpdated", ""),
                "source": "kaggle",
            })
        return datasets
    except Exception as e:
        print(f"  Kaggle error: {e}", file=sys.stderr)
        return []


def search_huggingface(query: str, limit: int = 10) -> list:
    """Search HuggingFace datasets."""
    try:
        resp = SESSION.get(
            "https://huggingface.co/api/datasets",
            params={"search": query, "limit": min(limit, 100)},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        datasets = []
        for d in resp.json()[:limit]:
            did = d.get("id", "")
            datasets.append({
                "title": did.split("/")[-1] if did else "",
                "id": did,
                "description": (d.get("description", "") or "")[:200],
                "url": f"https://huggingface.co/datasets/{did}",
                "downloads": d.get("downloads", 0),
                "likes": d.get("likes", 0),
                "tags": d.get("tags", []),
                "source": "huggingface",
            })
        return datasets
    except Exception as e:
        print(f"  HuggingFace error: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

PAPER_PROVIDERS = {
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "pubmed": search_pubmed,
}

DATASET_PROVIDERS = {
    "kaggle": search_kaggle,
    "huggingface": search_huggingface,
}


def search_concurrent(query: str, providers: dict, sources: list, limit: int) -> list:
    """Search multiple providers concurrently and deduplicate."""
    valid = [s for s in sources if s in providers]
    if not valid:
        print(f"No valid sources in: {sources}. Available: {list(providers.keys())}", file=sys.stderr)
        return []

    all_results = []
    with ThreadPoolExecutor(max_workers=len(valid)) as pool:
        futures = {pool.submit(providers[s], query, limit): s for s in valid}
        for future in as_completed(futures):
            source = futures[future]
            try:
                results = future.result()
                print(f"  {source}: {len(results)} results", file=sys.stderr)
                all_results.extend(results)
            except Exception as e:
                print(f"  {source}: error — {e}", file=sys.stderr)

    # Deduplicate by title
    seen = set()
    deduped = []
    for item in all_results:
        key = item.get("title", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:limit]


def format_papers(papers: list) -> str:
    """Format paper results for display."""
    lines = []
    for i, p in enumerate(papers, 1):
        authors = p.get("authors", [])
        if isinstance(authors, list) and authors:
            if isinstance(authors[0], dict):
                author_str = ", ".join(a.get("name", "") for a in authors[:3])
            else:
                author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
        else:
            author_str = "Unknown"

        year = p.get("year", "?")
        citations = p.get("citationCount", "")
        cite_str = f"  [{citations} citations]" if citations else ""
        source = p.get("source", "")

        lines.append(f"{i}. [{source}] {p.get('title', 'Untitled')} ({year}){cite_str}")
        lines.append(f"   {author_str}")
        lines.append(f"   {p.get('url', '')}")

        abstract = p.get("abstract", "")
        if abstract:
            lines.append(f"   {abstract[:200]}...")
        lines.append("")

    return "\n".join(lines)


def format_datasets(datasets: list) -> str:
    """Format dataset results for display."""
    lines = []
    for i, d in enumerate(datasets, 1):
        source = d.get("source", "")
        downloads = d.get("downloads", d.get("download_count", 0))
        likes = d.get("likes", d.get("votes", 0))
        stats = []
        if downloads:
            stats.append(f"{downloads:,} downloads")
        if likes:
            stats.append(f"{likes} likes")
        stat_str = f"  [{', '.join(stats)}]" if stats else ""

        lines.append(f"{i}. [{source}] {d.get('title', 'Untitled')}{stat_str}")
        lines.append(f"   {d.get('url', '')}")
        desc = d.get("description", "")
        if desc:
            lines.append(f"   {desc[:150]}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-source academic search & dataset discovery")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--source", type=str, default=None,
                        help="Comma-separated sources (papers: arxiv,semantic_scholar,pubmed | datasets: kaggle,huggingface)")
    parser.add_argument("--datasets", action="store_true", help="Search for datasets instead of papers")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.datasets:
        providers = DATASET_PROVIDERS
        default_sources = list(DATASET_PROVIDERS.keys())
        label = "datasets"
    else:
        providers = PAPER_PROVIDERS
        default_sources = ["arxiv", "semantic_scholar"]
        label = "papers"

    sources = args.source.split(",") if args.source else default_sources

    print(f"Searching {label}: \"{args.query}\" across {sources}...", file=sys.stderr)
    start = time.time()
    results = search_concurrent(args.query, providers, sources, args.max)
    elapsed = time.time() - start
    print(f"Found {len(results)} unique {label} in {elapsed:.1f}s\n", file=sys.stderr)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if args.datasets:
            print(format_datasets(results))
        else:
            print(format_papers(results))


if __name__ == "__main__":
    main()
