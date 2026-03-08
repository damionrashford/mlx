#!/usr/bin/env python3
"""Search and download ML/AI datasets from 5 free sources (no API keys).

Usage:
    datasets.py search <query> [--source SOURCE] [--limit N]
    datasets.py info <dataset_id> [--source SOURCE]
    datasets.py download <dataset_id> [--source SOURCE] [--output DIR] [--split train]

Sources: huggingface (default), paperswithcode, openml, uci, kaggle
All free, no API keys required.

Examples:
    datasets.py search "sentiment analysis"
    datasets.py search "image classification" --source openml --limit 5
    datasets.py info imdb --source huggingface
    datasets.py download imdb --source huggingface --output ./datasets --split train
    datasets.py download 61 --source openml --output ./datasets
"""

import gzip
import json
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


_UA = "ml-paper-research-skill/1.0"


def _get_json(url, timeout=30):
    """Fetch JSON from URL with standard headers."""
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url, timeout=30):
    """Fetch text/HTML from URL."""
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        encoding = resp.headers.get_content_charset() or "utf-8"
        return data.decode(encoding, errors="replace")


def _download_file(url, dest_path, timeout=120):
    """Download a file from URL to disk."""
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)
    return len(data)


# ---------------------------------------------------------------------------
# Standardized output
# ---------------------------------------------------------------------------

def _print_results(results, source_label, query):
    """Print dataset results in consistent format."""
    print(f"# {source_label}: {query}")
    print(f"Found {len(results)} datasets")
    print("---")
    for i, d in enumerate(results, 1):
        title = (d.get("title") or d.get("name") or "")[:80]
        print(f"\n## [{i}] {d.get('id', '')} — {title}")
        desc = (d.get("description") or "")[:250]
        if desc:
            print(f"Description: {desc}")
        stats = []
        if d.get("downloads"):
            stats.append(f"Downloads: {d['downloads']:,}")
        if d.get("likes"):
            stats.append(f"Likes: {d['likes']}")
        if d.get("instances"):
            stats.append(f"Instances: {d['instances']:,}")
        if d.get("features"):
            stats.append(f"Features: {d['features']}")
        if d.get("license"):
            stats.append(f"License: {d['license']}")
        if stats:
            print(f"Stats: {' | '.join(stats)}")
        if d.get("tags"):
            tags = d["tags"][:6] if isinstance(d["tags"], list) else [d["tags"]]
            print(f"Tags: {', '.join(str(t) for t in tags)}")
        print(f"URL: {d.get('url', '(none)')}")
        print(f"Source: {d.get('source', '')}")


def _print_info(d, source_label):
    """Print detailed dataset info."""
    print(f"# {source_label}: {d.get('id', '')}")
    print("---")
    print(f"Name: {d.get('title') or d.get('name', '')}")
    if d.get("description"):
        print(f"Description: {d['description'][:500]}")
    if d.get("authors"):
        print(f"Authors: {d['authors']}")
    stats = []
    if d.get("downloads"):
        stats.append(f"Downloads: {d['downloads']:,}")
    if d.get("likes"):
        stats.append(f"Likes: {d['likes']}")
    if d.get("instances"):
        stats.append(f"Instances: {d['instances']:,}")
    if d.get("features"):
        print(f"Features: {d['features']}")
    if stats:
        print(f"Stats: {' | '.join(stats)}")
    if d.get("license"):
        print(f"License: {d['license']}")
    if d.get("tags"):
        print(f"Tags: {', '.join(str(t) for t in d['tags'][:10])}")
    if d.get("url"):
        print(f"URL: {d['url']}")
    if d.get("download_urls"):
        print(f"\nDownloadable files:")
        for name, url in d["download_urls"].items():
            print(f"  {name}: {url}")
    if d.get("splits"):
        print(f"Splits: {', '.join(d['splits'])}")
    if d.get("columns"):
        print(f"Columns: {', '.join(d['columns'][:15])}")


# ===========================================================================
# HuggingFace Datasets
# ===========================================================================

_HF_API = "https://huggingface.co/api/datasets"


def search_huggingface(query, limit=10):
    """Search HuggingFace datasets. Free, no key required."""
    params = urlencode({"search": query, "limit": min(limit, 100), "sort": "downloads", "direction": "-1"})
    url = f"{_HF_API}?{params}"
    try:
        data = _get_json(url)
    except (HTTPError, URLError) as e:
        print(f"HuggingFace error: {e}", file=sys.stderr)
        return []

    results = []
    for d in data[:limit]:
        did = d.get("id", "")
        tags = d.get("tags", [])
        license_tag = ""
        task_tags = []
        for t in tags:
            if t.startswith("license:"):
                license_tag = t.replace("license:", "")
            elif t.startswith("task_categories:"):
                task_tags.append(t.replace("task_categories:", ""))

        results.append({
            "id": did,
            "name": did.split("/")[-1] if "/" in did else did,
            "title": did,
            "description": (d.get("description") or "")[:250],
            "url": f"https://huggingface.co/datasets/{did}",
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "license": license_tag,
            "tags": task_tags[:5] if task_tags else tags[:5],
            "source": "HuggingFace",
        })
    return results


def info_huggingface(dataset_id):
    """Get detailed info for a HuggingFace dataset."""
    url = f"{_HF_API}/{dataset_id}"
    try:
        d = _get_json(url)
    except (HTTPError, URLError) as e:
        print(f"HuggingFace error: {e}", file=sys.stderr)
        return None

    tags = d.get("tags", [])
    license_tag = ""
    for t in tags:
        if t.startswith("license:"):
            license_tag = t.replace("license:", "")

    # Try to get split/config info
    splits = []
    try:
        info_data = _get_json(f"https://datasets-server.huggingface.co/info?dataset={quote(dataset_id)}")
        ds_info = info_data.get("dataset_info", {})
        if ds_info:
            first_config = next(iter(ds_info.values()), {})
            split_info = first_config.get("splits", {})
            splits = list(split_info.keys())
    except Exception:
        pass

    # Try to get column names from first rows
    columns = []
    try:
        first_rows = _get_json(f"https://datasets-server.huggingface.co/first-rows?dataset={quote(dataset_id)}&config=default&split=train")
        features = first_rows.get("features", [])
        columns = [f.get("name", "") for f in features if f.get("name")]
    except Exception:
        pass

    return {
        "id": dataset_id,
        "title": dataset_id,
        "description": (d.get("description") or d.get("cardData", {}).get("description", "") or "")[:500],
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "downloads": d.get("downloads", 0),
        "likes": d.get("likes", 0),
        "license": license_tag,
        "tags": tags[:10],
        "splits": splits,
        "columns": columns,
        "source": "HuggingFace",
        "download_urls": {
            "parquet (via API)": f"https://datasets-server.huggingface.co/parquet?dataset={quote(dataset_id)}",
        },
    }


def download_huggingface(dataset_id, output_dir=".", split="train"):
    """Download a HuggingFace dataset split as parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    # Get parquet file URLs from the datasets server
    try:
        parquet_info = _get_json(f"https://datasets-server.huggingface.co/parquet?dataset={quote(dataset_id)}")
    except (HTTPError, URLError) as e:
        print(f"Error fetching parquet info: {e}", file=sys.stderr)
        # Fallback: try CSV download via first-rows
        print("Trying CSV fallback...", file=sys.stderr)
        return _download_hf_csv_fallback(dataset_id, output_dir, split)

    parquet_files = parquet_info.get("parquet_files", [])
    if not parquet_files:
        print("No parquet files available. Try a different dataset or use the HuggingFace `datasets` library.", file=sys.stderr)
        print(f"  pip install datasets && python3 -c \"from datasets import load_dataset; ds = load_dataset('{dataset_id}'); ds.save_to_disk('{output_dir}')\"")
        return

    # Filter by split
    split_files = [f for f in parquet_files if f.get("split") == split]
    if not split_files:
        available = sorted(set(f.get("split", "") for f in parquet_files))
        print(f"Split '{split}' not found. Available splits: {', '.join(available)}", file=sys.stderr)
        if available:
            split = available[0]
            split_files = [f for f in parquet_files if f.get("split") == split]
            print(f"Downloading '{split}' instead.", file=sys.stderr)
        else:
            return

    total_size = 0
    downloaded = []
    for pf in split_files:
        purl = pf.get("url", "")
        if not purl:
            continue
        fname = pf.get("filename", purl.split("/")[-1])
        safe_name = re.sub(r"[^\w.\-]", "_", fname)
        dest = os.path.join(output_dir, safe_name)
        print(f"Downloading {fname}...", file=sys.stderr)
        try:
            size = _download_file(purl, dest, timeout=300)
            total_size += size
            downloaded.append(dest)
            print(f"  Saved: {dest} ({size:,} bytes)")
        except (HTTPError, URLError) as e:
            print(f"  Failed: {e}", file=sys.stderr)

    if downloaded:
        print(f"\nDownloaded {len(downloaded)} file(s), {total_size:,} bytes total")
        print(f"Format: Parquet (read with pandas: pd.read_parquet('{downloaded[0]}'))")
    else:
        print("No files downloaded.", file=sys.stderr)


def _download_hf_csv_fallback(dataset_id, output_dir, split):
    """Fallback: download first N rows as CSV via the datasets server."""
    try:
        rows_data = _get_json(
            f"https://datasets-server.huggingface.co/first-rows?dataset={quote(dataset_id)}&config=default&split={split}&length=100"
        )
    except (HTTPError, URLError) as e:
        print(f"CSV fallback also failed: {e}", file=sys.stderr)
        print(f"Install the datasets library for full download:")
        print(f"  pip install datasets")
        print(f"  python3 -c \"from datasets import load_dataset; ds = load_dataset('{dataset_id}'); ds.save_to_disk('{output_dir}')\"")
        return

    rows = rows_data.get("rows", [])
    features = rows_data.get("features", [])
    if not rows:
        print("No rows returned.", file=sys.stderr)
        return

    import csv

    columns = [f.get("name", f"col_{i}") for i, f in enumerate(features)]
    dest = os.path.join(output_dir, f"{dataset_id.replace('/', '_')}_{split}_sample.csv")
    os.makedirs(output_dir, exist_ok=True)

    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_data = row.get("row", {})
            writer.writerow(row_data)

    print(f"Downloaded {len(rows)} sample rows to: {dest}")
    print(f"Note: This is a sample. For the full dataset, use:")
    print(f"  pip install datasets")
    print(f"  python3 -c \"from datasets import load_dataset; ds = load_dataset('{dataset_id}')\"")


# ===========================================================================
# Papers with Code Datasets
# ===========================================================================

_PWC_API = "https://paperswithcode.com/api/v1"


def search_paperswithcode(query, limit=10):
    """Search Papers with Code datasets. Free, no key."""
    time.sleep(3)
    params = urlencode({"q": query, "items_per_page": min(limit, 50), "page": 1})
    url = f"{_PWC_API}/datasets/?{params}"
    try:
        data = _get_json(url)
    except (HTTPError, URLError) as e:
        print(f"Papers with Code error: {e}", file=sys.stderr)
        return []

    results = []
    for d in data.get("results", [])[:limit]:
        if not isinstance(d, dict):
            continue
        num_papers = d.get("num_papers", 0)
        results.append({
            "id": d.get("id", "") or d.get("url", "").rstrip("/").split("/")[-1],
            "name": d.get("name", ""),
            "title": d.get("name", ""),
            "description": (d.get("description") or "")[:250],
            "url": d.get("url", ""),
            "instances": None,
            "tags": [f"{num_papers} papers"] if num_papers else [],
            "license": "",
            "source": "Papers with Code",
        })
    return results


def info_paperswithcode(dataset_id):
    """Get dataset info from Papers with Code."""
    time.sleep(3)
    url = f"{_PWC_API}/datasets/{quote(dataset_id)}/"
    try:
        d = _get_json(url)
    except (HTTPError, URLError) as e:
        print(f"Papers with Code error: {e}", file=sys.stderr)
        return None

    return {
        "id": dataset_id,
        "title": d.get("name", dataset_id),
        "description": (d.get("description") or "")[:500],
        "url": d.get("url", f"https://paperswithcode.com/dataset/{dataset_id}"),
        "license": "",
        "tags": [],
        "source": "Papers with Code",
        "download_urls": {"homepage": d.get("url", "")},
    }


# ===========================================================================
# OpenML
# ===========================================================================

_OPENML_API = "https://www.openml.org/api/v1/json"


def search_openml(query, limit=10):
    """Search OpenML datasets. Free, no key."""
    time.sleep(2)
    params = urlencode({"data_name": query, "limit": min(limit, 100), "status": "active"})
    url = f"{_OPENML_API}/data/list?{params}"
    try:
        data = _get_json(url)
    except (HTTPError, URLError) as e:
        # OpenML returns 412 when no results found
        if hasattr(e, "code") and e.code == 412:
            return []
        print(f"OpenML error: {e}", file=sys.stderr)
        return []

    datasets = data.get("data", {}).get("dataset", [])
    if not isinstance(datasets, list):
        return []

    results = []
    for d in datasets[:limit]:
        did = d.get("did", "")
        qualities = {q.get("name", ""): q.get("value", "") for q in d.get("quality", []) if isinstance(q, dict)}
        results.append({
            "id": str(did),
            "name": d.get("name", ""),
            "title": d.get("name", ""),
            "description": "",
            "url": f"https://www.openml.org/d/{did}",
            "instances": int(float(qualities.get("NumberOfInstances", 0))) if qualities.get("NumberOfInstances") else None,
            "features": qualities.get("NumberOfFeatures", ""),
            "downloads": int(float(qualities.get("NumberOfDownloads", 0))) if qualities.get("NumberOfDownloads") else None,
            "tags": [],
            "license": "",
            "source": "OpenML",
        })
    return results


def info_openml(dataset_id):
    """Get detailed OpenML dataset info."""
    time.sleep(2)
    url = f"{_OPENML_API}/data/{dataset_id}"
    try:
        data = _get_json(url)
    except (HTTPError, URLError) as e:
        print(f"OpenML error: {e}", file=sys.stderr)
        return None

    d = data.get("data_set_description", {})
    did = d.get("id", dataset_id)

    # Get features
    columns = []
    try:
        feat_data = _get_json(f"{_OPENML_API}/data/features/{dataset_id}")
        features = feat_data.get("data_features", {}).get("feature", [])
        columns = [f.get("name", "") for f in features if f.get("name")]
    except Exception:
        pass

    csv_url = d.get("url", "")

    return {
        "id": str(did),
        "title": d.get("name", ""),
        "description": (d.get("description") or "")[:500],
        "url": f"https://www.openml.org/d/{did}",
        "license": d.get("licence", ""),
        "tags": (d.get("tag") or []) if isinstance(d.get("tag"), list) else [d.get("tag", "")] if d.get("tag") else [],
        "columns": columns,
        "source": "OpenML",
        "download_urls": {"ARFF": csv_url} if csv_url else {},
    }


def download_openml(dataset_id, output_dir="."):
    """Download an OpenML dataset (ARFF format, converts to CSV if possible)."""
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset description to find the download URL
    time.sleep(2)
    try:
        data = _get_json(f"{_OPENML_API}/data/{dataset_id}")
    except (HTTPError, URLError) as e:
        print(f"OpenML error: {e}", file=sys.stderr)
        return

    d = data.get("data_set_description", {})
    csv_url = d.get("url", "")
    name = d.get("name", f"openml_{dataset_id}")

    if not csv_url:
        print(f"No download URL found for OpenML dataset {dataset_id}", file=sys.stderr)
        return

    # Download the ARFF file
    dest = os.path.join(output_dir, f"{name}.arff")
    print(f"Downloading {name} from OpenML...", file=sys.stderr)
    try:
        req = Request(csv_url, headers={"User-Agent": _UA, "Accept-Encoding": "gzip"})
        with urlopen(req, timeout=120) as resp:
            raw = resp.read()
            # OpenML often returns gzip
            try:
                raw = gzip.decompress(raw)
            except (gzip.BadGzipFile, OSError):
                pass
        with open(dest, "wb") as f:
            f.write(raw)
        print(f"Downloaded: {dest} ({len(raw):,} bytes)")
    except (HTTPError, URLError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return

    # Try to convert ARFF to CSV
    try:
        csv_dest = os.path.join(output_dir, f"{name}.csv")
        _arff_to_csv(dest, csv_dest)
        print(f"Converted to CSV: {csv_dest}")
        print(f"Read with: pd.read_csv('{csv_dest}')")
    except Exception:
        print(f"ARFF saved (install liac-arff or scipy for CSV conversion)")
        print(f"Read with: from scipy.io import arff; data, meta = arff.loadarff('{dest}')")


def _arff_to_csv(arff_path, csv_path):
    """Simple ARFF to CSV conversion (handles basic ARFF files)."""
    import csv

    with open(arff_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    headers = []
    data_started = False
    rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.upper().startswith("@ATTRIBUTE"):
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                attr_name = parts[1].strip("'\"")
                headers.append(attr_name)
        elif stripped.upper().startswith("@DATA"):
            data_started = True
        elif data_started and stripped:
            # Handle quoted fields
            reader = csv.reader([stripped])
            for row in reader:
                rows.append(row)

    if not headers or not rows:
        raise ValueError("Could not parse ARFF")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ===========================================================================
# UCI ML Repository
# ===========================================================================

_UCI_API = "https://archive.ics.uci.edu/api/datasets"


def search_uci(query, limit=10):
    """Search UCI ML Repository datasets. Free, no key."""
    time.sleep(2)
    params = urlencode({"search": query})
    url = f"{_UCI_API}/list?{params}"
    try:
        data = _get_json(url)
    except (HTTPError, URLError) as e:
        print(f"UCI error: {e}", file=sys.stderr)
        return []

    datasets = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []

    results = []
    for d in datasets[:limit]:
        if not isinstance(d, dict):
            continue
        did = d.get("id", "")
        name = d.get("name", "")
        slug = re.sub(r"[^\w\-]", "-", name.lower()).strip("-") if name else ""
        results.append({
            "id": str(did),
            "name": name,
            "title": name,
            "description": (d.get("abstract") or d.get("description") or "")[:250],
            "url": f"https://archive.ics.uci.edu/dataset/{did}/{slug}" if did else "",
            "instances": d.get("numInstances") or d.get("num_instances"),
            "features": str(d.get("numFeatures") or d.get("num_features", "")) or None,
            "tags": d.get("tasks", []) if isinstance(d.get("tasks"), list) else [],
            "license": "",
            "source": "UCI",
        })
    return results


def info_uci(dataset_id):
    """Get detailed UCI dataset info."""
    time.sleep(2)
    url = f"{_UCI_API}/{dataset_id}"
    try:
        data = _get_json(url)
    except (HTTPError, URLError) as e:
        # Try the list endpoint filtered by ID
        try:
            list_data = _get_json(f"{_UCI_API}/list?search={dataset_id}")
            datasets = list_data.get("data", []) if isinstance(list_data, dict) else []
            if datasets:
                data = datasets[0]
            else:
                print(f"UCI error: {e}", file=sys.stderr)
                return None
        except Exception:
            print(f"UCI error: {e}", file=sys.stderr)
            return None

    d = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(d, dict):
        return None
    did = d.get("id", dataset_id)
    name = d.get("name", "")
    slug = re.sub(r"[^\w\-]", "-", name.lower()).strip("-") if name else ""

    variables = d.get("variables", [])
    columns = [v.get("name", "") for v in variables if isinstance(v, dict) and v.get("name")]

    return {
        "id": str(did),
        "title": name,
        "description": (d.get("abstract") or d.get("description") or "")[:500],
        "url": f"https://archive.ics.uci.edu/dataset/{did}/{slug}",
        "instances": d.get("numInstances") or d.get("num_instances"),
        "features": str(d.get("numFeatures") or d.get("num_features", "")) or None,
        "license": "",
        "columns": columns,
        "tags": d.get("tasks", []) if isinstance(d.get("tasks"), list) else [],
        "source": "UCI",
        "download_urls": {"ZIP": f"https://archive.ics.uci.edu/static/public/{did}/data.csv"} if did else {},
    }


def download_uci(dataset_id, output_dir="."):
    """Download a UCI dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset info for name and download URL
    info = info_uci(dataset_id)
    if not info:
        return

    name = re.sub(r"[^\w\-]", "_", info.get("title", f"uci_{dataset_id}"))

    # UCI provides a ZIP download endpoint
    zip_url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{name}.zip"
    csv_url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv"

    # Try CSV first (simpler)
    for url, ext in [(csv_url, ".csv"), (zip_url, ".zip")]:
        dest = os.path.join(output_dir, f"{name}{ext}")
        print(f"Trying {ext.upper()} download...", file=sys.stderr)
        try:
            size = _download_file(url, dest, timeout=120)
            print(f"Downloaded: {dest} ({size:,} bytes)")
            if ext == ".csv":
                print(f"Read with: pd.read_csv('{dest}')")
            else:
                print(f"Extract with: unzip '{dest}' -d '{output_dir}/{name}/'")
            return
        except (HTTPError, URLError):
            continue

    # Fallback: try the generic download endpoint
    fallback_url = f"https://archive.ics.uci.edu/dataset/{dataset_id}/download"
    dest = os.path.join(output_dir, f"{name}.zip")
    print(f"Trying fallback download...", file=sys.stderr)
    try:
        size = _download_file(fallback_url, dest, timeout=120)
        print(f"Downloaded: {dest} ({size:,} bytes)")
        print(f"Extract with: unzip '{dest}' -d '{output_dir}/{name}/'")
    except (HTTPError, URLError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(f"Visit: {info.get('url', '')} to download manually.")


# ===========================================================================
# Kaggle (public, no auth — scrapes search page)
# ===========================================================================

def search_kaggle(query, limit=10):
    """Search Kaggle datasets via public search page scraping. No API key needed."""
    time.sleep(2)
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={quote(query)}&sortBy=hottest&filetype=all&page=1"
    try:
        data = _get_json(url)
    except (HTTPError, URLError):
        # Kaggle API often requires auth, try scraping instead
        return _search_kaggle_scrape(query, limit)

    if not isinstance(data, list):
        return _search_kaggle_scrape(query, limit)

    results = []
    for d in data[:limit]:
        ref = d.get("ref", "")
        results.append({
            "id": ref,
            "name": d.get("title", d.get("titleNullable", ref)),
            "title": d.get("title", d.get("titleNullable", ref)),
            "description": (d.get("subtitle", "") or "")[:250],
            "url": f"https://www.kaggle.com/datasets/{ref}" if ref else "",
            "downloads": d.get("downloadCount", 0),
            "likes": d.get("voteCount", 0),
            "instances": None,
            "license": d.get("licenseName", ""),
            "tags": [],
            "source": "Kaggle",
        })
    return results


def _search_kaggle_scrape(query, limit=10):
    """Fallback: scrape Kaggle search results from HTML."""
    try:
        url = f"https://www.kaggle.com/search?q={quote(query)}+in%3Adatasets"
        html = _get_text(url, timeout=15)

        results = []
        # Look for dataset links in the HTML
        for match in re.finditer(r'href="(/datasets/[^"]+)"[^>]*>([^<]*)</a>', html):
            path = match.group(1)
            title = match.group(2).strip()
            if not title or "..." in path:
                continue
            did = path.replace("/datasets/", "")
            results.append({
                "id": did,
                "name": title or did.split("/")[-1],
                "title": title or did.split("/")[-1],
                "description": "",
                "url": f"https://www.kaggle.com{path}",
                "downloads": None,
                "tags": [],
                "license": "",
                "source": "Kaggle",
            })
            if len(results) >= limit:
                break
        return results
    except Exception as e:
        print(f"Kaggle scraping failed: {e}", file=sys.stderr)
        return []


# ===========================================================================
# CLI
# ===========================================================================

SEARCH_PROVIDERS = {
    "huggingface": search_huggingface,
    "hf": search_huggingface,
    "paperswithcode": search_paperswithcode,
    "pwc": search_paperswithcode,
    "openml": search_openml,
    "uci": search_uci,
    "kaggle": search_kaggle,
}

INFO_PROVIDERS = {
    "huggingface": info_huggingface,
    "hf": info_huggingface,
    "paperswithcode": info_paperswithcode,
    "pwc": info_paperswithcode,
    "openml": info_openml,
    "uci": info_uci,
}

DOWNLOAD_PROVIDERS = {
    "huggingface": download_huggingface,
    "hf": download_huggingface,
    "openml": download_openml,
    "uci": download_uci,
}

ALL_SOURCES = "huggingface, paperswithcode, openml, uci, kaggle"


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if args else 1)

    command = args[0]
    remaining = args[1:]

    # Parse flags
    query_parts = []
    source = "huggingface"
    limit = 10
    output_dir = "./datasets"
    split = "train"
    i = 0
    while i < len(remaining):
        if remaining[i] == "--source" and i + 1 < len(remaining):
            source = remaining[i + 1].lower().replace("-", "")
            i += 2
        elif remaining[i] == "--limit" and i + 1 < len(remaining):
            limit = int(remaining[i + 1])
            i += 2
        elif remaining[i] in ("--output", "-o") and i + 1 < len(remaining):
            output_dir = remaining[i + 1]
            i += 2
        elif remaining[i] == "--split" and i + 1 < len(remaining):
            split = remaining[i + 1]
            i += 2
        elif not remaining[i].startswith("-"):
            query_parts.append(remaining[i])
            i += 1
        else:
            i += 1

    query = " ".join(query_parts)

    if command == "search":
        if not query:
            print("Error: no search query provided", file=sys.stderr)
            sys.exit(1)
        fn = SEARCH_PROVIDERS.get(source)
        if not fn:
            print(f"Unknown source: {source}. Available: {ALL_SOURCES}", file=sys.stderr)
            sys.exit(1)
        results = fn(query, limit=limit)
        source_names = {
            "huggingface": "HuggingFace", "hf": "HuggingFace",
            "paperswithcode": "Papers with Code", "pwc": "Papers with Code",
            "openml": "OpenML", "uci": "UCI", "kaggle": "Kaggle",
        }
        _print_results(results, source_names.get(source, source), query)

    elif command == "info":
        if not query:
            print("Error: no dataset ID provided", file=sys.stderr)
            sys.exit(1)
        fn = INFO_PROVIDERS.get(source)
        if not fn:
            print(f"Info not supported for: {source}. Available: huggingface, paperswithcode, openml, uci", file=sys.stderr)
            sys.exit(1)
        result = fn(query)
        if result:
            source_names = {
                "huggingface": "HuggingFace", "hf": "HuggingFace",
                "paperswithcode": "Papers with Code", "pwc": "Papers with Code",
                "openml": "OpenML", "uci": "UCI",
            }
            _print_info(result, source_names.get(source, source))
        else:
            print(f"Dataset not found: {query}", file=sys.stderr)
            sys.exit(1)

    elif command == "download":
        if not query:
            print("Error: no dataset ID provided", file=sys.stderr)
            sys.exit(1)
        fn = DOWNLOAD_PROVIDERS.get(source)
        if not fn:
            print(f"Download not supported for: {source}. Available: huggingface, openml, uci", file=sys.stderr)
            print(f"For Kaggle: kaggle datasets download -d {query}")
            print(f"For Papers with Code: visit the dataset page for download links")
            sys.exit(1)
        if source in ("huggingface", "hf"):
            fn(query, output_dir=output_dir, split=split)
        else:
            fn(query, output_dir=output_dir)

    else:
        print(f"Unknown command: {command}. Use: search, info, download", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
