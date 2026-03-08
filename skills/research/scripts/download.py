#!/usr/bin/env python3
"""Download a paper PDF by arXiv ID, ACL ID, JMLR ID, or direct URL.

Usage:
    download.py <id_or_url> [--output DIR]

Examples:
    download.py 2401.12345
    download.py 2401.12345v2 --output ./papers
    download.py 2022.acl-long.220
    download.py v22/19-920
    download.py https://aclanthology.org/2022.acl-long.220.pdf
    download.py https://jmlr.org/papers/volume22/19-920/19-920.pdf
"""

import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def resolve_pdf_url(identifier):
    """Resolve an identifier to a PDF URL."""
    # Already a URL
    if identifier.startswith("http://") or identifier.startswith("https://"):
        return identifier, identifier.split("/")[-1].replace(".pdf", "")

    # ACL Anthology ID (e.g. 2022.acl-long.220)
    if re.match(r"^\d{4}\.[a-zA-Z]+-[a-zA-Z]+\.\d+$", identifier):
        return f"https://aclanthology.org/{identifier}.pdf", identifier.replace("/", "_")

    # JMLR ID (e.g. v22/19-920 or 22/19-920)
    jmlr_match = re.match(r"^v?(\d+)/(\d+-\d+)$", identifier)
    if jmlr_match:
        vol, pid = jmlr_match.groups()
        return f"https://jmlr.org/papers/volume{vol}/{pid}/{pid}.pdf", f"jmlr-v{vol}-{pid}"

    # arXiv ID (e.g. 2401.12345 or 2401.12345v2 or hep-ex/0307015)
    clean_id = re.sub(r"v\d+$", "", identifier)
    return f"https://arxiv.org/pdf/{identifier}.pdf", clean_id.replace("/", "_")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if not args else 1)

    identifier = None
    output_dir = "."
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            identifier = args[i]
            i += 1
        else:
            i += 1

    if not identifier:
        print("Error: no paper ID or URL provided", file=sys.stderr)
        sys.exit(1)

    pdf_url, filename = resolve_pdf_url(identifier)
    out_path = os.path.join(output_dir, f"{filename}.pdf")
    os.makedirs(output_dir, exist_ok=True)

    time.sleep(3)  # Rate limit for arXiv/ACL
    try:
        req = Request(pdf_url, headers={"User-Agent": "ml-paper-research-skill/1.0"})
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Downloaded: {out_path}")
        print(f"Path: {os.path.abspath(out_path)}")
        print(f"Size: {len(data)} bytes")
    except (HTTPError, URLError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
