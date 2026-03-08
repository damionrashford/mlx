#!/usr/bin/env python3
"""Extract text from a PDF file.

Usage:
    extract.py <pdf_path> [--max-pages N]

Uses pdftotext (poppler). macOS: brew install poppler. Linux: apt install poppler-utils.
"""

import os
import subprocess
import sys


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if args else 1)

    pdf_path = None
    max_pages = 20
    i = 0
    while i < len(args):
        if args[i] == "--max-pages" and i + 1 < len(args):
            max_pages = int(args[i + 1])
            i += 2
        elif not args[i].startswith("-"):
            pdf_path = args[i]
            i += 1
        else:
            i += 1

    if not pdf_path or not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(max_pages), pdf_path, "-"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode != 0:
            print("Requires pdftotext: brew install poppler (macOS) or apt install poppler-utils (Linux)", file=sys.stderr)
            sys.exit(1)
        print(result.stdout if result.stdout else "(No text extracted — PDF may be image-based.)")
    except FileNotFoundError:
        print("Requires pdftotext: brew install poppler (macOS) or apt install poppler-utils (Linux)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
