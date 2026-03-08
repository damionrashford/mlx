# /// script
# requires-python = ">=3.10"
# ///
"""
Jupyter notebook assessment CLI — analyze notebook structure, identify issues,
and report metrics. No external dependencies (uses stdlib json only).
"""

import argparse
import json
import re
import sys


def assess_notebook(path: str) -> dict:
    with open(path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    empty = [c for c in code_cells if not "".join(c["source"]).strip()]
    has_outputs = sum(1 for c in code_cells if c.get("outputs"))

    # Detect imports
    imports = set()
    for cell in code_cells:
        for line in cell["source"]:
            m = re.match(r"^(?:from|import)\s+(\w+)", line)
            if m:
                imports.add(m.group(1))

    # Check for section headers in markdown
    headers = []
    for cell in md_cells:
        for line in cell["source"]:
            if line.startswith("#"):
                headers.append(line.strip())

    # Detect issues
    issues = []
    if empty:
        issues.append(f"{len(empty)} empty code cells")
    if has_outputs > 0:
        issues.append(f"{has_outputs} cells with stale outputs (clear before commit)")

    # Check if imports are consolidated
    import_cells = []
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        if re.search(r"^(?:from|import)\s+", src, re.MULTILINE):
            import_cells.append(i)
    if len(import_cells) > 1 and import_cells[0] != 0:
        issues.append("Imports scattered across multiple cells (consolidate to first)")

    if not headers:
        issues.append("No markdown section headers (add structure)")
    if not md_cells:
        issues.append("No markdown cells (add documentation)")
    if len(code_cells) > 0 and len(md_cells) / max(len(code_cells), 1) < 0.2:
        issues.append("Low documentation ratio (add more markdown explanations)")

    # Check for hardcoded paths
    for cell in code_cells:
        src = "".join(cell["source"])
        if re.search(r'["\'][/~][^"\']+\.(csv|json|parquet|xlsx)', src):
            issues.append("Hardcoded file paths detected (use variables or relative paths)")
            break

    # Check for missing seeds
    has_ml = any(
        lib in imports
        for lib in ["sklearn", "torch", "tensorflow", "xgboost", "lightgbm"]
    )
    if has_ml:
        all_src = "\n".join("".join(c["source"]) for c in code_cells)
        if "random_state" not in all_src and "seed" not in all_src:
            issues.append("ML imports found but no random seeds set")

    result = {
        "file": path,
        "total_cells": len(cells),
        "code_cells": len(code_cells),
        "markdown_cells": len(md_cells),
        "empty_cells": len(empty),
        "cells_with_outputs": has_outputs,
        "imports": sorted(imports),
        "headers": headers,
        "issues": issues,
        "score": max(0, 10 - len(issues)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Assess Jupyter notebook quality: structure, documentation, and common issues.",
        epilog=(
            "Examples:\n"
            "  %(prog)s notebook.ipynb\n"
            "  %(prog)s analysis.ipynb --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to .ipynb file")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of text"
    )

    args = parser.parse_args()

    try:
        result = assess_notebook(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        print(f"Usage: python {sys.argv[0]} <path-to-ipynb>", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid notebook format: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"=== Notebook Assessment: {result['file']} ===")
        print(
            f"Cells: {result['total_cells']} total "
            f"({result['code_cells']} code, {result['markdown_cells']} markdown, "
            f"{result['empty_cells']} empty)"
        )
        print(f"Outputs: {result['cells_with_outputs']} cells with outputs")
        print(f"Imports: {', '.join(result['imports']) if result['imports'] else 'none'}")
        if result["headers"]:
            print(f"Sections: {len(result['headers'])}")
        if result["issues"]:
            print(f"\nIssues ({len(result['issues'])}):")
            for issue in result["issues"]:
                print(f"  - {issue}")
        else:
            print("\nNo issues found.")
        print(f"\nScore: {result['score']}/10")


if __name__ == "__main__":
    main()
