#!/usr/bin/env python3
# prepare_dataset.py — Convert raw CSV/JSONL to alpaca or sharegpt format.
# Validates, deduplicates, and writes to output file.

import sys
import csv
import json
import argparse
import hashlib
from pathlib import Path


def load_input(path: str) -> list[dict]:
    """Load CSV or JSONL file."""
    p = Path(path)
    rows = []
    if p.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif p.suffix.lower() in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        sys.exit(f"Unsupported format: {p.suffix}. Use .csv or .jsonl")
    return rows


def to_alpaca(row: dict, instruction_col: str, input_col: str, output_col: str) -> dict | None:
    """Convert row to alpaca format."""
    instruction = str(row.get(instruction_col, "")).strip()
    output = str(row.get(output_col, "")).strip()
    if not instruction or not output:
        return None
    return {
        "instruction": instruction,
        "input": str(row.get(input_col, "")).strip(),
        "output": output,
        "text": (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{row.get(input_col, '')}\n\n"
            f"### Response:\n{output}"
        ),
    }


def to_sharegpt(row: dict, human_col: str, assistant_col: str) -> dict | None:
    """Convert row to sharegpt format."""
    human = str(row.get(human_col, "")).strip()
    assistant = str(row.get(assistant_col, "")).strip()
    if not human or not assistant:
        return None
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": assistant},
        ]
    }


def deduplicate(rows: list[dict], key_field: str) -> tuple[list[dict], int]:
    """Remove exact duplicates by hashing a key field."""
    seen = set()
    unique = []
    for row in rows:
        h = hashlib.md5(str(row.get(key_field, row)).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(row)
    return unique, len(rows) - len(unique)


def validate(rows: list[dict], fmt: str) -> list[str]:
    """Return list of validation warnings."""
    warnings = []
    if not rows:
        warnings.append("No rows after processing")
        return warnings

    if fmt == "alpaca":
        empty_input = sum(1 for r in rows if not r.get("input", "").strip())
        warnings.append(f"{empty_input}/{len(rows)} rows have empty 'input' field (OK for instruction-only)")

        avg_len = sum(len(r.get("text", "")) for r in rows) / len(rows)
        warnings.append(f"Average text length: {avg_len:.0f} chars")

        too_short = sum(1 for r in rows if len(r.get("output", "")) < 10)
        if too_short > 0:
            warnings.append(f"WARNING: {too_short} rows have output < 10 chars — check quality")

    elif fmt == "sharegpt":
        avg_turns = sum(len(r.get("conversations", [])) for r in rows) / len(rows)
        warnings.append(f"Average conversation turns: {avg_turns:.1f}")

    return warnings


def write_jsonl(rows: list[dict], output_path: str):
    """Write rows as JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument("input", help="Input file (.csv or .jsonl)")
    parser.add_argument("--format", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument("--output", help="Output .jsonl path (default: input_<format>.jsonl)")
    parser.add_argument("--instruction-col", default="instruction")
    parser.add_argument("--input-col", default="input")
    parser.add_argument("--output-col", default="output")
    parser.add_argument("--human-col", default="human")
    parser.add_argument("--assistant-col", default="assistant")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    output_path = args.output or Path(args.input).stem + f"_{args.format}.jsonl"

    print(f"Loading: {args.input}")
    raw_rows = load_input(args.input)
    print(f"Loaded: {len(raw_rows)} rows")

    # Convert
    converted = []
    skipped = 0
    for row in raw_rows:
        if args.format == "alpaca":
            result = to_alpaca(row, args.instruction_col, args.input_col, args.output_col)
        else:
            result = to_sharegpt(row, args.human_col, args.assistant_col)

        if result is None:
            skipped += 1
        else:
            converted.append(result)

    print(f"Converted: {len(converted)} | Skipped (empty fields): {skipped}")

    # Deduplicate
    if not args.no_dedup:
        key_field = "text" if args.format == "alpaca" else "conversations"
        converted, n_dupes = deduplicate(converted, key_field)
        print(f"Deduplicated: removed {n_dupes} exact duplicates")

    # Validate
    warnings = validate(converted, args.format)
    for w in warnings:
        print(f"  {w}")

    # Write
    write_jsonl(converted, output_path)
    print(f"\nWrote {len(converted)} examples to: {output_path}")


if __name__ == "__main__":
    main()
