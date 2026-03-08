---
name: prototype
description: >
  Convert research papers, articles, or technical documents into working code prototypes.
  Supports PDF, Markdown, Jupyter notebooks, and web URLs as input.
  Generates complete projects in Python, JavaScript/TypeScript, Rust, or Go.
  Use when the user wants to implement a paper, create code from an article, prototype an
  algorithm, or convert research to code.
allowed-tools: Bash, Read, Write, WebFetch, Glob, Grep
disable-model-invocation: true
argument-hint: path to PDF, URL, .ipynb, or .md file to convert
---

# Article-to-Code Prototyping

Instructions and pipeline for converting research papers and articles into working code projects.

## Prerequisites

```bash
python3 --version
which pdftotext || echo "WARNING: pdftotext not installed. PDF extraction limited."
```

## Usage

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/main.py <source> -o ./prototype [-l python] [-v]
```

| Argument | Description |
|----------|-------------|
| `<source>` | PDF file, URL, .ipynb, .md, or .txt |
| `-o` | Output directory (default: `./prototype`) |
| `-l` | Language: python, javascript, typescript, rust, go |
| `-v` | Verbose output |

## Output structure

| Language | Entry point | Dependencies |
|----------|------------|--------------|
| Python (default) | `src/main.py` | `requirements.txt` |
| JavaScript | `index.js` | `package.json` |
| TypeScript | `index.ts` | `package.json` |
| Rust | `src/main.rs` | `Cargo.toml` |
| Go | `main.go` | `go.mod` |

Every generated project includes: main implementation, dependency manifest, test file, README, .gitignore.

## Pipeline stages

1. **Extract** — format-specific parser (PDF, web, notebook, markdown)
2. **Analyze** — detect algorithms, architectures, domain, dependencies
3. **Select language** — user hint > code detection > domain default > Python
4. **Generate** — complete project scaffold with algorithm implementations

## Language selection logic

1. Explicit `-l python` flag
2. Code fragments found in the source
3. Domain default: ML→Python, Web→TypeScript, Systems→Rust
4. Library mentions: numpy→Python, react→JavaScript, tokio→Rust
5. Fallback: Python

## Code quality requirements

- No TODOs or placeholders — all code must be complete
- Type hints on all functions (Python, TypeScript)
- Docstrings on all public functions
- Error handling with specific exception types
- At least one test file
- README includes: title, install, usage, source attribution
