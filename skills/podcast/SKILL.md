---
name: podcast
description: >
  Generate audio podcasts, video overviews, quizzes, flashcards, reports, infographics,
  and slide decks from research papers and documents using Google NotebookLM. Supports
  multiple formats (deep dive, debate, critique, brief), 80+ languages, and custom
  instructions. Use when the user wants to turn a paper into a podcast, generate an
  audio summary, create a quiz from a paper, make slides from research, produce a
  video overview, or convert any document into listenable content.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: path to paper PDF, URL, or notebook ID (e.g. "papers/attention.pdf" or "https://arxiv.org/abs/2401.12345")
---

# Paper-to-Podcast & Content Generation

Generate podcasts, videos, quizzes, reports, and more from research papers using Google NotebookLM.

## Prerequisites

NotebookLM requires a one-time browser login to Google. Check auth status before generating:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/auth.py check
```

If not authenticated, guide the user to run:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/auth.py login
```

This opens a browser for Google SSO. Credentials are cached at `~/.notebooklm/storage_state.json`.

**Required package**: `pip install notebooklm` (or `uv pip install notebooklm`)

## Available scripts

| Script | Usage |
|--------|-------|
| [auth.py](scripts/auth.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/auth.py check` |
| [generate.py](scripts/generate.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast paper.pdf -o podcast.mp3` |
| [manage.py](scripts/manage.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/manage.py list` |

## Workflow

### 1. Generate a podcast from a paper

```bash
# From a local PDF
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast ./papers/attention.pdf -o podcast.mp3

# From a URL (arXiv, web page, etc.)
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast https://arxiv.org/abs/2401.12345 -o podcast.mp3

# Debate format (two speakers arguing perspectives)
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast paper.pdf -o debate.mp3 --format debate

# Deep dive (20-30 min detailed analysis)
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast paper.pdf -o deep.mp3 --format deep-dive --length long

# With custom focus instructions
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast paper.pdf -o podcast.mp3 \
  --instructions "Focus on the methodology and experimental results, skip the related work"

# In another language
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast paper.pdf -o podcast.mp3 --language ja
```

### 2. Generate other content types

```bash
# Video overview (MP4 with AI visuals)
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py video paper.pdf -o overview.mp4 --style cinematic

# Quiz from paper content
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py quiz paper.pdf -o quiz.json --difficulty hard

# Flashcards for study
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py flashcards paper.pdf -o cards.json

# Report (study guide, briefing doc, or blog post)
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py report paper.pdf -o guide.md --format study-guide

# Slide deck
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py slides paper.pdf -o slides.pdf

# Infographic
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py infographic paper.pdf -o infographic.png
```

### 3. Manage notebooks and artifacts

```bash
# List all notebooks
python3 ${CLAUDE_SKILL_DIR}/scripts/manage.py list

# List artifacts in a notebook
python3 ${CLAUDE_SKILL_DIR}/scripts/manage.py artifacts <notebook_id>

# Download an existing artifact
python3 ${CLAUDE_SKILL_DIR}/scripts/manage.py download <notebook_id> --artifact <artifact_id> -o output.mp3

# Delete a notebook
python3 ${CLAUDE_SKILL_DIR}/scripts/manage.py delete <notebook_id>
```

### 4. Multi-source notebooks (combine papers)

```bash
# Create a notebook, add multiple papers, then generate
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast \
  paper1.pdf paper2.pdf paper3.pdf \
  -o combined_podcast.mp3 \
  --title "Survey of Attention Mechanisms" \
  --format deep-dive \
  --instructions "Compare and contrast the approaches across all papers"
```

## Audio format reference

| Format | Description | Best for |
|--------|-------------|----------|
| `brief` | Quick 5-10 min overview (default) | Getting the gist of a paper |
| `deep-dive` | Detailed 20-30 min analysis | Thorough understanding |
| `critique` | Critical examination of methodology | Paper review prep |
| `debate` | Two-speaker debate format | Exploring multiple perspectives |

| Length | Duration |
|--------|----------|
| `short` | 5-10 minutes |
| `default` | 10-20 minutes |
| `long` | 20-30+ minutes |

## Video style reference

| Style | Description |
|-------|-------------|
| `auto` | AI selects best style (default) |
| `classic` | Clean, professional |
| `whiteboard` | Hand-drawn whiteboard aesthetic |
| `cinematic` | AI-generated documentary footage |
| `anime` | Anime-inspired visuals |
| `watercolor` | Watercolor painting style |
| `retro` | Vintage aesthetic |

## Integration with research skill

The research skill downloads papers to `./papers/`. Feed them directly:

```bash
# Step 1: Download paper (research skill)
python3 ${CLAUDE_SKILL_DIR}/../research/scripts/download.py 2401.12345 -o ./papers/

# Step 2: Generate podcast
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast ./papers/2401.12345.pdf -o podcast.mp3
```

## Rules

- Always check auth status before first generation attempt
- Generation takes 1-5 minutes — use `--wait` (default) to poll until complete
- NotebookLM has rate limits — add delays between bulk generations
- Session cookies expire — if auth fails, re-run `auth.py login`
- Downloaded audio is MP3, video is MP4
- Multi-source notebooks produce richer podcasts — combine related papers when possible
- The `--instructions` flag is powerful — use it to focus on specific sections or perspectives
