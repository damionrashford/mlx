---
name: media
description: >
  Extract content from YouTube videos and generate podcasts, video overviews, quizzes,
  flashcards, reports, and slide decks from research papers using Google NotebookLM.
  Use when the user wants to extract a YouTube transcript, analyze a video, turn a paper
  into a podcast, generate an audio summary, create a quiz from a paper, make slides
  from research, or automate any NotebookLM workflow.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: YouTube URL, paper PDF path, or notebook ID (e.g. "https://youtube.com/watch?v=..." or "papers/attention.pdf")
---

# Media: Extraction & Generation

This skill combines two capabilities: extracting metadata, transcripts, comments, chapters, and media from YouTube videos, and generating podcasts, videos, quizzes, reports, and more from research papers using Google NotebookLM.

## YouTube Extraction

Instructions and tools for extracting metadata, transcripts, comments, chapters, and media from YouTube videos.

### Prerequisites

```bash
pip install yt-dlp youtube-transcript-api youtube-comment-downloader
```

All three packages are pip-installable. No API keys required.

| Dependency | Used for |
|---|---|
| yt-dlp | Metadata, video/audio download |
| youtube-transcript-api | Captions and transcripts |
| youtube-comment-downloader | Comment scraping |

### Commands

#### Extract everything (metadata + transcript + comments)
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py all "$ARGUMENTS" --max-comments 20 --lang en
```

#### Extract metadata only
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py metadata "$ARGUMENTS"
```

#### Extract transcript only
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py transcript "$ARGUMENTS" --lang en
```

#### Extract comments
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py comments "$ARGUMENTS" --max 20
```

#### Extract for research (compact summary with style hints)
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py research "$ARGUMENTS" --lang en
```

#### Download video
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py download-video "$ARGUMENTS" --quality 720p --output ./cache/downloads
```

#### Download audio (WAV)
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py download-audio "$ARGUMENTS" --output ./cache/downloads
```

#### Extract chapters as scenes
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract.py chapters "$ARGUMENTS" --lang en
```

### Output

All commands output JSON to stdout by default. Use `--output <file.json>` to save to a file. Add `--no-pretty` to disable pretty-printing.

### Guidelines

- Start with `metadata` to check the video exists and get basic info.
- Use `transcript` for lecture analysis, summarization, or content extraction.
- Use `research` for a compact, LLM-friendly summary with style hints and top comments.
- Use `chapters` to get chapter-aligned transcript segments.
- Only use `download-video` or `download-audio` when the user explicitly needs local media files.
- Always cite: video title, channel, and URL in any summary.

## Content Generation (NotebookLM)

Generate podcasts, videos, quizzes, reports, and more from research papers using Google NotebookLM. This section covers both quick paper-to-podcast workflows and full NotebookLM CLI automation.

### Prerequisites

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

#### Agent Setup Verification

Before starting workflows, verify the CLI is ready:

1. `notebooklm status` -- Should show "Authenticated as: email@..."
2. `notebooklm list --json` -- Should return valid JSON (even if empty notebooks list)
3. If either fails -- Run `notebooklm login`

#### CI/CD, Multiple Accounts, and Parallel Agents

For automated environments, multiple accounts, or parallel agent workflows:

| Variable | Purpose |
|----------|---------|
| `NOTEBOOKLM_HOME` | Custom config directory (default: `~/.notebooklm`) |
| `NOTEBOOKLM_AUTH_JSON` | Inline auth JSON - no file writes needed |

**CI/CD setup:** Set `NOTEBOOKLM_AUTH_JSON` from a secret containing your `storage_state.json` contents.

**Multiple accounts:** Use different `NOTEBOOKLM_HOME` directories per account.

**Parallel agents:** The CLI stores notebook context in a shared file (`~/.notebooklm/context.json`). Multiple concurrent agents using `notebooklm use` can overwrite each other's context.

**Solutions for parallel workflows:**
1. **Always use explicit notebook ID** (recommended): Pass `-n <notebook_id>` (for `wait`/`download` commands) or `--notebook <notebook_id>` (for others) instead of relying on `use`
2. **Per-agent isolation:** Set unique `NOTEBOOKLM_HOME` per agent: `export NOTEBOOKLM_HOME=/tmp/agent-$ID`
3. **Use full UUIDs:** Avoid partial IDs in automation (they can become ambiguous)

### Available scripts

| Script | Usage |
|--------|-------|
| [auth.py](scripts/auth.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/auth.py check` |
| [generate.py](scripts/generate.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast paper.pdf -o podcast.mp3` |
| [manage.py](scripts/manage.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/manage.py list` |

### Quick Reference

| Task | Command |
|------|---------|
| Authenticate | `notebooklm login` |
| Diagnose auth issues | `notebooklm auth check` |
| Diagnose auth (full) | `notebooklm auth check --test` |
| List notebooks | `notebooklm list` |
| Create notebook | `notebooklm create "Title"` |
| Set context | `notebooklm use <notebook_id>` |
| Show context | `notebooklm status` |
| Add URL source | `notebooklm source add "https://..."` |
| Add file | `notebooklm source add ./file.pdf` |
| Add YouTube | `notebooklm source add "https://youtube.com/..."` |
| List sources | `notebooklm source list` |
| Delete source by ID | `notebooklm source delete <source_id>` |
| Delete source by exact title | `notebooklm source delete-by-title "Exact Title"` |
| Wait for source processing | `notebooklm source wait <source_id>` |
| Web research (fast) | `notebooklm source add-research "query"` |
| Web research (deep) | `notebooklm source add-research "query" --mode deep --no-wait` |
| Check research status | `notebooklm research status` |
| Wait for research | `notebooklm research wait --import-all` |
| Chat | `notebooklm ask "question"` |
| Chat (specific sources) | `notebooklm ask "question" -s src_id1 -s src_id2` |
| Chat (with references) | `notebooklm ask "question" --json` |
| Chat (save answer as note) | `notebooklm ask "question" --save-as-note` |
| Chat (save with title) | `notebooklm ask "question" --save-as-note --note-title "Title"` |
| Show conversation history | `notebooklm history` |
| Save all history as note | `notebooklm history --save` |
| Continue specific conversation | `notebooklm ask "question" -c <conversation_id>` |
| Save history with title | `notebooklm history --save --note-title "My Research"` |
| Get source fulltext | `notebooklm source fulltext <source_id>` |
| Get source guide | `notebooklm source guide <source_id>` |
| Generate podcast | `notebooklm generate audio "instructions"` |
| Generate podcast (JSON) | `notebooklm generate audio --json` |
| Generate podcast (specific sources) | `notebooklm generate audio -s src_id1 -s src_id2` |
| Generate video | `notebooklm generate video "instructions"` |
| Generate report | `notebooklm generate report --format briefing-doc` |
| Generate report (append instructions) | `notebooklm generate report --format study-guide --append "Target audience: beginners"` |
| Generate quiz | `notebooklm generate quiz` |
| Revise a slide | `notebooklm generate revise-slide "prompt" --artifact <id> --slide 0` |
| Check artifact status | `notebooklm artifact list` |
| Wait for completion | `notebooklm artifact wait <artifact_id>` |
| Download audio | `notebooklm download audio ./output.mp3` |
| Download video | `notebooklm download video ./output.mp4` |
| Download slide deck (PDF) | `notebooklm download slide-deck ./slides.pdf` |
| Download slide deck (PPTX) | `notebooklm download slide-deck ./slides.pptx --format pptx` |
| Download report | `notebooklm download report ./report.md` |
| Download mind map | `notebooklm download mind-map ./map.json` |
| Download data table | `notebooklm download data-table ./data.csv` |
| Download quiz | `notebooklm download quiz quiz.json` |
| Download quiz (markdown) | `notebooklm download quiz --format markdown quiz.md` |
| Download flashcards | `notebooklm download flashcards cards.json` |
| Download flashcards (markdown) | `notebooklm download flashcards --format markdown cards.md` |
| Delete notebook | `notebooklm notebook delete <id>` |
| List languages | `notebooklm language list` |
| Get language | `notebooklm language get` |
| Set language | `notebooklm language set zh_Hans` |

**Parallel safety:** Use explicit notebook IDs in parallel workflows. Commands supporting `-n` shorthand: `artifact wait`, `source wait`, `research wait/status`, `download *`. Download commands also support `-a/--artifact`. Other commands use `--notebook`. For chat, use `-c <conversation_id>` to target a specific conversation.

**Partial IDs:** Use first 6+ characters of UUIDs. Must be unique prefix (fails if ambiguous). Works for ID-based commands such as `use`, `source delete`, and `wait`. For exact source-title deletion, use `source delete-by-title "Title"`. For automation, prefer full UUIDs to avoid ambiguity.

### Workflow

#### 1. Generate a podcast from a paper

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

#### 2. Generate other content types

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

#### 3. Manage notebooks and artifacts

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

#### 4. Multi-source notebooks (combine papers)

```bash
# Create a notebook, add multiple papers, then generate
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast \
  paper1.pdf paper2.pdf paper3.pdf \
  -o combined_podcast.mp3 \
  --title "Survey of Attention Mechanisms" \
  --format deep-dive \
  --instructions "Compare and contrast the approaches across all papers"
```

#### 5. Research to podcast (automated with subagent)

When the user wants full automation (generate and download when ready):

1. Create notebook and add sources as usual
2. Wait for sources to be ready (use `source wait` or check `source list --json`)
3. Run `notebooklm generate audio "..." --json` -- parse `artifact_id` from output
4. **Spawn a background agent** using Task tool:
   ```
   Task(
     prompt="Wait for artifact {artifact_id} in notebook {notebook_id} to complete, then download.
             Use: notebooklm artifact wait {artifact_id} -n {notebook_id} --timeout 600
             Then: notebooklm download audio ./podcast.mp3 -a {artifact_id} -n {notebook_id}",
     subagent_type="general-purpose"
   )
   ```
5. Main conversation continues while agent waits

**Error handling in subagent:**
- If `artifact wait` returns exit code 2 (timeout): Report timeout, suggest checking `artifact list`
- If download fails: Check if artifact status is COMPLETED first

#### 6. Bulk import with source waiting (subagent pattern)

When adding multiple sources and needing to wait for processing before chat/generation:

1. Add sources with `--json` to capture IDs:
   ```bash
   notebooklm source add "https://url1.com" --json  # -> {"source_id": "abc..."}
   notebooklm source add "https://url2.com" --json  # -> {"source_id": "def..."}
   ```
2. **Spawn a background agent** to wait for all sources:
   ```
   Task(
     prompt="Wait for sources {source_ids} in notebook {notebook_id} to be ready.
             For each: notebooklm source wait {id} -n {notebook_id} --timeout 120
             Report when all ready or if any fail.",
     subagent_type="general-purpose"
   )
   ```
3. Main conversation continues while agent waits
4. Once sources are ready, proceed with chat or generation

**Why wait for sources?** Sources must be indexed before chat or generation. Takes 10-60 seconds per source.

#### 7. Deep web research (subagent pattern)

Deep research finds and analyzes web sources on a topic:

1. Create notebook: `notebooklm create "Research: [topic]"`
2. Start deep research (non-blocking):
   ```bash
   notebooklm source add-research "topic query" --mode deep --no-wait
   ```
3. **Spawn a background agent** to wait and import:
   ```
   Task(
     prompt="Wait for research in notebook {notebook_id} to complete and import sources.
             Use: notebooklm research wait -n {notebook_id} --import-all --timeout 300
             Report how many sources were imported.",
     subagent_type="general-purpose"
   )
   ```
4. Main conversation continues while agent waits
5. When agent completes, sources are imported automatically

**Alternative (blocking):** For simple cases, omit `--no-wait`:
```bash
notebooklm source add-research "topic" --mode deep --import-all
# Blocks for up to 5 minutes
```

**When to use each mode:**
- `--mode fast`: Specific topic, quick overview needed (5-10 sources, seconds)
- `--mode deep`: Broad topic, comprehensive analysis needed (20+ sources, 2-5 min)

**Research sources:**
- `--from web`: Search the web (default)
- `--from drive`: Search Google Drive

### Generation Types

All generate commands support:
- `-s, --source` to use specific source(s) instead of all sources
- `--language` to set output language (defaults to configured language or 'en')
- `--json` for machine-readable output (returns `task_id` and `status`)
- `--retry N` to automatically retry on rate limits with exponential backoff

| Type | Command | Options | Download |
|------|---------|---------|----------|
| Podcast | `generate audio` | `--format [deep-dive\|brief\|critique\|debate]`, `--length [short\|default\|long]` | .mp3 |
| Video | `generate video` | `--format [explainer\|brief]`, `--style [auto\|classic\|whiteboard\|kawaii\|anime\|watercolor\|retro-print\|heritage\|paper-craft]` | .mp4 |
| Slide Deck | `generate slide-deck` | `--format [detailed\|presenter]`, `--length [default\|short]` | .pdf / .pptx |
| Slide Revision | `generate revise-slide "prompt" --artifact <id> --slide N` | `--wait`, `--notebook` | *(re-downloads parent deck)* |
| Infographic | `generate infographic` | `--orientation [landscape\|portrait\|square]`, `--detail [concise\|standard\|detailed]`, `--style [auto\|sketch-note\|professional\|bento-grid\|editorial\|instructional\|bricks\|clay\|anime\|kawaii\|scientific]` | .png |
| Report | `generate report` | `--format [briefing-doc\|study-guide\|blog-post\|custom]`, `--append "extra instructions"` | .md |
| Mind Map | `generate mind-map` | *(sync, instant)* | .json |
| Data Table | `generate data-table` | description required | .csv |
| Quiz | `generate quiz` | `--difficulty [easy\|medium\|hard]`, `--quantity [fewer\|standard\|more]` | .json/.md/.html |
| Flashcards | `generate flashcards` | `--difficulty [easy\|medium\|hard]`, `--quantity [fewer\|standard\|more]` | .json/.md/.html |

### Audio format reference

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

### Video style reference

| Style | Description |
|-------|-------------|
| `auto` | AI selects best style (default) |
| `classic` | Clean, professional |
| `whiteboard` | Hand-drawn whiteboard aesthetic |
| `cinematic` | AI-generated documentary footage |
| `anime` | Anime-inspired visuals |
| `watercolor` | Watercolor painting style |
| `retro` | Vintage aesthetic |
| `kawaii` | Cute, Japanese-inspired style |
| `heritage` | Traditional, historical aesthetic |
| `paper-craft` | Paper cutout style |
| `retro-print` | Retro print aesthetic |

### Features Beyond the Web UI

These capabilities are available via CLI but not in NotebookLM's web interface:

| Feature | Command | Description |
|---------|---------|-------------|
| **Batch downloads** | `download <type> --all` | Download all artifacts of a type at once |
| **Quiz/Flashcard export** | `download quiz --format json` | Export as JSON, Markdown, or HTML (web UI only shows interactive view) |
| **Mind map extraction** | `download mind-map` | Export hierarchical JSON for visualization tools |
| **Data table export** | `download data-table` | Download structured tables as CSV |
| **Slide deck as PPTX** | `download slide-deck --format pptx` | Download slide deck as editable .pptx (web UI only offers PDF) |
| **Slide revision** | `generate revise-slide "prompt" --artifact <id> --slide N` | Modify individual slides with a natural-language prompt |
| **Report template append** | `generate report --format study-guide --append "..."` | Append custom instructions to built-in format templates without losing the format type |
| **Source fulltext** | `source fulltext <id>` | Retrieve the indexed text content of any source |
| **Save chat to note** | `ask "..." --save-as-note` / `history --save` | Save Q&A answers or conversation history as notebook notes |
| **Programmatic sharing** | `share` commands | Manage sharing permissions without the UI |

### Command Output Formats

Commands with `--json` return structured data for parsing:

**Create notebook:**
```
$ notebooklm create "Research" --json
{"id": "abc123de-...", "title": "Research"}
```

**Add source:**
```
$ notebooklm source add "https://example.com" --json
{"source_id": "def456...", "title": "Example", "status": "processing"}
```

**Generate artifact:**
```
$ notebooklm generate audio "Focus on key points" --json
{"task_id": "xyz789...", "status": "pending"}
```

**Chat with references:**
```
$ notebooklm ask "What is X?" --json
{"answer": "X is... [1] [2]", "conversation_id": "...", "turn_number": 1, "is_follow_up": false, "references": [{"source_id": "abc123...", "citation_number": 1, "cited_text": "Relevant passage from source..."}, {"source_id": "def456...", "citation_number": 2, "cited_text": "Another passage..."}]}
```

**Source fulltext (get indexed content):**
```
$ notebooklm source fulltext <source_id> --json
{"source_id": "...", "title": "...", "char_count": 12345, "content": "Full indexed text..."}
```

**Extract IDs:** Parse the `id`, `source_id`, or `task_id` field from JSON output.

**Status values:**
- Sources: `processing` -> `ready` (or `error`)
- Artifacts: `pending` or `in_progress` -> `completed` (or `unknown`)

### Language Configuration

Language setting controls the output language for generated artifacts (audio, video, etc.).

**Important:** Language is a **GLOBAL** setting that affects all notebooks in your account.

```bash
# List all 80+ supported languages with native names
notebooklm language list

# Show current language setting
notebooklm language get

# Set language for artifact generation
notebooklm language set zh_Hans  # Simplified Chinese
notebooklm language set ja       # Japanese
notebooklm language set en       # English (default)
```

**Common language codes:**

| Code | Language |
|------|----------|
| `en` | English |
| `zh_Hans` | Simplified Chinese |
| `zh_Hant` | Traditional Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `pt_BR` | Portuguese (Brasil) |

**Override per command:** Use `--language` flag on generate commands:
```bash
notebooklm generate audio --language ja   # Japanese podcast
notebooklm generate video --language zh_Hans  # Chinese video
```

**Offline mode:** Use `--local` flag to skip server sync:
```bash
notebooklm language set zh_Hans --local  # Save locally only
notebooklm language get --local  # Read local config only
```

### Integration with research skill

The research skill downloads papers to `./papers/`. Feed them directly:

```bash
# Step 1: Download paper (research skill)
python3 ${CLAUDE_SKILL_DIR}/../research/scripts/download.py 2401.12345 -o ./papers/

# Step 2: Generate podcast
python3 ${CLAUDE_SKILL_DIR}/scripts/generate.py podcast ./papers/2401.12345.pdf -o podcast.mp3
```

### Processing Times

| Operation | Typical time | Suggested timeout |
|-----------|--------------|-------------------|
| Source processing | 30s - 10 min | 600s |
| Research (fast) | 30s - 2 min | 180s |
| Research (deep) | 15 - 30+ min | 1800s |
| Notes | instant | n/a |
| Mind-map | instant (sync) | n/a |
| Quiz, flashcards | 5 - 15 min | 900s |
| Report, data-table | 5 - 15 min | 900s |
| Audio generation | 10 - 20 min | 1200s |
| Video generation | 15 - 45 min | 2700s |

**Polling intervals:** When checking status manually, poll every 15-30 seconds to avoid excessive API calls.

### Error Handling

| Error | Cause | Action |
|-------|-------|--------|
| Auth/cookie error | Session expired | Run `notebooklm auth check` then `notebooklm login` |
| "No notebook context" | Context not set | Use `-n <id>` or `--notebook <id>` flag (parallel), or `notebooklm use <id>` (single-agent) |
| "No result found for RPC ID" | Rate limiting | Wait 5-10 min, retry |
| `GENERATION_FAILED` | Google rate limit | Wait and retry later |
| Download fails | Generation incomplete | Check `artifact list` for status |
| Invalid notebook/source ID | Wrong ID | Run `notebooklm list` to verify |
| RPC protocol error | Google changed APIs | May need CLI update |

**On failure, offer the user a choice:**
1. Retry the operation
2. Skip and continue with something else
3. Investigate the error

### Exit Codes

All commands use consistent exit codes:

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Continue |
| 1 | Error (not found, processing failed) | Check stderr, see Error Handling |
| 2 | Timeout (wait commands only) | Extend timeout or check status manually |

**Examples:**
- `source wait` returns 1 if source not found or processing failed
- `artifact wait` returns 2 if timeout reached before completion
- `generate` returns 1 if rate limited (check stderr for details)

### Rules

#### Autonomy rules

**Run automatically (no confirmation):**
- `notebooklm status` - check context
- `notebooklm auth check` - diagnose auth issues
- `notebooklm list` - list notebooks
- `notebooklm source list` - list sources
- `notebooklm artifact list` - list artifacts
- `notebooklm language list` - list supported languages
- `notebooklm language get` - get current language
- `notebooklm language set` - set language (global setting)
- `notebooklm artifact wait` - wait for artifact completion (in subagent context)
- `notebooklm source wait` - wait for source processing (in subagent context)
- `notebooklm research status` - check research status
- `notebooklm research wait` - wait for research (in subagent context)
- `notebooklm use <id>` - set context (single-agent only -- use `-n` flag in parallel workflows)
- `notebooklm create` - create notebook
- `notebooklm ask "..."` - chat queries (without `--save-as-note`)
- `notebooklm history` - display conversation history (read-only)
- `notebooklm source add` - add sources

**Ask before running:**
- `notebooklm delete` - destructive
- `notebooklm generate *` - long-running, may fail
- `notebooklm download *` - writes to filesystem
- `notebooklm artifact wait` - long-running (when in main conversation)
- `notebooklm source wait` - long-running (when in main conversation)
- `notebooklm research wait` - long-running (when in main conversation)
- `notebooklm ask "..." --save-as-note` - writes a note
- `notebooklm history --save` - writes a note

#### General rules

- Always check auth status before first generation attempt
- Generation takes 1-5 minutes -- use `--wait` (default) to poll until complete
- NotebookLM has rate limits -- add delays between bulk generations
- Session cookies expire -- if auth fails, re-run `auth.py login`
- Downloaded audio is MP3, video is MP4
- Multi-source notebooks produce richer podcasts -- combine related papers when possible
- The `--instructions` flag is powerful -- use it to focus on specific sections or perspectives

### Troubleshooting

```bash
notebooklm --help              # Main commands
notebooklm auth check          # Diagnose auth issues
notebooklm auth check --test   # Full auth validation with network test
notebooklm notebook --help     # Notebook management
notebooklm source --help       # Source management
notebooklm research --help     # Research status/wait
notebooklm generate --help     # Content generation
notebooklm artifact --help     # Artifact management
notebooklm download --help     # Download content
notebooklm language --help     # Language settings
```

**Diagnose auth:** `notebooklm auth check` - shows cookie domains, storage path, validation status
**Re-authenticate:** `notebooklm login`
**Check version:** `notebooklm --version`
**Update skill:** `notebooklm skill install`
