# Media Skill: Supported Formats Quick Reference

## YouTube Extraction Modes

| Mode | Command | Description |
|------|---------|-------------|
| all | `extract.py all` | Metadata + transcript + comments |
| metadata | `extract.py metadata` | Title, channel, duration, views, description |
| transcript | `extract.py transcript` | Full captions/transcript text |
| comments | `extract.py comments` | Top comments (configurable max) |
| research | `extract.py research` | Compact LLM-friendly summary with style hints |
| chapters | `extract.py chapters` | Chapter-aligned transcript segments |
| download-video | `extract.py download-video` | Download video file (configurable quality) |
| download-audio | `extract.py download-audio` | Download audio as WAV |

## NotebookLM Generation Types

| Type | Command | Formats/Options | Output |
|------|---------|-----------------|--------|
| Podcast | `generate audio` | brief, deep-dive, critique, debate | .mp3 |
| Video | `generate video` | explainer, brief; 11 visual styles | .mp4 |
| Slide Deck | `generate slide-deck` | detailed, presenter | .pdf / .pptx |
| Slide Revision | `generate revise-slide` | per-slide prompt editing | *(parent deck)* |
| Infographic | `generate infographic` | 3 orientations, 3 detail levels, 11 styles | .png |
| Report | `generate report` | briefing-doc, study-guide, blog-post, custom | .md |
| Mind Map | `generate mind-map` | sync/instant | .json |
| Data Table | `generate data-table` | description required | .csv |
| Quiz | `generate quiz` | easy/medium/hard, fewer/standard/more | .json / .md / .html |
| Flashcards | `generate flashcards` | easy/medium/hard, fewer/standard/more | .json / .md / .html |

## Audio Formats

| Format | Duration | Best for |
|--------|----------|----------|
| brief | 5-10 min | Quick overview |
| deep-dive | 20-30 min | Thorough analysis |
| critique | 10-20 min | Methodology review |
| debate | 10-20 min | Multiple perspectives |

## Video Styles

auto, classic, whiteboard, cinematic, anime, watercolor, retro, kawaii, heritage, paper-craft, retro-print

## Infographic Styles

auto, sketch-note, professional, bento-grid, editorial, instructional, bricks, clay, anime, kawaii, scientific

## Report Formats

briefing-doc, study-guide, blog-post, custom
