# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "notebooklm>=0.3.0",
# ]
# ///
"""
Generate podcasts, videos, quizzes, reports, and more from documents
using Google NotebookLM. Full pipeline: source → notebook → generate → download.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


AUDIO_FORMATS = {
    "brief": "BRIEF",
    "deep-dive": "DEEP_DIVE",
    "critique": "CRITIQUE",
    "debate": "DEBATE",
}

AUDIO_LENGTHS = {
    "short": "SHORT",
    "default": "DEFAULT",
    "long": "LONG",
}

VIDEO_STYLES = {
    "auto": "AUTO_SELECT",
    "classic": "CLASSIC",
    "whiteboard": "WHITEBOARD",
    "kawaii": "KAWAII",
    "anime": "ANIME",
    "watercolor": "WATERCOLOR",
    "retro": "RETRO",
    "heritage": "HERITAGE",
    "paper-craft": "PAPER_CRAFT",
    "cinematic": "CINEMATIC",
}

VIDEO_FORMATS = {
    "explainer": "EXPLAINER",
    "brief": "BRIEF",
    "cinematic": "CINEMATIC",
}

REPORT_FORMATS = {
    "briefing": "BRIEFING_DOC",
    "study-guide": "STUDY_GUIDE",
    "blog": "BLOG_POST",
    "custom": "CUSTOM",
}

DIFFICULTIES = {
    "easy": "EASY",
    "medium": "MEDIUM",
    "hard": "HARD",
}


async def add_sources(client, notebook_id: str, sources: list[str]) -> list:
    """Add source files/URLs to a notebook. Returns list of Source objects."""
    added = []
    for source in sources:
        path = Path(source)
        if path.exists() and path.is_file():
            print(f"  Adding file: {path.name}", file=sys.stderr)
            src = await client.sources.add_file(
                notebook_id, str(path), title=path.stem, wait=True
            )
        else:
            print(f"  Adding URL: {source}", file=sys.stderr)
            src = await client.sources.add_url(
                notebook_id, source, wait=True
            )
        added.append(src)
    return added


async def generate_podcast(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Podcast: {Path(args.sources[0]).stem}"
        print(f"Creating notebook: {title}", file=sys.stderr)
        nb = await client.notebooks.create(title)

        await add_sources(client, nb.id, args.sources)

        fmt = AUDIO_FORMATS.get(args.format, "BRIEF")
        length = AUDIO_LENGTHS.get(args.length, "DEFAULT")

        print(
            f"Generating audio ({args.format}, {args.length})...",
            file=sys.stderr,
        )
        gen = await client.artifacts.generate_audio(
            nb.id,
            format=fmt,
            length=length,
            language=args.language,
            instructions=args.instructions,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        status = await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or "podcast.mp3"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_audio(nb.id, output)

        return {
            "type": "audio",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "format": args.format,
            "length": args.length,
            "output": str(path),
            "status": "completed",
        }


async def generate_video(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Video: {Path(args.sources[0]).stem}"
        nb = await client.notebooks.create(title)
        await add_sources(client, nb.id, args.sources)

        style = VIDEO_STYLES.get(args.style, "AUTO_SELECT")
        fmt = VIDEO_FORMATS.get(args.format, "EXPLAINER")

        print(f"Generating video ({args.format}, {args.style})...", file=sys.stderr)
        gen = await client.artifacts.generate_video(
            nb.id,
            format=fmt,
            style=style,
            language=args.language,
            instructions=args.instructions,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or "video.mp4"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_video(nb.id, output)

        return {
            "type": "video",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "output": str(path),
            "status": "completed",
        }


async def generate_quiz(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Quiz: {Path(args.sources[0]).stem}"
        nb = await client.notebooks.create(title)
        await add_sources(client, nb.id, args.sources)

        difficulty = DIFFICULTIES.get(args.difficulty, "MEDIUM")

        print(f"Generating quiz ({args.difficulty})...", file=sys.stderr)
        gen = await client.artifacts.generate_quiz(
            nb.id,
            difficulty=difficulty,
            language=args.language,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or "quiz.json"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_quiz(nb.id, output)

        return {
            "type": "quiz",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "output": str(path),
            "status": "completed",
        }


async def generate_flashcards(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Flashcards: {Path(args.sources[0]).stem}"
        nb = await client.notebooks.create(title)
        await add_sources(client, nb.id, args.sources)

        difficulty = DIFFICULTIES.get(args.difficulty, "MEDIUM")

        print(f"Generating flashcards ({args.difficulty})...", file=sys.stderr)
        gen = await client.artifacts.generate_flashcards(
            nb.id,
            difficulty=difficulty,
            language=args.language,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or "flashcards.json"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_flashcards(nb.id, output)

        return {
            "type": "flashcards",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "output": str(path),
            "status": "completed",
        }


async def generate_report(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Report: {Path(args.sources[0]).stem}"
        nb = await client.notebooks.create(title)
        await add_sources(client, nb.id, args.sources)

        fmt = REPORT_FORMATS.get(args.format, "CUSTOM")

        print(f"Generating report ({args.format})...", file=sys.stderr)
        gen = await client.artifacts.generate_report(
            nb.id,
            format=fmt,
            language=args.language,
            instructions=args.instructions,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or "report.md"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_report(nb.id, output)

        return {
            "type": "report",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "output": str(path),
            "status": "completed",
        }


async def generate_slides(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Slides: {Path(args.sources[0]).stem}"
        nb = await client.notebooks.create(title)
        await add_sources(client, nb.id, args.sources)

        print("Generating slide deck...", file=sys.stderr)
        gen = await client.artifacts.generate_slide_deck(
            nb.id,
            language=args.language,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or f"slides.{args.format}"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_slide_deck(
            nb.id, output, format=args.format
        )

        return {
            "type": "slides",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "output": str(path),
            "status": "completed",
        }


async def generate_infographic(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        title = args.title or f"Infographic: {Path(args.sources[0]).stem}"
        nb = await client.notebooks.create(title)
        await add_sources(client, nb.id, args.sources)

        style = VIDEO_STYLES.get(args.style, "AUTO_SELECT")

        print("Generating infographic...", file=sys.stderr)
        gen = await client.artifacts.generate_infographic(
            nb.id,
            style=style,
            language=args.language,
        )

        print("Waiting for generation to complete...", file=sys.stderr)
        await client.artifacts.wait_for_completion(
            nb.id, gen.task_id, timeout=args.timeout
        )

        output = args.output or "infographic.png"
        print(f"Downloading to {output}...", file=sys.stderr)
        path = await client.artifacts.download_infographic(nb.id, output)

        return {
            "type": "infographic",
            "notebook_id": nb.id,
            "artifact_id": gen.artifact_id,
            "output": str(path),
            "status": "completed",
        }


GENERATORS = {
    "podcast": generate_podcast,
    "video": generate_video,
    "quiz": generate_quiz,
    "flashcards": generate_flashcards,
    "report": generate_report,
    "slides": generate_slides,
    "infographic": generate_infographic,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate podcasts, videos, quizzes, and more from documents using Google NotebookLM.",
        epilog=(
            "Examples:\n"
            "  %(prog)s podcast paper.pdf -o podcast.mp3\n"
            "  %(prog)s podcast paper.pdf -o debate.mp3 --format debate --length long\n"
            "  %(prog)s podcast p1.pdf p2.pdf -o combined.mp3 --title 'Survey'\n"
            "  %(prog)s video paper.pdf -o overview.mp4 --style cinematic\n"
            "  %(prog)s quiz paper.pdf -o quiz.json --difficulty hard\n"
            "  %(prog)s flashcards paper.pdf -o cards.json\n"
            "  %(prog)s report paper.pdf -o guide.md --format study-guide\n"
            "  %(prog)s slides paper.pdf -o slides.pdf\n"
            "  %(prog)s infographic paper.pdf -o info.png\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- podcast ---
    p_podcast = sub.add_parser("podcast", help="Generate audio podcast from documents")
    p_podcast.add_argument("sources", nargs="+", help="PDF files or URLs to use as sources")
    p_podcast.add_argument("-o", "--output", help="Output file path (default: podcast.mp3)")
    p_podcast.add_argument("--title", help="Notebook title")
    p_podcast.add_argument(
        "--format",
        default="brief",
        choices=list(AUDIO_FORMATS.keys()),
        help="Audio format (default: brief)",
    )
    p_podcast.add_argument(
        "--length",
        default="default",
        choices=list(AUDIO_LENGTHS.keys()),
        help="Audio length (default: default)",
    )
    p_podcast.add_argument("--language", help="Language code, e.g. en, ja, es (default: en)")
    p_podcast.add_argument("--instructions", help="Custom instructions to guide content focus")
    p_podcast.add_argument("--timeout", type=int, default=600, help="Generation timeout in seconds (default: 600)")

    # --- video ---
    p_video = sub.add_parser("video", help="Generate video overview from documents")
    p_video.add_argument("sources", nargs="+", help="PDF files or URLs")
    p_video.add_argument("-o", "--output", help="Output file path (default: video.mp4)")
    p_video.add_argument("--title", help="Notebook title")
    p_video.add_argument(
        "--format",
        default="explainer",
        choices=list(VIDEO_FORMATS.keys()),
        help="Video format (default: explainer)",
    )
    p_video.add_argument(
        "--style",
        default="auto",
        choices=list(VIDEO_STYLES.keys()),
        help="Visual style (default: auto)",
    )
    p_video.add_argument("--language", help="Language code")
    p_video.add_argument("--instructions", help="Custom instructions")
    p_video.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")

    # --- quiz ---
    p_quiz = sub.add_parser("quiz", help="Generate quiz from documents")
    p_quiz.add_argument("sources", nargs="+", help="PDF files or URLs")
    p_quiz.add_argument("-o", "--output", help="Output file path (default: quiz.json)")
    p_quiz.add_argument("--title", help="Notebook title")
    p_quiz.add_argument(
        "--difficulty",
        default="medium",
        choices=list(DIFFICULTIES.keys()),
        help="Quiz difficulty (default: medium)",
    )
    p_quiz.add_argument("--language", help="Language code")
    p_quiz.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    # --- flashcards ---
    p_flash = sub.add_parser("flashcards", help="Generate flashcards from documents")
    p_flash.add_argument("sources", nargs="+", help="PDF files or URLs")
    p_flash.add_argument("-o", "--output", help="Output file path (default: flashcards.json)")
    p_flash.add_argument("--title", help="Notebook title")
    p_flash.add_argument(
        "--difficulty",
        default="medium",
        choices=list(DIFFICULTIES.keys()),
        help="Flashcard difficulty (default: medium)",
    )
    p_flash.add_argument("--language", help="Language code")
    p_flash.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    # --- report ---
    p_report = sub.add_parser("report", help="Generate report from documents")
    p_report.add_argument("sources", nargs="+", help="PDF files or URLs")
    p_report.add_argument("-o", "--output", help="Output file path (default: report.md)")
    p_report.add_argument("--title", help="Notebook title")
    p_report.add_argument(
        "--format",
        default="custom",
        choices=list(REPORT_FORMATS.keys()),
        help="Report format (default: custom)",
    )
    p_report.add_argument("--language", help="Language code")
    p_report.add_argument("--instructions", help="Custom instructions")
    p_report.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    # --- slides ---
    p_slides = sub.add_parser("slides", help="Generate slide deck from documents")
    p_slides.add_argument("sources", nargs="+", help="PDF files or URLs")
    p_slides.add_argument("-o", "--output", help="Output file path (default: slides.pdf)")
    p_slides.add_argument("--title", help="Notebook title")
    p_slides.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "pptx"],
        help="Output format (default: pdf)",
    )
    p_slides.add_argument("--language", help="Language code")
    p_slides.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    # --- infographic ---
    p_info = sub.add_parser("infographic", help="Generate infographic from documents")
    p_info.add_argument("sources", nargs="+", help="PDF files or URLs")
    p_info.add_argument("-o", "--output", help="Output file path (default: infographic.png)")
    p_info.add_argument("--title", help="Notebook title")
    p_info.add_argument(
        "--style",
        default="auto",
        choices=list(VIDEO_STYLES.keys()),
        help="Visual style (default: auto)",
    )
    p_info.add_argument("--language", help="Language code")
    p_info.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    args = parser.parse_args()

    generator = GENERATORS.get(args.command)
    if not generator:
        parser.error(f"Unknown command: {args.command}")

    try:
        result = asyncio.run(generator(args))
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\nGeneration cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        error_msg = str(e)
        if "auth" in error_msg.lower() or "cookie" in error_msg.lower():
            print(
                f"Error: Authentication failed. Run: python3 auth.py login\n{error_msg}",
                file=sys.stderr,
            )
        else:
            print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
