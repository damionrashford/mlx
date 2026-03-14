# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "notebooklm>=0.3.0",
# ]
# ///
"""
Manage NotebookLM notebooks and artifacts — list, inspect, download, delete.
"""

import argparse
import asyncio
import json
import sys


async def list_notebooks(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        notebooks = await client.notebooks.list()
        items = []
        for nb in notebooks:
            item = {
                "id": nb.id,
                "title": nb.title,
                "sources_count": nb.sources_count,
            }
            if nb.created_at:
                item["created_at"] = nb.created_at.isoformat()
            items.append(item)
        return {"notebooks": items, "count": len(items)}


async def list_artifacts(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        artifacts = await client.artifacts.list(args.notebook_id)
        items = []
        for art in artifacts:
            item = {
                "id": art.id,
                "title": art.title,
                "type": str(art.kind) if art.kind else "unknown",
                "status": str(art.status) if art.status else "unknown",
                "is_complete": art.is_complete,
            }
            if art.url:
                item["url"] = art.url
            if art.created_at:
                item["created_at"] = art.created_at.isoformat()
            items.append(item)
        return {"notebook_id": args.notebook_id, "artifacts": items, "count": len(items)}


async def download_artifact(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        output = args.output or f"artifact_{args.artifact_id}"

        artifact = await client.artifacts.get(args.notebook_id, args.artifact_id)
        if not artifact:
            return {"error": f"Artifact {args.artifact_id} not found"}

        kind = str(artifact.kind).lower() if artifact.kind else ""

        if "audio" in kind:
            path = await client.artifacts.download_audio(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "video" in kind:
            path = await client.artifacts.download_video(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "quiz" in kind:
            path = await client.artifacts.download_quiz(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "flashcard" in kind:
            path = await client.artifacts.download_flashcards(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "report" in kind:
            path = await client.artifacts.download_report(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "slide" in kind:
            path = await client.artifacts.download_slide_deck(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "infographic" in kind:
            path = await client.artifacts.download_infographic(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        elif "mind_map" in kind:
            path = await client.artifacts.download_mind_map(
                args.notebook_id, output, artifact_id=args.artifact_id
            )
        else:
            return {"error": f"Unknown artifact type: {kind}"}

        return {
            "artifact_id": args.artifact_id,
            "type": kind,
            "output": str(path),
            "status": "downloaded",
        }


async def delete_notebook(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        success = await client.notebooks.delete(args.notebook_id)
        return {
            "notebook_id": args.notebook_id,
            "deleted": success,
        }


async def delete_artifact(args) -> dict:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        success = await client.artifacts.delete(args.notebook_id, args.artifact_id)
        return {
            "notebook_id": args.notebook_id,
            "artifact_id": args.artifact_id,
            "deleted": success,
        }


COMMANDS = {
    "list": list_notebooks,
    "artifacts": list_artifacts,
    "download": download_artifact,
    "delete": delete_notebook,
    "delete-artifact": delete_artifact,
}


def main():
    parser = argparse.ArgumentParser(
        description="Manage NotebookLM notebooks and artifacts.",
        epilog=(
            "Examples:\n"
            "  %(prog)s list                                    # List all notebooks\n"
            "  %(prog)s artifacts <notebook_id>                 # List artifacts\n"
            "  %(prog)s download <notebook_id> --artifact <id> -o out.mp3\n"
            "  %(prog)s delete <notebook_id>                    # Delete notebook\n"
            "  %(prog)s delete-artifact <notebook_id> --artifact <id>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all notebooks")

    # artifacts
    p_art = sub.add_parser("artifacts", help="List artifacts in a notebook")
    p_art.add_argument("notebook_id", help="Notebook ID")

    # download
    p_dl = sub.add_parser("download", help="Download an artifact")
    p_dl.add_argument("notebook_id", help="Notebook ID")
    p_dl.add_argument("--artifact", dest="artifact_id", required=True, help="Artifact ID")
    p_dl.add_argument("-o", "--output", help="Output file path")

    # delete
    p_del = sub.add_parser("delete", help="Delete a notebook")
    p_del.add_argument("notebook_id", help="Notebook ID")

    # delete-artifact
    p_dela = sub.add_parser("delete-artifact", help="Delete an artifact")
    p_dela.add_argument("notebook_id", help="Notebook ID")
    p_dela.add_argument("--artifact", dest="artifact_id", required=True, help="Artifact ID")

    args = parser.parse_args()

    handler = COMMANDS.get(args.command)
    if not handler:
        parser.error(f"Unknown command: {args.command}")

    try:
        result = asyncio.run(handler(args))
        print(json.dumps(result, indent=2))
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
