#!/usr/bin/env python3
"""
YouTube Video Extractor — pulls ALL data from a YouTube video:
  - Metadata (title, description, tags, duration, views, likes, channel, etc.)
  - Full transcript/captions (auto-generated or manual)
  - Top comments
  - Thumbnail URLs
  - Chapter markers
  - Output as structured JSON

Usage:
  python extract.py all https://www.youtube.com/watch?v=VIDEO_ID
  python extract.py metadata VIDEO_ID
  python extract.py transcript VIDEO_ID --lang en
  python extract.py comments VIDEO_ID --max 20
  python extract.py research VIDEO_ID
  python extract.py download-video VIDEO_ID --quality 720p
  python extract.py download-audio VIDEO_ID
  python extract.py chapters VIDEO_ID
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_video_id(url_or_id: str) -> str:
    """Extract video ID from URL or return as-is if already an ID."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return url_or_id


def extract_metadata(video_id: str) -> dict:
    """Use yt-dlp to dump all metadata as JSON."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--dump-json", "--no-download", url],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0:
            raw = json.loads(r.stdout)
            return {
                "title": raw.get("title"),
                "description": raw.get("description"),
                "channel": raw.get("channel"),
                "channel_id": raw.get("channel_id"),
                "channel_url": raw.get("channel_url"),
                "upload_date": raw.get("upload_date"),
                "duration": raw.get("duration"),
                "duration_string": raw.get("duration_string"),
                "view_count": raw.get("view_count"),
                "like_count": raw.get("like_count"),
                "comment_count": raw.get("comment_count"),
                "tags": raw.get("tags", []),
                "categories": raw.get("categories", []),
                "thumbnails": raw.get("thumbnails", []),
                "chapters": raw.get("chapters", []),
                "age_limit": raw.get("age_limit"),
                "live_status": raw.get("live_status"),
                "language": raw.get("language"),
                "subtitles_available": list(raw.get("subtitles", {}).keys()),
                "auto_captions_available": list(raw.get("automatic_captions", {}).keys()),
                "webpage_url": raw.get("webpage_url"),
                "_raw_keys": list(raw.keys()),  # for discovery
            }
        else:
            return {"error": r.stderr.strip()}
    except Exception as e:
        return {"error": str(e)}


def extract_transcript(video_id: str, lang: str = "en") -> dict:
    """Use youtube-transcript-api to get captions."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang, "en"])
        full_text = " ".join(entry["text"] for entry in transcript)
        return {
            "language": lang,
            "segments": transcript,
            "full_text": full_text,
            "word_count": len(full_text.split()),
        }
    except Exception as e:
        # Try without language preference
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join(entry["text"] for entry in transcript)
            return {
                "language": "auto",
                "segments": transcript,
                "full_text": full_text,
                "word_count": len(full_text.split()),
            }
        except Exception as e2:
            return {"error": str(e2)}


def extract_comments(video_id: str, max_comments: int = 20) -> list:
    """Use youtube-comment-downloader to scrape comments."""
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader
        downloader = YoutubeCommentDownloader()
        comments = []
        url = f"https://www.youtube.com/watch?v={video_id}"
        for comment in downloader.get_comments_from_url(url, sort_by=0):
            comments.append({
                "author": comment.get("author"),
                "text": comment.get("text"),
                "likes": comment.get("votes"),
                "time": comment.get("time"),
                "is_reply": comment.get("reply", False),
            })
            if len(comments) >= max_comments:
                break
        return comments
    except Exception as e:
        return [{"error": str(e)}]


def extract_for_research(video_id: str, lang: str = "en") -> dict:
    """Condensed extraction optimized for LLM script learning.

    Returns a compact summary suitable for feeding to the ScriptWriterAgent
    so it can learn the style, pacing, and structure of reference videos.
    """
    metadata = extract_metadata(video_id)
    transcript = extract_transcript(video_id, lang)
    comments = extract_comments(video_id, max_comments=10)

    # Compute style hints from transcript
    style_hints = {}
    if "full_text" in transcript:
        words = transcript["full_text"].split()
        sentences = [s.strip() for s in transcript["full_text"].replace("?", ".").replace("!", ".").split(".") if s.strip()]
        style_hints = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 1),
            "duration_seconds": metadata.get("duration", 0),
            "words_per_minute": round(len(words) / max(metadata.get("duration", 1) / 60, 0.1), 1),
        }

    # Top comment themes (just text, trimmed)
    top_comments = []
    for c in comments:
        if isinstance(c, dict) and "text" in c and "error" not in c:
            top_comments.append(c["text"][:200])

    return {
        "video_id": video_id,
        "title": metadata.get("title", ""),
        "channel": metadata.get("channel", ""),
        "duration": metadata.get("duration_string", ""),
        "chapters": metadata.get("chapters", []),
        "tags": metadata.get("tags", [])[:15],
        "transcript_full": transcript.get("full_text", ""),
        "style_hints": style_hints,
        "top_comments": top_comments,
    }


def download_video(video_id: str, output_dir: str = "./cache/downloads", quality: str = "720p") -> str:
    """Download a YouTube video file via yt-dlp. Returns the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(Path(output_dir) / "%(title)s.%(ext)s")

    height = quality.replace("p", "")
    format_spec = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"

    try:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp",
             "-f", format_spec,
             "--merge-output-format", "mp4",
             "-o", output_template,
             "--print", "after_move:filepath",
             url],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode == 0:
            filepath = r.stdout.strip().split("\n")[-1]
            return filepath
        else:
            raise RuntimeError(f"yt-dlp error: {r.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timed out after 5 minutes")


def download_audio(video_id: str, output_dir: str = "./cache/downloads") -> str:
    """Extract audio-only track from a YouTube video. Returns WAV path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(Path(output_dir) / "%(title)s.%(ext)s")

    try:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp",
             "-x", "--audio-format", "wav",
             "-o", output_template,
             "--print", "after_move:filepath",
             url],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode == 0:
            filepath = r.stdout.strip().split("\n")[-1]
            return filepath
        else:
            raise RuntimeError(f"yt-dlp error: {r.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out after 5 minutes")


def extract_chapters_as_scenes(video_id: str, lang: str = "en") -> list[dict]:
    """Convert YouTube chapters into Scene-compatible dicts.

    Returns a list of dicts matching the Scene schema fields,
    enabling quick VideoPlan creation from existing video structure.
    """
    metadata = extract_metadata(video_id)
    transcript = extract_transcript(video_id, lang)
    chapters = metadata.get("chapters", [])

    if not chapters:
        return []

    segments = transcript.get("segments", [])
    scenes = []

    for i, chapter in enumerate(chapters):
        start = chapter.get("start_time", 0)
        end = chapters[i + 1]["start_time"] if i + 1 < len(chapters) else metadata.get("duration", start + 30)
        duration = end - start

        # Extract transcript text for this chapter's time range
        chapter_text = " ".join(
            seg["text"] for seg in segments
            if seg.get("start", 0) >= start and seg.get("start", 0) < end
        )

        title = chapter.get("title", f"Scene {i + 1}")
        scenes.append({
            "scene_number": i + 1,
            "narration": chapter_text or f"Content about: {title}",
            "duration": round(duration, 1),
            "visual_description": title,
            "search_keywords": title.lower().split()[:5],
            "capture_url": None,
            "transition": "crossfade",
        })

    return scenes


def extract_all(video_id: str, max_comments: int = 20, lang: str = "en") -> dict:
    """Extract everything from a YouTube video."""
    print(f"Extracting data for video: {video_id}")
    print("=" * 50)

    print("[1/3] Fetching metadata via yt-dlp...")
    metadata = extract_metadata(video_id)
    if "title" in metadata:
        print(f"  Title: {metadata['title']}")
        print(f"  Channel: {metadata['channel']}")
        print(f"  Views: {metadata.get('view_count', 'N/A'):,}" if isinstance(metadata.get('view_count'), int) else f"  Views: {metadata.get('view_count', 'N/A')}")
        print(f"  Duration: {metadata.get('duration_string', 'N/A')}")

    print("[2/3] Fetching transcript...")
    transcript = extract_transcript(video_id, lang)
    if "full_text" in transcript:
        print(f"  Words: {transcript['word_count']}")
        print(f"  Segments: {len(transcript['segments'])}")
    else:
        print(f"  {transcript.get('error', 'No transcript available')}")

    print(f"[3/3] Fetching top {max_comments} comments...")
    comments = extract_comments(video_id, max_comments)
    if comments and "error" not in comments[0]:
        print(f"  Got {len(comments)} comments")
    else:
        print(f"  {comments[0].get('error', 'No comments') if comments else 'No comments'}")

    result = {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "extracted_at": datetime.now().isoformat(),
        "metadata": metadata,
        "transcript": transcript,
        "comments": comments,
    }

    print("=" * 50)
    print("Done!")
    return result


def output_result(result, output_path: str = None, pretty: bool = True):
    """Serialize result as JSON to stdout or file."""
    json_str = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json_str)
        print(f"\nSaved to {output_path}", file=sys.stderr)
    else:
        print(json_str)


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 extract.py all https://youtube.com/watch?v=dQw4w9WgXcQ
  python3 extract.py metadata VIDEO_ID
  python3 extract.py transcript VIDEO_ID --lang en
  python3 extract.py comments VIDEO_ID --max 50
  python3 extract.py research VIDEO_ID
  python3 extract.py download-video VIDEO_ID --quality 1080p
  python3 extract.py download-audio VIDEO_ID --output ./audio
  python3 extract.py chapters VIDEO_ID
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # --- all ---
    p_all = subparsers.add_parser("all", help="Extract metadata, transcript, and comments")
    p_all.add_argument("video", help="YouTube URL or video ID")
    p_all.add_argument("--max-comments", type=int, default=20, help="Max comments to fetch (default: 20)")
    p_all.add_argument("--lang", default="en", help="Transcript language (default: en)")
    p_all.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    p_all.add_argument("--no-pretty", action="store_true", help="Disable pretty-print JSON")

    # --- metadata ---
    p_meta = subparsers.add_parser("metadata", help="Extract video metadata only")
    p_meta.add_argument("video", help="YouTube URL or video ID")
    p_meta.add_argument("--output", "-o", help="Output JSON file")
    p_meta.add_argument("--no-pretty", action="store_true")

    # --- transcript ---
    p_trans = subparsers.add_parser("transcript", help="Extract transcript/captions")
    p_trans.add_argument("video", help="YouTube URL or video ID")
    p_trans.add_argument("--lang", default="en", help="Language code (default: en)")
    p_trans.add_argument("--output", "-o", help="Output JSON file")
    p_trans.add_argument("--no-pretty", action="store_true")

    # --- comments ---
    p_comments = subparsers.add_parser("comments", help="Extract top comments")
    p_comments.add_argument("video", help="YouTube URL or video ID")
    p_comments.add_argument("--max", type=int, default=20, help="Max comments (default: 20)")
    p_comments.add_argument("--output", "-o", help="Output JSON file")
    p_comments.add_argument("--no-pretty", action="store_true")

    # --- research ---
    p_research = subparsers.add_parser("research", help="Compact extraction for LLM research")
    p_research.add_argument("video", help="YouTube URL or video ID")
    p_research.add_argument("--lang", default="en", help="Language code (default: en)")
    p_research.add_argument("--output", "-o", help="Output JSON file")
    p_research.add_argument("--no-pretty", action="store_true")

    # --- download-video ---
    p_dvideo = subparsers.add_parser("download-video", help="Download video file")
    p_dvideo.add_argument("video", help="YouTube URL or video ID")
    p_dvideo.add_argument("--quality", default="720p", help="Max video quality (default: 720p)")
    p_dvideo.add_argument("--output", "-o", default="./cache/downloads", help="Output directory")

    # --- download-audio ---
    p_daudio = subparsers.add_parser("download-audio", help="Download audio as WAV")
    p_daudio.add_argument("video", help="YouTube URL or video ID")
    p_daudio.add_argument("--output", "-o", default="./cache/downloads", help="Output directory")

    # --- chapters ---
    p_chapters = subparsers.add_parser("chapters", help="Extract chapters as scene dicts")
    p_chapters.add_argument("video", help="YouTube URL or video ID")
    p_chapters.add_argument("--lang", default="en", help="Language code (default: en)")
    p_chapters.add_argument("--output", "-o", help="Output JSON file")
    p_chapters.add_argument("--no-pretty", action="store_true")

    args = parser.parse_args()
    video_id = get_video_id(args.video)
    pretty = not getattr(args, "no_pretty", False)

    if args.command == "all":
        result = extract_all(video_id, args.max_comments, args.lang)
        output_result(result, args.output, pretty)

    elif args.command == "metadata":
        result = extract_metadata(video_id)
        output_result(result, args.output, pretty)

    elif args.command == "transcript":
        result = extract_transcript(video_id, args.lang)
        output_result(result, args.output, pretty)

    elif args.command == "comments":
        result = extract_comments(video_id, args.max)
        output_result(result, args.output, pretty)

    elif args.command == "research":
        result = extract_for_research(video_id, args.lang)
        output_result(result, args.output, pretty)

    elif args.command == "download-video":
        filepath = download_video(video_id, args.output, args.quality)
        print(json.dumps({"status": "ok", "filepath": filepath}, indent=2))

    elif args.command == "download-audio":
        filepath = download_audio(video_id, args.output)
        print(json.dumps({"status": "ok", "filepath": filepath}, indent=2))

    elif args.command == "chapters":
        result = extract_chapters_as_scenes(video_id, args.lang)
        output_result(result, args.output, pretty)


if __name__ == "__main__":
    main()
