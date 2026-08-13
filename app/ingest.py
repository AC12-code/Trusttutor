"""Turn a lesson into timestamped segments.

Two sources, so the demo always works:
  * a YouTube URL (fetches captions, no video download), or
  * a local file (JSON list of {start, text}, or plain .txt).
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Segment:
    start: float   # seconds
    text: str

    def ts(self) -> str:
        m, s = divmod(int(self.start), 60)
        return f"{m:02d}:{s:02d}"


def load_from_file(path: str) -> List[Segment]:
    if path.endswith(".json"):
        raw = json.load(open(path))
        return [Segment(float(r.get("start", i * 15)), r["text"].strip())
                for i, r in enumerate(raw)]
    # plain text -> pseudo-segment every ~1 sentence
    text = open(path, encoding="utf-8").read()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return [Segment(i * 15.0, p) for i, p in enumerate(parts)]


def _proxy_config():
    """Build a proxy config from env vars, if any are set. None means no proxy."""
    from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

    ws_user = os.getenv("WEBSHARE_PROXY_USERNAME")
    ws_pass = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if ws_user and ws_pass:
        return WebshareProxyConfig(proxy_username=ws_user, proxy_password=ws_pass)

    http_url = os.getenv("HTTP_PROXY_URL")
    https_url = os.getenv("HTTPS_PROXY_URL")
    if http_url or https_url:
        return GenericProxyConfig(http_url=http_url, https_url=https_url or http_url)

    return None


def load_from_youtube(url: str) -> List[Segment]:
    """Best-effort caption fetch. Raises with a clear message if unavailable.

    Set WEBSHARE_PROXY_USERNAME/WEBSHARE_PROXY_PASSWORD or HTTP_PROXY_URL/
    HTTPS_PROXY_URL to route requests through a proxy (works around YouTube
    blocking a residential IP after repeated requests).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise RuntimeError("pip install youtube-transcript-api") from e
    vid = _video_id(url)
    if not vid:
        raise ValueError(f"Could not parse a video id from: {url}")
    try:
        rows = YouTubeTranscriptApi(proxy_config=_proxy_config()).fetch(vid)
    except Exception as e:
        raise RuntimeError(
            f"No captions available for {vid} ({e}). "
            "Fall back to --transcript data/sample_transcript.json for the demo."
        ) from e
    return [Segment(float(r.start), r.text.strip()) for r in rows if r.text.strip()]


def _video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else (url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url) else "")


def segments_to_dicts(segs: List[Segment]):
    return [asdict(s) for s in segs]
