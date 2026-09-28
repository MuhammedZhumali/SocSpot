"""Internet candidates are downloaded and scored before becoming results."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import imageio_ffmpeg

from .video import LABELS, ROOT, duration, sha256, windows, write_json

QUERIES = {"backheel_pass": "football backheel passes highlights",
           "bicycle_kick": "football bicycle kick overhead kick highlights"}
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def run_download(command, timeout):
    """On timeout also stop FFmpeg, which yt-dlp launches as a child process."""
    options = {"start_new_session":True} if os.name != "nt" else {}
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(command, proc.returncode,
                                       stdout.decode("utf-8",errors="replace"), stderr.decode("utf-8",errors="replace"))


def excluded_video_ids():
    sources = json.loads((ROOT/"data/sources.json").read_text(encoding="utf-8"))
    return {s["source_id"] for s in sources}


def validate_action(action):
    if action not in QUERIES:
        raise ValueError("Choose backheel_pass or bicycle_kick")


def search_candidates(action, limit=3):
    validate_action(action)
    if not 1 <= limit <= 5:
        raise ValueError("Limit must be between 1 and 5")
    query, excluded = QUERIES[action], excluded_video_ids()
    requested = min(25, limit+len(excluded)+5)
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        params = urllib.parse.urlencode({"part":"snippet", "type":"video", "q":query,
                                         "maxResults":requested, "key":key, "safeSearch":"moderate"})
        try:
            with urllib.request.urlopen("https://www.googleapis.com/youtube/v3/search?"+params, timeout=30) as response:
                data = json.load(response)
        except Exception as exc:
            raise RuntimeError("YouTube Data API search failed; check the key, quota and network") from None
        entries = [{"id": r["id"].get("videoId"), "title":r["snippet"]["title"]} for r in data.get("items", [])]
        provider = "youtube_data_api"
    else:
        # An isolated subprocess gives the full search a finite wall-clock timeout.
        command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--flat-playlist", "--dump-single-json",
                   "--skip-download", "--no-warnings", "--socket-timeout", "20", "--retries", "1",
                   f"ytsearch{requested}:{query}"]
        proc = subprocess.run(command, capture_output=True, timeout=90, encoding="utf-8", errors="replace")
        if proc.returncode:
            raise RuntimeError("YouTube search is unavailable; check network or configure YOUTUBE_API_KEY")
        entries = json.loads(proc.stdout).get("entries", [])
        provider = "yt_dlp_youtube_search"
    candidates, seen = [], set()
    for entry in entries:
        if not entry:
            continue
        video_id = entry.get("id", "")
        if not VIDEO_ID.fullmatch(video_id) or video_id in excluded or video_id in seen or entry.get("is_live"):
            continue
        seen.add(video_id)
        candidates.append({"video_id":video_id, "title":entry.get("title") or video_id,
                           "url":f"https://www.youtube.com/watch?v={video_id}", "status":"candidate"})
        if len(candidates) == limit:
            break
    return {"query":query, "provider":provider, "excluded_dataset_video_ids":sorted(excluded), "candidates":candidates}


def download_video(video_id, seconds):
    if not VIDEO_ID.fullmatch(video_id) or not 3 <= seconds <= 180:
        raise ValueError("Invalid download request")
    folder = ROOT / "data/search/videos" / f"{video_id}-{seconds}s"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder/"media.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        cached = folder/saved["filename"]
        if cached.is_file() and sha256(cached) == saved["sha256"]:
            return cached, saved
    command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-progress",
               "--no-warnings", "--socket-timeout", "20", "--retries", "1", "--fragment-retries", "1",
               "--js-runtimes", "node", "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
               "--download-sections", f"*0-{seconds}", "--max-filesize", "100M",
               "-f", "bv*[height<=480][vcodec^=avc1]/b[height<=480]/bv*[height<=480]",
               "-o", str(folder/"video.%(ext)s"), f"https://www.youtube.com/watch?v={video_id}"]
    try:
        proc = run_download(command, timeout=240)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video download exceeded four minutes") from None
    if proc.returncode:
        # Don't forward signed CDN URLs or cookie data to API responses/log files.
        detail = "Video is unavailable for download (access restriction, site change or network error)"
        if "Sign in" in proc.stderr or "confirm" in proc.stderr:
            detail = "YouTube requires sign-in or verification for this video"
        raise RuntimeError(detail)
    files = [p for p in folder.glob("video.*") if p.suffix in {".mp4", ".webm", ".mkv"}]
    if len(files) != 1:
        raise RuntimeError("Download did not produce one complete video")
    path = files[0]
    measured = duration(path)
    saved = {"filename":path.name, "sha256":sha256(path), "duration_sec":measured,
             "source_start_sec":0.0, "requested_seconds":seconds}
    write_json(metadata, saved)
    return path, saved


def merge_events(scored_windows, action, threshold, max_gap=.01):
    """Merge overlapping positive windows. Times are window bounds, not contact time."""
    validate_action(action)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    events = []
    for row in sorted(scored_windows, key=lambda r:r["start_sec"]):
        if row["predicted_label"] != action or row["scores"][action] < threshold:
            continue
        score = row["scores"][action]
        if events and row["start_sec"] <= events[-1]["end_sec"]+max_gap:
            events[-1]["end_sec"] = max(events[-1]["end_sec"], row["end_sec"])
            events[-1]["model_score"] = max(events[-1]["model_score"], score)
            events[-1]["positive_windows"] += 1
        else:
            events.append({"label":action, "start_sec":row["start_sec"], "end_sec":row["end_sec"],
                           "model_score":score, "positive_windows":1})
    return events


def analyze_video(path, predictor, action, max_seconds=60, progress=None, cancelled=None):
    validate_action(action)
    analyzed = min(float(max_seconds), duration(Path(path)))
    intervals = windows(analyzed)
    scores = []
    for i, (start, end) in enumerate(intervals):
        if cancelled and cancelled():
            raise InterruptedError("Search cancelled")
        probabilities = predictor.predict_window(path, start, end)
        scores.append({"start_sec":start, "end_sec":end,
                       "predicted_label":LABELS[int(probabilities.argmax())],
                       "scores":{label:float(p) for label,p in zip(LABELS, probabilities, strict=True)}})
        if progress:
            progress(i+1, len(intervals))
    return {"analyzed_start_sec":0.0, "analyzed_end_sec":analyzed, "windows":scores,
            "events":merge_events(scores, action, predictor.thresholds[action]),
            "threshold":predictor.thresholds[action], "model_version":predictor.version}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=list(QUERIES))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"data/search/last-search.json")
    args = parser.parse_args()
    if not 3 <= args.seconds <= 180:
        parser.error("--seconds must be 3..180")
    found = search_candidates(args.action, args.limit)
    if not args.discover_only:
        from .model import Predictor
        current = json.loads((ROOT/"models/current.json").read_text(encoding="utf-8"))
        predictor = Predictor(ROOT/current["checkpoint"])
        for candidate in found["candidates"]:
            try:
                path, _ = download_video(candidate["video_id"], args.seconds)
                candidate.update(analyze_video(path, predictor, args.action, args.seconds))
                candidate["status"] = "analyzed"
            except Exception as exc:
                candidate.update(status="failed", error=str(exc))
    write_json(args.output, found)
    print(json.dumps(found, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
