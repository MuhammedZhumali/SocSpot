"""Local video inventory, timestamped contact sheets, clip export and validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools" / "python"))
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FFMPEG = os.environ.get("FFMPEG_BINARY") or imageio_ffmpeg.get_ffmpeg_exe()
try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf" if os.name == "nt" else "DejaVuSans.ttf", 17)
except OSError:
    FONT = ImageFont.load_default()


def run(args, *, check=True):
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", *map(str, args)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def probe(path):
    result = run(["-i", path], check=False)
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    stream = next((line for line in result.stderr.splitlines() if "Video:" in line), "")
    size = re.search(r"\b(\d{2,5})x(\d{2,5})\b", stream)
    fps = re.search(r"([\d.]+) fps", stream)
    if not duration or not size or not fps:
        raise ValueError(f"Cannot probe {path}: {result.stderr}")
    h, m, s = map(float, duration.groups())
    return {"duration_sec": round(h * 3600 + m * 60 + s, 3),
            "width": int(size[1]), "height": int(size[2]), "fps": float(fps[1])}


def inventory():
    sources = []
    for folder, label in [("backheel", "backheel_pass"), ("bycicle_kick", "bicycle_kick")]:
        for path in sorted((ROOT / "videos" / folder).glob("*.mp4")):
            with path.open("rb") as source_file:
                digest = hashlib.file_digest(source_file, "sha256").hexdigest()
            sources.append({"source_id": path.stem, "path": path.relative_to(ROOT).as_posix(),
                            "youtube_url": f"https://www.youtube.com/watch?v={path.stem}",
                            "collection_label": label, "bytes": path.stat().st_size,
                            "sha256": digest,
                            **probe(path)})
    write_json(ROOT / "data" / "sources.json", sources)
    print(json.dumps(sources, indent=2))


def timestamp(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def sheets(source_id, start, end, step, width, cols, rows):
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    source = next(s for s in sources if s["source_id"] == source_id)
    end = min(end if end is not None else source["duration_sec"], source["duration_sec"])
    out = ROOT / "videos" / "review" / source_id / f"{start:g}-{end:g}_step{step:g}"
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    # fps round=up keeps the requested first sample at the seek point.
    run(["-loglevel", "error", "-ss", start, "-i", ROOT / source["path"], "-t", end - start,
         "-an", "-vf", f"fps=1/{step}:round=up,scale={width}:-2", "-q:v", "3", "-y", frames / "%06d.jpg"])
    images = sorted(frames.glob("*.jpg"))
    per_page = cols * rows
    for page in range(math.ceil(len(images) / per_page)):
        batch = images[page * per_page:(page + 1) * per_page]
        height = Image.open(batch[0]).height
        sheet = Image.new("RGB", (cols * width, rows * (height + 27) + 32), "#141820")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 7), f"{source_id} | {source['collection_label']} | step {step}s", font=FONT, fill="white")
        for i, path in enumerate(batch):
            x, y = (i % cols) * width, (i // cols) * (height + 27) + 32
            with Image.open(path) as im:
                sheet.paste(im, (x, y))
            t = start + (page * per_page + i) * step
            draw.text((x + 7, y + height + 3), f"{t:.2f}s  {timestamp(t)}", font=FONT, fill="white")
        target = out / f"sheet_{page + 1:03d}.jpg"
        sheet.save(target, quality=90)
        print(target, flush=True)


def evidence(source_id):
    """Render 0.5-second sequences around the proposed actions, three per page."""
    sources = {s["source_id"]: s for s in json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))}
    seeds = json.loads((ROOT / "data" / "seeds.json").read_text(encoding="utf-8"))[source_id]
    out = ROOT / "videos/review" / source_id / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    strips = []
    for i, seed in enumerate(seeds):
        center = seed["time"]
        start = max(0, center - 2.5)
        frames = out / f"{i + 1:03d}"
        frames.mkdir(exist_ok=True)
        run(["-v", "error", "-ss", start, "-i", ROOT / sources[source_id]["path"], "-t", "6",
             "-an", "-vf", "fps=2:round=up,scale=240:-2", "-q:v", "3", "-frames:v", "12", "-y", frames / "%03d.jpg"])
        strip = Image.new("RGB", (1440, 355), "#141820")
        draw = ImageDraw.Draw(strip)
        draw.text((8, 4), f"{source_id} #{i + 1:03d} | {seed.get('note', '')}", fill="white", font=FONT)
        for j, path in enumerate(sorted(frames.glob("*.jpg"))):
            x, y = j % 6 * 240, 28 + j // 6 * 162
            with Image.open(path) as im:
                strip.paste(im, (x, y))
            draw.text((x + 4, y + 137), f"{start + j * .5:.2f}s", fill="white", font=FONT)
        strips.append(strip)
    for p in range(math.ceil(len(strips) / 3)):
        canvas = Image.new("RGB", (1440, 1065), "#141820")
        for j, strip in enumerate(strips[p * 3:p * 3 + 3]):
            canvas.paste(strip, (0, j * 355))
        target = out / f"page_{p + 1:03d}.jpg"
        canvas.save(target, quality=93)
        print(target, flush=True)


def load_annotations():
    doc = json.loads((ROOT / "data/annotations.json").read_text(encoding="utf-8"))
    sources = {s["source_id"]: s for s in doc["sources"]}
    ids = set()
    for event in doc["clips"]:
        assert event["clip_id"] not in ids, event["clip_id"]
        ids.add(event["clip_id"])
        source = sources[event["source_id"]]
        assert 0 <= event["start_sec"] < event["end_sec"] <= source["duration_sec"]
        if event.get("action_sec") is not None:
            assert event["start_sec"] <= event["action_sec"] <= event["end_sec"]
        assert event["label"] in {"backheel_pass", "bicycle_kick", "other", "uncertain"}
        assert event["event_id"] and event["split_group"]
    return sources, doc


def export_one(event, sources):
    output = ROOT / event["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    run(["-loglevel", "error", "-ss", event["start_sec"], "-i", ROOT / sources[event["source_id"]]["path"],
         "-t", round(event["end_sec"] - event["start_sec"], 3), "-map", "0:v:0", "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-threads", "2", "-movflags", "+faststart", "-y", output])
    return output.relative_to(ROOT).as_posix()


def export(revision=None, missing_only=False):
    sources, doc = load_annotations()
    events = [e for e in doc["clips"] if e.get("export", True)]
    if revision:
        events = [e for e in events if e.get("annotation_revision") == revision]
    if missing_only:
        events = [e for e in events if not (ROOT / e["path"]).is_file()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for path in pool.map(lambda e: export_one(e, sources), events):
            print(path, flush=True)


def validate():
    sources, doc = load_annotations()
    source_checks = []
    for source in sources.values():
        source_path = ROOT / source["path"]
        with source_path.open("rb") as source_file:
            digest = hashlib.file_digest(source_file, "sha256").hexdigest()
        assert digest == source["sha256"], f"Source changed: {source_path}"
        source_checks.append({"source_id": source["source_id"], "sha256_matches": True})
    checked = []
    for event in doc["clips"]:
        if not event.get("export", True):
            continue
        path = ROOT / event["path"]
        info = probe(path)
        expected = event["end_sec"] - event["start_sec"]
        assert abs(info["duration_sec"] - expected) <= max(.12, 2 / info["fps"]), (path, info, expected)
        source = sources[event["source_id"]]
        assert (info["width"], info["height"]) == (source["width"], source["height"])
        assert abs(info["fps"] - source["fps"]) < .02
        assert (ROOT / event["evidence_path"]).is_file(), event["evidence_path"]
        result = run(["-v", "error", "-i", path, "-map", "0:v:0", "-f", "null", "-"])
        assert not result.stderr.strip(), (path, result.stderr)
        checked.append({"clip_id": event["clip_id"], "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size, "full_decode_ok": True,
                        "duration_error_sec": round(info["duration_sec"] - expected, 4), **info})
    write_json(ROOT / "data" / "validation.json", {"clips_checked": len(checked),
               "validation_scope": "Technical file checks only; labels and action timestamps are provisional.",
               "sources": source_checks, "clips": checked})
    print(f"Decoded and validated {len(checked)} clips.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    contact = commands.add_parser("sheets")
    contact.add_argument("source_id")
    contact.add_argument("--start", type=float, default=0)
    contact.add_argument("--end", type=float)
    contact.add_argument("--step", type=float, default=3)
    contact.add_argument("--width", type=int, default=320)
    contact.add_argument("--cols", type=int, default=4)
    contact.add_argument("--rows", type=int, default=6)
    exporting = commands.add_parser("export")
    exporting.add_argument("--revision", help="Export only clips changed in this annotation revision")
    exporting.add_argument("--missing-only", action="store_true")
    detailed = commands.add_parser("evidence")
    detailed.add_argument("source_id")
    commands.add_parser("validate")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"inventory": inventory, "sheets": sheets, "evidence": evidence, "export": export, "validate": validate}[command](**args)
