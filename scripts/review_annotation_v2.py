"""Render denser visual evidence without modifying the baseline dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw
from video_dataset import ROOT, FONT, run


def snapshot():
    target = ROOT / "data/annotation_versions/v1"
    files = ["annotations.json", "review_decisions.json", "other_annotations.json", "split_config.json", "dataset_summary.json", "validation.json"]
    for name in files:
        dest = target / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "data" / name, dest)
    return json.loads((target / "annotations.json").read_text(encoding="utf-8"))


def render(clip, start=None, end=None, step=.125, width=320, cols=4):
    start = max(0, clip["action_sec"] - 1.5) if start is None else start
    end = start + 4 if end is None else end
    folder = ROOT / "videos/review/annotation-v2" / clip["clip_id"]
    frames = folder / f"{start:.3f}-{end:.3f}-{step:g}-{width}"
    frames.mkdir(parents=True, exist_ok=True)
    count = round((end - start) / step)
    run(["-v", "error", "-ss", start, "-i", ROOT / clip["source_path"],
         "-an", "-vf", f"fps=1/{step}:round=up,scale={width}:-2", "-frames:v", count,
         "-q:v", "2", "-y", frames / "%03d.jpg"])
    paths = sorted(frames.glob("*.jpg"))[:count]
    height = round(width * 9 / 16)
    canvas = Image.new("RGB", (cols * width, ((len(paths) + cols - 1) // cols) * (height + 24) + 35), "#141820")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"{clip['clip_id']} | {start:.3f} - {end:.3f}s | step {step}s", font=FONT, fill="white")
    for i, path in enumerate(paths):
        x, y = i % cols * width, 35 + i // cols * (height + 24)
        with Image.open(path) as im:
            canvas.paste(im.resize((width, height)), (x, y))
        draw.text((x + 6, y + height + 2), f"{start + i * step:.3f}s", font=FONT, fill="white")
    suffix = "" if width == 320 else f"-{width}"
    target = folder / f"{start:.3f}-{end:.3f}-{step:g}{suffix}.jpg"
    canvas.save(target, quality=94)
    return target.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--uncertain", action="store_true")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--step", type=float, default=.125)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()
    doc = snapshot()
    clips = [c for c in doc["clips"] if c["clip_id"] in args.ids or args.uncertain and c["label"] == "uncertain"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for path in pool.map(lambda c: render(c, args.start, args.end, args.step, args.width, args.cols), clips):
            print(path, flush=True)


if __name__ == "__main__":
    main()
