"""Shared sampling for training and inference; no action timestamp is needed."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["backheel_pass", "bicycle_kick", "other"]
PREPROCESS = {"frames": 16, "height": 128, "width": 171, "crop": 112,
              "sampling": "uniform_full_window", "window_sec": 3.0, "stride_sec": 1.0,
              "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def duration(path: Path):
    p = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)],
                       capture_output=True, timeout=30)
    text = p.stderr.decode("utf-8", errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    if not m:
        raise ValueError(f"Cannot determine video duration: {path.name}")
    h, minute, second = map(float, m.groups())
    return h * 3600 + minute * 60 + second


def sample_frames(path: Path, start: float, seconds: float) -> np.ndarray:
    if not math.isfinite(start) or not math.isfinite(seconds) or start < 0 or seconds <= 0:
        raise ValueError("Invalid sample interval")
    cfg = PREPROCESS
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "2", "-ss", str(start), "-i", str(path), "-t", str(seconds), "-an",
            "-vf", f"fps={cfg['frames']/seconds}:round=up,scale={cfg['width']}:{cfg['height']}",
            "-frames:v", str(cfg["frames"]), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.run(args, capture_output=True, timeout=90)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-1200:])
    frame_bytes = cfg["height"] * cfg["width"] * 3
    if not proc.stdout or len(proc.stdout) % frame_bytes:
        raise ValueError(f"No complete decoded frames: {path.name}")
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, cfg["height"], cfg["width"], 3).copy()
    if len(frames) < cfg["frames"]:
        frames = np.concatenate([frames, np.repeat(frames[-1:], cfg["frames"]-len(frames), axis=0)])
    return frames[:cfg["frames"]]


def windows(seconds: float, window: float = 3.0, stride: float = 1.0):
    if seconds <= 0 or window <= 0 or stride <= 0 or not all(map(math.isfinite, (seconds, window, stride))):
        raise ValueError("Window lengths must be finite and positive")
    if seconds <= window:
        return [(0.0, seconds)]
    starts = [round(i * stride, 6) for i in range(int((seconds-window)//stride)+1)]
    final = round(seconds-window, 6)
    if final-starts[-1] > 1e-5:
        starts.append(final)
    return [(s, min(seconds, s+window)) for s in starts]
