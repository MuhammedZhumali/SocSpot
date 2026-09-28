"""Prepare small frame tensors on CPU, independently of the training runtime."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import numpy as np
from .video import ROOT, duration, sample_frames, sha256


def prepare_cache(rows):
    cache = ROOT / "data/cache/frames-v1"
    cache.mkdir(parents=True, exist_ok=True)

    def one(row):
        path = ROOT/row["path"]
        key = sha256(path)
        cached = cache/f"{key}.npy"
        if not cached.exists():
            frames = sample_frames(path, 0, duration(path))
            # Atomic replacement lets an interrupted preparation safely resume.
            with cached.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, frames, allow_pickle=False)
            cached.with_suffix(".tmp").replace(cached)
        return row["clip_id"], cached

    with ThreadPoolExecutor(max_workers=4) as executor:
        return dict(executor.map(one, rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits/review_v2_source_holdout")
    args = parser.parse_args()
    rows = []
    for part in ("train", "validation"):
        rows.extend(json.loads((ROOT/args.splits/f"{part}.json").read_text(encoding="utf-8"))["clips"])
    result = prepare_cache(rows)
    print(f"Prepared {len(result)} training/validation clips")
