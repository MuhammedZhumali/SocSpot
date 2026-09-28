"""Download a pinned HF subset and prepare evidence for manual negative selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import urllib.request

from video_dataset import ROOT, FONT, Image, ImageDraw, probe, run, write_json

META = ROOT / "data/external/infactory-soccer-events"
REPO = "infactory-ai/soccer-events"


def match_id(row):
    identity = "|".join(row[k] for k in ("competition", "season", "game_date", "home_team", "away_team"))
    return "hf_match_" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def plan():
    with (META / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    revision = json.loads((META / "repository.json").read_text(encoding="utf-8"))["sha"]
    forbidden = {match_id(r) for r in rows if r["event_subtype"] in {"Back Heel", "Overhead/Bicycle"}}
    selected, used = [], set(forbidden)
    # Prefer rarer shot types before the abundant ordinary kicks, one source per match.
    priority = {"Header": 0, "Chip": 1, "Free kick shot": 2, "Kick": 3, "Lucky Situation": 4}
    for category, limit in [("goal", 30), ("cards", 10)]:
        candidates = sorted(
            (r for r in rows if (r["event_category"] == "goal") == (category == "goal") and float(r["duration_seconds"]) >= 14),
            key=lambda r: (priority.get(r["event_subtype"], 5), hashlib.sha256(r["asset_id"].encode()).hexdigest()),
        )
        count = 0
        for row in candidates:
            group = match_id(row)
            if group in used:
                continue
            selected.append({**row, "match_id": group})
            used.add(group)
            count += 1
            if count == limit:
                break
        if count != limit:
            raise ValueError(f"Not enough independent matches for {category}")
    selected.sort(key=lambda r: r["asset_id"])
    # Fixed split by match, balanced by source category before looking at frames.
    assignments = {}
    for category, train_count, validation_count in [("goal", 21, 4), ("cards", 7, 2)]:
        members = sorted(
            (r for r in selected if (r["event_category"] == "goal") == (category == "goal")),
            key=lambda r: hashlib.sha256(f"42:{r['match_id']}".encode()).hexdigest(),
        )
        for i, row in enumerate(members):
            assignments[row["match_id"]] = "train" if i < train_count else "validation" if i < train_count + validation_count else "test"
    for i, row in enumerate(selected, 1):
        row["review_number"] = i
        row["split"] = assignments[row["match_id"]]
        row["path"] = f"videos/external/infactory-soccer-events/{row['mp4_file']}"
        row["url"] = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/{row['mp4_file']}"
    write_json(META / "selection.json", {
        "repo_id": REPO, "revision": revision,
        "license": "proprietary; non-commercial research only per dataset card",
        "known_target_matches_excluded": sorted(forbidden),
        "selection": "30 goal clips and 10 card clips from 40 distinct metadata match identities; no automatic other labels",
        "sources": selected,
    })
    print(f"Selected {len(selected)} sources, 28 train / 6 validation / 6 test matches.")


def selection():
    return json.loads((META / "selection.json").read_text(encoding="utf-8"))


def download_one(row):
    path = ROOT / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        request = urllib.request.Request(row["url"], headers={"User-Agent": "SocSpot-dataset-research/1.0"})
        partial = path.with_suffix(".mp4.part")
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        partial.replace(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {**row, "sha256": digest, "bytes": path.stat().st_size, **probe(path)}


def download():
    doc = selection()
    sources = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download_one, doc["sources"]):
            sources.append(row)
            print(f"Downloaded {row['review_number']:02d}: {row['bytes'] / 2**20:.1f} MiB", flush=True)
    write_json(META / "sources.json", {**doc, "sources": sources})


def evidence_one(row):
    out = ROOT / "videos/review/hf_other" / f"{row['review_number']:02d}"
    out.mkdir(parents=True, exist_ok=True)
    # Four successive three-second candidate windows, sampled every half-second.
    run(["-v", "error", "-i", ROOT / row["path"], "-t", "12", "-an",
         "-vf", "fps=2:round=up,scale=280:-2", "-frames:v", "24", "-q:v", "2", "-y", out / "%03d.jpg"])
    frames = sorted(out.glob("[0-9][0-9][0-9].jpg"))
    with Image.open(frames[0]) as first:
        height = first.height
    canvas = Image.new("RGB", (1680, 4 * (height + 27) + 64), "#141820")
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 3), f"HF #{row['review_number']:02d} {row['asset_id']} | {row['event_category']} / {row['event_subtype']} | {row['split']}", font=FONT, fill="white")
    draw.text((7, 29), f"{row['home_team']} - {row['away_team']} | {row['game_date']} | candidates: row 1=0-3s, 2=3-6s, 3=6-9s, 4=9-12s", font=FONT, fill="white")
    for i, path in enumerate(frames):
        x, y = i % 6 * 280, 64 + i // 6 * (height + 27)
        with Image.open(path) as frame:
            canvas.paste(frame, (x, y))
        draw.text((x + 4, y + height + 3), f"{i / 2:.1f}s", font=FONT, fill="white")
    path = out / "contact.jpg"
    canvas.save(path, quality=93)
    return str(path)


def evidence():
    with ThreadPoolExecutor(max_workers=3) as pool:
        for path in pool.map(evidence_one, selection()["sources"]):
            print(path, flush=True)


def build():
    doc = json.loads((META / "sources.json").read_text(encoding="utf-8"))
    decisions = json.loads((META / "review_decisions.json").read_text(encoding="utf-8"))
    sources, clips, assignments = [], [], {}
    for row in doc["sources"]:
        sid = "hf_" + row["asset_id"]
        sources.append({
            "source_id": sid, "path": row["path"], "source_url": row["url"],
            "collection_label": "other_candidates", "dataset": REPO,
            "revision": doc["revision"], "license": doc["license"],
            "match_id": row["match_id"],
            **{k: row[k] for k in ("sha256", "bytes", "duration_sec", "width", "height", "fps")},
        })
        entries = decisions[str(row["review_number"])]
        if entries:
            assignments[sid] = row["split"]
        for i, (start, end, subtype, note) in enumerate(entries, 1):
            if not 0 <= start < end <= min(12, row["duration_sec"]):
                raise ValueError("Selected window must have reviewed evidence in the first 12 seconds")
            cid = f"hf_other_{row['review_number']:03d}_{i:02d}"
            clips.append({
                "clip_id": cid, "source_id": sid, "source_path": row["path"],
                "source_url": row["url"], "start_sec": start, "end_sec": end,
                "duration_sec": end - start, "action_sec": None, "action_offset_sec": None,
                "label": "other", "proposed_label": "other", "negative_subtype": subtype,
                "label_status": "provisional", "review_method": "assistant_contact_sheets_0.5_sec",
                "human_review_status": "pending", "event_id": f"hf_window_{row['asset_id']}_{i:02d}",
                "match_id": row["match_id"], "evaluation_group": row["match_id"],
                "duplicate_of": None, "group_status": "source_match_metadata",
                "split": row["split"], "split_group": row["match_id"],
                "training_candidate": True, "short_context": end - start < 2.5,
                "notes": f"{note} | {row['home_team']} — {row['away_team']}, {row['game_date']}",
                "export": True, "path": f"videos/clips/other/{cid}.mp4",
                "evidence_path": f"videos/review/hf_other/{row['review_number']:02d}/contact.jpg",
                "dataset": REPO, "dataset_revision": doc["revision"],
                "license": doc["license"],
            })
    write_json(ROOT / "data/other_annotations.json", {
        "schema_version": 1, "sources": sources, "clips": clips,
        "fixed_source_splits": assignments,
        "review_scope": "Selected three-second windows inspected at 0.5-second intervals; provisional negatives, not exhaustive full-motion labels.",
    })
    print(f"Built {len(clips)} other windows from {len(assignments)} sources.")


def export():
    from video_dataset import export_one
    doc = json.loads((ROOT / "data/other_annotations.json").read_text(encoding="utf-8"))
    sources = {s["source_id"]: s for s in doc["sources"]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for path in pool.map(lambda c: export_one(c, sources), doc["clips"]):
            print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "download", "evidence", "build", "export"])
    args = parser.parse_args()
    {"plan": plan, "download": download, "evidence": evidence, "build": build, "export": export}[args.command]()
