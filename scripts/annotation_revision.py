"""Apply the recorded visual review without rewriting baseline media or decisions."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data/reviews/annotation-v2.json"
FIELDS = ("label", "start_sec", "end_sec", "action_sec", "event_id")


def apply_revision(clips, sources):
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    sources = {s["source_id"]: s for s in sources}
    clips = deepcopy(clips)
    by_id = {c["clip_id"]: c for c in clips}
    audit = []
    for cid, change in review["reviews"].items():
        c = by_id[cid]
        before = {k: c[k] for k in FIELDS}
        start = max(0, c["action_sec"] - 1.5)
        folder = ROOT / "videos/review/annotation-v2" / cid
        evidence = [f"{folder.relative_to(ROOT).as_posix()}/{start:.3f}-{start + 4:.3f}-0.125.jpg"]
        evidence.extend(p.relative_to(ROOT).as_posix() for p in sorted(folder.glob("*-640.jpg")))
        evidence.extend(change.get("extra_evidence", []))
        c.update({k: v for k, v in change.items() if k not in {"review_note", "extra_evidence"}})
        c["notes"] = change["review_note"]
        c["evidence_path"] = evidence[0]
        c["review"] = {"revision": "v2", "before": before, "evidence_paths": evidence,
                       "sample_interval_sec": .125, "reason": change["review_note"]}
        audit.append({"clip_id": cid, "before": before, "after": {k: c[k] for k in FIELDS},
                      "reason": change["review_note"]})
    for addition in review["additions"]:
        source = sources[addition["source_id"]]
        c = {**addition, "label": "other", "action_sec": None,
             "source_path": source["path"], "source_url": source["youtube_url"],
             "notes": addition["review_note"], "export": True,
             "group_status": "provisional_visual_grouping", "split": None,
             "split_group": "initial_four_compilations",
             "review": {"revision": "v2", "before": None, "sample_interval_sec": .125,
                        "evidence_paths": [addition["evidence_path"]], "reason": addition["review_note"]}}
        c.pop("review_note")
        clips.append(c)
    for c in clips:
        if c.get("review", {}).get("revision") != "v2":
            continue
        c.update(annotation_revision="v2", label_status="provisional", human_review_status="pending",
                 review_method="assistant_contact_sheets_0.125_sec", proposed_label=c["label"])
        c["duration_sec"] = round(c["end_sec"] - c["start_sec"], 3)
        c["action_offset_sec"] = None if c["action_sec"] is None else round(c["action_sec"] - c["start_sec"], 3)
        c["short_context"] = c["duration_sec"] < 2.5
        c["path"] = f"videos/clips/v2/{c['label']}/{c['clip_id']}.mp4"
        for path in c["review"]["evidence_paths"]:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(path)
    groups = defaultdict(list)
    for c in clips:
        groups[c["event_id"]].append(c)
    for group in groups.values():
        labels = {c["label"] for c in group}
        for i, c in enumerate(group):
            c["duplicate_of"] = group[0]["clip_id"] if i else None
            c["training_candidate"] = len(labels) == 1 and "uncertain" not in labels and i == 0
    summary = {
        "revision": "v2", "reviewer": review["reviewer"], "method": review["method"],
        "selection": review["selection"], "taxonomy_decision": review["taxonomy_decision"],
        "review_ledger_sha256": hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest(),
        "reviewed_existing_clips": len(audit), "added_other_clips": len(review["additions"]),
        "resolved_uncertain": sum(a["before"]["label"] == "uncertain" and a["after"]["label"] != "uncertain" for a in audit),
        "removed_from_other": sum(a["before"]["label"] == "other" and a["after"]["label"] != "other" for a in audit),
        "changed_boundaries": sum(any(a["before"][k] != a["after"][k] for k in ("start_sec", "end_sec")) for a in audit),
        "changed_action_timestamps": sum(a["before"]["action_sec"] != a["after"]["action_sec"] for a in audit),
        "by_label": dict(Counter(c["label"] for c in clips)),
        "remaining_uncertain": [c["clip_id"] for c in clips if c["label"] == "uncertain"],
        "human_review_status": "pending", "model_retrained": False,
        "changes": audit,
    }
    return clips, summary
