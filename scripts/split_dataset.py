"""Create a provisional source-held-out test set, purging known event repeats.

This deliberately permits an experimental split of the initial collection.
It does not certify that the provisional event IDs identify every duplicate.
Only the standard library is required; source media and annotations stay intact.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("backheel_pass", "bicycle_kick", "other")
SPLITS = ("train", "validation", "test")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_partition(original, partitions, excluded, config):
    """Check complete accounting and separation using the available metadata."""
    expected = {c["clip_id"] for c in original}
    if len(expected) != len(original):
        raise ValueError("Duplicate clip_id in annotations")
    rows = [c for split in SPLITS for c in partitions[split]]
    accounted = [c["clip_id"] for c in rows + excluded]
    if len(accounted) != len(set(accounted)) or set(accounted) != expected:
        raise ValueError("Every input clip must be selected or excluded exactly once")
    events = [c["event_id"] for c in rows]
    if len(events) != len(set(events)):
        raise ValueError("A known event was selected more than once")
    fixed = config.get("fixed_source_splits", {})
    test_sources = set(config["test_sources"]) | {sid for sid, split in fixed.items() if split == "test"}
    test_events = {c["event_id"] for c in original if c["source_id"] in test_sources}
    development_sources = set(config["development_sources"])
    group_owners = {}
    for split in SPLITS:
        for c in partitions[split]:
            if c["label"] not in LABELS:
                raise ValueError("Uncertain or unknown class in a split")
            allowed = test_sources if split == "test" else development_sources | {sid for sid, assigned in fixed.items() if assigned == split}
            if c["source_id"] not in allowed:
                raise ValueError(f"Source leakage in {split}: {c['clip_id']}")
            if split != "test" and c["event_id"] in test_events:
                raise ValueError(f"Known test event in development: {c['clip_id']}")
            group = c.get("evaluation_group", c["event_id"])
            if group in group_owners and group_owners[group] != split:
                raise ValueError(f"Evaluation group crosses splits: {group}")
            group_owners[group] = split


def make_partition(clips, config):
    fixed = config.get("fixed_source_splits", {})
    development_sources = set(config["development_sources"])
    if set(fixed) & (development_sources | set(config["test_sources"])):
        raise ValueError("Fixed external sources must not override original sources")
    if any(split not in SPLITS for split in fixed.values()):
        raise ValueError("Invalid fixed source split")
    test_sources = set(config["test_sources"]) | {sid for sid, split in fixed.items() if split == "test"}
    if development_sources & test_sources:
        raise ValueError("Development and test sources must be disjoint")
    if development_sources | test_sources | set(fixed) != {c["source_id"] for c in clips}:
        raise ValueError("Configure all sources explicitly before rebuilding the split")
    fraction = config["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    groups = defaultdict(list)
    for c in clips:
        groups[c["event_id"]].append(c)
    partitions = {s: [] for s in SPLITS}
    excluded, development = [], []

    def exclude(c, reason, representative=None):
        excluded.append({
            "clip_id": c["clip_id"], "source_id": c["source_id"],
            "event_id": c["event_id"], "label": c["label"],
            "reason": reason, "selected_representative": representative,
        })

    for event_id in sorted(groups):
        group = sorted(groups[event_id], key=lambda c: c["clip_id"])
        labels = {c["label"] for c in group}
        if "uncertain" in labels or len(labels) != 1 or not labels <= set(LABELS):
            for c in group:
                exclude(c, "uncertain_or_conflicting_event_label")
            continue

        held_out = [c for c in group if c["source_id"] in test_sources]
        # Choose a test-source view first even if it was previously duplicate_of
        # a development-source clip. The development view is then excluded.
        chosen = (held_out or group)[0]
        row = {
            k: chosen[k] for k in (
                "clip_id", "path", "label", "source_id", "source_url",
                "event_id", "start_sec", "end_sec", "action_offset_sec",
                "label_status", "human_review_status",
            )
        }
        row["evaluation_group"] = chosen.get("evaluation_group", event_id)
        if held_out:
            partitions["test"].append(row)
        elif chosen["source_id"] in fixed:
            partitions[fixed[chosen["source_id"]]].append(row)
        else:
            development.append(row)
        for c in group:
            if c["clip_id"] != chosen["clip_id"]:
                reason = "known_test_event" if held_out and c["source_id"] not in test_sources else "repeated_event"
                exclude(c, reason, chosen["clip_id"])

    # Stable hash ordering avoids depending on input ordering or Python RNG versions.
    for label in LABELS:
        candidates = sorted(
            (c for c in development if c["label"] == label),
            key=lambda c: hashlib.sha256(f"{config['seed']}:{c['event_id']}".encode()).hexdigest(),
        )
        n_validation = min(len(candidates) - 1, max(1, round(len(candidates) * fraction))) if len(candidates) > 1 else 0
        partitions["validation"].extend(candidates[:n_validation])
        partitions["train"].extend(candidates[n_validation:])
    for split in SPLITS:
        partitions[split].sort(key=lambda c: c["clip_id"])
    excluded.sort(key=lambda c: c["clip_id"])
    validate_partition(clips, partitions, excluded, config)
    return partitions, excluded


def main():
    annotation_path = ROOT / "data/annotations.json"
    config_path = ROOT / "data/split_config.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    external_path = ROOT / "data/other_annotations.json"
    if external_path.is_file():
        external = json.loads(external_path.read_text(encoding="utf-8"))
        config["fixed_source_splits"] = external["fixed_source_splits"]
    partitions, excluded = make_partition(annotations["clips"], config)
    for rows in partitions.values():
        for c in rows:
            if not (ROOT / c["path"]).is_file():
                raise FileNotFoundError(c["path"])
    provenance = {
        "schema_version": 1,
        "status": "experimental",
        "annotations_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "external_annotations_sha256": hashlib.sha256(external_path.read_bytes()).hexdigest() if external_path.is_file() else None,
        "duplicate_audit_complete": False,
        "labels_review_complete": False,
        "suitable_for_final_quality_claim": False,
    }
    out = ROOT / "data/splits" / config["name"]
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in partitions.items():
        write_json(out / f"{split}.json", {**provenance, "split": split, "clips": rows})
    write_json(out / "excluded.json", {**provenance, "clips": excluded})
    summary = {
        **provenance,
        "config": config,
        "counts": {
            split: {"total": len(rows), "by_label": {label: sum(c["label"] == label for c in rows) for label in LABELS}}
            for split, rows in partitions.items()
        },
        "excluded": {"total": len(excluded), "by_reason": dict(Counter(c["reason"] for c in excluded))},
        "checks": {
            "every_input_clip_accounted_for": True,
            "selected_files_exist": True,
            "selected_event_ids_unique": True,
            "test_sources_absent_from_train_validation": True,
            "known_test_events_absent_from_train_validation": True,
            "evaluation_groups_do_not_cross_splits": True,
        },
        "limitations": [
            "Labels are provisional. Known event IDs may miss cross-compilation repeats or multiple events from the same match.",
            "Train and validation share development compilation sources; validation is not an independent-source estimate.",
            "Other class counts (train/validation/test): "
            + "/".join(str(sum(c["label"] == "other" for c in partitions[s])) for s in SPLITS)
            + ". HF windows from the same match are correlated; report support by match as well as by clip.",
            "External negatives differ in source, watermark and frame rate from the target-class compilations; this can create source shortcuts. Cross-dataset match overlap has not been fully audited.",
            "Two held-out compilations do not establish performance on arbitrary internet videos or continuous match footage.",
            "This is an explicit experimental partition of the conservative initial_four_compilations collection, not a completed match-level audit.",
        ],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"output": str(out), "counts": summary["counts"], "excluded": summary["excluded"]}, indent=2))


if __name__ == "__main__":
    main()
