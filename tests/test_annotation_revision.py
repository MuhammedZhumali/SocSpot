"""Regression checks for annotation provenance and event-separated splits."""
import json
from pathlib import Path

import pytest

from scripts.split_dataset import make_partition, validate_partition

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sample(cid, source="dev", event=None, group=None, label="bicycle_kick"):
    return {"clip_id": cid, "source_id": source, "event_id": event or cid,
            "evaluation_group": group or event or cid, "label": label,
            "path": f"videos/clips/{cid}.mp4", "source_url": "",
            "start_sec": 0, "end_sec": 3, "action_offset_sec": 1,
            "label_status": "provisional", "human_review_status": "pending"}


def config():
    return {"development_sources": ["dev"], "test_sources": ["test"],
            "validation_fraction": .2, "seed": 42,
            "fixed_evaluation_group_splits": {"old_group": "validation"}}


def test_new_background_and_old_positive_keep_same_assignment():
    rows = [sample("old", group="old_group"), sample("background", group="old_group", label="other"),
            sample("new"), sample("holdout", source="test")]
    parts, excluded = make_partition(rows, config())
    assert {r["clip_id"] for r in parts["validation"]} >= {"old", "background"}
    assert not excluded
    assert make_partition(list(reversed(rows)), config()) == (parts, excluded)


def test_test_event_replay_and_related_background_are_purged_from_development():
    rows = [sample("dev_view", event="kick", group="match"),
            sample("test_view", source="test", event="kick", group="match"),
            sample("celebration", group="match", label="other"), sample("old", group="old_group")]
    parts, excluded = make_partition(rows, config())
    assert [r["clip_id"] for r in parts["test"]] == ["test_view"]
    assert {r["clip_id"] for r in excluded} == {"dev_view", "celebration"}


def test_conflicting_or_uncertain_replay_cannot_enter_training():
    rows = [sample("clear", event="same"), sample("unclear", event="same", label="uncertain"),
            sample("test", source="test")]
    parts, excluded = make_partition(rows, config())
    assert not parts["train"] and not parts["validation"]
    assert {r["clip_id"] for r in excluded} == {"clear", "unclear"}


def test_validator_rejects_test_group_even_if_test_view_is_uncertain():
    dev = sample("dev_window", group="match")
    held_out = sample("test_window", source="test", group="match", label="uncertain")
    with pytest.raises(ValueError, match="Known test evaluation group"):
        validate_partition([dev, held_out], {"train": [dev], "validation": [], "test": []}, [held_out], config())


def test_validator_rejects_moving_previously_selected_validation_group():
    rows = [sample("old", group="old_group"), sample("test", source="test")]
    parts, excluded = make_partition(rows, config())
    parts["train"] += parts["validation"]
    parts["validation"] = []
    with pytest.raises(ValueError, match="Previously assigned group moved"):
        validate_partition(rows, parts, excluded, config())


def test_revision_records_provenance_and_keeps_old_media_paths():
    before = {c["clip_id"]: c for c in read("data/annotation_versions/v1/annotations.json")["clips"]}
    after = read("data/annotations.json")["clips"]
    ledger = read("data/reviews/annotation-v2.json")
    ids = set(ledger["reviews"]) | {c["clip_id"] for c in ledger["additions"]}
    for c in after:
        assert c["start_sec"] < c["end_sec"]
        if c["action_sec"] is None:
            assert c["action_offset_sec"] is None
        else:
            assert c["start_sec"] <= c["action_sec"] < c["end_sec"]
            assert c["action_offset_sec"] == pytest.approx(c["action_sec"] - c["start_sec"])
        if c["clip_id"] in ids:
            assert c["path"].startswith("videos/clips/v2/")
            assert c["human_review_status"] == "pending"
            if c["clip_id"] in before:
                assert c["path"] != before[c["clip_id"]]["path"]
                for key, value in c["review"]["before"].items():
                    assert before[c["clip_id"]][key] == value
        else:
            assert c["path"] == before[c["clip_id"]]["path"]
            assert c["label"] == before[c["clip_id"]]["label"]


def test_current_manifests_preserve_baseline_groups_and_account_for_every_clip():
    owners = {}
    for split in ("train", "validation", "test"):
        for c in read(f"data/splits/initial_source_holdout/{split}.json")["clips"]:
            owners[c.get("evaluation_group", c["event_id"])] = split
    summary = read("data/splits/review_v2_source_holdout/summary.json")
    parts = {s: read(f"data/splits/review_v2_source_holdout/{s}.json")["clips"] for s in ("train", "validation", "test")}
    excluded = read("data/splits/review_v2_source_holdout/excluded.json")["clips"]
    validate_partition(read("data/annotations.json")["clips"], parts, excluded, summary["config"])
    for split, rows in parts.items():
        for c in rows:
            assert owners.get(c["evaluation_group"], split) == split
    assert summary["test_previously_evaluated"] is True


def test_added_negatives_do_not_overlap_target_windows():
    clips = read("data/annotations.json")["clips"]
    additions = {c["clip_id"] for c in read("data/reviews/annotation-v2.json")["additions"]}
    for negative in (c for c in clips if c["clip_id"] in additions):
        for positive in clips:
            if positive["source_id"] == negative["source_id"] and positive["label"] in {"bicycle_kick", "backheel_pass"}:
                assert min(positive["end_sec"], negative["end_sec"]) <= max(positive["start_sec"], negative["start_sec"])
