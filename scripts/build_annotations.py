"""Build readable annotations and a local review page from visual review decisions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from annotation_revision import apply_revision

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    source_map = {s["source_id"]: s for s in sources}
    decisions = json.loads((ROOT / "data/review_decisions.json").read_text(encoding="utf-8"))
    clips, first_by_event = [], {}
    for source in sources:
        sid = source["source_id"]
        for number, start, end, action, label, note, *shared in decisions[sid]:
            clip_id = f"{sid}_{number:03d}"
            event_id = shared[0] if shared else f"event_{clip_id}"
            duplicate_of = first_by_event.get(event_id)
            first_by_event.setdefault(event_id, clip_id)
            clips.append({
                "clip_id": clip_id, "source_id": sid, "source_path": source["path"],
                "source_url": source["youtube_url"], "start_sec": start, "end_sec": end,
                "duration_sec": round(end - start, 3), "action_sec": action,
                "action_offset_sec": None if action is None else round(action - start, 3), "label": label,
                "proposed_label": "bicycle_kick" if label == "uncertain" and source["collection_label"] == "bicycle_kick" else label,
                "label_status": "provisional", "review_method": "assistant_contact_sheets_0.5_sec",
                "human_review_status": "pending", "event_id": event_id,
                "duplicate_of": duplicate_of,
                "group_status": "provisional_visual_grouping",
                "split": None, "split_group": "initial_four_compilations",
                "training_candidate": label != "uncertain" and duplicate_of is None,
                "short_context": end - start < 2.5, "notes": note, "export": True,
                "path": f"videos/clips/{label}/{clip_id}.mp4",
                "evidence_path": f"videos/review/{sid}/evidence/page_{(number - 1) // 3 + 1:03d}.jpg",
            })
    other_path = ROOT / "data/other_annotations.json"
    if other_path.is_file():
        other = json.loads(other_path.read_text(encoding="utf-8"))
        sources.extend(other["sources"])
        clips.extend(other["clips"])
        source_map.update({s["source_id"]: s for s in other["sources"]})
    clips, revision = apply_revision(clips, sources)
    ids = [c["clip_id"] for c in clips]
    assert len(ids) == len(set(ids)), "Duplicate clip IDs"
    for c in clips:
        assert 0 <= c["start_sec"] < c["end_sec"] <= source_map[c["source_id"]]["duration_sec"]
        if c["action_sec"] is not None:
            assert c["start_sec"] <= c["action_sec"] < c["end_sec"]
    doc = {
        "schema_version": 2, "annotation_version": "v2", "taxonomy_version": "v2",
        "split_config_path": "data/split_config_v2.json",
        "revision_summary": {k: v for k, v in revision.items() if k != "changes"},
        "dataset_name": "SocSpot local action clips and external negatives",
        "annotation_date": "2026-09-28", "status": "draft_for_review",
        "taxonomy": {
            "backheel_pass": "Передача мяча партнёру пяткой; удары, ведение и передачи плечом сюда не входят.",
            "bicycle_kick": "Удар через себя или боковые ножницы в прыжке. Оба вида включены по решению пользователя; подтип уточнённых боковых ударов — side_scissors.",
            "other": "Другие действия: обычные удары, передачи, борьба, паузы и несколько похожих действий. Дополнение из Hugging Face просмотрено отдельно; метки предварительные.",
            "uncertain": "Очередь проверки: неясное действие или граница класса. Это не четвёртый класс для обучения.",
        },
        "method": {
            "overview_sample_interval_sec": 2, "detail_sample_interval_sec": 0.5,
            "revision_sample_interval_sec": 0.125,
            "action_time_precision": "Approximate contact-sheet estimates: initial review 0.5 s, revised clips 0.125 s. Occlusion, blends and slow motion limit accuracy; decimal places are not frame-exact ground truth.",
            "coverage": "Selected action windows from four edited compilations; not exhaustive temporal annotation.",
            "excluded": "Many replays and editorial transitions remain omitted. V2 adds two intros and four celebrations as negatives. smNANjh2t2Q seed 97 repeats the end of seed 96 and is not exported.",
            "identity_notes": "Player/team names are visual navigation notes from footage and captions, not verified identity metadata.",
            "external_negatives": "Selected HF windows reviewed at 0.5-second intervals. Grouped by match metadata; action timestamps are null for background windows.",
        },
        "split_policy": {
            "assigned": False, "duplicate_audit_complete": False,
            "reason": "Known and suspected repeats are grouped. Cross-video/match audit is incomplete. All four compilations share one conservative split_group; do not randomly divide their clips for evaluation.",
            "next_step": "After label review, use this batch for a prototype and collect independent matches/events for validation and test, or complete the match/duplicate audit before assigning splits.",
            "experimental_manifests": "data/splits/review_v2_source_holdout; existing group assignments and HF match splits are preserved",
            "test_previously_evaluated": True,
        },
        "sources": sources,
        "stats": {
            "clips": len(clips), "by_label": dict(Counter(c["label"] for c in clips)),
            "by_source": dict(Counter(c["source_id"] for c in clips)),
            "training_candidates_by_label": dict(Counter(c["label"] for c in clips if c["training_candidate"])),
            "linked_repeats_or_possible_repeats": sum(c["duplicate_of"] is not None for c in clips),
            "provisional_event_groups": len({c["event_id"] for c in clips}),
            "total_clip_duration_sec": round(sum(c["duration_sec"] for c in clips), 2),
        },
        "clips": clips,
    }
    write_json(ROOT / "data/annotations.json", doc)
    write_json(ROOT / "data/dataset_summary.json", doc["stats"])
    write_json(ROOT / "data/reviews/annotation-v2-summary.json", revision)
    template = (ROOT / "scripts/review_template.html").read_text(encoding="utf-8")
    # Embed local metadata so the page also works directly via file://.
    payload = json.dumps(doc, ensure_ascii=False).replace("<", "\\u003c")
    (ROOT / "dataset-review.html").write_text(template.replace("__DATASET_JSON__", payload), encoding="utf-8")
    print(json.dumps(doc["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
