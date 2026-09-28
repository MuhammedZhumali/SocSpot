import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from socspot.metrics import classification_metrics, select_thresholds
from socspot.search import merge_events, search_candidates
from socspot.video import windows


def test_confusion_matrix_and_macro_f1():
    report = classification_metrics([0, 0, 1, 2, 2], [0, 1, 1, 2, 0])
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    assert report["accuracy"] == .6
    assert report["classes"]["backheel_pass"]["precision"] == .5
    assert report["classes"]["bicycle_kick"]["recall"] == 1
    assert report["macro_f1"] == pytest.approx((.5+2/3+2/3)/3)


def test_precision_threshold_can_reject_false_positive():
    thresholds = select_thresholds([0, 0, 2, 1], [[.95,.03,.02], [.85,.1,.05], [.6,.1,.3], [.1,.85,.05]])
    assert .6 < thresholds["backheel_pass"] <= .85


def test_window_coverage_includes_short_tail():
    assert windows(1.2) == [(0, 1.2)]
    assert windows(5.5) == [(0,3), (1,4), (2,5), (2.5,5.5)]
    with pytest.raises(ValueError):
        windows(float("nan"))


def row(start, end, predicted="backheel_pass", score=.8):
    return {"start_sec":start, "end_sec":end, "predicted_label":predicted,
            "scores":{"backheel_pass":score,"bicycle_kick":.1,"other":.1}}


def test_merge_requires_target_prediction_and_does_not_join_separated_events():
    events = merge_events([row(8,11), row(0,3), row(1,4), row(4,7,"other",.9), row(5,8,score=.4)],
                          "backheel_pass", .7)
    assert [(r["start_sec"], r["end_sec"], r["positive_windows"]) for r in events] == [(0,4,2),(8,11,1)]


def test_discovery_excludes_dataset_sources_and_duplicates(monkeypatch):
    import socspot.search as search
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(search, "excluded_video_ids", lambda:{"aVL9JkMTbV8"})
    class Result:
        returncode = 0
        stdout = json.dumps({"entries":[{"id":"aVL9JkMTbV8"}, {"id":"abcdefghijk","title":"new"},
                                        {"id":"abcdefghijk"}, {"id":"../../unsafe"}]})
    monkeypatch.setattr(search.subprocess, "run", lambda *a,**k:Result())
    found = search_candidates("backheel_pass",3)
    assert [c["video_id"] for c in found["candidates"]] == ["abcdefghijk"]
    assert found["candidates"][0]["status"] == "candidate"


def test_api_rejects_invalid_request_before_starting_job():
    from socspot.app import app
    client = TestClient(app)
    for body in [{"action":"other"}, {"action":"bicycle_kick","limit":100},
                 {"action":"backheel_pass","seconds":36000}]:
        assert client.post("/api/search",json=body).status_code == 422
    assert client.get("/api/jobs/unknown").status_code == 404


def test_split_loader_rejects_match_overlap(tmp_path, monkeypatch):
    import socspot.train as train
    monkeypatch.setattr(train, "SPLITS", tmp_path)
    for i, part in enumerate(("train","validation","test")):
        (tmp_path/f"{part}.json").write_text(json.dumps({"clips":[{
            "clip_id":f"clip{i}","event_id":f"event{i}","evaluation_group":"same-match",
            "label":"other","path":"readme.md"}]}),encoding="utf-8")
    with pytest.raises(ValueError, match="Evaluation group leakage"):
        train.read_splits()


def test_analysis_uses_model_predictions_not_search_label(tmp_path):
    from unittest.mock import patch
    from socspot.search import analyze_video
    class Predictor:
        thresholds = {"bicycle_kick":.7}
        version = "test-model"
        def predict_window(self, *args):
            return np.array([.05,.05,.9])
    with patch("socspot.search.duration",return_value=5):
        result = analyze_video(Path("unused.mp4"),Predictor(),"bicycle_kick",5)
    assert result["events"] == []
    assert len(result["windows"]) == 3
    assert all(r["predicted_label"] == "other" for r in result["windows"])


def test_worker_preserves_download_failure_and_empty_model_results(tmp_path, monkeypatch):
    import threading
    import socspot.app as service
    import socspot.model as model_module
    monkeypatch.setattr(service,"ROOT",tmp_path)
    monkeypatch.setattr(service,"JOBS",tmp_path/"jobs")
    (tmp_path/"models").mkdir()
    (tmp_path/"models/current.json").write_text(json.dumps({"checkpoint":"model.pt"}))
    class Predictor:
        version = "model-test"
        thresholds = {"backheel_pass":.7}
        def __init__(self,*args): pass
    monkeypatch.setattr(model_module,"Predictor",Predictor)
    monkeypatch.setattr(service,"search_candidates",lambda *args:{"candidates":[
        {"video_id":"aaaaaaaaaaa","title":"ordinary play","url":"https://www.youtube.com/watch?v=aaaaaaaaaaa"},
        {"video_id":"bbbbbbbbbbb","title":"unavailable","url":"https://www.youtube.com/watch?v=bbbbbbbbbbb"}]})
    def download(video_id, seconds):
        if video_id == "bbbbbbbbbbb": raise RuntimeError("Unavailable")
        return tmp_path/"video.mp4",{}
    monkeypatch.setattr(service,"download_video",download)
    monkeypatch.setattr(service,"analyze_video",lambda *args:{"analyzed_end_sec":30,"events":[],"windows":[]})
    manager=service.JobManager()
    job_id="a"*32
    manager.jobs[job_id]={"id":job_id,"status":"queued","action":"backheel_pass","limit":2,"seconds":30}
    manager.cancel_flags[job_id]=threading.Event()
    manager.active=job_id
    manager.run(job_id)
    result=manager.get(job_id)
    assert result["status"]=="partial"
    assert result["results"]==[]
    assert [c["status"] for c in result["candidates"]]==["analyzed","failed"]
    assert result["processed"]==2 and manager.active is None
    manager.executor.shutdown()
