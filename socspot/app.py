"""Local API + web UI. One worker serializes GPU work and limits downloads."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .search import analyze_video, download_video, search_candidates, VIDEO_ID
from .video import ROOT, write_json

JOBS = ROOT / "data/search/jobs"


def now():
    return datetime.now(timezone.utc).isoformat()


class SearchRequest(BaseModel):
    action: Literal["backheel_pass", "bicycle_kick"]
    limit: int = Field(default=3, ge=1, le=5)
    seconds: int = Field(default=60, ge=10, le=180)


class JobManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="socspot-search")
        self.jobs = {}
        self.active = None
        self.cancel_flags = {}
        self.predictor = None

    def update(self, job_id, **fields):
        with self.lock:
            self.jobs[job_id].update(fields, updated_utc=now())
            write_json(JOBS/f"{job_id}.json", self.jobs[job_id])

    def get(self, job_id):
        if not is_job_id(job_id):
            raise HTTPException(404, "Задача не найдена")
        with self.lock:
            if job_id in self.jobs:
                return json.loads(json.dumps(self.jobs[job_id]))
            path = JOBS/f"{job_id}.json"
            if not path.is_file():
                raise HTTPException(404, "Задача не найдена")
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] not in {"done", "partial", "failed", "cancelled", "interrupted"}:
                job.update(status="interrupted", message="Сервер был перезапущен; запустите поиск заново")
            return job

    def start(self, request):
        if not (ROOT/"models/current.json").is_file():
            raise HTTPException(503, "Модель ещё не готова")
        with self.lock:
            if self.active:
                raise HTTPException(409, "Дождитесь завершения текущего поиска или отмените его")
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {"id":job_id, **request.model_dump(), "status":"queued",
                                 "created_utc":now(), "updated_utc":now(), "message":"Поиск в очереди",
                                 "candidates":[], "results":[], "processed":0}
            self.cancel_flags[job_id] = threading.Event()
            self.active = job_id
            self.update(job_id)
            self.executor.submit(self.run, job_id)
        return self.get(job_id)

    def run(self, job_id):
        try:
            from .model import Predictor
            import torch
            torch.set_num_threads(4)
            current = json.loads((ROOT/"models/current.json").read_text(encoding="utf-8"))
            if self.predictor is None or self.checkpoint != current["checkpoint"]:
                self.predictor = Predictor(ROOT/current["checkpoint"])
                self.checkpoint = current["checkpoint"]
            job = self.get(job_id)
            cancelled = self.cancel_flags[job_id].is_set
            self.update(job_id, status="searching", message="Ищу новые видео на YouTube",
                        model_version=self.predictor.version, threshold=self.predictor.thresholds[job["action"]])
            discovery = search_candidates(job["action"], job["limit"])
            if cancelled():
                raise InterruptedError()
            candidates = discovery["candidates"]
            self.update(job_id, **discovery)
            results = []
            for index, candidate in enumerate(candidates):
                if cancelled():
                    raise InterruptedError()
                self.update(job_id, status="downloading", message=f"Получаю видео {index+1}/{len(candidates)}",
                            current_video_id=candidate["video_id"], windows_done=0, windows_total=0)
                try:
                    path, metadata = download_video(candidate["video_id"], job["seconds"])
                    candidate.update(local_path=path.relative_to(ROOT).as_posix(), media=metadata)
                    self.update(job_id, status="analyzing", message=f"Проверяю моделью: {candidate['title']}")

                    def progress(done, total):
                        self.update(job_id, windows_done=done, windows_total=total)

                    analyzed = analyze_video(path, self.predictor, job["action"], job["seconds"], progress, cancelled)
                    # Keep per-window scores in a separate audit record, not in every poll response.
                    audit = ROOT/"data/search/analyses"/f"{job_id}-{candidate['video_id']}.json"
                    write_json(audit, analyzed)
                    candidate.update(status="analyzed", analyzed_end_sec=analyzed["analyzed_end_sec"],
                                     event_count=len(analyzed["events"]), audit_path=audit.relative_to(ROOT).as_posix())
                    for event in analyzed["events"]:
                        results.append({**event, "video_id":candidate["video_id"], "title":candidate["title"],
                                        "url":candidate["url"]+f"&t={int(event['start_sec'])}s",
                                        "preview_url":f"/api/jobs/{job_id}/video/{candidate['video_id']}",
                                        "model_version":self.predictor.version})
                except InterruptedError:
                    raise
                except Exception as exc:
                    candidate.update(status="failed", error=str(exc)[:500])
                self.update(job_id, candidates=candidates, results=results, processed=index+1)
            failed = sum(c["status"] == "failed" for c in candidates)
            state = "failed" if candidates and failed == len(candidates) else "partial" if failed else "done"
            message = f"Проверено видео: {len(candidates)-failed}. Найдено фрагментов: {len(results)}."
            if state == "failed":
                message = "Видео найдены, но получить их для проверки не удалось"
            elif not candidates:
                message = "Поиск не вернул новых доступных кандидатов"
            self.update(job_id, status=state, message=message)
        except InterruptedError:
            self.update(job_id, status="cancelled", message="Поиск отменён; завершённые результаты сохранены")
        except Exception as exc:
            self.update(job_id, status="failed", message=str(exc)[:500])
        finally:
            with self.lock:
                self.active = None


def is_job_id(value):
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


manager = JobManager()
app = FastAPI(title="SocSpot", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.get("/")
def index():
    return FileResponse(ROOT/"web/index.html")


@app.get("/api/health")
def health():
    current_path = ROOT/"models/current.json"
    ready = current_path.is_file()
    if ready:
        current = json.loads(current_path.read_text(encoding="utf-8"))
        ready = (ROOT/current["checkpoint"]).is_file()
    return {"model_ready":ready, "active_job":manager.active, "status":"experimental"}


@app.get("/api/model")
def model_report():
    if not (ROOT/"models/current.json").is_file():
        raise HTTPException(503, "Модель ещё не готова")
    current = json.loads((ROOT/"models/current.json").read_text(encoding="utf-8"))
    return json.loads((ROOT/current["report"]).read_text(encoding="utf-8"))


@app.post("/api/search", status_code=202)
def start_search(request: SearchRequest):
    return manager.start(request)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return manager.get(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    manager.get(job_id)
    with manager.lock:
        if manager.active == job_id:
            manager.cancel_flags[job_id].set()
            manager.update(job_id, message="Отмена после текущей загрузки или окна анализа")
    return manager.get(job_id)


@app.get("/api/jobs/{job_id}/video/{video_id}")
def preview_video(job_id: str, video_id: str):
    if not VIDEO_ID.fullmatch(video_id):
        raise HTTPException(404)
    job = manager.get(job_id)
    candidate = next((c for c in job["candidates"] if c["video_id"] == video_id), None)
    if not candidate or not candidate.get("local_path"):
        raise HTTPException(404)
    path = (ROOT/candidate["local_path"]).resolve()
    if not path.is_relative_to((ROOT/"data/search/videos").resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
