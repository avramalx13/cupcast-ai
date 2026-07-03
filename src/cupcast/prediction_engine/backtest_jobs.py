from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cupcast.shared.constants import DEFAULT_BACKTEST_JOBS_DIR, DEFAULT_COMPARISON_RESULTS_PATH

from .compare import compare_models

API_BACKTEST_MODELS = [
    "majority_baseline",
    "uniform_random_baseline",
    "elo_logistic_regression",
    "logistic_regression",
    "poisson_goal_model",
]

STALE_RUNNING_JOB_AFTER = timedelta(minutes=10)


@dataclass
class BacktestJob:
    job_id: str
    status: str
    created_at: str
    completed_at: str | None = None
    result_path: str | None = None
    error_message: str | None = None


class BacktestJobStore:
    def __init__(self, jobs_dir: str | Path = DEFAULT_BACKTEST_JOBS_DIR) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create(self) -> BacktestJob:
        job = BacktestJob(
            job_id=str(uuid.uuid4()),
            status="pending",
            created_at=datetime.now(UTC).isoformat(),
        )
        self.save(job)
        return job

    def save(self, job: BacktestJob) -> None:
        self._job_path(job.job_id).write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")

    def get(self, job_id: str) -> BacktestJob | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._mark_stale_if_needed(BacktestJob(**raw))

    def list_jobs(self) -> list[BacktestJob]:
        jobs = []
        for path in self.jobs_dir.glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not _is_job_record(raw):
                continue
            jobs.append(self._mark_stale_if_needed(BacktestJob(**raw)))
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def latest_completed(self, preferred_dataset_source: str | None = None) -> BacktestJob | None:
        jobs = self.list_jobs()
        if preferred_dataset_source:
            for job in jobs:
                if job.status == "completed" and job.result_path and self._result_source(job) == preferred_dataset_source:
                    return job
        for job in self.list_jobs():
            if job.status == "completed" and job.result_path:
                return job
        return None

    def result_for(self, job: BacktestJob) -> dict[str, Any] | None:
        if not job.result_path:
            return None
        path = Path(job.result_path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _mark_stale_if_needed(self, job: BacktestJob) -> BacktestJob:
        if job.status not in {"pending", "running"}:
            return job
        try:
            created_at = datetime.fromisoformat(job.created_at)
        except ValueError:
            return job
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        if now - created_at < STALE_RUNNING_JOB_AFTER:
            return job
        job.status = "failed"
        job.completed_at = now.isoformat()
        job.error_message = "Backtest job exceeded the 10 minute safety timeout. Run the fast dashboard backtest again."
        self.save(job)
        return job

    def _result_source(self, job: BacktestJob) -> str | None:
        result = self.result_for(job)
        if not isinstance(result, dict):
            return None
        source = result.get("dataset_source")
        return str(source) if source else None


def _is_job_record(raw: object) -> bool:
    if not isinstance(raw, dict):
        return False
    required_keys = {"job_id", "status", "created_at"}
    return required_keys.issubset(raw.keys())


def run_backtest_job(
    job_id: str,
    config_path: str | Path,
    train_before: int,
    test_tournament: str,
    model_names: list[str] | None = None,
    store: BacktestJobStore | None = None,
) -> None:
    store = store or BacktestJobStore()
    job = store.get(job_id)
    if job is None:
        return

    job.status = "running"
    store.save(job)
    try:
        result_path = store.jobs_dir / f"{job_id}_result.json"
        leaderboard_path = store.jobs_dir / f"{job_id}_leaderboard.json"
        compare_models(
            config_path=config_path,
            train_before=train_before,
            test_tournament=test_tournament,
            output_path=result_path,
            leaderboard_path=leaderboard_path,
            model_names=model_names or API_BACKTEST_MODELS,
        )
        job.status = "completed"
        job.completed_at = datetime.now(UTC).isoformat()
        job.result_path = str(result_path)
    except Exception as exc:  # pragma: no cover - defensive status capture
        job.status = "failed"
        job.completed_at = datetime.now(UTC).isoformat()
        job.error_message = str(exc)
    finally:
        store.save(job)


def latest_completed_backtest_summary(
    store: BacktestJobStore | None = None,
    include_manual_fallback: bool = True,
    preferred_dataset_source: str | None = None,
) -> dict[str, Any]:
    store = store or BacktestJobStore()
    job = store.latest_completed(preferred_dataset_source=preferred_dataset_source)
    if job is not None:
        return {"status": "completed", "job": asdict(job), "result": store.result_for(job)}
    if include_manual_fallback and DEFAULT_COMPARISON_RESULTS_PATH.exists():
        fallback_result = json.loads(DEFAULT_COMPARISON_RESULTS_PATH.read_text(encoding="utf-8"))
        if not preferred_dataset_source or fallback_result.get("dataset_source") == preferred_dataset_source:
            return {
                "status": "completed",
                "job": None,
                "result": fallback_result,
            }
    return {"status": "empty", "job": None, "result": None}
