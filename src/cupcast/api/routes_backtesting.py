from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from cupcast.prediction_engine.backtest_jobs import (
    API_BACKTEST_MODELS,
    BacktestJobStore,
    latest_completed_backtest_summary,
    run_backtest_job,
)

from .schemas import BacktestJobResponse, BacktestRunRequest
from .services import api_config_path, get_api_dataset_source


router = APIRouter(prefix="/backtesting", tags=["backtesting"])


@router.post("/run", response_model=BacktestJobResponse)
def run_backtesting(request: BacktestRunRequest, background_tasks: BackgroundTasks) -> BacktestJobResponse:
    store = BacktestJobStore()
    job = store.create()
    background_tasks.add_task(
        run_backtest_job,
        job.job_id,
        api_config_path(),
        request.train_before,
        request.test_tournament,
        API_BACKTEST_MODELS,
    )
    return BacktestJobResponse(**asdict(job))


@router.get("/jobs/{job_id}", response_model=BacktestJobResponse)
def get_backtesting_job(job_id: str) -> BacktestJobResponse:
    job = BacktestJobStore().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Backtest job not found: {job_id}")
    return BacktestJobResponse(**asdict(job))


@router.get("/summary")
def backtesting_summary() -> dict[str, object]:
    return latest_completed_backtest_summary(preferred_dataset_source=get_api_dataset_source())
