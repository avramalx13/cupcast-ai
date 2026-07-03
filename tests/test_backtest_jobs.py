from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from cupcast.prediction_engine.backtest_jobs import (
    BacktestJob,
    BacktestJobStore,
    latest_completed_backtest_summary,
)
from scripts.create_sample_data import main as create_sample_data


def test_backtest_store_returns_empty_summary_with_no_completed_jobs(tmp_path) -> None:
    summary = latest_completed_backtest_summary(
        BacktestJobStore(tmp_path),
        include_manual_fallback=False,
    )

    assert summary["status"] == "empty"
    assert summary["result"] is None


def test_backtest_store_ignores_result_artifacts(tmp_path) -> None:
    (tmp_path / "example_leaderboard.json").write_text('{"models": []}', encoding="utf-8")
    (tmp_path / "example_result.json").write_text('{"models": []}', encoding="utf-8")
    store = BacktestJobStore(tmp_path)

    assert store.list_jobs() == []


def test_backtest_store_marks_stale_running_jobs_failed(tmp_path) -> None:
    store = BacktestJobStore(tmp_path)
    job = BacktestJob(
        job_id="stale",
        status="running",
        created_at=(datetime.now(UTC) - timedelta(minutes=11)).isoformat(),
    )
    store.save(job)

    loaded = store.get("stale")

    assert loaded is not None
    assert loaded.status == "failed"
    assert "safety timeout" in str(loaded.error_message)


def test_backtest_summary_prefers_matching_dataset_source(tmp_path) -> None:
    store = BacktestJobStore(tmp_path)
    synthetic_job = BacktestJob(
        job_id="synthetic",
        status="completed",
        created_at="2026-01-02T00:00:00+00:00",
        completed_at="2026-01-02T00:00:01+00:00",
        result_path=str(tmp_path / "synthetic_result.json"),
    )
    real_job = BacktestJob(
        job_id="real",
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        result_path=str(tmp_path / "real_result.json"),
    )
    store.save(synthetic_job)
    store.save(real_job)
    (tmp_path / "synthetic_result.json").write_text(json.dumps({"dataset_source": "synthetic"}), encoding="utf-8")
    (tmp_path / "real_result.json").write_text(json.dumps({"dataset_source": "international-results"}), encoding="utf-8")

    summary = latest_completed_backtest_summary(
        store,
        include_manual_fallback=False,
        preferred_dataset_source="international-results",
    )

    assert summary["job"] == asdict(real_job)
    assert summary["result"] == {"dataset_source": "international-results"}


def test_backtesting_api_creates_and_reads_job() -> None:
    create_sample_data()
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.post(
        "/backtesting/run",
        json={"train_before": 2022, "test_tournament": "World Cup 2022"},
    )
    assert response.status_code == 200
    job = response.json()

    status_response = client.get(f"/backtesting/jobs/{job['job_id']}")

    assert status_response.status_code == 200
    assert status_response.json()["status"] in {"pending", "running", "completed", "failed"}


def test_backtesting_api_returns_404_for_invalid_job_id() -> None:
    from cupcast.api.main import app

    client = TestClient(app)
    response = client.get("/backtesting/jobs/not-a-real-job")

    assert response.status_code == 404
