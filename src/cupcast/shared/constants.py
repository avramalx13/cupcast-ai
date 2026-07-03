from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATCHES_PATH = PROJECT_ROOT / "data" / "raw" / "historical_matches.csv"
DEFAULT_TEAMS_PATH = PROJECT_ROOT / "data" / "raw" / "teams.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "prediction_model.joblib"
DEFAULT_COMPARISON_RESULTS_PATH = PROJECT_ROOT / "models" / "comparison_results.json"
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "processed" / "probability_snapshots.csv"
DEFAULT_BACKTEST_JOBS_DIR = PROJECT_ROOT / "data" / "processed" / "backtest_jobs"

OUTCOME_LABELS = ("A_WIN", "DRAW", "B_WIN")
