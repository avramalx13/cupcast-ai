from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import pandas as pd

from cupcast.analyst.explanation_service import ExplanationService
from cupcast.prediction_engine.data_loader import load_dataset
from cupcast.prediction_engine.data_sources import load_player_events, validate_dataset_files
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model import PredictionModel, load_model, train_prediction_model
from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import (
    DEFAULT_MATCHES_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SNAPSHOT_PATH,
    DEFAULT_TEAMS_PATH,
    PROJECT_ROOT,
)


DEFAULT_API_CONFIG_PATH = PROJECT_ROOT / "configs" / "prediction_model.yaml"
REAL_API_CONFIG_PATH = PROJECT_ROOT / "configs" / "prediction_model_real.yaml"
REAL_TEAMS_PATH = PROJECT_ROOT / "data" / "processed" / "teams_real.csv"
REAL_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "real_completed_matches.csv"
REAL_MODEL_PATH = PROJECT_ROOT / "models" / "prediction_model_real.joblib"
FULL_TOURNAMENT_JSON_PATH = PROJECT_ROOT / "models" / "full_tournament_simulation.json"
FULL_TOURNAMENT_MD_PATH = PROJECT_ROOT / "models" / "full_tournament_simulation.md"


def api_config_path() -> Path:
    configured_path = os.environ.get("CUPCAST_API_CONFIG")
    if configured_path:
        return resolve_project_path(configured_path, PROJECT_ROOT)
    if _real_api_artifacts_available():
        return REAL_API_CONFIG_PATH
    return DEFAULT_API_CONFIG_PATH


def _real_api_artifacts_available() -> bool:
    return all(
        path.exists()
        for path in (
            REAL_API_CONFIG_PATH,
            REAL_MATCHES_PATH,
            REAL_TEAMS_PATH,
            REAL_MODEL_PATH,
        )
    )


@lru_cache(maxsize=1)
def get_api_config() -> dict[str, object]:
    try:
        config = load_yaml(api_config_path())
    except FileNotFoundError:
        config = {}
    return config


@lru_cache(maxsize=1)
def get_data_config() -> dict[str, object]:
    config = get_api_config()
    return dict(config.get("data", {})) if isinstance(config.get("data", {}), dict) else {}


@lru_cache(maxsize=1)
def get_model_config() -> dict[str, object]:
    config = get_api_config()
    return dict(config.get("model", {})) if isinstance(config.get("model", {}), dict) else {}


def _resolve_data_paths() -> tuple[str, Path, Path | None, Path | None, Path | None]:
    data_cfg = get_data_config()
    mode = str(data_cfg.get("mode") or data_cfg.get("source") or "synthetic")
    matches_path = resolve_project_path(
        data_cfg.get("matches_path", DEFAULT_MATCHES_PATH),
        PROJECT_ROOT,
    )
    teams_value = data_cfg.get("teams_path", DEFAULT_TEAMS_PATH)
    teams_path = resolve_project_path(teams_value, PROJECT_ROOT) if teams_value else None
    shootouts_value = data_cfg.get("shootouts_path")
    shootouts_path = resolve_project_path(shootouts_value, PROJECT_ROOT) if shootouts_value else None
    player_events_value = data_cfg.get("player_events_path") or data_cfg.get("goalscorers_path")
    goalscorers_path = resolve_project_path(player_events_value, PROJECT_ROOT) if player_events_value else None
    source_type = "international-results" if mode in {"real", "international-results"} else "csv"
    return source_type, matches_path, teams_path, shootouts_path, goalscorers_path


def _resolve_model_path() -> Path:
    model_cfg = get_model_config()
    return resolve_project_path(model_cfg.get("output_path", DEFAULT_MODEL_PATH), PROJECT_ROOT)


def get_api_dataset_source() -> str:
    source_type, _matches_path, _teams_path, _shootouts_path, _goalscorers_path = _resolve_data_paths()
    return source_type


@lru_cache(maxsize=1)
def get_matches():
    source_type, matches_path, teams_path, shootouts_path, _goalscorers_path = _resolve_data_paths()
    matches, _teams = load_dataset(
        matches_path,
        teams_path,
        source_type=source_type,
        shootouts_path=shootouts_path,
    )
    return matches


@lru_cache(maxsize=1)
def get_teams():
    source_type, matches_path, teams_path, shootouts_path, _goalscorers_path = _resolve_data_paths()
    _matches, teams = load_dataset(
        matches_path,
        teams_path,
        source_type=source_type,
        shootouts_path=shootouts_path,
    )
    return teams


@lru_cache(maxsize=1)
def get_prediction_model() -> PredictionModel:
    model_path = _resolve_model_path()
    try:
        return load_model(model_path)
    except FileNotFoundError:
        source_type, _matches_path, _teams_path, _shootouts_path, goalscorers_path = _resolve_data_paths()
        player_events = load_player_events(goalscorers_path)
        features = build_feature_table(get_matches(), get_teams(), player_events=player_events)
        return train_prediction_model(features, model_version="api-in-memory")


@lru_cache(maxsize=1)
def get_explanation_service() -> ExplanationService:
    return ExplanationService()


@lru_cache(maxsize=1)
def get_full_tournament_context():
    if REAL_MODEL_PATH.exists() and REAL_MATCHES_PATH.exists() and REAL_TEAMS_PATH.exists():
        matches, teams = load_dataset(
            REAL_MATCHES_PATH,
            REAL_TEAMS_PATH,
            source_type="international-results",
        )
        return load_model(REAL_MODEL_PATH), matches, teams, "real"
    return get_prediction_model(), get_matches(), get_teams(), "api-config"


def snapshot_path():
    return DEFAULT_SNAPSHOT_PATH


def full_tournament_report_paths() -> tuple[Path, Path]:
    return FULL_TOURNAMENT_JSON_PATH, FULL_TOURNAMENT_MD_PATH


def get_dataset_status() -> dict[str, object]:
    source_type, matches_path, teams_path, shootouts_path, goalscorers_path = _resolve_data_paths()
    mode = "real" if source_type == "international-results" else "synthetic"
    if not matches_path.exists():
        return {
            "mode": mode,
            "matches_loaded": 0,
            "date_range": [None, None],
            "teams_loaded": 0,
            "last_validation_valid": False,
            "warnings": [f"Matches file not found: {matches_path}"],
        }
    report = validate_dataset_files(
        matches_path=matches_path,
        teams_path=teams_path,
        shootouts_path=shootouts_path,
        goalscorers_path=goalscorers_path,
        source_type=source_type,
    )
    return {
        "mode": mode,
        "matches_loaded": report.matches_count,
        "date_range": [report.date_min, report.date_max],
        "teams_loaded": report.teams_count,
        "last_validation_valid": report.is_valid,
        "warnings": report.warnings + report.errors,
    }


def get_team_directory() -> dict[str, object]:
    if REAL_TEAMS_PATH.exists():
        teams = pd.read_csv(REAL_TEAMS_PATH)
        return {"teams": _team_records(teams), "source": "real"}
    return {"teams": _team_records(get_teams()), "source": "synthetic"}


def _team_records(teams) -> list[dict[str, object]]:
    if teams is None or teams.empty or "team" not in teams:
        return []
    frame = teams.copy()
    frame["team"] = frame["team"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["team"].ne("")]
    if frame.empty:
        return []
    for column, default in [
        ("confederation", "UNKNOWN"),
        ("match_count", None),
        ("first_match_date", None),
        ("last_match_date", None),
    ]:
        if column not in frame:
            frame[column] = default
    frame["confederation"] = frame["confederation"].fillna("UNKNOWN").astype(str).str.strip()
    frame["match_count"] = pd.to_numeric(frame["match_count"], errors="coerce")
    frame = frame.sort_values(["team", "match_count"], ascending=[True, False]).drop_duplicates("team")
    records = []
    for row in frame.sort_values("team").itertuples(index=False):
        match_count = getattr(row, "match_count", None)
        records.append(
            {
                "name": str(row.team),
                "confederation": str(getattr(row, "confederation", "UNKNOWN") or "UNKNOWN"),
                "match_count": None if pd.isna(match_count) else int(match_count),
                "first_match_date": _optional_text(getattr(row, "first_match_date", None)),
                "last_match_date": _optional_text(getattr(row, "last_match_date", None)),
            }
        )
    return records


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
