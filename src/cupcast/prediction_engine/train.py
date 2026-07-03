from __future__ import annotations

import argparse
from pathlib import Path

from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT

from .data_loader import load_dataset
from .features import build_feature_table
from .model import save_model, train_prediction_model
from .rating_sources import LocalEloRatingSource, LocalFifaRankingSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CupCast structured prediction model")
    parser.add_argument("--config", default="configs/prediction_model.yaml")
    parser.add_argument("--output", default=None, help="Override model output path")
    return parser.parse_args()


def train_from_config(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    feature_cfg = config.get("features", {})
    calibration_cfg = config.get("calibration", {})
    ensemble_cfg = config.get("ensemble", {})

    source_type = str(data_cfg.get("mode") or data_cfg.get("source") or "csv")
    matches_path = resolve_project_path(data_cfg.get("matches_path", "data/raw/historical_matches.csv"), PROJECT_ROOT)
    teams_value = data_cfg.get("teams_path", "data/raw/teams.csv")
    teams_path = resolve_project_path(teams_value, PROJECT_ROOT) if teams_value else None
    shootouts_value = data_cfg.get("shootouts_path")
    shootouts_path = resolve_project_path(shootouts_value, PROJECT_ROOT) if shootouts_value else None
    output_path = resolve_project_path(model_cfg.get("output_path", "models/prediction_model.joblib"), PROJECT_ROOT)
    recent_window = int(feature_cfg.get("recent_window", 5))
    elo_k = float(feature_cfg.get("elo_k", 28.0))
    elo_ratings_path = _optional_path(data_cfg.get("elo_ratings_path") or data_cfg.get("elo_path"))
    fifa_rankings_path = _optional_path(data_cfg.get("fifa_rankings_path") or data_cfg.get("fifa_path"))

    matches, teams = load_dataset(
        matches_path,
        teams_path,
        source_type=source_type,
        shootouts_path=shootouts_path,
    )
    features = build_feature_table(
        matches,
        teams,
        recent_window=recent_window,
        elo_k=elo_k,
        external_elo_ratings=LocalEloRatingSource(elo_ratings_path).load() if elo_ratings_path else None,
        external_fifa_rankings=LocalFifaRankingSource(fifa_rankings_path).load() if fifa_rankings_path else None,
        feature_flags=dict(feature_cfg.get("enable_groups", {})) if isinstance(feature_cfg.get("enable_groups"), dict) else {},
    )
    model = train_prediction_model(
        features,
        model_type=str(model_cfg.get("type", "logistic_regression")),
        random_state=int(model_cfg.get("random_state", 42)),
        max_iter=int(model_cfg.get("max_iter", 1000)),
        recent_window=recent_window,
        elo_k=elo_k,
        model_version=f"{Path(config_path).stem}-local",
        model_params=_model_params(calibration_cfg, ensemble_cfg),
    )
    save_model(model, output_path)
    return output_path


def _optional_path(value: object) -> Path | None:
    if not value:
        return None
    candidate = resolve_project_path(value, PROJECT_ROOT)
    return candidate if candidate.exists() else None


def _model_params(calibration_cfg: object, ensemble_cfg: object) -> dict[str, object]:
    calibration = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    ensemble = ensemble_cfg if isinstance(ensemble_cfg, dict) else {}
    params: dict[str, object] = {
        "calibration_method": calibration.get("method", "sigmoid"),
        "calibration_cv": calibration.get("cv", 3),
        "ensemble_mode": ensemble.get("mode", "validation_weighted"),
    }
    if isinstance(ensemble.get("candidates"), list):
        params["ensemble_candidates"] = ensemble["candidates"]
    return params


def main() -> None:
    args = parse_args()
    output_path = train_from_config(args.config)
    if args.output:
        configured_path = output_path
        output_path = Path(args.output)
        output_path.write_bytes(configured_path.read_bytes())
    print(f"model={output_path}")


if __name__ == "__main__":
    main()
