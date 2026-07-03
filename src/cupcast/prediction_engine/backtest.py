from __future__ import annotations

import argparse
from pathlib import Path

from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT

from .data_loader import load_dataset
from .evaluate import evaluate_model
from .features import build_feature_table
from .model import train_prediction_model


def run_backtest(
    train_before: int,
    test_tournament: str,
    matches_path: str | Path = PROJECT_ROOT / "data" / "raw" / "historical_matches.csv",
    teams_path: str | Path = PROJECT_ROOT / "data" / "raw" / "teams.csv",
    source_type: str = "csv",
    shootouts_path: str | Path | None = None,
    recent_window: int = 5,
    elo_k: float = 28.0,
    model_type: str = "logistic_regression",
) -> dict[str, object]:
    matches, teams = load_dataset(
        matches_path,
        teams_path,
        source_type=source_type,
        shootouts_path=shootouts_path,
    )
    features = build_feature_table(matches, teams, recent_window=recent_window, elo_k=elo_k)
    test_mask = features["tournament"].astype(str).str.lower() == test_tournament.lower()
    train_mask = (features["date"].dt.year < int(train_before)) & ~test_mask
    train_features = features.loc[train_mask].reset_index(drop=True)
    test_features = features.loc[test_mask].reset_index(drop=True)
    if train_features.empty:
        raise ValueError(f"No training matches found before {train_before}")
    if test_features.empty:
        raise ValueError(f"No matches found for tournament {test_tournament!r}")

    model = train_prediction_model(
        train_features,
        model_type=model_type,
        recent_window=recent_window,
        elo_k=elo_k,
        model_version=f"backtest-before-{train_before}",
    )
    metrics = evaluate_model(model, test_features)
    metrics["train_matches"] = int(len(train_features))
    metrics["test_tournament"] = test_tournament
    metrics["model_version"] = model.model_version
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the CupCast prediction model")
    parser.add_argument("--config", default="configs/prediction_model.yaml")
    parser.add_argument("--train-before", type=int, default=2022)
    parser.add_argument("--test-tournament", default="World Cup 2022")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})
    model_cfg = config.get("model", {})
    metrics = run_backtest(
        train_before=args.train_before,
        test_tournament=args.test_tournament,
        matches_path=resolve_project_path(data_cfg.get("matches_path", "data/raw/historical_matches.csv"), PROJECT_ROOT),
        teams_path=resolve_project_path(data_cfg.get("teams_path", "data/raw/teams.csv"), PROJECT_ROOT),
        source_type=str(data_cfg.get("mode") or data_cfg.get("source") or "csv"),
        shootouts_path=(
            resolve_project_path(data_cfg["shootouts_path"], PROJECT_ROOT)
            if data_cfg.get("shootouts_path")
            else None
        ),
        recent_window=int(feature_cfg.get("recent_window", 5)),
        elo_k=float(feature_cfg.get("elo_k", 28.0)),
        model_type=str(model_cfg.get("type", "logistic_regression")),
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
