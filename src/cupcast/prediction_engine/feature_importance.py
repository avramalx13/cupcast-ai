from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT

from .data_loader import load_dataset
from .features import build_feature_table
from .model import train_prediction_model
from .rating_sources import LocalEloRatingSource, LocalFifaRankingSource


DEFAULT_IMPORTANCE_MODELS = ["random_forest", "gradient_boosting", "logistic_regression"]


def analyze_feature_importance(
    config_path: str | Path,
    output_json: str | Path = PROJECT_ROOT / "models" / "feature_importance.json",
    output_md: str | Path = PROJECT_ROOT / "models" / "feature_importance.md",
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    features = _features_from_config(config)
    rows = []
    for model_name in model_names or DEFAULT_IMPORTANCE_MODELS:
        try:
            model = train_prediction_model(features, model_type=model_name)
            rows.append(_importance_for_model(model_name, model.estimator, model.feature_columns))
        except Exception as exc:
            rows.append({"model_name": model_name, "status": "failed", "error_message": str(exc), "features": []})
    payload = {"models": rows}
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(output_md).write_text(render_feature_importance_markdown(payload), encoding="utf-8")
    return payload


def render_feature_importance_markdown(payload: dict[str, Any]) -> str:
    lines = ["# CupCast AI Feature Importance", ""]
    for model in payload.get("models", []):
        lines.extend([f"## {model.get('model_name')}", ""])
        if model.get("status") != "ok":
            lines.extend([f"Unavailable: {model.get('error_message')}", ""])
            continue
        lines.extend(["| Feature | Importance |", "|---|---:|"])
        for row in model.get("features", [])[:20]:
            lines.append(f"| `{row['feature']}` | {float(row['importance']):.6f} |")
        lines.append("")
    return "\n".join(lines)


def _importance_for_model(model_name: str, estimator: Any, feature_columns: list[str]) -> dict[str, Any]:
    raw_estimator = estimator
    if hasattr(raw_estimator, "named_steps"):
        raw_estimator = raw_estimator.named_steps.get("classifier", raw_estimator)
    if hasattr(raw_estimator, "feature_importances_"):
        values = list(raw_estimator.feature_importances_)
    elif hasattr(raw_estimator, "coef_"):
        values = list(pd.DataFrame(raw_estimator.coef_).abs().mean(axis=0))
    else:
        return {
            "model_name": model_name,
            "status": "unsupported",
            "error_message": "Model does not expose feature_importances_ or coef_",
            "features": [],
        }
    feature_rows = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(feature_columns, values, strict=False)
    ]
    feature_rows.sort(key=lambda row: row["importance"], reverse=True)
    return {"model_name": model_name, "status": "ok", "features": feature_rows}


def _features_from_config(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})
    mode = str(data_cfg.get("mode") or data_cfg.get("source") or "csv")
    source_type = "international-results" if mode in {"real", "international-results"} else "csv"
    matches_path = resolve_project_path(data_cfg.get("matches_path", "data/raw/historical_matches.csv"), PROJECT_ROOT)
    teams_value = data_cfg.get("teams_path", "data/raw/teams.csv")
    teams_path = resolve_project_path(teams_value, PROJECT_ROOT) if teams_value else None
    shootouts_value = data_cfg.get("shootouts_path")
    shootouts_path = resolve_project_path(shootouts_value, PROJECT_ROOT) if shootouts_value else None
    matches, teams = load_dataset(matches_path, teams_path, source_type=source_type, shootouts_path=shootouts_path)
    elo_path = _optional_path(data_cfg.get("elo_ratings_path") or data_cfg.get("elo_path"))
    fifa_path = _optional_path(data_cfg.get("fifa_rankings_path") or data_cfg.get("fifa_path"))
    return build_feature_table(
        matches,
        teams,
        recent_window=int(feature_cfg.get("recent_window", 5)),
        elo_k=float(feature_cfg.get("elo_k", 28.0)),
        external_elo_ratings=LocalEloRatingSource(elo_path).load() if elo_path else None,
        external_fifa_rankings=LocalFifaRankingSource(fifa_path).load() if fifa_path else None,
        feature_flags=dict(feature_cfg.get("enable_groups", {})) if isinstance(feature_cfg.get("enable_groups"), dict) else {},
    )


def _optional_path(value: object) -> Path | None:
    if not value:
        return None
    path = resolve_project_path(value, PROJECT_ROOT)
    return path if path.exists() else None
