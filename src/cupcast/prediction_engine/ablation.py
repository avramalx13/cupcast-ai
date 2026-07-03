from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cupcast.shared.config import load_yaml
from cupcast.shared.constants import PROJECT_ROOT

from .evaluate import evaluate_model
from .feature_importance import _features_from_config, _optional_path
from .model import train_prediction_model


ABLATION_GROUPS = {
    "elo_only": "elo_logistic_regression",
    "form_only": "recent_form_only",
    "ranking_only": "ranking_only",
    "schedule_only": "schedule_only",
    "goals_only": "goals_only",
    "elo_plus_form": "elo_plus_form",
    "all_features": "logistic_regression",
    "all_features_plus_external_ratings": "logistic_regression",
}


def run_ablation_study(
    config_path: str | Path,
    output_json: str | Path = PROJECT_ROOT / "models" / "ablation_study.json",
    output_md: str | Path = PROJECT_ROOT / "models" / "ablation_study.md",
    max_rows: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    features = _features_from_config(config)
    total_feature_rows = int(len(features))
    if max_rows is not None and len(features) > int(max_rows):
        features = features.sort_values("date").tail(int(max_rows)).reset_index(drop=True)
    train_features, test_features = _chronological_split(features)
    data_cfg = config.get("data", {})
    external_available = bool(
        _optional_path(data_cfg.get("elo_ratings_path") or data_cfg.get("elo_path"))
        or _optional_path(data_cfg.get("fifa_rankings_path") or data_cfg.get("fifa_path"))
    )
    rows = []
    for group, model_name in ABLATION_GROUPS.items():
        if group == "all_features_plus_external_ratings" and not external_available:
            rows.append(
                {
                    "feature_group": group,
                    "model_name": model_name,
                    "status": "unavailable",
                    "reason": "external rating CSVs not found",
                    "accuracy": None,
                    "log_loss": None,
                    "brier_score": None,
                    "ece": None,
                }
            )
            continue
        try:
            model = train_prediction_model(train_features, model_type=model_name, max_iter=400)
            metrics = evaluate_model(model, test_features)
            rows.append(
                {
                    "feature_group": group,
                    "model_name": model_name,
                    "status": "ok",
                    "reason": None,
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                    "ece": metrics["expected_calibration_error"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature_group": group,
                    "model_name": model_name,
                    "status": "failed",
                    "reason": str(exc),
                    "accuracy": None,
                    "log_loss": None,
                    "brier_score": None,
                    "ece": None,
                }
            )
    payload = {
        "train_matches": int(len(train_features)),
        "test_matches": int(len(test_features)),
        "total_feature_rows": total_feature_rows,
        "ablation_rows": int(len(features)),
        "data_window_note": (
            f"ablation uses the most recent {len(features)} rows for runtime"
            if max_rows is not None and total_feature_rows > int(max_rows)
            else "ablation uses all available rows"
        ),
        "groups": rows,
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(output_md).write_text(render_ablation_markdown(payload), encoding="utf-8")
    return payload


def render_ablation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CupCast AI Ablation Study",
        "",
        f"Train matches: {payload.get('train_matches')}; test matches: {payload.get('test_matches')}",
        f"Rows used: {payload.get('ablation_rows')} of {payload.get('total_feature_rows')}",
        f"Note: {payload.get('data_window_note')}",
        "",
        "| Feature group | Status | Accuracy | Log loss | Brier | ECE | Reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("groups", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature_group"]),
                    str(row["status"]),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("log_loss")),
                    _fmt(row.get("brier_score")),
                    _fmt(row.get("ece")),
                    str(row.get("reason") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _chronological_split(features):
    ordered = features.sort_values("date").reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * 0.75))
    if split_idx >= len(ordered):
        split_idx = max(1, len(ordered) - 1)
    return ordered.iloc[:split_idx].reset_index(drop=True), ordered.iloc[split_idx:].reset_index(drop=True)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
