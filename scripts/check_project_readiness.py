from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cupcast.prediction_engine.data_loader import load_dataset
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.real_data_pipeline import run_real_data_pipeline
from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT


REQUIRED_SCRIPTS = [
    "scripts/validate_dataset.py",
    "scripts/validate_ratings.py",
    "scripts/train_prediction_model.py",
    "scripts/compare_models.py",
    "scripts/run_backtest.py",
    "scripts/run_simulation.py",
    "scripts/simulate_full_tournament.py",
    "scripts/update_after_match.py",
    "scripts/run_real_data_pipeline.py",
    "scripts/export_frontend_reports.py",
    "scripts/analyze_feature_importance.py",
    "scripts/run_ablation_study.py",
    "scripts/generate_portfolio_docs.py",
]

README_REQUIRED_SNIPPETS = [
    "pytest",
    "scripts/train_prediction_model.py",
    "scripts/compare_models.py",
    "scripts/run_backtest.py",
    "scripts/run_real_data_pipeline.py",
    "scripts/export_frontend_reports.py",
]

BANNED_README_PHRASES = [
    "is betting-grade",
    "guaranteed winner",
    "guaranteed prediction",
    "fake real-data",
    "fake metrics",
]


def check_project_readiness(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    _check_required_scripts(project_root, errors)
    _check_readme(project_root, errors, warnings)
    _check_gitignore(project_root, errors)
    _check_synthetic_mode(project_root, errors)
    _check_real_pipeline_missing_file_behavior(errors)

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
    }


def _check_required_scripts(project_root: Path, errors: list[str]) -> None:
    missing = [path for path in REQUIRED_SCRIPTS if not (project_root / path).exists()]
    if missing:
        errors.append(f"required scripts missing: {missing}")


def _check_readme(project_root: Path, errors: list[str], warnings: list[str]) -> None:
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        errors.append("README.md is missing")
        return
    readme = readme_path.read_text(encoding="utf-8")
    missing_snippets = [snippet for snippet in README_REQUIRED_SNIPPETS if snippet not in readme]
    if missing_snippets:
        errors.append(f"README.md missing reproducibility command snippets: {missing_snippets}")
    if "This project does not use paid prediction APIs" not in readme:
        warnings.append("README.md should state that forecasts do not use paid prediction APIs")
    banned = [phrase for phrase in BANNED_README_PHRASES if phrase.lower() in readme.lower()]
    if banned:
        errors.append(f"README.md contains banned or overclaiming phrases: {banned}")


def _check_gitignore(project_root: Path, errors: list[str]) -> None:
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        errors.append(".gitignore is missing")
        return
    gitignore = gitignore_path.read_text(encoding="utf-8")
    if "data/real/*.csv" not in gitignore.replace("\\", "/"):
        errors.append(".gitignore must exclude data/real/*.csv")


def _check_synthetic_mode(project_root: Path, errors: list[str]) -> None:
    try:
        config = load_yaml(project_root / "configs" / "prediction_model.yaml")
        data_cfg = config.get("data", {})
        feature_cfg = config.get("features", {})
        matches_path = resolve_project_path(data_cfg.get("matches_path", "data/raw/historical_matches.csv"), project_root)
        teams_path = resolve_project_path(data_cfg.get("teams_path", "data/raw/teams.csv"), project_root)
        matches, teams = load_dataset(matches_path, teams_path, source_type="csv")
        features = build_feature_table(
            matches,
            teams,
            recent_window=int(feature_cfg.get("recent_window", 5)),
            elo_k=float(feature_cfg.get("elo_k", 28.0)),
            feature_flags=dict(feature_cfg.get("enable_groups", {}))
            if isinstance(feature_cfg.get("enable_groups"), dict)
            else {},
        )
        if features.empty:
            errors.append("synthetic prediction feature table is empty")
    except Exception as exc:
        errors.append(f"synthetic mode failed: {exc}")


def _check_real_pipeline_missing_file_behavior(errors: list[str]) -> None:
    try:
        run_real_data_pipeline(
            matches_path=PROJECT_ROOT / "data" / "real" / "__missing_readiness_results.csv",
            shootouts_path=None,
            years=[2022],
        )
    except FileNotFoundError as exc:
        if "Real dataset not found" not in str(exc):
            errors.append("real-data pipeline missing-file error lacks clear setup instructions")
    except Exception as exc:
        errors.append(f"real-data pipeline missing-file behavior is not graceful: {exc}")
    else:
        errors.append("real-data pipeline unexpectedly succeeded with a missing matches file")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CupCast AI portfolio readiness")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = check_project_readiness()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Project readiness: {result['status']}")
        print(f"Warnings: {result['warnings']}")
        if result["errors"]:
            print(f"Errors: {result['errors']}", file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
