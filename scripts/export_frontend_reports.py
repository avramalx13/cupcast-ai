from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from cupcast.prediction_engine.model import load_model
from cupcast.prediction_engine.data_loader import load_dataset
from cupcast.shared.constants import PROJECT_ROOT


DEFAULT_DEMO_MATCHUPS = [
    ("France", "Brazil"),
    ("Argentina", "Spain"),
    ("Morocco", "Senegal"),
    ("England", "Portugal"),
    ("Germany", "Netherlands"),
    ("Japan", "South Korea"),
    ("Mexico", "United States"),
    ("Ghana", "Ivory Coast"),
]

REPORT_FILES = [
    "model_leaderboard.json",
    "world_cup_backtest_results.json",
    "world_cup_error_analysis.json",
    "feature_importance.json",
    "ablation_study.json",
    "real_data_results_summary.md",
    "full_tournament_simulation.json",
    "full_tournament_simulation.md",
]


def export_frontend_reports(
    teams_path: str | Path = PROJECT_ROOT / "data" / "processed" / "teams_real.csv",
    output_dir: str | Path = PROJECT_ROOT / "frontend" / "public" / "reports",
    matches_path: str | Path = PROJECT_ROOT / "data" / "processed" / "real_completed_matches.csv",
    model_path: str | Path = PROJECT_ROOT / "models" / "prediction_model_real.joblib",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    teams_payload = _teams_payload(Path(teams_path), warnings)
    teams_output = output / "teams_real.json"
    teams_output.write_text(json.dumps(teams_payload, indent=2), encoding="utf-8")

    copied_reports = _copy_report_files(output, warnings)
    demo_payload = _demo_matchups_payload(
        teams=teams_payload["teams"],
        matches_path=Path(matches_path),
        teams_path=Path(teams_path),
        model_path=Path(model_path),
        warnings=warnings,
    )
    demo_output = output / "demo_matchups.json"
    demo_output.write_text(json.dumps(demo_payload, indent=2), encoding="utf-8")

    return {
        "teams": str(teams_output),
        "demo_matchups": str(demo_output),
        "reports": copied_reports,
        "warnings": warnings,
    }


def _teams_payload(teams_path: Path, warnings: list[str]) -> dict[str, Any]:
    if not teams_path.exists():
        warnings.append(f"Teams file not found: {teams_path}")
        return {"source": "missing", "teams": []}
    teams = pd.read_csv(teams_path)
    if teams.empty or "team" not in teams:
        warnings.append(f"Teams file is empty or missing team column: {teams_path}")
        return {"source": "real", "teams": []}
    frame = teams.copy()
    frame["team"] = frame["team"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["team"].ne("")]
    for column, default in [
        ("confederation", "UNKNOWN"),
        ("match_count", None),
        ("first_match_date", None),
        ("last_match_date", None),
    ]:
        if column not in frame:
            frame[column] = default
    frame["match_count"] = pd.to_numeric(frame["match_count"], errors="coerce")
    frame = frame.sort_values(["team", "match_count"], ascending=[True, False]).drop_duplicates("team")
    rows = []
    for row in frame.sort_values("team").itertuples(index=False):
        match_count = getattr(row, "match_count", None)
        rows.append(
            {
                "name": str(row.team),
                "confederation": str(getattr(row, "confederation", "UNKNOWN") or "UNKNOWN"),
                "match_count": None if pd.isna(match_count) else int(match_count),
                "first_match_date": _optional_text(getattr(row, "first_match_date", None)),
                "last_match_date": _optional_text(getattr(row, "last_match_date", None)),
            }
        )
    return {"source": "real", "teams": rows}


def _copy_report_files(output: Path, warnings: list[str]) -> list[str]:
    copied = []
    for name in REPORT_FILES:
        source = PROJECT_ROOT / "models" / name
        if not source.exists():
            warnings.append(f"Report not found: {source}")
            continue
        target = output / name
        shutil.copyfile(source, target)
        copied.append(str(target))
    return copied


def _demo_matchups_payload(
    teams: list[dict[str, Any]],
    matches_path: Path,
    teams_path: Path,
    model_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    available_teams = {str(team["name"]) for team in teams}
    if not model_path.exists():
        warnings.append(f"Real prediction model not found for static demo matchups: {model_path}")
        return {"source": "unavailable", "matchups": [], "warnings": warnings}
    if not matches_path.exists() or not teams_path.exists():
        warnings.append("Real matches/teams are unavailable for static demo matchup generation")
        return {"source": "unavailable", "matchups": [], "warnings": warnings}
    model = load_model(model_path)
    matches, team_frame = load_dataset(matches_path, teams_path, source_type="international-results")
    rows = []
    for team_a, team_b in DEFAULT_DEMO_MATCHUPS:
        if team_a not in available_teams or team_b not in available_teams:
            warnings.append(f"Skipped static matchup with unavailable team: {team_a} vs {team_b}")
            continue
        try:
            result = model.predict_match(team_a, team_b, teams=team_frame, matches=matches)
        except Exception as exc:
            warnings.append(f"Skipped static matchup {team_a} vs {team_b}: {exc}")
            continue
        rows.append(result.as_dict())
    return {"source": "model", "matchups": rows, "warnings": warnings}


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frontend static report data")
    parser.add_argument("--teams", default="data/processed/teams_real.csv")
    parser.add_argument("--matches", default="data/processed/real_completed_matches.csv")
    parser.add_argument("--model", default="models/prediction_model_real.joblib")
    parser.add_argument("--output-dir", default="frontend/public/reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_frontend_reports(
        teams_path=PROJECT_ROOT / args.teams,
        matches_path=PROJECT_ROOT / args.matches,
        model_path=PROJECT_ROOT / args.model,
        output_dir=PROJECT_ROOT / args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
