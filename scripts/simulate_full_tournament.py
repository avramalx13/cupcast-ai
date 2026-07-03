from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cupcast.analyst.explanation_service import ExplanationService
from cupcast.prediction_engine.data_loader import load_dataset
from cupcast.prediction_engine.model import load_model
from cupcast.shared.config import resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT
from cupcast.simulator.full_tournament import (
    TournamentConfigError,
    TournamentPredictionService,
    load_tournament_config,
    simulate_full_tournament,
    write_full_tournament_reports,
)


REAL_MODEL_PATH = PROJECT_ROOT / "models" / "prediction_model_real.joblib"
SYNTHETIC_MODEL_PATH = PROJECT_ROOT / "models" / "prediction_model.joblib"
REAL_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "real_completed_matches.csv"
REAL_TEAMS_PATH = PROJECT_ROOT / "data" / "processed" / "teams_real.csv"
SYNTHETIC_MATCHES_PATH = PROJECT_ROOT / "data" / "raw" / "historical_matches.csv"
SYNTHETIC_TEAMS_PATH = PROJECT_ROOT / "data" / "raw" / "teams.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Simulate a full 48-team World Cup tournament")
    parser.add_argument("--groups", default="data/tournaments/world_cup_2026_groups.yaml")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=str(defaults["model"].relative_to(PROJECT_ROOT)))
    parser.add_argument("--matches", default=str(defaults["matches"].relative_to(PROJECT_ROOT)))
    parser.add_argument("--teams", default=str(defaults["teams"].relative_to(PROJECT_ROOT)))
    parser.add_argument("--source-type", default=str(defaults["source_type"]))
    parser.add_argument("--out-json", default="models/full_tournament_simulation.json")
    parser.add_argument("--out-md", default="models/full_tournament_simulation.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    groups_path = resolve_project_path(args.groups, PROJECT_ROOT)
    model_path = resolve_project_path(args.model, PROJECT_ROOT)
    matches_path = resolve_project_path(args.matches, PROJECT_ROOT)
    teams_path = resolve_project_path(args.teams, PROJECT_ROOT)
    out_json = resolve_project_path(args.out_json, PROJECT_ROOT)
    out_md = resolve_project_path(args.out_md, PROJECT_ROOT)

    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(
            f"Prediction model not found: {model_path}\n"
            "Train a model first, for example:\n"
            "  python scripts/run_real_data_pipeline.py --matches data/real/results.csv --shootouts data/real/shootouts.csv --years 2014 2018 2022\n"
            "or:\n"
            "  python scripts/train_prediction_model.py --config configs/prediction_model_real.yaml",
            file=sys.stderr,
        )
        return 1

    try:
        matches, teams = load_dataset(matches_path, teams_path, source_type=args.source_type)
        config = load_tournament_config(groups_path, teams=teams)
        service = TournamentPredictionService(model, teams=teams, matches=matches)
        summary = simulate_full_tournament(config, service, n_simulations=args.simulations, seed=args.seed)
    except TournamentConfigError as exc:
        print(f"Full tournament simulation config error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Full tournament simulation input missing: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Full tournament simulation failed: {exc}", file=sys.stderr)
        return 1

    summary["analyst_explanation"] = ExplanationService().explain_full_tournament_summary(summary)
    write_full_tournament_reports(summary, out_json=out_json, out_md=out_md)
    print(f"simulations={summary['simulations']}")
    print(f"top_champion={summary['top_champions'][0]['team']} title_probability={summary['top_champions'][0]['title_probability']:.3f}")
    print(f"output_json={out_json}")
    print(f"output_md={out_md}")
    return 0


def _default_paths() -> dict[str, Path | str]:
    if REAL_MODEL_PATH.exists() and REAL_MATCHES_PATH.exists() and REAL_TEAMS_PATH.exists():
        return {
            "model": REAL_MODEL_PATH,
            "matches": REAL_MATCHES_PATH,
            "teams": REAL_TEAMS_PATH,
            "source_type": "international-results",
        }
    return {
        "model": SYNTHETIC_MODEL_PATH,
        "matches": SYNTHETIC_MATCHES_PATH,
        "teams": SYNTHETIC_TEAMS_PATH,
        "source_type": "csv",
    }


if __name__ == "__main__":
    raise SystemExit(main())
