from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cupcast.prediction_engine.real_data_pipeline import run_real_data_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full CupCast real-data evaluation pipeline")
    parser.add_argument("--matches", default="data/real/results.csv")
    parser.add_argument("--shootouts", default="data/real/shootouts.csv")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--teams-out", default="data/processed/teams_real.csv")
    parser.add_argument("--elo", default="data/real/elo_ratings.csv", help="Optional local Elo snapshot CSV")
    parser.add_argument("--fifa", default="data/real/fifa_rankings.csv", help="Optional local FIFA ranking snapshot CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches_path = Path(args.matches)
    if not matches_path.exists():
        print(
            "Real dataset not found.\n"
            "Place results.csv in data/real/results.csv.\n"
            "Optional: place shootouts.csv in data/real/shootouts.csv.",
            file=sys.stderr,
        )
        return 1
    try:
        result = run_real_data_pipeline(
            matches_path=matches_path,
            shootouts_path=args.shootouts,
            years=args.years,
            models_dir=args.models_dir,
            teams_output_path=args.teams_out,
            elo_ratings_path=args.elo,
            fifa_rankings_path=args.fifa,
        )
    except Exception as exc:
        print(f"Real-data pipeline failed: {exc}", file=sys.stderr)
        return 1
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(json.dumps(result["outputs"], indent=2))
    print(f"summary={result['outputs']['markdown_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
