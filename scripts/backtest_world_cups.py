from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cupcast.prediction_engine.world_cup_backtest import backtest_world_cups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest CupCast on real historical World Cup tournaments")
    parser.add_argument("--matches", required=True, help="Path to data/real/results.csv")
    parser.add_argument("--shootouts", default=None, help="Optional path to data/real/shootouts.csv")
    parser.add_argument("--teams", default=None, help="Optional teams CSV; generated from matches if omitted")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--output", default="models/world_cup_backtest_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches_path = Path(args.matches)
    if not matches_path.exists():
        print(
            f"Real matches file not found: {matches_path}. Place the real CSV at data/real/results.csv and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        result = backtest_world_cups(
            matches_path=matches_path,
            teams_path=args.teams,
            shootouts_path=args.shootouts,
            years=args.years,
            output_path=args.output,
        )
    except ValueError as exc:
        print(f"World Cup backtest failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
