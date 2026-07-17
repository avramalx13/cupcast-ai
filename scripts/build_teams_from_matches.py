from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cupcast.prediction_engine.data_sources import build_teams_from_matches, normalize_international_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CupCast team metadata from match results")
    parser.add_argument("--matches", required=True, help="Path to real international results CSV")
    parser.add_argument("--fifa", help="Optional FIFA rankings CSV to merge into team metadata")
    parser.add_argument("--out", required=True, help="Output path for generated teams CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches_path = Path(args.matches)
    if not matches_path.exists():
        print(
            f"Matches CSV not found: {matches_path}. Place real data in data/real/results.csv first.",
            file=sys.stderr,
        )
        sys.exit(1)
    raw_matches = pd.read_csv(matches_path)
    matches = normalize_international_results(raw_matches)
    fifa_rankings = pd.read_csv(args.fifa) if args.fifa else None
    teams = build_teams_from_matches(matches, fifa_rankings=fifa_rankings)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    teams.to_csv(output_path, index=False)
    print(f"teams={len(teams)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
