from __future__ import annotations

import argparse
import json
from pathlib import Path

from cupcast.prediction_engine.rating_sources import validate_rating_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate optional local CupCast rating/ranking CSV files")
    parser.add_argument("--elo", default="data/real/elo_ratings.csv")
    parser.add_argument("--fifa", default="data/real/fifa_rankings.csv")
    parser.add_argument("--output", default="models/rating_validation_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_rating_sources(
        elo_path=Path(args.elo),
        fifa_path=Path(args.fifa),
        output_path=Path(args.output),
    )
    print(json.dumps(payload, indent=2))
    print(f"output={args.output}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
