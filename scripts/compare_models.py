from __future__ import annotations

import argparse
import json
from pathlib import Path

from cupcast.prediction_engine.compare import compare_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CupCast match outcome models")
    parser.add_argument("--config", default="configs/prediction_model.yaml")
    parser.add_argument("--train-before", type=int, default=2022)
    parser.add_argument("--test-tournament", default="World Cup 2022")
    parser.add_argument("--output", default="models/comparison_results.json")
    parser.add_argument("--leaderboard-output", default="models/model_leaderboard.json")
    parser.add_argument("--matches", default=None)
    parser.add_argument("--teams", default=None)
    parser.add_argument("--shootouts", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--elo", default=None, help="Optional local Elo snapshot CSV")
    parser.add_argument("--fifa", default=None, help="Optional local FIFA ranking snapshot CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_models(
        config_path=args.config,
        train_before=args.train_before,
        test_tournament=args.test_tournament,
        output_path=Path(args.output),
        leaderboard_path=Path(args.leaderboard_output),
        matches_path=args.matches,
        teams_path=args.teams,
        source_type=args.source,
        shootouts_path=args.shootouts,
        elo_ratings_path=args.elo,
        fifa_rankings_path=args.fifa,
    )
    print(json.dumps(result, indent=2))
    print(f"output={args.output}")
    print(f"leaderboard={args.leaderboard_output}")


if __name__ == "__main__":
    main()
