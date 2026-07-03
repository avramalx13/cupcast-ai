from __future__ import annotations

import argparse
import json

from cupcast.prediction_engine.data_loader import load_dataset
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model import load_model, train_prediction_model
from cupcast.shared.constants import DEFAULT_MATCHES_PATH, DEFAULT_MODEL_PATH, DEFAULT_TEAMS_PATH
from cupcast.simulator.bracket import default_bracket
from cupcast.simulator.update_after_result import apply_match_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a CupCast live result update")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--score-a", type=int, required=True)
    parser.add_argument("--score-b", type=int, required=True)
    parser.add_argument("--penalty-winner", default=None)
    parser.add_argument("--simulations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches, teams = load_dataset(DEFAULT_MATCHES_PATH, DEFAULT_TEAMS_PATH)
    try:
        model = load_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        model = train_prediction_model(build_feature_table(matches, teams), model_version="update-in-memory")
    update = apply_match_result(
        bracket=default_bracket(),
        prediction_model=model,
        teams=teams,
        matches=matches,
        team_a=args.team_a,
        team_b=args.team_b,
        score_a=args.score_a,
        score_b=args.score_b,
        penalty_winner=args.penalty_winner,
        simulations=args.simulations,
    )
    print(
        json.dumps(
            {
                "event": update.event,
                "eliminated_team": update.eliminated_team,
                "advanced_team": update.advanced_team,
                "probability_changes": update.probability_changes,
                "elo_update": update.elo_update,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
