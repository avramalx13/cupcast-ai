from __future__ import annotations

import argparse

from cupcast.shared.constants import DEFAULT_MATCHES_PATH, DEFAULT_MODEL_PATH, DEFAULT_TEAMS_PATH

from .data_loader import load_dataset, load_matches, load_teams
from .features import build_feature_table
from .model import load_model, train_prediction_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a match with the CupCast structured model")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--matches", default=str(DEFAULT_MATCHES_PATH))
    parser.add_argument("--teams", default=str(DEFAULT_TEAMS_PATH))
    parser.add_argument("--stage", default="group")
    parser.add_argument("--neutral", action="store_true", default=True)
    return parser.parse_args()


def load_or_train_default_model(model_path: str = str(DEFAULT_MODEL_PATH)):
    try:
        return load_model(model_path)
    except FileNotFoundError:
        matches = load_matches(DEFAULT_MATCHES_PATH)
        teams = load_teams(DEFAULT_TEAMS_PATH)
        features = build_feature_table(matches, teams)
        return train_prediction_model(features, model_version="in-memory-default")


def main() -> None:
    args = parse_args()
    matches, teams = load_dataset(args.matches, args.teams)
    model = load_or_train_default_model(args.model)
    result = model.predict_match(
        team_a=args.team_a,
        team_b=args.team_b,
        teams=teams,
        matches=matches,
        neutral=args.neutral,
        stage=args.stage,
    )
    print(f"{result.team_a} win: {result.team_a_win_probability:.1%}")
    print(f"Draw: {result.draw_probability:.1%}")
    print(f"{result.team_b} win: {result.team_b_win_probability:.1%}")


if __name__ == "__main__":
    main()
