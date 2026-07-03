from __future__ import annotations

import argparse

from cupcast.prediction_engine.data_loader import load_dataset
from cupcast.prediction_engine.features import build_feature_table
from cupcast.prediction_engine.model import load_model, save_model, train_prediction_model
from cupcast.shared.config import load_yaml, resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT
from cupcast.simulator.bracket import default_bracket
from cupcast.simulator.formats import WorldCup2026Format
from cupcast.simulator.monte_carlo import simulate_tournament
from cupcast.simulator.probability_snapshots import save_probability_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CupCast Monte Carlo simulation")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--simulations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    sim_cfg = config.get("simulation", {})
    if not isinstance(sim_cfg, dict):
        sim_cfg = {}
    merged = {**config, **sim_cfg}
    matches_path = resolve_project_path(merged.get("matches_path", "data/raw/historical_matches.csv"), PROJECT_ROOT)
    teams_path = resolve_project_path(merged.get("teams_path", "data/raw/teams.csv"), PROJECT_ROOT)
    model_path = resolve_project_path(merged.get("model_path", "models/prediction_model.joblib"), PROJECT_ROOT)
    snapshot_path = resolve_project_path(merged.get("snapshot_path", "data/processed/probability_snapshots.csv"), PROJECT_ROOT)

    matches, teams = load_dataset(matches_path, teams_path)
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        model = train_prediction_model(build_feature_table(matches, teams), model_version="simulation-in-memory")
        save_model(model, model_path)

    n_simulations = args.simulations or int(merged.get("simulations", 10000))
    seed = int(merged.get("random_seed", 42))
    simulation_format = str(merged.get("format", "simple_16_team_knockout"))
    if simulation_format == "world_cup_2026":
        groups = _groups_from_config(merged)
        result = WorldCup2026Format(groups=groups).simulate(
            prediction_model=model,
            teams=teams,
            matches=matches,
            n_simulations=n_simulations,
            seed=seed,
        )
    else:
        result = simulate_tournament(
            bracket=default_bracket(),
            prediction_model=model,
            teams=teams,
            matches=matches,
            n_simulations=n_simulations,
            seed=seed,
            simulation_version=str(merged.get("simulation_version", "local")),
        )
    save_probability_snapshot(result, snapshot_path)
    for row in result.team_probabilities[:10]:
        print(
            f"{row['team']}: title={float(row['win_tournament_probability']):.1%} "
            f"final={float(row['reach_final_probability']):.1%}"
        )
    print(f"snapshot={snapshot_path}")

def _groups_from_config(config: dict) -> dict[str, list[str]]:
    groups = config.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("world_cup_2026 simulation requires a groups mapping in the config")
    return {str(name): [str(team) for team in teams] for name, teams in groups.items()}


if __name__ == "__main__":
    main()
