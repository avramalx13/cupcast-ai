from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from cupcast.shared.config import resolve_project_path
from cupcast.shared.constants import PROJECT_ROOT
from cupcast.simulator.bracket import default_bracket
from cupcast.simulator.full_tournament import (
    TournamentConfigError,
    TournamentPredictionService,
    load_tournament_config,
    simulate_full_tournament,
    write_full_tournament_reports,
)
from cupcast.simulator.monte_carlo import simulate_tournament
from cupcast.simulator.probability_snapshots import load_latest_snapshot, save_probability_snapshot
from cupcast.simulator.update_after_result import apply_match_result

from .schemas import FullTournamentRunRequest, ResultUpdateRequest, SimulationRunRequest
from .services import (
    full_tournament_report_paths,
    get_explanation_service,
    get_full_tournament_context,
    get_matches,
    get_prediction_model,
    get_teams,
    snapshot_path,
)


router = APIRouter(tags=["simulation"])


@router.get("/simulation/latest")
def latest_simulation() -> dict[str, object]:
    latest = load_latest_snapshot(snapshot_path())
    return {"results": latest}


@router.post("/simulation/run")
def run_simulation(request: SimulationRunRequest) -> dict[str, object]:
    result = simulate_tournament(
        bracket=default_bracket(),
        prediction_model=get_prediction_model(),
        teams=get_teams(),
        matches=get_matches(),
        n_simulations=request.simulations,
        seed=42,
        simulation_version="api-run",
    )
    save_probability_snapshot(result, snapshot_path())
    return {
        "n_simulations": result.n_simulations,
        "model_version": result.model_version,
        "results": result.team_probabilities,
    }


@router.get("/simulation/full-tournament/latest")
def latest_full_tournament_simulation() -> dict[str, object]:
    json_path, _md_path = full_tournament_report_paths()
    if not json_path.exists():
        return {
            "available": False,
            "message": "Run full tournament simulation first.",
        }
    return json.loads(json_path.read_text(encoding="utf-8"))


@router.post("/simulation/full-tournament/run")
def run_full_tournament_simulation(request: FullTournamentRunRequest) -> dict[str, object]:
    try:
        model, matches, teams, source = get_full_tournament_context()
        config = load_tournament_config(
            resolve_project_path(request.groups_path, PROJECT_ROOT),
            teams=teams,
        )
        service = TournamentPredictionService(model, teams=teams, matches=matches)
        summary = simulate_full_tournament(
            config,
            service,
            n_simulations=request.simulations,
            seed=request.seed,
        )
    except TournamentConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary["data_source"] = source
    summary["analyst_explanation"] = get_explanation_service().explain_full_tournament_summary(summary)
    json_path, md_path = full_tournament_report_paths()
    write_full_tournament_reports(summary, out_json=json_path, out_md=md_path)
    return {
        "status": "completed",
        "available": True,
        "simulations": summary["simulations"],
        "top_champions": summary["top_champions"],
        "group_predictions": summary["group_predictions"],
        "analyst_explanation": summary["analyst_explanation"],
        "report_path": str(json_path),
    }


@router.post("/matches/update-result")
def update_result(request: ResultUpdateRequest) -> dict[str, object]:
    try:
        update = apply_match_result(
            bracket=default_bracket(),
            prediction_model=get_prediction_model(),
            teams=get_teams(),
            matches=get_matches(),
            team_a=request.team_a,
            team_b=request.team_b,
            score_a=request.score_a,
            score_b=request.score_b,
            penalty_winner=request.penalty_winner,
            simulations=request.simulations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "event": update.event,
        "eliminated_team": update.eliminated_team,
        "advanced_team": update.advanced_team,
        "probability_changes": update.probability_changes,
        "elo_update": update.elo_update,
    }
