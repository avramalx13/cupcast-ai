from __future__ import annotations

from pydantic import BaseModel, Field


class MatchPredictionRequest(BaseModel):
    team_a: str
    team_b: str
    neutral: bool = True
    stage: str = "group"


class MatchPredictionResponse(BaseModel):
    team_a: str
    team_b: str
    team_a_win_probability: float
    draw_probability: float
    team_b_win_probability: float
    explanation: str | None = None


class SimulationRunRequest(BaseModel):
    simulations: int = Field(default=1000, ge=1, le=100000)


class FullTournamentRunRequest(BaseModel):
    groups_path: str = "data/tournaments/world_cup_2026_groups.yaml"
    simulations: int = Field(default=1000, ge=1, le=10000)
    seed: int = 42


class ResultUpdateRequest(BaseModel):
    team_a: str
    team_b: str
    score_a: int = Field(ge=0)
    score_b: int = Field(ge=0)
    penalty_winner: str | None = None
    simulations: int = Field(default=1000, ge=1, le=100000)


class AnalystRequest(BaseModel):
    kind: str = "match_prediction"
    team_a: str | None = None
    team_b: str | None = None
    team_a_win_probability: float | None = None
    draw_probability: float | None = None
    team_b_win_probability: float | None = None
    event: str | None = None
    probability_changes: dict[str, dict[str, float]] | None = None
    use_mini_llm: bool = False


class AnalystResponse(BaseModel):
    explanation: str


class TeamMetadata(BaseModel):
    name: str
    confederation: str = "UNKNOWN"
    match_count: int | None = None
    first_match_date: str | None = None
    last_match_date: str | None = None


class TeamsResponse(BaseModel):
    teams: list[TeamMetadata]
    source: str


class BacktestRunRequest(BaseModel):
    train_before: int = 2022
    test_tournament: str = "World Cup 2022"


class BacktestJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: str | None = None
    result_path: str | None = None
    error_message: str | None = None
