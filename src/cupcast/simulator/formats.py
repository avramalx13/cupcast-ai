from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from .bracket import TournamentBracket
from .monte_carlo import MatchPredictor, SimulationResult
from .world_cup_2026 import simulate_world_cup_2026


KnockoutBracket = TournamentBracket


class TournamentFormat(ABC):
    @abstractmethod
    def simulate(
        self,
        prediction_model: MatchPredictor,
        teams: pd.DataFrame,
        matches: pd.DataFrame | None,
        n_simulations: int,
        seed: int | None,
    ) -> SimulationResult:
        ...


@dataclass
class WorldCup2026Format(TournamentFormat):
    groups: dict[str, list[str]]

    def simulate(
        self,
        prediction_model: MatchPredictor,
        teams: pd.DataFrame,
        matches: pd.DataFrame | None,
        n_simulations: int,
        seed: int | None,
    ) -> SimulationResult:
        return simulate_world_cup_2026(
            groups=self.groups,
            prediction_model=prediction_model,
            teams=teams,
            matches=matches,
            n_simulations=n_simulations,
            seed=seed,
        )
