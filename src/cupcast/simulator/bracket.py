from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_TEAMS = [
    "France",
    "Brazil",
    "Argentina",
    "Germany",
    "Spain",
    "England",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Croatia",
    "Morocco",
    "Paraguay",
    "Mexico",
    "United States",
    "Japan",
    "Uruguay",
]


@dataclass(frozen=True)
class KnockoutMatch:
    team_a: str
    team_b: str
    stage: str


@dataclass
class TournamentBracket:
    round_of_16: list[KnockoutMatch]
    eliminated: set[str] = field(default_factory=set)
    completed_round_of_16: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def teams(self) -> list[str]:
        names: list[str] = []
        for match in self.round_of_16:
            names.extend([match.team_a, match.team_b])
        return names

    def without_team(self, team: str) -> "TournamentBracket":
        bracket = TournamentBracket(
            round_of_16=list(self.round_of_16),
            eliminated=set(self.eliminated),
            completed_round_of_16=dict(self.completed_round_of_16),
        )
        bracket.eliminated.add(team)
        return bracket

    def with_completed_match(self, team_a: str, team_b: str, winner: str) -> "TournamentBracket":
        match = self.find_round_of_16_match(team_a, team_b)
        if match is None:
            raise ValueError(f"{team_a} and {team_b} are not paired in the current round of 16 bracket")
        if winner not in {team_a, team_b}:
            raise ValueError(f"Winner must be {team_a} or {team_b}, got {winner}")
        loser = team_b if winner == team_a else team_a
        bracket = TournamentBracket(
            round_of_16=list(self.round_of_16),
            eliminated={*self.eliminated, loser},
            completed_round_of_16=dict(self.completed_round_of_16),
        )
        bracket.completed_round_of_16[_match_key(match.team_a, match.team_b)] = winner
        return bracket

    def find_round_of_16_match(self, team_a: str, team_b: str) -> KnockoutMatch | None:
        wanted = {team_a, team_b}
        for match in self.round_of_16:
            if {match.team_a, match.team_b} == wanted:
                return match
        return None


def default_bracket() -> TournamentBracket:
    teams = list(DEFAULT_TEAMS)
    pairings = [
        (teams[0], teams[15]),
        (teams[7], teams[8]),
        (teams[3], teams[11]),
        (teams[4], teams[12]),
        (teams[1], teams[14]),
        (teams[6], teams[9]),
        (teams[2], teams[13]),
        (teams[5], teams[10]),
    ]
    return TournamentBracket(
        round_of_16=[
            KnockoutMatch(team_a=team_a, team_b=team_b, stage="round_of_16")
            for team_a, team_b in pairings
        ]
    )


def _match_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted((team_a, team_b)))
