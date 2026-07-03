from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TEAMS = [
    ("France", "UEFA", 2042, 2),
    ("Brazil", "CONMEBOL", 2056, 5),
    ("Argentina", "CONMEBOL", 2018, 1),
    ("Germany", "UEFA", 1964, 10),
    ("Spain", "UEFA", 1986, 4),
    ("England", "UEFA", 1988, 3),
    ("Portugal", "UEFA", 1974, 6),
    ("Netherlands", "UEFA", 1945, 7),
    ("Belgium", "UEFA", 1908, 8),
    ("Croatia", "UEFA", 1886, 12),
    ("Morocco", "CAF", 1845, 14),
    ("Paraguay", "CONMEBOL", 1768, 45),
    ("Mexico", "CONCACAF", 1812, 16),
    ("United States", "CONCACAF", 1788, 18),
    ("Canada", "CONCACAF", 1705, 31),
    ("Japan", "AFC", 1824, 17),
    ("Switzerland", "UEFA", 1836, 19),
    ("Senegal", "CAF", 1802, 20),
    ("Uruguay", "CONMEBOL", 1868, 11),
    ("Colombia", "CONMEBOL", 1852, 13),
]

TOURNAMENTS = [
    "World Cup 2014",
    "Euro/Copa 2016",
    "World Cup 2018",
    "Continental Finals 2020",
    "World Cup 2022",
    "World Cup Qualifying 2026",
]

STAGES = ["group", "group", "group", "round_of_16", "quarterfinal", "semifinal", "final"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    write_teams()
    matches = build_matches()
    write_matches(matches)
    examples = write_football_text(matches)
    print(f"teams={len(TEAMS)}")
    print(f"matches={len(matches)}")
    print(f"football_text_examples={examples}")


def write_teams() -> None:
    with (RAW / "teams.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "confederation", "initial_elo", "fifa_rank"])
        writer.writerows(TEAMS)


def build_matches() -> list[dict[str, object]]:
    rng = random.Random(42)
    team_names = [team for team, *_ in TEAMS]
    team_strength = {team: elo for team, _confed, elo, _rank in TEAMS}
    rows: list[dict[str, object]] = []
    start = date(2013, 9, 1)
    for idx in range(180):
        team_a, team_b = rng.sample(team_names, 2)
        tournament = TOURNAMENTS[min(idx // 30, len(TOURNAMENTS) - 1)]
        stage = STAGES[idx % len(STAGES)]
        neutral = stage != "qualifier" and rng.random() < 0.78
        diff = (team_strength[team_a] - team_strength[team_b]) / 180.0
        base_a = 1.25 + max(diff, -1.1) * 0.45
        base_b = 1.15 + max(-diff, -1.1) * 0.45
        score_a = bounded_goal_sample(rng, base_a)
        score_b = bounded_goal_sample(rng, base_b)
        if idx % 11 == 0:
            score_b = score_a
        if stage == "final" and score_a == score_b and idx % 3 == 0:
            score_a += 1
        rows.append(
            {
                "date": (start + timedelta(days=24 * idx)).isoformat(),
                "team_a": team_a,
                "team_b": team_b,
                "team_a_score": score_a,
                "team_b_score": score_b,
                "tournament": tournament,
                "neutral": int(neutral),
                "stage": stage,
            }
        )
    return rows


def bounded_goal_sample(rng: random.Random, mean: float) -> int:
    threshold = max(0.2, min(3.4, mean))
    goals = 0
    remaining = threshold
    while remaining > 0:
        if rng.random() < min(remaining, 1.0) * 0.42:
            goals += 1
        remaining -= 1.0
    if rng.random() < 0.08:
        goals += 1
    return min(goals, 5)


def write_matches(matches: list[dict[str, object]]) -> None:
    with (RAW / "historical_matches.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "date",
            "team_a",
            "team_b",
            "team_a_score",
            "team_b_score",
            "tournament",
            "neutral",
            "stage",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def write_football_text(matches: list[dict[str, object]]) -> int:
    templates = []
    rng = random.Random(7)
    team_elo = {team: elo for team, _confed, elo, _rank in TEAMS}
    for idx in range(72):
        match = matches[idx * 2]
        team_a = str(match["team_a"])
        team_b = str(match["team_b"])
        elo_a = team_elo[team_a] + rng.randint(-25, 25)
        elo_b = team_elo[team_b] + rng.randint(-25, 25)
        prob_a, draw, prob_b = simple_probabilities(elo_a, elo_b)
        favorite = team_a if prob_a >= prob_b else team_b
        underdog = team_b if favorite == team_a else team_a
        templates.append(
            "### Context:\n"
            f"Team A: {team_a}\n"
            f"Team B: {team_b}\n"
            f"{team_a} Elo: {elo_a}\n"
            f"{team_b} Elo: {elo_b}\n"
            f"{team_a} win probability: {prob_a}%\n"
            f"Draw probability: {draw}%\n"
            f"{team_b} win probability: {prob_b}%\n\n"
            "### Analysis:\n"
            f"{favorite} is favored because the structured model gives them the stronger Elo profile "
            f"and a cleaner path against {underdog}. The draw remains relevant, so the forecast should be "
            "read as a probability distribution rather than a guaranteed result.\n"
        )
        if idx % 8 == 0:
            templates.append(
                "### Context:\n"
                f"Event: {underdog} eliminated {favorite} on penalties.\n"
                f"{favorite} title probability before: {rng.randint(6, 18)}.{rng.randint(0, 9)}%\n"
                f"{favorite} title probability after: 0.0%\n"
                f"{underdog} title probability before: {rng.randint(1, 4)}.{rng.randint(0, 9)}%\n"
                f"{underdog} title probability after: {rng.randint(3, 8)}.{rng.randint(0, 9)}%\n\n"
                "### Analysis:\n"
                f"{underdog}'s probability increased because the bracket removed a stronger opponent. "
                f"{favorite} falls to zero title probability because they are eliminated, while the remaining "
                "teams on that side inherit a slightly easier tournament path.\n"
            )

    (RAW / "football_text.txt").write_text("\n".join(templates), encoding="utf-8")
    return len(templates)


def simple_probabilities(elo_a: int, elo_b: int) -> tuple[int, int, int]:
    diff = elo_a - elo_b
    draw = max(18, min(30, 25 - abs(diff) // 60))
    team_a = int(round((100 - draw) * (1 / (1 + 10 ** (-diff / 400)))))
    team_b = 100 - draw - team_a
    return team_a, draw, team_b


if __name__ == "__main__":
    main()
