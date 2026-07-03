from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .monte_carlo import SimulationResult


SNAPSHOT_COLUMNS = [
    "timestamp",
    "team",
    "round_probability",
    "title_probability",
    "simulation_version",
    "model_version",
]


def save_probability_snapshot(
    result: SimulationResult,
    path: str | Path,
    timestamp: datetime | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (timestamp or datetime.now(UTC)).isoformat()
    rows = [
        {
            "timestamp": stamp,
            "team": row["team"],
            "round_probability": row["reach_final_probability"],
            "title_probability": row["win_tournament_probability"],
            "simulation_version": result.simulation_version,
            "model_version": result.model_version,
        }
        for row in result.team_probabilities
    ]
    snapshot = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        snapshot = pd.concat([existing, snapshot], ignore_index=True)
    snapshot.to_csv(output_path, index=False)
    return output_path


def load_latest_snapshot(path: str | Path) -> list[dict[str, object]]:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return []
    frame = pd.read_csv(snapshot_path)
    if frame.empty:
        return []
    latest_timestamp = frame["timestamp"].max()
    return frame.loc[frame["timestamp"] == latest_timestamp].to_dict(orient="records")
