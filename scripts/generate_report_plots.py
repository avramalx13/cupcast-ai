from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple PNG plots from CupCast report JSON files")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed. Install it with `pip install matplotlib` to generate report plots.",
            file=sys.stderr,
        )
        return 0

    models_dir = Path(args.models_dir)
    leaderboard_path = models_dir / "model_leaderboard.json"
    backtest_path = models_dir / "world_cup_backtest_results.json"
    missing = [str(path) for path in [leaderboard_path, backtest_path] if not path.exists()]
    if missing:
        print(
            "Report plots were not generated because required reports are missing: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    backtests = json.loads(backtest_path.read_text(encoding="utf-8"))
    output_dir = models_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [row for row in leaderboard.get("models", []) if row.get("status", "ok") == "ok"]
    _bar_plot(
        plt=plt,
        rows=models,
        metric="accuracy",
        title="Model Leaderboard Accuracy",
        ylabel="Accuracy",
        output_path=output_dir / "model_leaderboard_accuracy.png",
    )
    _bar_plot(
        plt=plt,
        rows=models,
        metric="log_loss",
        title="Model Leaderboard Log Loss",
        ylabel="Log loss",
        output_path=output_dir / "model_leaderboard_log_loss.png",
    )
    _bar_plot(
        plt=plt,
        rows=models,
        metric="ece",
        title="Calibration ECE",
        ylabel="ECE",
        output_path=output_dir / "calibration_ece.png",
    )

    backtest_rows = [row for row in backtests.get("results", []) if row.get("status") == "ok"]
    if backtest_rows:
        plt.figure(figsize=(8, 4.5))
        plt.bar([str(row["year"]) for row in backtest_rows], [float(row["accuracy"]) for row in backtest_rows])
        plt.title("World Cup Backtest Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("World Cup year")
        plt.tight_layout()
        plt.savefig(output_dir / "world_cup_backtest_accuracy.png", dpi=160)
        plt.close()

    print(f"plots={output_dir}")
    return 0


def _bar_plot(plt: Any, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str, output_path: Path) -> None:
    filtered = [row for row in rows if row.get(metric) is not None]
    if not filtered:
        return
    plt.figure(figsize=(10, 4.8))
    plt.bar([str(row["model_name"]) for row in filtered], [float(row[metric]) for row in filtered])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
