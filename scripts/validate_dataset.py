from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cupcast.prediction_engine.data_sources import validate_dataset_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CupCast historical match and team CSV files")
    parser.add_argument("--matches", required=True, help="Path to historical matches CSV")
    parser.add_argument("--teams", default=None, help="Path to teams CSV")
    parser.add_argument("--shootouts", default=None, help="Optional shootouts CSV for international results")
    parser.add_argument("--goalscorers", default=None, help="Optional goalscorers CSV")
    parser.add_argument("--source", default="csv", help="csv | synthetic | international-results")
    parser.add_argument("--report", default="models/dataset_validation_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset_files(
        matches_path=args.matches,
        teams_path=args.teams,
        shootouts_path=args.shootouts,
        goalscorers_path=args.goalscorers,
        source_type=args.source,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(report.format_text())
    print(f"output={report_path}")
    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
