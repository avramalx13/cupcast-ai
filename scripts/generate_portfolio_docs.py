from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cupcast.prediction_engine.report_docs import RealReportError, write_portfolio_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio docs from CupCast real-data reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--readme", default="README.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = write_portfolio_docs(models_dir=Path(args.models_dir), readme_path=Path(args.readme))
    except RealReportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
