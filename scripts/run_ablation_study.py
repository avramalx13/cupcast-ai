from __future__ import annotations

import argparse
import json

from cupcast.prediction_engine.ablation import run_ablation_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CupCast feature-group ablation study")
    parser.add_argument("--config", default="configs/prediction_model.yaml")
    parser.add_argument("--output-json", default="models/ablation_study.json")
    parser.add_argument("--output-md", default="models/ablation_study.md")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional most-recent-row cap for large real datasets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_ablation_study(
        args.config,
        output_json=args.output_json,
        output_md=args.output_md,
        max_rows=args.max_rows,
    )
    print(json.dumps({"groups": [row["feature_group"] for row in payload["groups"]]}, indent=2))
    print(f"output={args.output_json}")
    print(f"markdown={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
