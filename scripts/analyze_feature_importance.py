from __future__ import annotations

import argparse
import json

from cupcast.prediction_engine.feature_importance import analyze_feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CupCast model feature importance")
    parser.add_argument("--config", default="configs/prediction_model.yaml")
    parser.add_argument("--output-json", default="models/feature_importance.json")
    parser.add_argument("--output-md", default="models/feature_importance.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze_feature_importance(args.config, output_json=args.output_json, output_md=args.output_md)
    print(json.dumps({"models": [row["model_name"] for row in payload["models"]]}, indent=2))
    print(f"output={args.output_json}")
    print(f"markdown={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
