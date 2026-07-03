from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import TokenBlockDataset, read_text_file
from .generate import generate_text
from .model import GPTLanguageModel
from .tokenizer import tokenizer_from_dict
from .utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CupCast AI mini-LLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", "--eval-data", dest="data", required=True)
    parser.add_argument("--output", default="models/mini_llm_eval_report.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_loss(
    model: GPTLanguageModel,
    dataset: TokenBlockDataset,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    losses: list[float] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return loss during evaluation")
        losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    eval_data_path: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
    device_override: str | None = None,
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_file.exists():
        report = {
            "status": "missing_checkpoint",
            "message": f"Mini-LLM checkpoint not found: {checkpoint_file}",
        }
        output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    device = get_device(device_override)
    checkpoint = load_checkpoint(checkpoint_file, map_location=device)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    model = GPTLanguageModel(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])

    text = read_text_file(eval_data_path)
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    block_size = min(config.model.context_length, max(1, len(token_ids) - 1))
    dataset = TokenBlockDataset(token_ids, block_size=block_size)
    loss = evaluate_loss(model, dataset, batch_size, device)

    generations = []
    for prompt in _extract_eval_prompts(text):
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=80,
            temperature=0.5,
            top_k=20,
        )
        analysis = generated.split("### Analysis:", 1)[-1].strip()
        generations.append(
            {
                "prompt": prompt,
                "generated": generated,
                "analysis": analysis,
                "generation_length": len(analysis.split()),
                "repetition_rate": repetition_rate(analysis),
                "format_compliance": format_compliance_score(prompt, analysis),
            }
        )

    report = {
        "status": "ok",
        "validation_loss": loss,
        "perplexity": math.exp(loss),
        "average_generation_length": _average([item["generation_length"] for item in generations]),
        "average_repetition_rate": _average([item["repetition_rate"] for item in generations]),
        "unknown_token_rate": unknown_token_rate(
            tokenizer,
            "\n".join(str(item["generated"]) for item in generations),
        ),
        "average_format_compliance": _average(
            [item["format_compliance"]["score"] for item in generations]
        ),
        "generations": generations,
    }
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def repetition_rate(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    return repeats / (len(tokens) - 1)


def unknown_token_rate(tokenizer: object, text: str) -> float:
    unknown_tokens = getattr(tokenizer, "unknown_tokens", None)
    if not callable(unknown_tokens):
        return 0.0
    tokens = [token for token in re.findall(r"\S+", text) if token.strip()]
    if not tokens:
        return 0.0
    return min(1.0, len(unknown_tokens(text)) / len(tokens))


def format_compliance_score(prompt: str, analysis: str) -> dict[str, Any]:
    lower = analysis.lower()
    prompt_lower = prompt.lower()
    prompt_teams = {team.strip() for team in re.findall(r"Team [AB]:\s*([^\n]+)", prompt)}
    known_teams = {
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
    }
    unrelated = [
        team
        for team in known_teams
        if team not in prompt_teams and re.search(rf"\b{re.escape(team.lower())}\b", lower)
    ]
    checks = {
        "non_empty_analysis": bool(analysis.strip()),
        "avoids_false_certainty": not any(
            phrase in lower
            for phrase in ["will definitely win", "guaranteed", "certain to win", "cannot lose"]
        ),
        "mentions_uncertainty_when_probabilities_present": (
            "%" not in prompt_lower
            or "probability" in lower
            or "chance" in lower
            or "likely" in lower
            or "uncertain" in lower
        ),
        "avoids_unrelated_teams": not unrelated,
    }
    score = sum(1 for value in checks.values() if value) / len(checks)
    return {"score": score, "checks": checks, "unrelated_teams": unrelated}


def _extract_eval_prompts(text: str) -> list[str]:
    prompts = []
    for chunk in text.split("### Context:"):
        if "### Analysis:" not in chunk:
            continue
        context, _analysis = chunk.split("### Analysis:", 1)
        prompts.append("### Context:" + context + "### Analysis:\n")
    return prompts[:5]


def _average(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def main() -> None:
    args = parse_args()
    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        eval_data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        device_override=args.device,
    )
    if report["status"] == "missing_checkpoint":
        print(report["message"])
        print(f"output={args.output}")
        return
    print(
        f"loss={report['validation_loss']:.4f} "
        f"perplexity={report['perplexity']:.2f} "
        f"format_compliance={report['average_format_compliance']:.2f}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
