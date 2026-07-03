from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .model import GPTLanguageModel
from .tokenizer import BPETokenizer, WordTokenizer, tokenizer_from_dict
from .utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate football analysis with CupCast AI")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def generate_text(
    model: GPTLanguageModel,
    tokenizer: BPETokenizer | WordTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
) -> str:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True)
    if isinstance(tokenizer, WordTokenizer):
        unknown = tokenizer.unknown_tokens(prompt)
        if unknown:
            preview = ", ".join(unknown[:10])
            suffix = "..." if len(unknown) > 10 else ""
            print(f"warning: prompt has {len(unknown)} unknown token(s): {preview}{suffix}")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_token_logits = logits[0, -1, :] / temperature
        next_token_logits = _filter_logits(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if int(next_id.item()) == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPTLanguageModel, BPETokenizer | WordTokenizer]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    model = GPTLanguageModel(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer


def _filter_logits(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    filtered = logits.clone()

    if top_k is not None and top_k > 0:
        k = min(top_k, filtered.numel())
        threshold = torch.topk(filtered, k).values[-1]
        filtered[filtered < threshold] = -float("inf")

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        filtered[sorted_indices[sorted_indices_to_remove]] = -float("inf")

    return filtered


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
