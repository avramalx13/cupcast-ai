from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .dataset import TokenBlockDataset, read_text_file, split_token_ids
from .model import GPTLanguageModel
from .tokenizer import tokenizer_from_dict, train_tokenizer_from_texts
from .utils import count_parameters, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CupCast AI mini language model")
    parser.add_argument("--config", default="configs/small.yaml", help="Path to YAML config")
    parser.add_argument("--data", required=True, help="Path to raw training text")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override training.eval_interval")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override training.checkpoint_interval",
    )
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return a validation loss")
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def build_training_state(
    config: ExperimentConfig,
    text: str,
    resume_path: str | None,
    device: torch.device,
) -> tuple[ExperimentConfig, object, GPTLanguageModel, int, dict | None]:
    optimizer_state = None
    start_step = 0

    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        config = ExperimentConfig.from_dict(checkpoint["config"])
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        model = GPTLanguageModel(config.model).to(device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        start_step = int(checkpoint.get("step", 0))
        return config, tokenizer, model, start_step, optimizer_state

    tokenizer = train_tokenizer_from_texts(
        tokenizer_type=config.tokenizer.type,
        texts=[text],
        vocab_size=config.model.vocab_size,
        min_pair_freq=config.tokenizer.min_pair_freq,
        max_merges=config.tokenizer.max_merges,
        max_training_bytes=config.tokenizer.max_training_bytes,
    )
    config.model.vocab_size = tokenizer.vocab_size
    model = GPTLanguageModel(config.model).to(device)
    return config, tokenizer, model, start_step, optimizer_state


def train() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_training_overrides(config, args)
    set_seed(config.data.seed)
    device = get_device(args.device)

    text = read_text_file(args.data)
    config, tokenizer, model, start_step, optimizer_state = build_training_state(
        config=config,
        text=text,
        resume_path=args.resume,
        device=device,
    )
    apply_training_overrides(config, args)

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    train_ids, val_ids = split_token_ids(
        token_ids=token_ids,
        block_size=config.model.context_length,
        validation_split=config.data.validation_split,
    )
    train_dataset = TokenBlockDataset(train_ids, block_size=config.model.context_length)
    val_dataset = TokenBlockDataset(val_ids, block_size=config.model.context_length)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = start_step

    print(f"device={device}")
    if start_step:
        print(f"resuming_from_step={start_step} target_max_steps={config.training.max_steps}")
    print(f"tokens={len(token_ids)} train_blocks={len(train_dataset)} val_blocks={len(val_dataset)}")
    print(f"vocab_size={tokenizer.vocab_size} parameters={count_parameters(model):,}")

    model.train()
    while global_step < config.training.max_steps:
        for inputs, targets in train_loader:
            global_step += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Model did not return a training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()

            if global_step == 1 or global_step % config.training.eval_interval == 0:
                val_loss = estimate_loss(model, val_loader, device)
                print(
                    f"step={global_step} train_loss={loss.item():.4f} "
                    f"val_loss={val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        config=config,
                        tokenizer=tokenizer,
                        val_loss=val_loss,
                    )

            if global_step % config.training.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{global_step}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    config=config,
                    tokenizer=tokenizer,
                    val_loss=best_val_loss if best_val_loss < float("inf") else None,
                )

            if global_step >= config.training.max_steps:
                break

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=global_step,
        config=config,
        tokenizer=tokenizer,
        val_loss=best_val_loss if best_val_loss < float("inf") else None,
    )
    print(f"finished step={global_step}; wrote checkpoints to {checkpoint_dir}")


def apply_training_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.eval_interval is not None:
        config.training.eval_interval = args.eval_interval
    if args.checkpoint_interval is not None:
        config.training.checkpoint_interval = args.checkpoint_interval
    config.training.validate()


if __name__ == "__main__":
    train()
