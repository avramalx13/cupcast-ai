from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length next-token prediction dataset."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Need at least {block_size + 1} tokens, got {len(token_ids)}"
            )
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


def read_text_file(path: str | Path) -> str:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    text = data_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Data file is empty: {data_path}")
    return text


def split_token_ids(
    token_ids: list[int],
    block_size: int,
    validation_split: float,
) -> tuple[list[int], list[int]]:
    minimum = block_size + 1
    if len(token_ids) < minimum * 2:
        raise ValueError(
            f"Need at least {minimum * 2} tokens for train/validation split, got {len(token_ids)}"
        )

    split_at = int(len(token_ids) * (1.0 - validation_split))
    split_at = max(minimum, min(split_at, len(token_ids) - minimum))
    return token_ids[:split_at], token_ids[split_at:]
