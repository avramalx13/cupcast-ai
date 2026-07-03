from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z]+|\d|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class MergeRule:
    left: int
    right: int
    new_id: int

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "right": self.right, "new_id": self.new_id}

    @classmethod
    def from_dict(cls, raw: dict[str, int]) -> "MergeRule":
        return cls(left=int(raw["left"]), right=int(raw["right"]), new_id=int(raw["new_id"]))


class BPETokenizer:
    """Small byte-level BPE tokenizer trained from local text only."""

    special_tokens = ("<pad>", "<bos>", "<eos>")

    def __init__(
        self,
        merges: list[MergeRule] | None = None,
        id_to_bytes: dict[int, bytes] | None = None,
    ) -> None:
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self._byte_offset = len(self.special_tokens)
        self.id_to_bytes = id_to_bytes or {
            self._byte_offset + byte_value: bytes([byte_value]) for byte_value in range(256)
        }
        self.merges = merges or []
        self._merge_lookup = {(rule.left, rule.right): rule.new_id for rule in self.merges}

    @property
    def vocab_size(self) -> int:
        return max([self.eos_id, *self.id_to_bytes.keys()]) + 1

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_pair_freq: int = 2,
        max_merges: int | None = None,
        max_training_bytes: int | None = None,
    ) -> "BPETokenizer":
        tokenizer = cls()
        target_vocab_size = max(vocab_size, tokenizer.vocab_size)
        if max_merges is not None:
            target_vocab_size = min(target_vocab_size, tokenizer.vocab_size + max_merges)
        sequences = [
            tokenizer._bytes_to_ids(text.encode("utf-8"))
            for text in _prepare_training_texts(texts, max_training_bytes=max_training_bytes)
        ]

        while tokenizer.vocab_size < target_vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            (left, right), count = pair_counts.most_common(1)[0]
            if count < min_pair_freq:
                break

            new_id = tokenizer.vocab_size
            tokenizer.id_to_bytes[new_id] = tokenizer.id_to_bytes[left] + tokenizer.id_to_bytes[right]
            rule = MergeRule(left=left, right=right, new_id=new_id)
            tokenizer.merges.append(rule)
            tokenizer._merge_lookup[(left, right)] = new_id
            sequences = [cls._replace_pair(sequence, (left, right), new_id) for sequence in sequences]

        return tokenizer

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = self._bytes_to_ids(text.encode("utf-8"))
        for rule in self.merges:
            token_ids = self._replace_pair(token_ids, (rule.left, rule.right), rule.new_id)

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        chunks: list[bytes] = []
        for token_id in token_ids:
            idx = int(token_id)
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                if skip_special_tokens:
                    continue
                chunks.append(self.special_tokens[idx].encode("utf-8"))
                continue
            piece = self.id_to_bytes.get(idx)
            if piece is None:
                raise ValueError(f"Token id {idx} is not in the tokenizer vocabulary")
            chunks.append(piece)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "byte_bpe",
            "special_tokens": list(self.special_tokens),
            "id_to_bytes": {str(idx): value.hex() for idx, value in self.id_to_bytes.items()},
            "merges": [rule.to_dict() for rule in self.merges],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BPETokenizer":
        if raw.get("type") != "byte_bpe":
            raise ValueError("Checkpoint tokenizer is not a byte_bpe tokenizer")
        id_to_bytes = {int(idx): bytes.fromhex(value) for idx, value in raw["id_to_bytes"].items()}
        merges = [MergeRule.from_dict(item) for item in raw["merges"]]
        return cls(merges=merges, id_to_bytes=id_to_bytes)

    def _bytes_to_ids(self, raw_bytes: bytes) -> list[int]:
        return [self._byte_offset + byte_value for byte_value in raw_bytes]

    @staticmethod
    def _replace_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        if len(sequence) < 2:
            return sequence
        result: list[int] = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(sequence[i])
                i += 1
        return result


class WordTokenizer:
    """Simple whitespace-preserving word/punctuation tokenizer trained from local text."""

    special_tokens = ("<pad>", "<bos>", "<eos>", "<unk>")
    always_tokens = tuple("0123456789")

    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
    ) -> None:
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.token_to_id = token_to_id or {
            token: idx for idx, token in enumerate((*self.special_tokens, *self.always_tokens))
        }
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        max_training_bytes: int | None = None,
        **_: Any,
    ) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for text in _prepare_training_texts(texts, max_training_bytes=max_training_bytes):
            counter.update(_word_tokens(text))

        token_to_id = {token: idx for idx, token in enumerate((*cls.special_tokens, *cls.always_tokens))}
        target_vocab_size = max(vocab_size, len(token_to_id))
        for token, _count in counter.most_common(max(0, target_vocab_size - len(token_to_id))):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in _word_tokens(text)]
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def unknown_tokens(self, text: str) -> list[str]:
        unknown: list[str] = []
        seen: set[str] = set()
        for token in _word_tokens(text):
            if token.strip() and token not in self.token_to_id and token not in seen:
                unknown.append(token)
                seen.add(token)
        return unknown

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            idx = int(token_id)
            token = self.id_to_token.get(idx, self.special_tokens[self.unk_id])
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                if skip_special_tokens:
                    continue
            tokens.append(token)
        return "".join(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "word",
            "special_tokens": list(self.special_tokens),
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WordTokenizer":
        if raw.get("type") != "word":
            raise ValueError("Checkpoint tokenizer is not a word tokenizer")
        return cls(token_to_id={str(token): int(idx) for token, idx in raw["token_to_id"].items()})


def train_tokenizer_from_texts(
    tokenizer_type: str,
    texts: Iterable[str],
    vocab_size: int,
    min_pair_freq: int = 2,
    max_merges: int | None = None,
    max_training_bytes: int | None = None,
) -> BPETokenizer | WordTokenizer:
    if tokenizer_type == "byte_bpe":
        return BPETokenizer.train_from_texts(
            texts=texts,
            vocab_size=vocab_size,
            min_pair_freq=min_pair_freq,
            max_merges=max_merges,
            max_training_bytes=max_training_bytes,
        )
    if tokenizer_type == "word":
        return WordTokenizer.train_from_texts(
            texts=texts,
            vocab_size=vocab_size,
            max_training_bytes=max_training_bytes,
        )
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def tokenizer_from_dict(raw: dict[str, Any]) -> BPETokenizer | WordTokenizer:
    tokenizer_type = raw.get("type")
    if tokenizer_type == "byte_bpe":
        return BPETokenizer.from_dict(raw)
    if tokenizer_type == "word":
        return WordTokenizer.from_dict(raw)
    raise ValueError(f"Unsupported checkpoint tokenizer type: {tokenizer_type}")


def _word_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _prepare_training_texts(
    texts: Iterable[str],
    max_training_bytes: int | None,
) -> list[str]:
    prepared: list[str] = []
    remaining = max_training_bytes
    for text in texts:
        if not text:
            continue
        if remaining is None:
            prepared.append(text)
            continue

        encoded = text.encode("utf-8")
        if remaining <= 0:
            break
        if len(encoded) <= remaining:
            prepared.append(text)
            remaining -= len(encoded)
            continue

        prepared.append(encoded[:remaining].decode("utf-8", errors="ignore"))
        break
    return prepared
