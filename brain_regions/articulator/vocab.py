"""Word-level tokenizer for the state-influenced articulator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tokenizer import clean_tokenize

_SPECIAL = ("<pad>", "<bos>", "<sep>", "<eos>", "<unk>")


@dataclass
class ArticulatorTokenizer:
    tokens: List[str]
    token_to_id: Dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<sep>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @classmethod
    def build(cls, texts: Iterable[str], *, min_frequency: int = 1) -> "ArticulatorTokenizer":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in clean_tokenize(text.lower(), stem=False):
                counts[token] = counts.get(token, 0) + 1
        vocab = list(_SPECIAL)
        for token, count in sorted(counts.items()):
            if count >= min_frequency and token not in vocab:
                vocab.append(token)
        return cls(vocab, {token: index for index, token in enumerate(vocab)})

    @classmethod
    def from_dict(cls, payload: dict) -> "ArticulatorTokenizer":
        tokens = list(payload["tokens"])
        return cls(tokens, {token: index for index, token in enumerate(tokens)})

    def to_dict(self) -> dict:
        return {"version": 1, "tokens": self.tokens}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ArticulatorTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for token in clean_tokenize(text.lower(), stem=False):
            ids.append(self.token_to_id.get(token, self.unk_id))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        words = []
        for index in ids:
            if index in {self.pad_id, self.bos_id, self.sep_id}:
                continue
            if index == self.eos_id:
                break
            token = self.tokens[int(index)]
            if token == "<unk>":
                continue
            words.append(token)
        text = " ".join(words)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text.strip()
