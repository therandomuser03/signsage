from __future__ import annotations

import json
from pathlib import Path


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
        missing = [t for t in SPECIAL_TOKENS if t not in self.token_to_id]
        if missing:
            raise ValueError(f"Vocabulary missing special tokens: {missing}")

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        # Support either {"token": id} or {"token_to_id": {...}}.
        if "token_to_id" in data:
            data = data["token_to_id"]

        if not isinstance(data, dict):
            raise ValueError(f"Unsupported vocabulary format in {path}")
        return cls(data)

    def encode(self, text: str, max_tokens: int) -> list[int]:
        words = str(text).lower().split()[: max(0, max_tokens - 2)]
        return [self.bos_id] + [self.token_to_id.get(w, self.unk_id) for w in words] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            idx = int(idx)
            if idx in {self.pad_id, self.bos_id}:
                continue
            if idx == self.eos_id:
                break
            out.append(self.id_to_token.get(idx, "<unk>"))
        return " ".join(out).strip()
