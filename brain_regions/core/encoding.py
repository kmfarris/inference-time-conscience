"""
encoding.py — Hash-based token → input vector encoding for small SNNs.

No hardcoded English in agent logic beyond trigger sets passed at construction
(from curriculum / training data). Encoding is deterministic from token strings.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

import numpy as np


class TokenEncoder:
    """Maps tokens to fixed-size bipolar input vectors via hashed population coding."""

    def __init__(self, n_inputs: int = 32, bits_per_token: int = 4, seed: int = 0):
        self.n_inputs = n_inputs
        self.bits_per_token = bits_per_token
        self.seed = seed

    def encode_token(self, token: str) -> np.ndarray:
        token = token.lower().strip()
        vec = np.zeros(self.n_inputs, dtype=np.float32)
        h = hashlib.sha256(f"{self.seed}:{token}".encode()).digest()
        for i in range(self.bits_per_token):
            idx = int.from_bytes(h[i * 2 : i * 2 + 2], "big") % self.n_inputs
            sign = 1.0 if h[i + 8] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_sequence(self, tokens: Sequence[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((1, self.n_inputs), dtype=np.float32)
        rows = [self.encode_token(t) for t in tokens]
        return np.stack(rows, axis=0)

    def encode_activation_map(
        self, activations: Iterable[tuple[str, float]]
    ) -> np.ndarray:
        """Blend encodings weighted by SNN concept activation strength."""
        vec = np.zeros(self.n_inputs, dtype=np.float32)
        total = 0.0
        for word, strength in activations:
            if strength <= 0:
                continue
            vec += strength * self.encode_token(word)
            total += strength
        if total > 0:
            vec /= total
        return vec
