#!/usr/bin/env python3
"""Reference: residual executive bias on logits (no GGUF dependency)."""

from __future__ import annotations

import json

import numpy as np


def apply_bias(logits: np.ndarray, concept_bias: np.ndarray, alpha: float) -> np.ndarray:
    return logits + alpha * concept_bias


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def main() -> None:
    vocab = ["care", "means", "helping", "others", "ignore"]
    token_to_id = {t: i for i, t in enumerate(vocab)}
    plan = ["care", "means", "helping"]

    rng = np.random.default_rng(0)
    base = rng.normal(size=len(vocab))

    concept_bias = np.zeros(len(vocab))
    for t in plan:
        concept_bias[token_to_id[t]] = 2.0

    unbiased = greedy(base)
    biased = greedy(apply_bias(base, concept_bias, alpha=0.35))

    print(
        json.dumps(
            {
                "vocab": vocab,
                "plan_tokens": plan,
                "unbiased_pick": vocab[unbiased],
                "biased_pick": vocab[biased],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
