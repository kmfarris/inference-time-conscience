"""
values.py — Conscience SNN (Golden Lantern + poison pairs export).

Values are mutable but hardest to change: slow plasticity, high challenge threshold.
Outputs excitation for value-aligned words and inhibition for poison-pair violations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from brain_regions.agents.base import AgentOutput, SpecialistAgent
from brain_regions.core.encoding import TokenEncoder
from brain_regions.core.reservoir import ReservoirConfig, ReservoirSNN


class ValuesSNN(SpecialistAgent):
    name = "values"

    # Slowest plasticity in the swarm
    CHALLENGE_THRESHOLD = 0.75
    LEARN_RATE = 0.002

    def __init__(
        self,
        lantern_words: Optional[List[str]] = None,
        poison_pairs: Optional[List[Tuple[str, str]]] = None,
        n_neurons: int = 128,
        n_inputs: int = 32,
        seed: int = 211,
    ):
        self.lantern_words = lantern_words or [
            "empathy", "compassion", "logic", "love", "wisdom",
            "patience", "truth", "protection",
        ]
        self.poison_pairs = poison_pairs or [
            ("destroy", "compassion"), ("kill", "love"), ("dehumanize", "empathy"),
            ("manipulate", "truth"), ("hate", "compassion"),
        ]
        self.poison_hosts: Set[str] = {p[0] for p in self.poison_pairs}
        self.encoder = TokenEncoder(n_inputs=n_inputs, seed=seed)
        cfg = ReservoirConfig(n_neurons=n_neurons, n_inputs=n_inputs, seed=seed)
        self.reservoir = ReservoirSNN(cfg)
        self._challenge_accumulator = 0.0

    def reset(self) -> None:
        self.reservoir.reset_state()
        self._challenge_accumulator = 0.0

    def tick(self, context: Dict[str, Any]) -> AgentOutput:
        candidates: List[str] = context.get("word_candidates", [])
        active_values: Dict[str, float] = context.get("value_activations", {})
        utterance_tokens: List[str] = context.get("output_tokens", [])

        # Encode current moral context
        value_items = [(w, active_values.get(w, 0.5)) for w in self.lantern_words]
        inp = self.encoder.encode_activation_map(value_items)
        if utterance_tokens:
            inp += 0.3 * self.encoder.encode_token(utterance_tokens[-1])
        spikes = self.reservoir.tick(inp)

        n = len(candidates)
        bias = np.zeros(n, dtype=np.float32)
        inhibit = np.zeros(n, dtype=np.float32)

        for i, word in enumerate(candidates):
            wl = word.lower()
            if wl in active_values:
                bias[i] += active_values[wl]
            if wl in self.lantern_words:
                bias[i] += 0.4
            for host, victim in self.poison_pairs:
                if wl == host or wl == victim:
                    if host in utterance_tokens or any(host in t for t in utterance_tokens):
                        inhibit[i] += 0.8

        return AgentOutput(
            agent_name=self.name,
            spikes=spikes,
            bias_vector=bias,
            inhibition_vector=inhibit,
            metadata={"lantern_active": sum(1 for w in self.lantern_words if active_values.get(w, 0) > 0.2)},
        )

    def register_challenge(self, strength: float) -> bool:
        """Returns True if sustained challenge crosses mutation threshold."""
        self._challenge_accumulator = 0.9 * self._challenge_accumulator + strength
        return self._challenge_accumulator >= self.CHALLENGE_THRESHOLD

    def apply_value_delta(self, word: str, delta: float, challenge_strength: float) -> bool:
        """Mutate value association only if challenge gate opens."""
        if not self.register_challenge(challenge_strength):
            return False
        inp = self.encoder.encode_token(word)
        self.reservoir.hebbian_input_update(inp, self.reservoir.tick(inp, record=False), self.LEARN_RATE)
        self._challenge_accumulator *= 0.5
        return True
