"""
reservoir.py — Small leaky integrate-and-fire reservoir SNN.

Used by all specialist agents (Grammar, Values, Selection, Chemicals, Encoder).
Lightweight numpy implementation — no GPU required for v0 agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ReservoirConfig:
    n_neurons: int = 64
    n_inputs: int = 32
    leak: float = 0.85
    threshold: float = 1.0
    reset: float = 0.0
    refractory_ticks: int = 2
    input_gain: float = 0.4
    recurrent_gain: float = 0.25
    noise: float = 0.02
    seed: int = 42


class ReservoirSNN:
    """Fixed-size reservoir with optional plastic input weights."""

    def __init__(self, config: Optional[ReservoirConfig] = None):
        self.cfg = config or ReservoirConfig()
        rng = np.random.default_rng(self.cfg.seed)
        c = self.cfg

        self.W_in = rng.normal(0, 0.3, (c.n_neurons, c.n_inputs)).astype(np.float32)
        # Sparse recurrent — 15% connectivity
        mask = rng.random((c.n_neurons, c.n_neurons)) < 0.15
        np.fill_diagonal(mask, False)
        self.W_rec = (rng.normal(0, 0.2, (c.n_neurons, c.n_neurons)) * mask).astype(
            np.float32
        )

        self.membrane = np.zeros(c.n_neurons, dtype=np.float32)
        self.refractory = np.zeros(c.n_neurons, dtype=np.int32)
        self.spike_history: list[np.ndarray] = []
        self._rng = rng

    def reset_state(self) -> None:
        c = self.cfg
        self.membrane.fill(0)
        self.refractory.fill(0)
        self.spike_history.clear()

    def tick(self, input_vector: np.ndarray, record: bool = True) -> np.ndarray:
        """One simulation step. input_vector shape (n_inputs,). Returns spikes (n_neurons,) bool."""
        c = self.cfg
        inp = np.asarray(input_vector, dtype=np.float32).reshape(-1)
        if inp.shape[0] != c.n_inputs:
            raise ValueError(f"expected {c.n_inputs} inputs, got {inp.shape[0]}")

        # Decay refractory counters
        self.refractory = np.maximum(self.refractory - 1, 0)
        can_fire = self.refractory == 0

        drive = c.input_gain * (self.W_in @ inp)
        if self.spike_history:
            last = self.spike_history[-1].astype(np.float32)
            drive += c.recurrent_gain * (self.W_rec @ last)
        drive += self._rng.normal(0, c.noise, c.n_neurons).astype(np.float32)

        self.membrane = c.leak * self.membrane + drive
        spikes = (self.membrane >= c.threshold) & can_fire
        self.membrane[spikes] = c.reset
        self.refractory[spikes] = c.refractory_ticks

        if record:
            self.spike_history.append(spikes.copy())
        return spikes

    def run(self, input_sequence: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run a sequence of input vectors. Returns (firing_rate, last_spikes)."""
        self.reset_state()
        total_spikes = 0
        last = np.zeros(self.cfg.n_neurons, dtype=bool)
        for t in range(input_sequence.shape[0]):
            last = self.tick(input_sequence[t])
            total_spikes += int(last.sum())
        ticks = max(len(self.spike_history), 1)
        rate = total_spikes / (self.cfg.n_neurons * ticks)
        return float(rate), last

    def concentration(self, baseline: float = 0.0, ceiling: float = 1.0) -> float:
        """Map recent activity to [baseline, ceiling] chemical concentration."""
        if not self.spike_history:
            return baseline
        window = self.spike_history[-min(8, len(self.spike_history)) :]
        rate = np.mean([s.mean() for s in window])
        return float(baseline + (ceiling - baseline) * min(1.0, rate * 4.0))

    def hebbian_input_update(
        self, input_vector: np.ndarray, target_neurons: np.ndarray, lr: float = 0.01
    ) -> None:
        """Strengthen input weights for co-active input + target reservoir neurons."""
        inp = np.asarray(input_vector, dtype=np.float32).reshape(-1)
        for idx in np.where(target_neurons)[0]:
            self.W_in[idx] += lr * inp * (1.0 - np.abs(self.W_in[idx]))
        np.clip(self.W_in, -1.0, 1.0, out=self.W_in)
