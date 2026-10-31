"""Persistent-memory decoder with mandatory per-step executive influence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from brain_regions.articulator.influence import InfluenceSignal


@dataclass
class ArticulatorConfig:
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    influence_size: int = 29
    max_seq_len: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "influence_size": self.influence_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ArticulatorConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


@dataclass
class ArticulatorCache:
    hidden: Optional[Tensor] = None
    position: int = 0


class StateInfluencedArticulator(nn.Module):
    """GRU decoder with persistent hidden state and per-step influence injection."""

    def __init__(self, config: ArticulatorConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.token_embed = nn.Embedding(config.vocab_size, hidden)
        self.influence_proj = nn.Sequential(
            nn.Linear(config.influence_size, hidden),
            nn.Tanh(),
        )
        self.core = nn.GRU(
            hidden,
            hidden,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, config.vocab_size, bias=False)

    def initial_cache(self) -> ArticulatorCache:
        return ArticulatorCache(hidden=None, position=0)

    def step(
        self,
        token_id: Tensor,
        cache: ArticulatorCache,
        influence: InfluenceSignal,
    ) -> Tuple[Tensor, ArticulatorCache]:
        if token_id.dim() == 0:
            token_id = token_id.unsqueeze(0)
        token_id = token_id.unsqueeze(0)
        embedded = self.token_embed(token_id)
        influence_hidden = self.influence_proj(influence.embedding_vector()).view(1, 1, -1)
        step_input = embedded + influence_hidden
        output, next_hidden = self.core(step_input, cache.hidden)
        logits = self.lm_head(self.final_norm(output[:, -1, :])).squeeze(0)
        return logits, ArticulatorCache(hidden=next_hidden, position=cache.position + 1)

    def forward_sequence(
        self,
        input_ids: Tensor,
        influences: List[InfluenceSignal],
    ) -> Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        cache = self.initial_cache()
        logits_steps = []
        for step in range(input_ids.shape[1]):
            logits, cache = self.step(input_ids[0, step], cache, influences[step])
            logits_steps.append(logits.unsqueeze(0))
        return torch.stack(logits_steps, dim=1)

    @staticmethod
    def apply_influence(
        lm_logits: Tensor,
        influence: InfluenceSignal,
        *,
        vocab_size: int,
        concept_ids: List[int],
        plan_ids: List[int],
        inhibition_ids: List[int],
    ) -> Tensor:
        bias = influence.project_logits(
            vocab_size=vocab_size,
            concept_ids=concept_ids,
            plan_ids=plan_ids,
            inhibition_ids=inhibition_ids,
        )
        return lm_logits + influence.alpha * bias
