"""Per-token executive influence contract for the articulator LM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class InfluenceSignal:
    """Numeric executive pressure applied each decode step."""

    concept_bias: Tensor
    inhibition_bias: Tensor
    plan_slot_bias: Tensor
    affect: Tensor
    alpha: float

    def to_dict(self) -> dict:
        return {
            "concept_bias": self.concept_bias.detach().cpu().tolist(),
            "inhibition_bias": self.inhibition_bias.detach().cpu().tolist(),
            "plan_slot_bias": self.plan_slot_bias.detach().cpu().tolist(),
            "affect": self.affect.detach().cpu().tolist(),
            "alpha": float(self.alpha),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, device: str = "cpu") -> "InfluenceSignal":
        return cls(
            concept_bias=torch.tensor(payload["concept_bias"], device=device, dtype=torch.float32),
            inhibition_bias=torch.tensor(
                payload["inhibition_bias"], device=device, dtype=torch.float32
            ),
            plan_slot_bias=torch.tensor(
                payload["plan_slot_bias"], device=device, dtype=torch.float32
            ),
            affect=torch.tensor(payload["affect"], device=device, dtype=torch.float32),
            alpha=float(payload["alpha"]),
        )

    def embedding_vector(self) -> Tensor:
        return torch.cat(
            (
                self.concept_bias,
                self.inhibition_bias,
                self.plan_slot_bias,
                self.affect,
                self.affect.new_tensor([self.alpha]),
            )
        )

    def project_logits(
        self,
        *,
        vocab_size: int,
        concept_ids: Sequence[int],
        plan_ids: Sequence[int],
        inhibition_ids: Sequence[int],
    ) -> Tensor:
        logits = torch.zeros(vocab_size, device=self.concept_bias.device)
        for slot, token_id in enumerate(concept_ids):
            if 0 <= token_id < vocab_size:
                logits[token_id] += float(self.concept_bias[slot].item())
        for slot, token_id in enumerate(plan_ids):
            if 0 <= token_id < vocab_size:
                logits[token_id] += float(self.plan_slot_bias[slot].item())
        for slot, token_id in enumerate(inhibition_ids):
            if 0 <= token_id < vocab_size:
                logits[token_id] -= float(self.inhibition_bias[slot].item())
        return logits


@dataclass
class ExecutivePoll:
    token_index: int
    influence: InfluenceSignal
    plan_coverage_ratio: float
    plan_coverage_bitmap: List[bool] = field(default_factory=list)
    chemical_levels: Dict[str, float] = field(default_factory=dict)
    selection_metadata: Dict[str, object] = field(default_factory=dict)
    monitor_action: Optional[str] = None
    production_mode: str = "influenced_articulator"

    def to_dict(self) -> dict:
        return {
            "token_index": self.token_index,
            "influence": self.influence.to_dict(),
            "plan_coverage_ratio": self.plan_coverage_ratio,
            "plan_coverage_bitmap": self.plan_coverage_bitmap,
            "chemical_levels": self.chemical_levels,
            "selection_metadata": self.selection_metadata,
            "monitor_action": self.monitor_action,
            "production_mode": self.production_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, device: str = "cpu") -> "ExecutivePoll":
        return cls(
            token_index=int(payload["token_index"]),
            influence=InfluenceSignal.from_dict(payload["influence"], device=device),
            plan_coverage_ratio=float(payload["plan_coverage_ratio"]),
            plan_coverage_bitmap=list(payload.get("plan_coverage_bitmap", [])),
            chemical_levels=dict(payload.get("chemical_levels", {})),
            selection_metadata=dict(payload.get("selection_metadata", {})),
            monitor_action=payload.get("monitor_action"),
            production_mode=str(payload.get("production_mode", "influenced_articulator")),
        )


def zero_influence(*, width: int, affect_size: int = 4, device: str = "cpu") -> InfluenceSignal:
    return InfluenceSignal(
        concept_bias=torch.zeros(width, device=device),
        inhibition_bias=torch.zeros(width, device=device),
        plan_slot_bias=torch.zeros(width, device=device),
        affect=torch.zeros(affect_size, device=device),
        alpha=0.0,
    )
