"""Turn-level executive state and per-token SNN poll for articulator decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from brain_regions.articulator.influence import ExecutivePoll, InfluenceSignal
from guidance import GuidanceSignal
from response_guidance import ResponseGuidance
from tokenizer import clean_tokenize


@dataclass
class TurnExecutiveState:
    plan_tokens: List[str]
    plan_ids: List[int]
    plan_coverage: List[bool]
    concept_slots: List[Tuple[str, float]]
    inhibition_tokens: List[str]
    guidance_valence: float = 0.0
    inhibitory_suppression: float = 0.0
    chemical_levels: Dict[str, float] = field(default_factory=dict)
    conversational_mode: str = "general"
    energy_level: float = 0.5
    confidence: float = 0.5


class ExecutivePoller:
    """Poll executive influence each token without recomputing conscience state."""

    def __init__(
        self,
        *,
        token_to_id: Dict[str, int],
        concept_slots: int = 8,
        device: str = "cpu",
        seed: int = 2026,
    ):
        self.token_to_id = token_to_id
        self.concept_slots = concept_slots
        self.device = device
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        self._state: Optional[TurnExecutiveState] = None

    def begin_turn(
        self,
        *,
        guidance: GuidanceSignal,
        response_guidance: Optional[ResponseGuidance],
        chemical_levels: Optional[Dict[str, float]],
        plan_tokens: Sequence[str],
    ) -> TurnExecutiveState:
        concepts = list(guidance.top_concepts[: self.concept_slots])
        while len(concepts) < self.concept_slots:
            concepts.append(("", 0.0))
        plan = [token for token in plan_tokens if token]
        coverage = [False] * len(plan)
        inhibition = list(guidance.suppressed_concepts[: self.concept_slots])
        mode = "general"
        energy = 0.5
        confidence = 0.5
        if response_guidance is not None:
            mode = response_guidance.conversational_mode
            energy = float(response_guidance.energy_level)
            confidence = float(response_guidance.confidence)
        self._state = TurnExecutiveState(
            plan_tokens=plan,
            plan_ids=[self.token_to_id.get(token, self.token_to_id.get("<unk>", 0)) for token in plan],
            plan_coverage=coverage,
            concept_slots=concepts,
            inhibition_tokens=inhibition,
            guidance_valence=float(guidance.emotional_valence),
            inhibitory_suppression=float(guidance.inhibitory_suppression),
            chemical_levels=dict(chemical_levels or {}),
            conversational_mode=mode,
            energy_level=energy,
            confidence=confidence,
        )
        return self._state

    def _current_plan_slot(self) -> int:
        assert self._state is not None
        for index, covered in enumerate(self._state.plan_coverage):
            if not covered:
                return index
        return max(0, len(self._state.plan_coverage) - 1)

    def _advance_plan(self, emitted_token: str) -> None:
        assert self._state is not None
        if not self._state.plan_tokens:
            return
        slot = self._current_plan_slot()
        if slot >= len(self._state.plan_tokens):
            return
        target = self._state.plan_tokens[slot].lower()
        if emitted_token.lower() == target:
            self._state.plan_coverage[slot] = True

    def _alpha(self) -> float:
        assert self._state is not None
        cortisol = float(self._state.chemical_levels.get("cortisol", 0.1))
        dopamine = float(self._state.chemical_levels.get("dopamine", 0.15))
        base = 0.25 + 0.35 * self._state.confidence
        base += 0.15 * self._state.energy_level
        base += 0.20 * self._state.inhibitory_suppression
        base -= 0.10 * cortisol
        base += 0.05 * dopamine
        return float(max(0.05, min(0.85, base)))

    def poll(
        self,
        emitted_tokens: Sequence[str],
        token_index: int,
        *,
        monitor_action: Optional[str] = None,
    ) -> ExecutivePoll:
        if self._state is None:
            raise RuntimeError("ExecutivePoller.begin_turn() must be called first")
        if token_index > 0 and emitted_tokens:
            self._advance_plan(emitted_tokens[-1])

        concept_bias = torch.zeros(self.concept_slots, device=self.device)
        for slot, (word, strength) in enumerate(self._state.concept_slots):
            concept_bias[slot] = float(strength)

        inhibition_bias = torch.zeros(self.concept_slots, device=self.device)
        for slot, word in enumerate(self._state.inhibition_tokens[: self.concept_slots]):
            inhibition_bias[slot] = 1.0 if word else 0.0

        plan_slot_bias = torch.zeros(self.concept_slots, device=self.device)
        if self._state.plan_tokens:
            slot = self._current_plan_slot()
            plan_slot_bias[min(slot, self.concept_slots - 1)] = 2.0
            if slot + 1 < self.concept_slots:
                plan_slot_bias[slot + 1] = 0.5

        affect = torch.tensor(
            [
                self._state.guidance_valence,
                self._state.energy_level,
                self._state.confidence,
                self._state.inhibitory_suppression,
            ],
            device=self.device,
            dtype=torch.float32,
        )

        influence = InfluenceSignal(
            concept_bias=concept_bias,
            inhibition_bias=inhibition_bias,
            plan_slot_bias=plan_slot_bias,
            affect=affect,
            alpha=self._alpha(),
        )
        covered = sum(1 for flag in self._state.plan_coverage if flag)
        ratio = covered / max(1, len(self._state.plan_coverage))
        return ExecutivePoll(
            token_index=token_index,
            influence=influence,
            plan_coverage_ratio=ratio,
            plan_coverage_bitmap=list(self._state.plan_coverage),
            chemical_levels=dict(self._state.chemical_levels),
            selection_metadata={
                "conversational_mode": self._state.conversational_mode,
                "plan_slot": self._current_plan_slot(),
            },
            monitor_action=monitor_action,
        )

    def concept_ids(self) -> List[int]:
        assert self._state is not None
        return [
            self.token_to_id.get(word, self.token_to_id.get("<unk>", 0))
            for word, _ in self._state.concept_slots
        ]

    def plan_ids(self) -> List[int]:
        assert self._state is not None
        return list(self._state.plan_ids)

    def inhibition_ids(self) -> List[int]:
        assert self._state is not None
        return [
            self.token_to_id.get(word, self.token_to_id.get("<unk>", 0))
            for word in self._state.inhibition_tokens[: self.concept_slots]
        ]


def plan_tokens_from_response(response: str, *, max_tokens: int = 8) -> List[str]:
    tokens = clean_tokenize(response.lower(), stem=False)
    content = [token for token in tokens if token not in {"i", "am", "the", "a", "an", "to", "of"}]
    return content[:max_tokens]
