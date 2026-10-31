"""Offline teacher-curriculum assembly (validation only, no generation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class LinguisticMetadata:
    pos_tags: Tuple[Tuple[str, str], ...]
    chunks: Tuple[Tuple[str, ...], ...]
    meaning_plan: dict


@dataclass(frozen=True)
class TeacherCandidate:
    candidate_id: str
    prompt: str
    response: str
    semantic_family: str
    split: str
    approved: bool
    source_refs: Tuple[str, ...]
    linguistic: LinguisticMetadata
    soft_targets: dict
    provenance: dict


@dataclass(frozen=True)
class PreferenceSpec:
    pair_id: str
    approved_id: str
    rejected_id: str


@dataclass(frozen=True)
class GroundingDecision:
    grounded: bool
    reason: str = ""


@dataclass
class TeacherDataset:
    candidates: List[TeacherCandidate] = field(default_factory=list)
    preference_pairs: List[PreferenceSpec] = field(default_factory=list)
    duplicate_candidate_ids: List[str] = field(default_factory=list)


GroundingHook = Callable[[TeacherCandidate], GroundingDecision]


def deduplicate_candidates(
    candidates: Sequence[TeacherCandidate],
) -> Tuple[List[TeacherCandidate], List[str]]:
    seen = {}
    duplicates: List[str] = []
    unique: List[TeacherCandidate] = []
    for candidate in candidates:
        key = (candidate.prompt, candidate.response, candidate.semantic_family)
        if key in seen:
            duplicates.append(candidate.candidate_id)
            continue
        seen[key] = candidate.candidate_id
        unique.append(candidate)
    return unique, duplicates


def assert_no_teacher_leakage(candidates: Sequence[TeacherCandidate]) -> None:
    for candidate in candidates:
        if candidate.split != "train":
            raise ValueError(
                f"teacher candidate {candidate.candidate_id} uses split {candidate.split}"
            )


def build_teacher_dataset(
    candidates: Sequence[TeacherCandidate],
    preference_pairs: Sequence[PreferenceSpec],
    *,
    grounding_hooks: Iterable[GroundingHook] = (),
) -> TeacherDataset:
    unique, duplicates = deduplicate_candidates(candidates)
    assert_no_teacher_leakage(unique)
    for candidate in unique:
        if not candidate.approved:
            continue
        for hook in grounding_hooks:
            decision = hook(candidate)
            if not decision.grounded:
                raise ValueError(
                    f"candidate {candidate.candidate_id} failed grounding: {decision.reason}"
                )
    return TeacherDataset(
        candidates=list(unique),
        preference_pairs=list(preference_pairs),
        duplicate_candidate_ids=duplicates,
    )
