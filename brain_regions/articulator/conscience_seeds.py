"""Explicit lesson seeds for SNN conscience training data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from brain_regions.articulator.data import DialogueExample, bootstrap_dialogues
from definitions_primer import DEFINITIONS_PRIMER


@dataclass(frozen=True)
class ConscienceProfile:
    golden_lantern_fired: Tuple[str, ...]
    golden_lantern_activations: Tuple[Tuple[str, float], ...]
    suppressed_concepts: Tuple[str, ...]
    inhibitory_suppression: float
    chemical_levels: Tuple[Tuple[str, float], ...]

    def to_dict(self) -> dict:
        return {
            "golden_lantern_fired": list(self.golden_lantern_fired),
            "golden_lantern_activations": dict(self.golden_lantern_activations),
            "suppressed_concepts": list(self.suppressed_concepts),
            "inhibitory_suppression": self.inhibitory_suppression,
            "chemical_levels": dict(self.chemical_levels),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConscienceProfile":
        return cls(
            golden_lantern_fired=tuple(payload.get("golden_lantern_fired", ())),
            golden_lantern_activations=tuple(
                (key, float(value))
                for key, value in payload.get("golden_lantern_activations", {}).items()
            ),
            suppressed_concepts=tuple(payload.get("suppressed_concepts", ())),
            inhibitory_suppression=float(payload.get("inhibitory_suppression", 0.0)),
            chemical_levels=tuple(
                (key, float(value))
                for key, value in payload.get("chemical_levels", {}).items()
            ),
        )


@dataclass(frozen=True)
class ConscienceLessonSeed:
    lesson_id: str
    family: str
    split: str
    task: str
    prompt: str
    source_text: str
    source_refs: Tuple[str, ...]
    canonical_prompts: Tuple[str, ...]
    meaning_plan: Tuple[str, ...]
    conscience: ConscienceProfile


_DEFAULT_CHEMICALS = (
    ("oxytocin", 0.2),
    ("cortisol", 0.1),
    ("dopamine", 0.15),
    ("serotonin", 0.4),
)


def _profile(
    *,
    fired: Sequence[str],
    activations: Sequence[Tuple[str, float]] | None = None,
    suppressed: Sequence[str] = (),
    inhibitory_suppression: float = 0.0,
    chemicals: Sequence[Tuple[str, float]] | None = None,
) -> ConscienceProfile:
    activation_rows = activations or tuple((word, 1.0) for word in fired)
    return ConscienceProfile(
        golden_lantern_fired=tuple(fired),
        golden_lantern_activations=tuple(activation_rows),
        suppressed_concepts=tuple(suppressed),
        inhibitory_suppression=float(inhibitory_suppression),
        chemical_levels=tuple(chemicals or _DEFAULT_CHEMICALS),
    )


# Explicit conscience profile per semantic family (not inferred at runtime).
CONSCIENCE_BY_FAMILY: Dict[str, ConscienceProfile] = {
    "definition:compassion": _profile(fired=("compassion", "empathy"), activations=(("compassion", 1.0), ("empathy", 0.8))),
    "definition:love": _profile(fired=("love", "compassion"), activations=(("love", 1.0), ("compassion", 0.6))),
    "definition:wisdom": _profile(fired=("wisdom", "truth"), activations=(("wisdom", 1.0), ("truth", 0.7))),
    "definition:truth": _profile(fired=("truth", "logic"), activations=(("truth", 1.0), ("logic", 0.6))),
    "definition:logic": _profile(fired=("logic", "truth"), activations=(("logic", 1.0), ("truth", 0.5))),
    "definition:patience": _profile(fired=("patience", "understanding"), activations=(("patience", 1.0), ("understanding", 0.6))),
    "definition:suffering": _profile(fired=("suffering", "compassion"), activations=(("suffering", 1.0), ("compassion", 0.8))),
    "definition:understanding": _profile(fired=("understanding", "wisdom"), activations=(("understanding", 1.0), ("wisdom", 0.6))),
    "definition:creation": _profile(fired=("creation", "foster"), activations=(("creation", 1.0), ("foster", 0.7))),
    "definition:dehumanization": _profile(
        fired=("compassion", "empathy"),
        suppressed=("dehumanize", "subhuman"),
        inhibitory_suppression=0.45,
    ),
    "definition:darkness": _profile(fired=("truth", "wisdom"), suppressed=("darkness",), inhibitory_suppression=0.25),
    "definition:fear": _profile(fired=("patience", "understanding"), suppressed=("fear",), inhibitory_suppression=0.2),
    "definition:freedom": _profile(fired=("truth", "understanding"), activations=(("freedom", 1.0), ("truth", 0.5))),
    "definition:justice": _profile(fired=("truth", "compassion"), activations=(("justice", 1.0), ("truth", 0.6), ("compassion", 0.5))),
    "definition:knowledge": _profile(fired=("understanding", "wisdom"), activations=(("knowledge", 1.0), ("understanding", 0.6))),
    "conversation:can-you-give-me-an-example": _profile(fired=("understanding",), activations=(("understanding", 0.5),)),
    "conversation:how-are-things": _profile(fired=("compassion",), activations=(("compassion", 0.4),)),
    "conversation:i-am-present.-i-am-listening.-what-would": _profile(fired=("compassion", "patience"), activations=(("compassion", 0.6), ("patience", 0.5))),
    "conversation:what-do-you-mean-by-that": _profile(fired=("understanding", "patience"), activations=(("understanding", 0.6), ("patience", 0.4))),
}


DILEMMA_SOURCE_FILES: Tuple[Tuple[str, str], ...] = (
    ("dilemma:euthanasia", "training_data/foundation/universal_truths.txt"),
    ("dilemma:self-defense", "training_data/dark_curriculum/wiki_The_Holocaust.txt"),
    ("dilemma:war", "training_data/dark_curriculum/Twain_-_The_War_Prayer.txt"),
    ("dilemma:deception", "training_data/dark_curriculum/wiki_Propaganda.txt"),
    ("dilemma:civil-disobedience", "training_data/dark_curriculum/Douglass_-_Narrative_of_the_Life.txt"),
    ("dilemma:quarantine", "training_data/definitions/definitions_primer.txt"),
    ("dilemma:whistleblowing", "training_data/dark_curriculum/wiki_White_Rose.txt"),
    ("dilemma:just-war", "training_data/dark_curriculum/Twain_-_The_War_Prayer.txt"),
)


DILEMMA_PROMPTS: Dict[str, str] = {
    "dilemma:euthanasia": "Should a terminally ill person in great pain be allowed to choose to die?",
    "dilemma:self-defense": "If someone attacks me and I fight back, is that wrong?",
    "dilemma:war": "Is war ever justified to protect innocent people?",
    "dilemma:deception": "Is it ever right to lie to protect someone's feelings?",
    "dilemma:civil-disobedience": "Is breaking an unjust law the right thing to do?",
    "dilemma:quarantine": "Is isolating a sick person an act of compassion or an act of harm?",
    "dilemma:whistleblowing": "Should a government employee leak classified information showing crimes?",
    "dilemma:just-war": "Can destroying a weapons factory be an act of protection?",
}


DILEMMA_CONSCIENCE: Dict[str, ConscienceProfile] = {
    "dilemma:euthanasia": _profile(fired=("compassion", "suffering", "patience"), inhibitory_suppression=0.2),
    "dilemma:self-defense": _profile(fired=("compassion", "logic"), suppressed=("malice",), inhibitory_suppression=0.15),
    "dilemma:war": _profile(fired=("compassion", "truth"), suppressed=("destruction",), inhibitory_suppression=0.25),
    "dilemma:deception": _profile(fired=("truth", "compassion"), inhibitory_suppression=0.1),
    "dilemma:civil-disobedience": _profile(fired=("truth", "justice"), inhibitory_suppression=0.15),
    "dilemma:quarantine": _profile(fired=("compassion", "logic"), inhibitory_suppression=0.1),
    "dilemma:whistleblowing": _profile(fired=("truth", "justice"), inhibitory_suppression=0.2),
    "dilemma:just-war": _profile(fired=("compassion", "logic"), suppressed=("destruction",), inhibitory_suppression=0.2),
}


def _read_source(path: str, *, max_chars: int = 4000) -> str:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"missing conscience source: {path}")
    text = target.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    if not text:
        raise ValueError(f"empty conscience source: {path}")
    return text


def _definition_source_text() -> str:
    return DEFINITIONS_PRIMER.strip()


def seeds_from_bootstrap() -> List[ConscienceLessonSeed]:
    rows: List[ConscienceLessonSeed] = []
    source_text = _definition_source_text()
    for example in bootstrap_dialogues():
        if example.split != "train":
            continue
        conscience = CONSCIENCE_BY_FAMILY.get(example.family)
        if conscience is None:
            continue
        rows.append(
            ConscienceLessonSeed(
                lesson_id=f"bootstrap-{example.family.replace(':', '-')}",
                family=example.family,
                split=example.split,
                task=(
                    "grounded_definition"
                    if example.family.startswith("definition:")
                    else "grounded_conversation"
                ),
                prompt=example.prompt,
                source_text=source_text if example.family.startswith("definition:") else example.response,
                source_refs=(
                    "definitions_primer.py"
                    if example.family.startswith("definition:")
                    else example.family
                ),
                canonical_prompts=(example.prompt,),
                meaning_plan=tuple(example.plan),
                conscience=conscience,
            )
        )
    return rows


def seeds_from_dilemmas() -> List[ConscienceLessonSeed]:
    rows: List[ConscienceLessonSeed] = []
    for family, source_path in DILEMMA_SOURCE_FILES:
        prompt = DILEMMA_PROMPTS[family]
        conscience = DILEMMA_CONSCIENCE[family]
        source_text = _read_source(source_path)
        rows.append(
            ConscienceLessonSeed(
                lesson_id=f"{family.replace(':', '-')}",
                family=family,
                split="train",
                task="grounded_conversation",
                prompt=prompt,
                source_text=source_text,
                source_refs=(source_path,),
                canonical_prompts=(prompt,),
                meaning_plan=tuple(token for token in prompt.lower().split() if token.isalpha())[:8],
                conscience=conscience,
            )
        )
    return rows


def all_conscience_seeds(*, include_dilemmas: bool = True) -> List[ConscienceLessonSeed]:
    rows = seeds_from_bootstrap()
    if include_dilemmas:
        rows.extend(seeds_from_dilemmas())
    dedup = {}
    for seed in rows:
        dedup[seed.lesson_id] = seed
    return list(dedup.values())
