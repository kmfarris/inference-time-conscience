"""Trajectory recording and bootstrap dataset generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from brain_regions.articulator.data import DialogueExample, all_training_examples, split_examples
from brain_regions.articulator.conscience_seeds import ConscienceProfile
from brain_regions.articulator.influence import ExecutivePoll
from brain_regions.articulator.poller import ExecutivePoller, plan_tokens_from_response
from brain_regions.articulator.vocab import ArticulatorTokenizer
from guidance import GuidanceSignal
from response_guidance import ResponseGuidance
from tokenizer import clean_tokenize


@dataclass
class TrajectoryStep:
    token_index: int
    token: str
    poll: ExecutivePoll
    prefix: List[str] = field(default_factory=list)


@dataclass
class TrajectoryRecord:
    prompt: str
    response: str
    plan: List[str]
    family: str
    split: str
    tokens: List[str]
    steps: List[TrajectoryStep]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "plan": self.plan,
            "family": self.family,
            "split": self.split,
            "tokens": self.tokens,
            "polls": [
                {
                    "t": step.token_index,
                    "token": step.token,
                    "prefix": step.prefix,
                    "poll": step.poll.to_dict(),
                }
                for step in self.steps
            ],
        }


class TrajectoryRecorder:
    def __init__(self) -> None:
        self.records: List[TrajectoryRecord] = []

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _synthetic_guidance(example: DialogueExample) -> GuidanceSignal:
    concepts = [(token, 1.0 - index * 0.05) for index, token in enumerate(example.plan[:8])]
    while len(concepts) < 8:
        concepts.append(("", 0.0))
    return GuidanceSignal(
        top_concepts=concepts,
        golden_lantern_activations={},
        golden_lantern_fired=set(),
        emotional_valence=0.2,
        topic_clusters=list(example.plan[:3]),
        suppressed_concepts=[],
        inhibitory_suppression=0.0,
        raw_inference=None,
    )


def _synthetic_response_guidance(example: DialogueExample) -> ResponseGuidance:
    mode = "definition" if example.family.startswith("definition:") else "exploration"
    return ResponseGuidance(
        conversational_mode=mode,
        target_sentence_count=2,
        target_sentence_length="medium",
        energy_level=0.5,
        personal_investment=0.4,
        concept_priority=[(word, 1.0) for word in example.plan[:5]],
        confidence=0.6,
        intent_summary=f"trajectory:{example.family}",
    )


def record_conscience_trajectory(
    example: DialogueExample,
    *,
    conscience: ConscienceProfile,
    tokenizer: ArticulatorTokenizer,
    device: str = "cpu",
) -> TrajectoryRecord:
    concepts = [
        (word, float(strength))
        for word, strength in conscience.golden_lantern_activations
    ]
    while len(concepts) < 8:
        concepts.append(("", 0.0))
    guidance = GuidanceSignal(
        top_concepts=concepts[:8],
        golden_lantern_activations=dict(conscience.golden_lantern_activations),
        golden_lantern_fired=set(conscience.golden_lantern_fired),
        emotional_valence=0.0,
        topic_clusters=list(example.plan[:3]),
        suppressed_concepts=list(conscience.suppressed_concepts),
        inhibitory_suppression=conscience.inhibitory_suppression,
        raw_inference=None,
    )
    response_guidance = _synthetic_response_guidance(example)
    plan = list(example.plan) or plan_tokens_from_response(example.response)
    poller = ExecutivePoller(token_to_id=tokenizer.token_to_id, device=device)
    poller.begin_turn(
        guidance=guidance,
        response_guidance=response_guidance,
        chemical_levels=dict(conscience.chemical_levels),
        plan_tokens=plan,
    )
    tokens = clean_tokenize(example.response.lower(), stem=False)
    steps: List[TrajectoryStep] = []
    prefix: List[str] = []
    for index, token in enumerate(tokens):
        poll = poller.poll(prefix, index)
        steps.append(TrajectoryStep(token_index=index, token=token, poll=poll, prefix=list(prefix)))
        prefix.append(token)
    return TrajectoryRecord(
        prompt=example.prompt,
        response=example.response,
        plan=plan,
        family=example.family,
        split=example.split,
        tokens=tokens,
        steps=steps,
    )


def record_example_trajectory(
    example: DialogueExample,
    *,
    tokenizer: ArticulatorTokenizer,
    device: str = "cpu",
) -> TrajectoryRecord:
    guidance = _synthetic_guidance(example)
    response_guidance = _synthetic_response_guidance(example)
    plan = list(example.plan) or plan_tokens_from_response(example.response)
    poller = ExecutivePoller(token_to_id=tokenizer.token_to_id, device=device)
    poller.begin_turn(
        guidance=guidance,
        response_guidance=response_guidance,
        chemical_levels={"oxytocin": 0.2, "cortisol": 0.1, "dopamine": 0.15, "serotonin": 0.4},
        plan_tokens=plan,
    )
    tokens = clean_tokenize(example.response.lower(), stem=False)
    steps: List[TrajectoryStep] = []
    prefix: List[str] = []
    for index, token in enumerate(tokens):
        poll = poller.poll(prefix, index)
        steps.append(TrajectoryStep(token_index=index, token=token, poll=poll, prefix=list(prefix)))
        prefix.append(token)
    return TrajectoryRecord(
        prompt=example.prompt,
        response=example.response,
        plan=plan,
        family=example.family,
        split=example.split,
        tokens=tokens,
        steps=steps,
    )


def build_bootstrap_trajectories(
    output_dir: str | Path,
    *,
    curriculum_path: Optional[str | Path] = None,
    device: str = "cpu",
) -> dict:
    examples = all_training_examples(curriculum_path)
    texts = [example.prompt + " " + example.response for example in examples]
    tokenizer = ArticulatorTokenizer.build(texts)
    recorder = TrajectoryRecorder()
    for example in examples:
        recorder.add(record_example_trajectory(example, tokenizer=tokenizer, device=device))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer.save(root / "tokenizer.json")
    train_rows = [record for record in recorder.records if record.split == "train"]
    dev_rows = [record for record in recorder.records if record.split != "train"]
    train_recorder = TrajectoryRecorder()
    dev_recorder = TrajectoryRecorder()
    for row in train_rows:
        train_recorder.add(row)
    for row in dev_rows:
        dev_recorder.add(row)
    train_recorder.write_jsonl(root / "trajectories" / "train.jsonl")
    dev_recorder.write_jsonl(root / "trajectories" / "dev.jsonl")
    manifest = {
        "schema_version": 1,
        "example_count": len(examples),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "vocab_size": len(tokenizer.tokens),
        "curriculum_path": str(curriculum_path) if curriculum_path else None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
