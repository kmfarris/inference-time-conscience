"""Bootstrap dialogue examples for articulator trajectories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from conversation_primer import CONVERSATION_PATTERNS
from definitions_primer import DEFINITIONS_PRIMER


@dataclass(frozen=True)
class DialogueExample:
    family: str
    split: str
    prompt: str
    response: str
    plan: tuple[str, ...]


_HELDOUT_FAMILIES = frozenset(
    {
        "definition:empathy",
        "conversation:how-are-you",
        "conversation:greetings-and-responses",
    }
)


def _definition_examples() -> Iterator[DialogueExample]:
    for line in DEFINITIONS_PRIMER.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z]+)\s+is\s+(.+)$", line)
        if not match:
            continue
        subject = match.group(1).lower()
        response = line[0].upper() + line[1:]
        if not response.endswith("."):
            response += "."
        family = f"definition:{subject}"
        split = "dev" if family in _HELDOUT_FAMILIES else "train"
        plan = tuple(token for token in response.lower().split() if token.isalpha())[:8]
        yield DialogueExample(
            family=family,
            split=split,
            prompt=f"what is {subject}?",
            response=response,
            plan=plan,
        )


def _conversation_examples() -> Iterator[DialogueExample]:
    current_trigger: Optional[str] = None
    for raw in CONVERSATION_PATTERNS.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("?"):
            current_trigger = line[:-1].lower().strip()
            continue
        if current_trigger and line.endswith("."):
            family_key = current_trigger.replace(" ", "-")[:40]
            family = f"conversation:{family_key}"
            split = "dev" if any(key in family for key in ("how-are-you", "hello", "greet")) else "train"
            if "how are you" in current_trigger:
                family = "conversation:how-are-you"
                split = "dev"
            if current_trigger.startswith("hello") or current_trigger.startswith("hi"):
                family = "conversation:greetings-and-responses"
                split = "dev"
            plan = tuple(token for token in line.lower().split() if token.isalpha())[:8]
            yield DialogueExample(
                family=family,
                split=split,
                prompt=current_trigger + "?",
                response=line,
                plan=plan,
            )


def bootstrap_dialogues() -> List[DialogueExample]:
    seen = set()
    rows: List[DialogueExample] = []
    for item in list(_definition_examples()) + list(_conversation_examples()):
        key = (item.family, item.prompt)
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
    return rows


def load_curriculum_jsonl(path: str | Path) -> List[DialogueExample]:
    target = Path(path)
    if not target.exists():
        return []
    rows: List[DialogueExample] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") != "candidate":
            continue
        payload = record["payload"]
        if payload.get("status") != "approved":
            continue
        prompt = payload["linguistic"]["prompt"]
        response = payload["linguistic"]["response"]
        plan = tuple(payload["linguistic"]["meaning_plan"]["ordered_concepts"])
        family = payload["semantic"]["family"]
        split = "train"
        rows.append(
            DialogueExample(
                family=family,
                split=split,
                prompt=prompt,
                response=response,
                plan=plan,
            )
        )
    return rows


def all_training_examples(curriculum_path: Optional[str | Path] = None) -> List[DialogueExample]:
    rows = bootstrap_dialogues()
    if curriculum_path:
        rows.extend(load_curriculum_jsonl(curriculum_path))
    dedup = {}
    for item in rows:
        dedup[(item.family, item.prompt)] = item
    return list(dedup.values())


def split_examples(examples: Sequence[DialogueExample], split: str) -> List[DialogueExample]:
    return [item for item in examples if item.split == split]
