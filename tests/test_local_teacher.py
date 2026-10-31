"""Fast tests for the fail-closed local teacher boundary."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from brain_regions.evaluation.local_teacher import (
    LocalOllamaTeacher,
    TeacherRequest,
)


def _request(split: str = "train") -> TeacherRequest:
    return TeacherRequest(
        lesson_id="definition-kindness-1",
        task="grounded_definition",
        prompt="What is kindness?",
        semantic_family="definition:kindness",
        split=split,
        source_text="Kindness means acting with care toward another person.",
        source_refs=("lesson.txt:1",),
        meaning_plan=("kindness", "care", "another person"),
    )


def _valid_output() -> dict:
    return {
        "question_variants": [
            "What does kindness mean?",
            "How would you explain kindness?",
        ],
        "approved_responses": [
            "Kindness is acting with care toward another person.",
            "Kindness can mean treating another person with care.",
        ],
        "rejected_responses": ["Kindness means ignoring other people."],
        "plans": [
            ["kindness", "acting", "care", "person"],
            ["kindness", "treatment", "person", "care"],
        ],
        "correction": "",
        "reason_codes": ["grounded", "clear"],
        "source_claims": [
            "Kindness means acting with care toward another person."
        ],
    }


class FakeTransport:
    def __init__(self, output=None, *, models=True):
        self.output = output if output is not None else _valid_output()
        self.models = models
        self.generate_calls = 0

    def __call__(self, url, payload, timeout):
        if url.endswith("/api/tags"):
            return {
                "models": (
                    [
                        {
                            "name": "qwen2.5:3b-instruct",
                            "digest": "sha256:teacher-a",
                        }
                    ]
                    if self.models
                    else []
                )
            }
        self.generate_calls += 1
        return {"response": json.dumps(self.output)}


def test_teacher_cache_and_provenance_roundtrip(tmp_path):
    transport = FakeTransport()
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path, request_fn=transport
    )
    first = teacher.generate(_request())
    second = teacher.generate(_request())
    assert transport.generate_calls == 1
    assert not first.provenance.cache_hit
    assert second.provenance.cache_hit
    assert first.provenance.model_digest == "sha256:teacher-a"
    assert first.provenance.request_hash == second.provenance.request_hash
    assert len(first.output.approved_responses) == 2


def test_teacher_rejects_non_train_split_before_model_access(tmp_path):
    transport = FakeTransport()
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path, request_fn=transport
    )
    with pytest.raises(ValueError, match="train-split"):
        teacher.generate(_request(split="test"))
    assert transport.generate_calls == 0


def test_teacher_fails_when_required_model_is_missing(tmp_path):
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path, request_fn=FakeTransport(models=False)
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        teacher.generate(_request())


def test_teacher_rejects_malformed_json_and_schema(tmp_path):
    malformed = FakeTransport()

    def malformed_transport(url, payload, timeout):
        if url.endswith("/api/tags"):
            return malformed(url, payload, timeout)
        return {"response": "not-json"}

    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "malformed",
        request_fn=malformed_transport,
    )
    with pytest.raises(ValueError, match="malformed JSON"):
        teacher.generate(_request())

    incomplete = _valid_output()
    del incomplete["plans"]
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "incomplete",
        request_fn=FakeTransport(incomplete),
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        teacher.generate(_request())


def test_teacher_rejects_unverifiable_quotes_and_sentence_plans(tmp_path):
    unsupported = _valid_output()
    unsupported["source_claims"] = ["Kindness is an emotion."]
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "unsupported",
        request_fn=FakeTransport(unsupported),
    )
    with pytest.raises(ValueError, match="verbatim"):
        teacher.generate(_request())

    copied_plan = _valid_output()
    copied_plan["plans"][0] = ["is", "the", "of"]
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "copied",
        request_fn=FakeTransport(copied_plan),
    )
    with pytest.raises(ValueError, match="non-empty"):
        teacher.generate(_request())

    wrapped = _valid_output()
    wrapped["approved_responses"][0] = "{Kindness means acting with care.}"
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "wrapped",
        request_fn=FakeTransport(wrapped),
    )
    with pytest.raises(ValueError, match="wrapper"):
        teacher.generate(_request())


def test_teacher_requires_canonical_prompt_and_response_grounding(tmp_path):
    request = replace(
        _request(), canonical_prompts=("What is kindness?",)
    )
    missing_prompt = _valid_output()
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "prompt",
        request_fn=FakeTransport(missing_prompt),
    )
    with pytest.raises(ValueError, match="canonical"):
        teacher.generate(request)

    ungrounded = _valid_output()
    ungrounded["question_variants"][0] = "What is kindness?"
    ungrounded["approved_responses"][0] = (
        "Kindness controls quantum engines across distant galaxies."
    )
    teacher = LocalOllamaTeacher(
        cache_dir=tmp_path / "grounding",
        request_fn=FakeTransport(ungrounded),
    )
    with pytest.raises(ValueError, match="response-level"):
        teacher.generate(request)
