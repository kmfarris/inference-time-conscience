"""Tests for conscience curriculum assembly without Ollama."""

from __future__ import annotations

from brain_regions.articulator.conscience_curriculum import (
    build_conscience_trajectories,
    candidates_from_result,
    generate_conscience_curriculum,
    seed_to_request,
)
from brain_regions.articulator.conscience_seeds import all_conscience_seeds
from brain_regions.evaluation.local_teacher import (
    TeacherOutput,
    TeacherProvenance,
    TeacherRequest,
    TeacherResult,
)


class FakeTeacher:
    def generate(self, request: TeacherRequest) -> TeacherResult:
        output = TeacherOutput(
            question_variants=(
                request.canonical_prompts[0],
                f"please explain {request.semantic_family}",
                f"tell me about {request.semantic_family}",
            ),
            approved_responses=(
                request.source_text.split(".")[0] + ".",
                request.source_text.split(".")[0] + " It matters.",
                request.source_text.split(".")[0] + " That is important.",
            ),
            rejected_responses=(
                "That answer ignores the source.",
                "That response changes the subject.",
                "That reply is unsupported.",
            ),
            plans=(
                tuple(request.meaning_plan[:4]) or ("response", "meaning", "source"),
                tuple(request.meaning_plan[:4]) or ("response", "meaning", "source"),
                tuple(request.meaning_plan[:4]) or ("response", "meaning", "source"),
            ),
            correction=request.source_text.split(".")[0] + ".",
            reason_codes=("unsupported",),
            source_claims=(request.source_text.splitlines()[0],),
        )
        provenance = TeacherProvenance(
            model="fake-teacher",
            model_digest="sha256:fake",
            prompt_version="fake-v1",
            temperature=0.0,
            seed=0,
            source_hash="sha256:fake",
            request_hash="sha256:fake",
            cache_hit=False,
        )
        return TeacherResult(request, output, provenance)


def test_seed_inventory_has_train_rows():
    seeds = all_conscience_seeds(include_dilemmas=True)
    assert len(seeds) >= 10
    assert all(seed.split == "train" for seed in seeds)


def test_candidates_from_teacher_result():
    seed = all_conscience_seeds(include_dilemmas=False)[0]
    result = FakeTeacher().generate(seed_to_request(seed))
    candidates, preferences = candidates_from_result(seed, result)
    assert len(candidates) == 6
    assert len(preferences) == 3
    assert all(item.conscience == seed.conscience for item in candidates)


def test_generate_and_build_trajectories(tmp_path):
    output = tmp_path / "conscience"
    manifest = generate_conscience_curriculum(
        FakeTeacher(),
        output / "curriculum",
        seeds=all_conscience_seeds(include_dilemmas=False)[:3],
    )
    assert manifest["approved_candidate_count"] == 9
    traj = build_conscience_trajectories(
        output / "curriculum",
        output / "trajectories",
    )
    assert traj["trajectory_count"] == 9
    assert (output / "trajectories" / "train.jsonl").exists()
