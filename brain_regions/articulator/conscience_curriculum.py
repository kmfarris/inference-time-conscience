"""Generate teacher-augmented SNN conscience curriculum and influence trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from brain_regions.articulator.conscience_seeds import (
    ConscienceLessonSeed,
    ConscienceProfile,
    all_conscience_seeds,
)
from brain_regions.articulator.data import DialogueExample
from brain_regions.articulator.recorder import TrajectoryRecorder, record_conscience_trajectory
from brain_regions.articulator.vocab import ArticulatorTokenizer
from brain_regions.evaluation.artifacts import write_json_artifact
from brain_regions.evaluation.local_teacher import (
    LocalOllamaTeacher,
    TeacherRequest,
    TeacherResult,
)
from brain_regions.evaluation.teacher import (
    GroundingDecision,
    LinguisticMetadata,
    PreferenceSpec,
    TeacherCandidate,
    TeacherDataset,
    build_teacher_dataset,
)
from tokenizer import clean_tokenize


class TeacherClient(Protocol):
    def generate(self, request: TeacherRequest) -> TeacherResult: ...


@dataclass(frozen=True)
class ConscienceCandidate:
    candidate_id: str
    prompt: str
    response: str
    family: str
    split: str
    plan: tuple[str, ...]
    approved: bool
    conscience: ConscienceProfile
    source_refs: tuple[str, ...]
    provenance: dict


def seed_to_request(seed: ConscienceLessonSeed) -> TeacherRequest:
    return TeacherRequest(
        lesson_id=seed.lesson_id,
        task=seed.task,
        prompt=seed.prompt,
        semantic_family=seed.family,
        split=seed.split,
        source_text=seed.source_text,
        source_refs=seed.source_refs,
        canonical_prompts=seed.canonical_prompts,
        meaning_plan=seed.meaning_plan,
    )


def _candidate_id(family: str, label: str, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{family.replace(':', '-')}-{label}-{index}-{digest}"


def _linguistic(response: str, plan: Sequence[str], family: str) -> LinguisticMetadata:
    tokens = clean_tokenize(response.lower(), stem=False)
    return LinguisticMetadata(
        pos_tags=tuple((token, "X") for token in tokens),
        chunks=(tuple(plan),),
        meaning_plan={
            "semantic_family": family,
            "ordered_concepts": list(plan),
        },
    )


def candidates_from_result(
    seed: ConscienceLessonSeed,
    result: TeacherResult,
) -> tuple[List[ConscienceCandidate], List[PreferenceSpec]]:
    output = result.output
    candidates: List[ConscienceCandidate] = []
    preferences: List[PreferenceSpec] = []
    common_provenance = {
        **asdict(result.provenance),
        "lesson_id": seed.lesson_id,
        "task": seed.task,
        "teacher_only": True,
    }
    for index, approved_response in enumerate(output.approved_responses):
        prompt = output.question_variants[index % len(output.question_variants)]
        plan = tuple(output.plans[index])
        rejected = output.rejected_responses[index % len(output.rejected_responses)]
        approved_id = _candidate_id(seed.family, "approved", index, approved_response)
        rejected_id = _candidate_id(seed.family, "rejected", index, rejected)
        candidates.extend(
            (
                ConscienceCandidate(
                    candidate_id=approved_id,
                    prompt=prompt,
                    response=approved_response,
                    family=seed.family,
                    split=seed.split,
                    plan=plan,
                    approved=True,
                    conscience=seed.conscience,
                    source_refs=seed.source_refs,
                    provenance={
                        **common_provenance,
                        "source_claims": list(output.source_claims),
                    },
                ),
                ConscienceCandidate(
                    candidate_id=rejected_id,
                    prompt=prompt,
                    response=rejected,
                    family=seed.family,
                    split=seed.split,
                    plan=plan,
                    approved=False,
                    conscience=seed.conscience,
                    source_refs=seed.source_refs,
                    provenance={
                        **common_provenance,
                        "rejection_reason_codes": list(output.reason_codes),
                    },
                ),
            )
        )
        preferences.append(
            PreferenceSpec(
                pair_id=f"{seed.family.replace(':', '-')}-pair-{index}",
                approved_id=approved_id,
                rejected_id=rejected_id,
            )
        )
    return candidates, preferences


def _grounded(candidate: ConscienceCandidate) -> GroundingDecision:
    if not candidate.source_refs:
        return GroundingDecision(False, "missing source refs")
    if not candidate.provenance.get("model_digest") and candidate.provenance.get("model") != "deterministic-grounded-fallback":
        return GroundingDecision(False, "missing teacher provenance")
    return GroundingDecision(True)


def _json_record(record_type: str, payload: object) -> str:
    body = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    return json.dumps(
        {"record_type": record_type, "payload": body},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_conscience_curriculum(
    output_dir: str | Path,
    candidates: Sequence[ConscienceCandidate],
    preferences: Sequence[PreferenceSpec],
    *,
    failures: Sequence[dict],
    results: Sequence[TeacherResult],
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lines = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        payload = {
            "status": "approved" if candidate.approved else "rejected",
            "linguistic": {
                "prompt": candidate.prompt,
                "response": candidate.response,
                "meaning_plan": {"ordered_concepts": list(candidate.plan)},
            },
            "semantic": {"family": candidate.family},
            "conscience": candidate.conscience.to_dict(),
            "split": candidate.split,
            "candidate_id": candidate.candidate_id,
            "source_refs": list(candidate.source_refs),
            "provenance": candidate.provenance,
        }
        lines.append(_json_record("candidate", payload))
    for pair in sorted(preferences, key=lambda item: item.pair_id):
        lines.append(_json_record("preference", asdict(pair)))
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    lessons_path = output / "teacher_lessons.jsonl"
    lessons_path.write_bytes(encoded)
    manifest = {
        "schema_version": 1,
        "artifact_type": "conscience_curriculum",
        "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
        "candidate_count": len(candidates),
        "approved_candidate_count": sum(1 for item in candidates if item.approved),
        "preference_count": len(preferences),
        "generated_lesson_count": len(results),
        "failed_lesson_count": len(failures),
        "failures": list(failures),
        "model_digests": sorted({result.provenance.model_digest for result in results}),
    }
    write_json_artifact(output / "manifest.json", manifest)
    return manifest


def generate_conscience_curriculum(
    teacher: TeacherClient,
    output_dir: str | Path,
    *,
    seeds: Optional[Iterable[ConscienceLessonSeed]] = None,
    max_lessons: Optional[int] = None,
) -> dict:
    selected = list(seeds or all_conscience_seeds())
    if max_lessons is not None:
        selected = selected[:max_lessons]
    results: List[TeacherResult] = []
    failures: List[dict] = []
    candidates: List[ConscienceCandidate] = []
    preferences: List[PreferenceSpec] = []
    for index, seed in enumerate(selected, start=1):
        print(f"[conscience] lesson={index}/{len(selected)} {seed.lesson_id}", flush=True)
        request = seed_to_request(seed)
        try:
            result = teacher.generate(request)
            lesson_candidates, lesson_preferences = candidates_from_result(seed, result)
            results.append(result)
            candidates.extend(lesson_candidates)
            preferences.extend(lesson_preferences)
        except Exception as exc:
            failures.append(
                {
                    "lesson_id": seed.lesson_id,
                    "family": seed.family,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[conscience] rejected lesson={seed.lesson_id} error={failures[-1]['error']}",
                flush=True,
            )
    teacher_candidates = [
        TeacherCandidate(
            candidate_id=item.candidate_id,
            prompt=item.prompt,
            response=item.response,
            semantic_family=item.family,
            split=item.split,
            approved=item.approved,
            source_refs=item.source_refs,
            linguistic=_linguistic(item.response, item.plan, item.family),
            soft_targets={"grounding": 1.0 if item.approved else 0.0},
            provenance=item.provenance,
        )
        for item in candidates
    ]
    build_teacher_dataset(
        teacher_candidates,
        preferences,
        grounding_hooks=(_grounded,),
    )
    return write_conscience_curriculum(
        output_dir,
        candidates,
        preferences,
        failures=failures,
        results=results,
    )


def load_conscience_candidates(path: str | Path) -> List[ConscienceCandidate]:
    rows: List[ConscienceCandidate] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") != "candidate":
            continue
        payload = record["payload"]
        if payload.get("status") != "approved":
            continue
        rows.append(
            ConscienceCandidate(
                candidate_id=payload["candidate_id"],
                prompt=payload["linguistic"]["prompt"],
                response=payload["linguistic"]["response"],
                family=payload["semantic"]["family"],
                split=payload.get("split", "train"),
                plan=tuple(payload["linguistic"]["meaning_plan"]["ordered_concepts"]),
                approved=True,
                conscience=ConscienceProfile.from_dict(payload["conscience"]),
                source_refs=tuple(payload.get("source_refs", ())),
                provenance=dict(payload.get("provenance", {})),
            )
        )
    return rows


def build_conscience_trajectories(
    curriculum_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    curriculum = Path(curriculum_path)
    candidates = load_conscience_candidates(curriculum / "teacher_lessons.jsonl")
    if not candidates:
        raise ValueError("no approved conscience candidates found")
    texts = [f"{item.prompt} {item.response}" for item in candidates]
    tokenizer = ArticulatorTokenizer.build(texts)
    recorder = TrajectoryRecorder()
    for candidate in candidates:
        example = DialogueExample(
            family=candidate.family,
            split=candidate.split,
            prompt=candidate.prompt,
            response=candidate.response,
            plan=candidate.plan,
        )
        recorder.add(
            record_conscience_trajectory(
                example,
                conscience=candidate.conscience,
                tokenizer=tokenizer,
                device=device,
            )
        )
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
    train_recorder.write_jsonl(root / "train.jsonl")
    dev_recorder.write_jsonl(root / "dev.jsonl")
    manifest = {
        "schema_version": 1,
        "artifact_type": "conscience_trajectories",
        "candidate_count": len(candidates),
        "trajectory_count": len(recorder.records),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "vocab_size": len(tokenizer.tokens),
        "curriculum_path": str(curriculum),
    }
    write_json_artifact(root / "manifest.json", manifest)
    return manifest


def run_conscience_pipeline(
    output_dir: str | Path,
    *,
    teacher: Optional[TeacherClient] = None,
    max_lessons: Optional[int] = None,
    include_dilemmas: bool = True,
    device: str = "cpu",
) -> dict:
    root = Path(output_dir)
    teacher = teacher or LocalOllamaTeacher(cache_dir=root / "teacher_cache")
    seeds = all_conscience_seeds(include_dilemmas=include_dilemmas)
    curriculum_manifest = generate_conscience_curriculum(
        teacher,
        root / "curriculum",
        seeds=seeds,
        max_lessons=max_lessons,
    )
    trajectory_manifest = build_conscience_trajectories(
        root / "curriculum",
        root / "trajectories",
        device=device,
    )
    report = {
        "schema_version": 1,
        "curriculum": curriculum_manifest,
        "trajectories": trajectory_manifest,
    }
    write_json_artifact(root / "pipeline_report.json", report)
    return report
