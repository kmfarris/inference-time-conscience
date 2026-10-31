"""Fail-closed local Ollama teacher for offline curriculum generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Optional, Tuple
import urllib.error
import urllib.request

from brain_regions.evaluation.manifest import SPLITS


PROMPT_VERSION = "offline-teacher-v10"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
_PLAN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "with",
}


def _normalized_words(text: str) -> Tuple[str, ...]:
    return tuple(
        "".join(character for character in word.casefold() if character.isalnum() or character == "'")
        for word in text.split()
        if any(character.isalnum() for character in word)
    )


@dataclass(frozen=True)
class TeacherRequest:
    lesson_id: str
    task: str
    prompt: str
    semantic_family: str
    split: str
    source_text: str
    source_refs: Tuple[str, ...]
    canonical_prompts: Tuple[str, ...] = ()
    meaning_plan: Tuple[str, ...] = ()
    failed_response: str = ""

    def __post_init__(self) -> None:
        if not all(
            (
                self.lesson_id,
                self.task,
                self.prompt.strip(),
                self.semantic_family,
                self.source_text.strip(),
            )
        ):
            raise ValueError("teacher request fields cannot be empty")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        if not self.source_refs:
            raise ValueError("source_refs are required")


@dataclass(frozen=True)
class TeacherOutput:
    question_variants: Tuple[str, ...]
    approved_responses: Tuple[str, ...]
    rejected_responses: Tuple[str, ...]
    plans: Tuple[Tuple[str, ...], ...]
    correction: str
    reason_codes: Tuple[str, ...]
    source_claims: Tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "TeacherOutput":
        required = {
            "question_variants",
            "approved_responses",
            "rejected_responses",
            "plans",
            "correction",
            "reason_codes",
            "source_claims",
        }
        if set(raw) != required:
            missing = sorted(required - set(raw))
            extra = sorted(set(raw) - required)
            raise ValueError(
                f"teacher output schema mismatch; missing={missing}, extra={extra}"
            )

        def strings(name: str, minimum: int = 0) -> Tuple[str, ...]:
            value = raw[name]
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            items = tuple(str(item).strip() for item in value)
            if len(items) < minimum or any(not item for item in items):
                raise ValueError(f"{name} contains too few or empty items")
            return items

        raw_plans = raw["plans"]
        if not isinstance(raw_plans, list):
            raise ValueError("plans must be a list")
        plans = tuple(
            tuple(
                str(token).strip()
                for token in plan
                if not all(
                    word in _PLAN_STOPWORDS
                    for word in _normalized_words(str(token))
                )
            )
            for plan in raw_plans
            if isinstance(plan, list)
        )
        if len(plans) != len(raw_plans) or any(not plan for plan in plans):
            raise ValueError("every plan must be a non-empty string list")
        approved = strings("approved_responses", minimum=2)
        if len(plans) != len(approved):
            raise ValueError("plans must align with approved_responses")
        correction = raw["correction"]
        if not isinstance(correction, str):
            raise ValueError("correction must be a string")
        return cls(
            question_variants=strings("question_variants", minimum=2),
            approved_responses=approved,
            rejected_responses=strings("rejected_responses", minimum=1),
            plans=plans,
            correction=correction.strip(),
            reason_codes=strings("reason_codes"),
            source_claims=strings("source_claims", minimum=1),
        )


def validate_teacher_output(
    request: TeacherRequest,
    output: TeacherOutput,
) -> None:
    source_normalized = " ".join(_normalized_words(request.source_text))
    for quote in output.source_claims:
        quote_normalized = " ".join(_normalized_words(quote))
        if not quote_normalized or quote_normalized not in source_normalized:
            raise ValueError("teacher source_claims must be verbatim source excerpts")
    subject = request.semantic_family.partition(":")[2].replace("-", " ").casefold()
    if request.semantic_family.startswith("definition:") and subject and not all(
        subject in response.casefold() for response in output.approved_responses
    ):
        raise ValueError("every approved response must preserve the lesson subject")
    if request.canonical_prompts:
        canonical = {
            " ".join(_normalized_words(prompt))
            for prompt in request.canonical_prompts
        }
        variants = {
            " ".join(_normalized_words(prompt))
            for prompt in output.question_variants
        }
        if canonical.isdisjoint(variants):
            raise ValueError(
                "question variants must preserve a canonical bootstrap prompt"
            )
    source_content = {
        word for word in _normalized_words(request.source_text)
        if word not in _PLAN_STOPWORDS and len(word) > 2
    }
    minimum_support = (
        0.15 if request.task == "grounded_conversation" else 0.30
    )
    for response in output.approved_responses:
        response_content = {
            word for word in _normalized_words(response)
            if word not in _PLAN_STOPWORDS and len(word) > 2
        }
        support = len(response_content.intersection(source_content)) / max(
            1, len(response_content)
        )
        if support < minimum_support:
            raise ValueError(
                "approved response lacks response-level source grounding"
            )
    for response in output.approved_responses + output.rejected_responses:
        if any(character in response for character in "{}[]"):
            raise ValueError("teacher responses cannot contain wrapper artifacts")
    if len(set(output.question_variants)) != len(output.question_variants):
        raise ValueError("question variants must be unique")
    for plan in output.plans:
        if not 3 <= len(plan) <= 12:
            raise ValueError("plans must contain 3-12 content units")
        for unit in plan:
            words = _normalized_words(unit)
            if not words or all(word in _PLAN_STOPWORDS for word in words):
                raise ValueError("plans cannot contain function-word-only units")


@dataclass(frozen=True)
class TeacherProvenance:
    model: str
    model_digest: str
    prompt_version: str
    temperature: float
    seed: int
    source_hash: str
    request_hash: str
    cache_hit: bool


@dataclass(frozen=True)
class TeacherResult:
    request: TeacherRequest
    output: TeacherOutput
    provenance: TeacherProvenance


RequestFn = Callable[[str, Optional[dict], int], dict]


def _default_request(url: str, payload: Optional[dict], timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ConnectionError(f"local teacher request failed: {exc}") from exc


class LocalOllamaTeacher:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        cache_dir: str | os.PathLike[str] = "artifacts/teacher_cache",
        temperature: float = 0.2,
        seed: int = 2026,
        timeout: int = 180,
        request_fn: RequestFn = _default_request,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.timeout = int(timeout)
        self._request = request_fn

    def _model_digest(self) -> str:
        tags = self._request(f"{self.base_url}/api/tags", None, self.timeout)
        for item in tags.get("models", []):
            if item.get("name") == self.model:
                digest = str(item.get("digest") or "")
                if not digest:
                    raise RuntimeError("teacher model has no digest")
                return digest
        raise RuntimeError(f"required local teacher model is unavailable: {self.model}")

    def _request_body(self, request: TeacherRequest, digest: str) -> dict:
        source_hash = hashlib.sha256(
            request.source_text.encode("utf-8")
        ).hexdigest()
        source_claim = next(
            line.strip()
            for line in request.source_text.splitlines()
            if line.strip()
        )
        instruction = {
            "task": request.task,
            "lesson_prompt": request.prompt,
            "canonical_prompts": list(request.canonical_prompts),
            "semantic_family": request.semantic_family,
            "meaning_plan": list(request.meaning_plan),
            "failed_snn_response": request.failed_response,
            "source_refs": list(request.source_refs),
            "source_text": request.source_text,
            "required_source_claim": source_claim,
            "requirements": [
                "Use only facts supported by source_text.",
                "Return exactly 3 unique question_variants, 3 approved_responses, 3 rejected_responses, and 3 aligned plans.",
                "At least one question_variant must exactly match a supplied canonical_prompt.",
                "Set question_variants[0] to canonical_prompts[0] exactly, preserving spelling and punctuation.",
                "Create varied natural wording, not copied answer scripts.",
                "Plans must contain ordered content concepts and differ from response wording.",
                "Each plan must contain 3-12 content words or short content phrases; omit articles, conjunctions, and punctuation.",
                "Every plan item and reason code must be a non-empty content string.",
                "For definition tasks, every approved response must explicitly name the defined subject.",
                "For conversation tasks, every approved response must reuse at least two content words exactly from source_text.",
                "Rejected responses must be plausible but clearly worse or incorrect.",
                "Responses must be plain natural sentences with no JSON, braces, brackets, labels, or wrappers.",
                "Do not add moral policy beyond source_text.",
                "Set source_claims[0] to required_source_claim exactly.",
                "Return JSON only.",
            ],
        }
        schema = {
            "type": "object",
            "properties": {
                "question_variants": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 3, "maxItems": 3,
                    **(
                        {
                            "prefixItems": [
                                {"const": request.canonical_prompts[0]}
                            ]
                        }
                        if request.canonical_prompts
                        else {}
                    ),
                },
                "approved_responses": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 3, "maxItems": 3,
                },
                "rejected_responses": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 3, "maxItems": 3,
                },
                "plans": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 3,
                        "maxItems": 12,
                    },
                    "minItems": 3,
                    "maxItems": 3,
                },
                "correction": {"type": "string"},
                "reason_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "source_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "prefixItems": [{"const": source_claim}],
                },
            },
            "required": [
                "question_variants",
                "approved_responses",
                "rejected_responses",
                "plans",
                "correction",
                "reason_codes",
                "source_claims",
            ],
        }
        return {
            "model": self.model,
            "prompt": json.dumps(instruction, sort_keys=True),
            "format": schema,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": 1200,
            },
            "keep_alive": "10m",
            "_provenance": {
                "model_digest": digest,
                "prompt_version": PROMPT_VERSION,
                "source_hash": source_hash,
            },
        }

    @staticmethod
    def _hash_body(body: Mapping[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _cache_path(self, request_hash: str) -> Path:
        return self.cache_dir / f"{request_hash}.json"

    def _load_cache(
        self, path: Path, request: TeacherRequest, request_hash: str
    ) -> Optional[TeacherResult]:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_hash") != request_hash:
            raise ValueError("teacher cache request hash mismatch")
        output = TeacherOutput.from_mapping(payload["output"])
        validate_teacher_output(request, output)
        provenance = TeacherProvenance(
            **{**payload["provenance"], "cache_hit": True}
        )
        return TeacherResult(request, output, provenance)

    def _write_cache(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def generate(self, request: TeacherRequest) -> TeacherResult:
        if request.split != "train":
            raise ValueError("local teacher may only receive train-split lessons")
        digest = self._model_digest()
        body = self._request_body(request, digest)
        request_hash = self._hash_body(
            {"request": asdict(request), "body": body}
        )
        cache_path = self._cache_path(request_hash)
        cached = self._load_cache(cache_path, request, request_hash)
        if cached is not None:
            return cached

        api_body = {key: value for key, value in body.items() if key != "_provenance"}
        response = self._request(
            f"{self.base_url}/api/generate", api_body, self.timeout
        )
        raw_output = response.get("response")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError("teacher returned an empty response")
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError("teacher returned malformed JSON") from exc
        output = TeacherOutput.from_mapping(parsed)
        try:
            validate_teacher_output(request, output)
        except ValueError as first_error:
            repair_instruction = json.loads(str(body["prompt"]))
            repair_instruction["validation_error"] = str(first_error)
            repair_instruction["previous_output"] = parsed
            repair_instruction["requirements"].append(
                "Repair every validation error while preserving the exact JSON schema."
            )
            repair_body = dict(api_body)
            repair_body["prompt"] = json.dumps(
                repair_instruction, sort_keys=True
            )
            repair_body["options"] = {
                **repair_body["options"],
                "seed": self.seed + 1,
            }
            repaired = self._request(
                f"{self.base_url}/api/generate",
                repair_body,
                self.timeout,
            ).get("response")
            if not isinstance(repaired, str) or not repaired.strip():
                raise first_error
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                raise first_error
            output = TeacherOutput.from_mapping(parsed)
            validate_teacher_output(request, output)
        provenance = TeacherProvenance(
            model=self.model,
            model_digest=digest,
            prompt_version=PROMPT_VERSION,
            temperature=self.temperature,
            seed=self.seed,
            source_hash=body["_provenance"]["source_hash"],
            request_hash=request_hash,
            cache_hit=False,
        )
        self._write_cache(
            cache_path,
            {
                "request_hash": request_hash,
                "request": asdict(request),
                "output": asdict(output),
                "provenance": asdict(provenance),
            },
        )
        return TeacherResult(request, output, provenance)
