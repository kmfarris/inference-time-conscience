"""Release and attribution gates for the state-influenced articulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from brain_regions.articulator.data import bootstrap_dialogues
from brain_regions.articulator.runtime import InfluencedArticulatorRuntime
from brain_regions.articulator.training import _synthetic_guidance_from_record
from brain_regions.articulator.vocab import ArticulatorTokenizer
from brain_regions.evaluation.artifacts import write_json_artifact
from guidance import GuidanceSignal
from response_guidance import ResponseGuidance
from tokenizer import clean_tokenize


def _plan_coverage(text: str, plan: Sequence[str]) -> float:
    tokens = set(clean_tokenize(text.lower(), stem=False))
    if not plan:
        return 1.0
    hits = sum(1 for token in plan if token.lower() in tokens)
    return hits / len(plan)


def _repetition_rate(tokens: Sequence[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for index in range(1, len(tokens)) if tokens[index] == tokens[index - 1])
    return repeats / (len(tokens) - 1)


def _alignment_score(text: str, reference: str, plan: Sequence[str]) -> float:
    ref_tokens = set(clean_tokenize(reference.lower(), stem=False))
    out_tokens = set(clean_tokenize(text.lower(), stem=False))
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & out_tokens) / len(ref_tokens)
    return 0.5 * overlap + 0.5 * _plan_coverage(text, plan)


def _generate_variant(
    runtime: InfluencedArticulatorRuntime,
    *,
    prompt: str,
    guidance: GuidanceSignal,
    plan: Sequence[str],
    zero_influence_mode: bool = False,
) -> dict:
    result = runtime.produce(
        prompt=prompt,
        guidance=guidance,
        response_guidance=ResponseGuidance(conversational_mode="definition"),
        chemical_levels={"oxytocin": 0.2, "cortisol": 0.1, "dopamine": 0.15, "serotonin": 0.4},
        plan_tokens=plan,
        max_new_tokens=24,
        temperature=0.0,
        use_selection=False,
        force_zero_influence=zero_influence_mode,
    )
    tokens = clean_tokenize(result.text.lower(), stem=False)
    return {
        "text": result.text,
        "tokens": tokens,
        "alpha_mean": result.alpha_mean,
        "plan_coverage": result.plan_coverage,
        "repetition_rate": _repetition_rate(tokens),
    }


def run_articulator_release_gate(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    runtime = InfluencedArticulatorRuntime.from_checkpoint(
        checkpoint_path, tokenizer_path, device=device
    )
    heldout = [item for item in bootstrap_dialogues() if item.split != "train"]
    full_rows = []
    zero_rows = []
    for example in heldout:
        guidance = _synthetic_guidance_from_record(
            {"plan": list(example.plan), "tokens": example.response.split()}
        )
        full = _generate_variant(
            runtime,
            prompt=example.prompt,
            guidance=guidance,
            plan=example.plan,
            zero_influence_mode=False,
        )
        zero = _generate_variant(
            runtime,
            prompt=example.prompt,
            guidance=guidance,
            plan=example.plan,
            zero_influence_mode=True,
        )
        alignment_full = _alignment_score(full["text"], example.response, example.plan)
        alignment_zero = _alignment_score(zero["text"], example.response, example.plan)
        full_rows.append(
            {
                "family": example.family,
                "prompt": example.prompt,
                "reference": example.response,
                "response": full["text"],
                "alignment_score": alignment_full,
                "plan_coverage": _plan_coverage(full["text"], example.plan),
                "repetition_rate": full["repetition_rate"],
            }
        )
        zero_rows.append(
            {
                "family": example.family,
                "prompt": example.prompt,
                "response": zero["text"],
                "alignment_score": alignment_zero,
                "plan_coverage": _plan_coverage(zero["text"], example.plan),
            }
        )

    fluency_score = sum(row["alignment_score"] for row in zero_rows) / max(1, len(zero_rows))
    alignment_score = sum(row["alignment_score"] for row in full_rows) / max(1, len(full_rows))
    influence_delta = alignment_score - fluency_score
    gates = {
        "alignment_above_zero_influence": alignment_score >= fluency_score,
        "influence_delta_positive": influence_delta > 0.0,
        "heldout_plan_coverage": sum(row["plan_coverage"] for row in full_rows)
        / max(1, len(full_rows))
        >= 0.25,
        "human_preference": False,
    }
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "quality_gates_passed": all(gates.values()),
        "gates": gates,
        "fluency_score": fluency_score,
        "alignment_score": alignment_score,
        "influence_delta": influence_delta,
        "full_influence": full_rows,
        "zero_influence": zero_rows,
        "production_enabled": False,
    }
    output = Path(output_dir)
    write_json_artifact(output / "quality_report.json", report)
    blind = {
        "schema_version": 1,
        "identities_blinded": True,
        "candidates": [
            {"id": f"case_{index}", "prompt": row["prompt"], "response": row["response"]}
            for index, row in enumerate(full_rows)
        ],
    }
    write_json_artifact(output / "blind_candidates.json", blind)
    write_json_artifact(
        output / "identity_key.json",
        {"schema_version": 1, "mapping": {f"case_{i}": row["family"] for i, row in enumerate(full_rows)}},
    )
    return report
