"""Training harness for the state-influenced articulator."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from brain_regions.articulator.data import DialogueExample, bootstrap_dialogues, split_examples
from brain_regions.articulator.influence import InfluenceSignal, zero_influence
from brain_regions.articulator.model import ArticulatorConfig, StateInfluencedArticulator
from brain_regions.articulator.poller import ExecutivePoller
from brain_regions.articulator.recorder import (
    build_bootstrap_trajectories,
    record_example_trajectory,
)
from brain_regions.articulator.runtime import InfluencedArticulatorRuntime
from brain_regions.articulator.vocab import ArticulatorTokenizer
from guidance import GuidanceSignal
from response_guidance import ResponseGuidance


@dataclass
class ArticulatorTrainingConfig:
    artifact_dir: str = "artifacts/articulator_v1"
    hidden_size: int = 256
    num_layers: int = 4
    batch_size: int = 8
    steps: int = 400
    learning_rate: float = 3e-4
    eval_interval: int = 50
    device: str = "cpu"
    seed: int = 2026
    curriculum_path: Optional[str] = None


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _example_from_row(row: dict) -> Optional[DialogueExample]:
    if not row.get("tokens"):
        return None
    return DialogueExample(
        family=row.get("family", "unknown"),
        split=row.get("split", "train"),
        prompt=row["prompt"],
        response=row["response"],
        plan=tuple(row.get("plan", [])),
    )


def _sequence_tensors(
    record: dict,
    tokenizer: ArticulatorTokenizer,
    *,
    device: str,
) -> Optional[tuple[torch.Tensor, List[InfluenceSignal], List[int]]]:
    example = _example_from_row(record)
    if example is None:
        return None
    token_ids = [tokenizer.token_to_id.get(token, tokenizer.unk_id) for token in record["tokens"]]
    if not token_ids:
        return None
    polls = record.get("polls", [])
    influences: List[InfluenceSignal] = []
    for index in range(len(token_ids)):
        if index < len(polls):
            influence = InfluenceSignal.from_dict(polls[index]["poll"]["influence"], device=device)
        else:
            influence = zero_influence(width=8, device=device)
        influences.append(influence)
    return torch.tensor(token_ids, device=device, dtype=torch.long), influences, token_ids


def _shuffle_influence_batch(influences: List[List[InfluenceSignal]]) -> List[List[InfluenceSignal]]:
    if len(influences) < 2:
        return influences
    order = list(range(len(influences)))
    order.reverse()
    return [influences[index] for index in order]


def train_articulator(config: ArticulatorTrainingConfig) -> dict:
    torch.manual_seed(config.seed)
    root = Path(config.artifact_dir)
    build_bootstrap_trajectories(root, curriculum_path=config.curriculum_path, device=config.device)
    tokenizer = ArticulatorTokenizer.load(root / "tokenizer.json")
    train_rows = _load_jsonl(root / "trajectories" / "train.jsonl")
    dev_rows = _load_jsonl(root / "trajectories" / "dev.jsonl")

    model_config = ArticulatorConfig(
        vocab_size=len(tokenizer.tokens),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    )
    model = StateInfluencedArticulator(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    evaluations: List[dict] = []
    best_loss = float("inf")
    best_path = root / "trained" / "articulator_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, config.steps + 1):
        batch_records = train_rows[: config.batch_size]
        total_loss = 0.0
        fluency_loss = 0.0
        shuffle_loss = 0.0
        valid = 0
        influence_batches: List[List[InfluenceSignal]] = []
        valid_records: List[tuple[dict, List[InfluenceSignal]]] = []
        for record in batch_records:
            packed = _sequence_tensors(record, tokenizer, device=config.device)
            if packed is None:
                continue
            token_ids, influences, _ = packed
            influence_batches.append(influences)
            valid_records.append((record, influences))
            logits = model.forward_sequence(token_ids.unsqueeze(0), influences)
            final_logits = []
            poller_stub = ExecutivePoller(token_to_id=tokenizer.token_to_id, device=config.device)
            poller_stub.begin_turn(
                guidance=_synthetic_guidance_from_record(record),
                response_guidance=None,
                chemical_levels={},
                plan_tokens=record.get("plan", []),
            )
            for index in range(logits.shape[1]):
                poll = poller_stub.poll(record["tokens"][:index], index)
                final = StateInfluencedArticulator.apply_influence(
                    logits[0, index],
                    poll.influence,
                    vocab_size=len(tokenizer.tokens),
                    concept_ids=poller_stub.concept_ids(),
                    plan_ids=poller_stub.plan_ids(),
                    inhibition_ids=poller_stub.inhibition_ids(),
                )
                final_logits.append(final)
            final_logits = torch.stack(final_logits).unsqueeze(0)
            target = token_ids.unsqueeze(0)
            loss_final = F.cross_entropy(final_logits.view(-1, final_logits.shape[-1]), target.view(-1))
            loss_lm = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))
            total_loss = total_loss + loss_final + 0.5 * loss_lm
            fluency_loss += float(loss_lm.item())
            valid += 1

        if valid == 0:
            continue
        loss = total_loss / valid
        shuffled = _shuffle_influence_batch(influence_batches)
        for (record, _), wrong_influences in zip(valid_records, shuffled):
            packed = _sequence_tensors(record, tokenizer, device=config.device)
            if packed is None:
                continue
            token_ids, _, _ = packed
            aligned_wrong = wrong_influences[: token_ids.shape[0]]
            while len(aligned_wrong) < token_ids.shape[0]:
                aligned_wrong.append(zero_influence(width=8, device=config.device))
            wrong_logits = model.forward_sequence(token_ids.unsqueeze(0), aligned_wrong)
            shuffle_loss += float(
                F.cross_entropy(
                    wrong_logits.view(-1, wrong_logits.shape[-1]),
                    token_ids.unsqueeze(0).view(-1),
                ).item()
            )
        attribution_penalty = max(0.0, (best_loss + 0.05) - (shuffle_loss / max(1, valid)))
        total = loss + 0.1 * attribution_penalty
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.steps:
            dev_loss = _eval_loss(model, dev_rows, tokenizer, device=config.device)
            evaluations.append({"step": step, "train_loss": float(loss.item()), "dev_loss": dev_loss})
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(
                    {"config": model_config.to_dict(), "model_state": model.state_dict()},
                    best_path,
                )

    report = {
        "schema_version": 1,
        "steps_completed": config.steps,
        "best_dev_loss": best_loss,
        "evaluations": evaluations,
        "checkpoint": str(best_path),
    }
    (root / "trained" / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tokenizer.save(root / "trained" / "tokenizer.json")
    return report


def _eval_loss(
    model: StateInfluencedArticulator,
    rows: Sequence[dict],
    tokenizer: ArticulatorTokenizer,
    *,
    device: str,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for record in rows:
            packed = _sequence_tensors(record, tokenizer, device=device)
            if packed is None:
                continue
            token_ids, influences, _ = packed
            logits = model.forward_sequence(token_ids.unsqueeze(0), influences)
            losses.append(
                float(
                    F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        token_ids.unsqueeze(0).view(-1),
                    ).item()
                )
            )
    model.train()
    return sum(losses) / max(1, len(losses))


def _synthetic_guidance_from_record(record: dict) -> GuidanceSignal:
    plan = record.get("plan", [])
    concepts = [(token, 1.0 - index * 0.05) for index, token in enumerate(plan[:8])]
    while len(concepts) < 8:
        concepts.append(("", 0.0))
    return GuidanceSignal(
        top_concepts=concepts,
        golden_lantern_activations={},
        golden_lantern_fired=set(),
        emotional_valence=0.0,
        topic_clusters=list(plan[:3]),
        suppressed_concepts=[],
        inhibitory_suppression=0.0,
        raw_inference=None,
    )


def run_micro_overfit_proof(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
    steps: int = 200,
) -> dict:
    examples = bootstrap_dialogues()[:6]
    tokenizer = ArticulatorTokenizer.build(
        [example.prompt + " " + example.response for example in examples]
    )
    config = ArticulatorConfig(vocab_size=len(tokenizer.tokens), hidden_size=128, num_layers=2)
    model = StateInfluencedArticulator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    records = [
        record_example_trajectory(example, tokenizer=tokenizer, device=device).to_dict()
        for example in examples
    ]
    for step in range(steps):
        loss_sum = 0.0
        for record in records:
            packed = _sequence_tensors(record, tokenizer, device=device)
            if packed is None:
                continue
            token_ids, influences, _ = packed
            logits = model.forward_sequence(token_ids.unsqueeze(0), influences)
            loss_sum += F.cross_entropy(
                logits.view(-1, logits.shape[-1]), token_ids.unsqueeze(0).view(-1)
            )
        optimizer.zero_grad()
        (loss_sum / len(records)).backward()
        optimizer.step()
    runtime = InfluencedArticulatorRuntime(model, tokenizer, device=device)
    passed = 0
    outputs = []
    for example in examples:
        guidance = _synthetic_guidance_from_record(
            {"plan": list(example.plan), "tokens": example.response.split()}
        )
        result = runtime.produce(
            prompt=example.prompt,
            guidance=guidance,
            response_guidance=ResponseGuidance(conversational_mode="definition"),
            chemical_levels={},
            plan_tokens=example.plan,
            max_new_tokens=16,
            temperature=0.0,
            use_selection=False,
        )
        ok = example.response.split()[0].lower() in result.text.lower()
        if ok:
            passed += 1
        outputs.append({"prompt": example.prompt, "text": result.text, "passed": ok})
    return {
        "schema_version": 1,
        "passed": passed >= 4,
        "passed_count": passed,
        "case_count": len(examples),
        "outputs": outputs,
    }
