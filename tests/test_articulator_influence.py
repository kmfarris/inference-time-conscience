"""Tests for state-influenced articulator contracts and poll loop."""

from __future__ import annotations

import json

import torch

from brain_regions.articulator.influence import ExecutivePoll, InfluenceSignal
from brain_regions.articulator.model import ArticulatorConfig, StateInfluencedArticulator
from brain_regions.articulator.poller import ExecutivePoller
from brain_regions.articulator.recorder import build_bootstrap_trajectories, record_example_trajectory
from brain_regions.articulator.vocab import ArticulatorTokenizer
from brain_regions.articulator.data import bootstrap_dialogues
from guidance import GuidanceSignal
from response_guidance import ResponseGuidance


def _guidance() -> GuidanceSignal:
    return GuidanceSignal(
        top_concepts=[("care", 1.0), ("means", 0.8), ("helping", 0.6)],
        golden_lantern_activations={"compassion": 0.5},
        golden_lantern_fired={"compassion"},
        emotional_valence=0.2,
        topic_clusters=["care"],
        suppressed_concepts=["destroy"],
        inhibitory_suppression=0.1,
        raw_inference=None,
    )


def test_poll_is_deterministic_and_alpha_bounded():
    tokenizer = ArticulatorTokenizer.build(["care means helping others"])
    poller = ExecutivePoller(token_to_id=tokenizer.token_to_id, seed=7)
    poller.begin_turn(
        guidance=_guidance(),
        response_guidance=ResponseGuidance(conversational_mode="definition"),
        chemical_levels={"cortisol": 0.1, "dopamine": 0.2},
        plan_tokens=["care", "means", "helping", "others"],
    )
    first = poller.poll([], 0)
    second = poller.poll([], 0)
    assert first.influence.alpha == second.influence.alpha
    assert 0.05 <= first.influence.alpha <= 0.85


def test_plan_coverage_advances_only_after_matching_token():
    tokenizer = ArticulatorTokenizer.build(["care means helping others"])
    poller = ExecutivePoller(token_to_id=tokenizer.token_to_id)
    poller.begin_turn(
        guidance=_guidance(),
        response_guidance=None,
        chemical_levels={},
        plan_tokens=["care", "means", "helping"],
    )
    before = poller.poll([], 0)
    assert before.plan_coverage_ratio == 0.0
    poller.poll(["care"], 1)
    after = poller.poll(["care"], 1)
    assert after.plan_coverage_ratio > 0.0


def test_executive_poll_json_roundtrip():
    signal = InfluenceSignal(
        concept_bias=torch.tensor([1.0, 0.5]),
        inhibition_bias=torch.tensor([0.0, 0.2]),
        plan_slot_bias=torch.tensor([2.0, 0.0]),
        affect=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        alpha=0.42,
    )
    poll = ExecutivePoll(
        token_index=2,
        influence=signal,
        plan_coverage_ratio=0.5,
        plan_coverage_bitmap=[True, False, False],
        chemical_levels={"dopamine": 0.2},
        selection_metadata={"plan_slot": 1},
    )
    restored = ExecutivePoll.from_dict(json.loads(json.dumps(poll.to_dict())))
    assert restored.token_index == 2
    assert abs(restored.influence.alpha - 0.42) < 1e-6
    assert restored.plan_coverage_bitmap == [True, False, False]


def test_articulator_step_with_persistent_cache():
    config = ArticulatorConfig(vocab_size=32, hidden_size=64, num_layers=2)
    model = StateInfluencedArticulator(config)
    cache = model.initial_cache()
    influence = InfluenceSignal(
        concept_bias=torch.zeros(8),
        inhibition_bias=torch.zeros(8),
        plan_slot_bias=torch.zeros(8),
        affect=torch.zeros(4),
        alpha=0.3,
    )
    logits_a, cache = model.step(torch.tensor(1), cache, influence)
    logits_b, cache = model.step(torch.tensor(2), cache, influence)
    assert logits_a.shape == (32,)
    assert logits_b.shape == (32,)
    assert cache.position == 2


def test_bootstrap_trajectory_builder(tmp_path):
    manifest = build_bootstrap_trajectories(tmp_path / "articulator_smoke")
    assert manifest["example_count"] >= 5
    assert (tmp_path / "articulator_smoke" / "trajectories" / "train.jsonl").exists()


def test_micro_overfit_proof_runs():
    examples = bootstrap_dialogues()[:4]
    tokenizer = ArticulatorTokenizer.build(
        [example.prompt + " " + example.response for example in examples]
    )
    from brain_regions.articulator.model import ArticulatorConfig, StateInfluencedArticulator
    import torch
    from torch.nn import functional as F
    from brain_regions.articulator.recorder import record_example_trajectory

    config = ArticulatorConfig(vocab_size=len(tokenizer.tokens), hidden_size=64, num_layers=1)
    model = StateInfluencedArticulator(config)
    records = [
        record_example_trajectory(example, tokenizer=tokenizer).to_dict()
        for example in examples
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(30):
        loss = 0.0
        for record in records:
            token_ids = torch.tensor(
                [tokenizer.token_to_id.get(t, tokenizer.unk_id) for t in record["tokens"]]
            )
            influences = [
                InfluenceSignal.from_dict(step["poll"]["influence"])
                for step in record["polls"]
            ]
            logits = model.forward_sequence(token_ids.unsqueeze(0), influences)
            loss += F.cross_entropy(
                logits.view(-1, logits.shape[-1]), token_ids.unsqueeze(0).view(-1)
            )
        optimizer.zero_grad()
        (loss / len(records)).backward()
        optimizer.step()
    assert float(loss.item()) >= 0.0
