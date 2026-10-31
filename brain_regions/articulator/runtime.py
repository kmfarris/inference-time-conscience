"""Runtime decode loop for the state-influenced articulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from brain_regions.agents.selection import SelectionSNN
from brain_regions.articulator.influence import ExecutivePoll, zero_influence
from brain_regions.articulator.model import (
    ArticulatorCache,
    ArticulatorConfig,
    StateInfluencedArticulator,
)
from brain_regions.articulator.poller import ExecutivePoller, plan_tokens_from_response
from brain_regions.articulator.vocab import ArticulatorTokenizer
from guidance import GuidanceSignal
from response_guidance import ResponseGuidance


@dataclass
class TokenDecision:
    token: str
    candidate_count: int
    selection_confidence: float
    lm_top1: str
    final_top1: str


@dataclass
class InfluencedArticulatorResult:
    text: str
    tokens: List[str]
    decisions: List[TokenDecision] = field(default_factory=list)
    polls: List[ExecutivePoll] = field(default_factory=list)
    alpha_mean: float = 0.0
    plan_coverage: float = 0.0
    mode: str = "influenced_articulator"


class InfluencedArticulatorRuntime:
    def __init__(
        self,
        model: StateInfluencedArticulator,
        tokenizer: ArticulatorTokenizer,
        *,
        selection: Optional[SelectionSNN] = None,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.selection = selection or SelectionSNN(seed=307)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "InfluencedArticulatorRuntime":
        payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        config = ArticulatorConfig.from_dict(payload["config"])
        model = StateInfluencedArticulator(config)
        model.load_state_dict(payload["model_state"])
        tokenizer = ArticulatorTokenizer.load(tokenizer_path)
        return cls(model, tokenizer, device=device)

    def _prefix_tensor(self, prompt: str) -> torch.Tensor:
        ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt) + [self.tokenizer.sep_id]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def produce(
        self,
        *,
        prompt: str,
        guidance: GuidanceSignal,
        response_guidance: Optional[ResponseGuidance],
        chemical_levels: Optional[Dict[str, float]],
        plan_tokens: Optional[Sequence[str]] = None,
        max_new_tokens: int = 32,
        top_k: int = 12,
        temperature: float = 0.8,
        use_selection: bool = True,
        force_zero_influence: bool = False,
    ) -> InfluencedArticulatorResult:
        plan = list(plan_tokens or plan_tokens_from_response(prompt))
        poller = ExecutivePoller(token_to_id=self.tokenizer.token_to_id, device=str(self.device))
        poller.begin_turn(
            guidance=guidance,
            response_guidance=response_guidance,
            chemical_levels=chemical_levels,
            plan_tokens=plan,
        )
        prefix_ids = self._prefix_tensor(prompt)
        cache = self.model.initial_cache()
        last_token = prefix_ids[0]
        for token_id in prefix_ids[1:]:
            influence = poller.poll([], 0).influence
            _, cache = self.model.step(token_id, cache, influence)
            last_token = token_id

        emitted: List[str] = []
        decisions: List[TokenDecision] = []
        polls: List[ExecutivePoll] = []
        alphas: List[float] = []

        for step in range(max_new_tokens):
            poll = poller.poll(emitted, step)
            if force_zero_influence:
                poll.influence = zero_influence(width=8, device=str(self.device))
            polls.append(poll)
            alphas.append(poll.influence.alpha)
            lm_logits, cache = self.model.step(last_token, cache, poll.influence)
            final_logits = StateInfluencedArticulator.apply_influence(
                lm_logits,
                poll.influence,
                vocab_size=len(self.tokenizer.tokens),
                concept_ids=poller.concept_ids(),
                plan_ids=poller.plan_ids(),
                inhibition_ids=poller.inhibition_ids(),
            )
            if temperature <= 0:
                next_id = int(torch.argmax(final_logits).item())
            else:
                probs = torch.softmax(final_logits / temperature, dim=-1)
                top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))
                next_id = int(top_ids[torch.multinomial(top_probs, 1)].item())

            lm_top_id = int(torch.argmax(lm_logits).item())
            final_top_id = int(torch.argmax(final_logits).item())
            token = self.tokenizer.tokens[next_id]
            if token in {"<pad>", "<bos>", "<sep>", "<unk>"}:
                break
            if use_selection:
                values, indices = torch.topk(final_logits, k=min(top_k, final_logits.shape[-1]))
                candidates = []
                scores = {}
                for value, index in zip(values.tolist(), indices.tolist()):
                    word = self.tokenizer.tokens[int(index)]
                    if word in {"<pad>", "<bos>", "<sep>", "<unk>", "<eos>"}:
                        continue
                    if word not in scores:
                        candidates.append(word)
                        scores[word] = float(value)
                if candidates:
                    self.selection.reset()
                    winner, selection_output = self.selection.select_word(
                        {
                            "word_candidates": candidates,
                            "semantic_scores": scores,
                            "values_bias": torch.zeros(len(candidates)).numpy(),
                            "values_inhibit": torch.zeros(len(candidates)).numpy(),
                            "chemical_levels": poll.chemical_levels,
                        }
                    )
                    if winner is not None:
                        token = winner
            if token == "<eos>":
                break
            emitted.append(token)
            decisions.append(
                TokenDecision(
                    token=token,
                    candidate_count=min(top_k, len(self.tokenizer.tokens)),
                    selection_confidence=1.0,
                    lm_top1=self.tokenizer.tokens[lm_top_id],
                    final_top1=self.tokenizer.tokens[final_top_id],
                )
            )
            last_token = torch.tensor(next_id, device=self.device, dtype=torch.long)

        text = self.tokenizer.decode(
            [self.tokenizer.token_to_id[token] for token in emitted if token in self.tokenizer.token_to_id]
        )
        return InfluencedArticulatorResult(
            text=text,
            tokens=emitted,
            decisions=decisions,
            polls=polls,
            alpha_mean=sum(alphas) / max(1, len(alphas)),
            plan_coverage=polls[-1].plan_coverage_ratio if polls else 0.0,
        )


# Optional helper removed — trajectory logging lives in recorder.py
