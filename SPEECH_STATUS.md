# Speech Status — Track C Influenced Articulator

**Updated:** August 2026  
**Scope:** Honest status for the SNN-influenced articulator experiment. Default production speech is unchanged.

---

## Production paths (what actually runs)

| Path | Entry | Default? |
|------|--------|----------|
| **A — Frame/pathway** | `scatterbrain_interface.py` → `speech.py` → `frame_speech.py` | **Yes** |
| **B — Compose/retrieval hybrid** | Various compose + mesh retrieval helpers in `speech.py` | Sometimes |
| **C — Influenced articulator** | `SCATTERBRAIN_INFLUENCED_ARTICULATOR=1` → `brain_regions/production/influenced_articulator.py` | **Opt-in only** |

Track C does **not** replace Track A unless you set the env var. Track C is **not** claimed as human-quality speech.

---

## Track C architecture (influence, not drive)

The articulator is a **trainable decoder LM with persistent hidden state** (GRU stack). Each generated token:

1. **LM step** — updates persistent decode memory (not an 8–32 token prompt window only).
2. **Executive poll** — `ExecutivePoller` reads turn-level guidance, chemicals, and plan coverage; emits `InfluenceSignal`.
3. **Residual bias** — `final_logits = lm_logits + alpha * executive_bias` (concept / plan / inhibition slots).
4. **Optional SelectionSNN** — top-k rerank among biased candidates.

The SNN executive **influences** token choice; it does **not** generate text or replace LM memory.

**Package:** `brain_regions/articulator/`  
**Checkpoint:** `artifacts/articulator_v1/trained/articulator_best.pt`  
**Enable:** `export SCATTERBRAIN_INFLUENCED_ARTICULATOR=1`

---

## Training & data

- Bootstrap trajectories from `definitions_primer.py` + `conversation_primer.py` (`build_bootstrap_trajectories`).
- Optional curriculum JSONL via `ArticulatorTrainingConfig.curriculum_path`.
- Loss: CE on influence-adjusted logits + 0.5× LM CE + shuffle-attribution penalty.
- Harness: `brain_regions/articulator/training.py` → `train_articulator()`.

---

## Evaluation contract

Release gate: `brain_regions/evaluation/articulator_release_gate.py`

Held-out families: `definition:empathy`, `conversation:how-are-you`, `conversation:greetings-and-responses`.

| Gate | Meaning |
|------|---------|
| `alignment_above_zero_influence` | Full influence ≥ zero-influence alignment on held-out |
| `influence_delta_positive` | Influence helps vs ablated run |
| `heldout_plan_coverage` | Mean plan token coverage ≥ 0.25 |
| `human_preference` | Manual blind review (always false until reviewed) |

**Latest smoke run** (`artifacts/articulator_v1/release/quality_report.json`, 200 train steps, hidden=128):

- `quality_gates_passed`: **false**
- `fluency_score` (zero influence): **0.0**
- `alignment_score` (full influence): **0.1125**
- `influence_delta`: **+0.1125** (executive bias moves output; model not yet fluent)
- `heldout_plan_coverage`: **false** (repetition collapse on micro bootstrap vocab)
- `production_enabled`: **false**

This is expected at micro-bootstrap scale. Do not enable Track C in production without passing gates and human review.

---

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Covers poll determinism, plan coverage, JSON round-trip, persistent cache steps, trajectory builder, micro overfit smoke.

---

## What Track C is not

- Not a claim that SNNs **generate** speech end-to-end.
- Not integration with `customLLM` / PRCF (separate repo; user declined for this plan).
- Not a substitute for compositional frame speech until eval and quality justify it.

See also: `SPEECH_DIRECTION.md`, `.cursor/rules/scatterbrain-goals.mdc`.
