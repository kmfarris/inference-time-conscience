# Track C influenced articulator

*August 2026*

This extract documents **path C**: a GRU decoder with per-token executive logit bias. In the full Scatterbrain tree, **path A** (frame / pathway speech) remains the default; path C requires `SCATTERBRAIN_INFLUENCED_ARTICULATOR=1` in that repository.

## Decoder loop

Each token:

1. **LM step** — GRU state carries across the utterance.
2. **Executive poll** — `ExecutivePoller` → `InfluenceSignal` from guidance, chemicals, plan coverage.
3. **Residual bias** — `final_logits = lm_logits + alpha * executive_bias`.

**Code in this repo:** `brain_regions/articulator/`  
**Enable (monorepo):** `export SCATTERBRAIN_INFLUENCED_ARTICULATOR=1`

## Release gates

Evaluator: `brain_regions/evaluation/articulator_release_gate.py`

Held-out families: `definition:empathy`, `conversation:how-are-you`, `conversation:greetings-and-responses`.

**Last smoke run** (200 train steps, hidden 128, bootstrap vocab — artifact paths refer to monorepo layout when generated there):

| Metric | Value |
|--------|--------|
| `quality_gates_passed` | false |
| `fluency_score` (zero influence) | 0.0 |
| `alignment_score` (full influence) | 0.1125 |
| `influence_delta` | +0.1125 |
| `heldout_plan_coverage` | false |
| `production_enabled` | false |

## Tests (this repository)

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

## Default speech

Frame / pathway production and the full mesh live in [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain).
