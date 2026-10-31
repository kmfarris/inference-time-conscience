# Track C influenced articulator

*August 2026*

This repository is **path C**: a small language model that writes a reply while an executive poll nudges each next word. In the full Scatterbrain tree, everyday chat is still **path A** (frame / pathway). Path C there requires `SCATTERBRAIN_INFLUENCED_ARTICULATOR=1`.

## Decoder loop

1. **LM step** — GRU memory lasts for the whole utterance.
2. **Executive poll** — `ExecutivePoller` → `InfluenceSignal` from guidance, chemicals, plan coverage.
3. **Residual bias** — `final_logits = lm_logits + alpha * executive_bias`.

**Code:** `brain_regions/articulator/`

## Release gates

Evaluator: `brain_regions/evaluation/articulator_release_gate.py`

Held-out families: `definition:empathy`, `conversation:how-are-you`, `conversation:greetings-and-responses`.

**Last smoke run** (200 train steps, hidden 128, bootstrap vocab):

| Metric | Value |
|--------|--------|
| `quality_gates_passed` | false |
| `fluency_score` (zero influence) | 0.0 |
| `alignment_score` (full influence) | 0.1125 |
| `influence_delta` | +0.1125 |
| `heldout_plan_coverage` | false |
| `production_enabled` | false |

The nudge moves scores; fluency and coverage have not cleared on this checkpoint.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Parent project: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain).
