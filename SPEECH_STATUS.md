# Track C influenced articulator

*August 2026*

This page tracks whether the steered speech system is ready for everyday use. It lists the quality checks the system has to pass and the results of the latest test run. In the main Scatterbrain project, everyday conversation uses a different speech method ("path A"). This one ("path C") stays switched off until it passes these checks, and can be turned on there with `SCATTERBRAIN_INFLUENCED_ARTICULATOR=1`.

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

In plain terms: the executive's push does change which words get chosen, but the language model can't yet write fluent sentences and repeats itself, so the plan's points aren't reliably covered.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Parent project: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain).
