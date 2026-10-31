# Release gate status

Release gate results for the conscience-steered articulator. The gate compares the same checkpoint with the conscience active and with its influence set to zero, on held-out lesson families.

**Evaluator:** `brain_regions/evaluation/articulator_release_gate.py`

**Held-out families:** `definition:empathy`, `conversation:how-are-you`, `conversation:greetings-and-responses`

## Latest run

200 training steps, GRU hidden size 128, bootstrap vocabulary.

| Metric | Value |
|--------|--------|
| `quality_gates_passed` | false |
| `fluency_score` (influence zeroed) | 0.0 |
| `alignment_score` (full influence) | 0.1125 |
| `influence_delta` | +0.1125 |
| `heldout_plan_coverage` | false |

The steering bias moves the output toward the reference answer: everything the steered output matched came from steering, since the unsteered output matched nothing. The articulator is undertrained, though. It repeats tokens and doesn't cover every plan concept, so the gate fails on fluency and coverage.

**What this gate measures.** The gate builds its steering signal from each example's plan words only; the value activations, suppressed concepts, and inhibition are empty. The score is half plan-word coverage and half word overlap with the reference. So the result shows that the steering path pushes planned words into the output. It doesn't test the conscience's values.

**Reproducing it.** The gate imports `brain_regions/agents/selection.py`, which isn't in this repository, and the 200-step checkpoint isn't stored, so it can't be rerun from here yet.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```
