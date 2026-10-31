# scatterbrain-steering

Most decoders treat executive intent as a prefix: plan once, then generate until context slips. **Track C** keeps a **GRU decoder state** across the whole utterance and **polls executive signals every token**—plan coverage, chemical levels, structured guidance—turning them into a bounded bias on vocabulary logits before sampling.

```text
final_logits = lm_logits + α · executive_bias
```

The recurrent model still owns memory and fluency; the executive nudges *which* token wins, step by step. That split is what this repository implements, tests, and documents—not mesh-scale spiking speech, not a production chat path.

## What problem this targets

You want a persistent decode state (not an 8-token window) and you want turn-level control (plans, affect, guidance) to **steer** tokens without replacing the LM. Track C is the experiment: influence tensors, JSON poll roundtrips, ablation vs zero-influence, and a curriculum that asks a local teacher for grounded definition/dialogue trajectories—rejecting bad JSON instead of silently keeping it.

## Status

| Item | State |
|------|--------|
| Executive poll + influence contract | Implemented; **6** articulator influence tests + teacher/curriculum tests (**15** total) pass |
| Conscience curriculum v1 | **39** approved trajectories; failed lessons retained in manifest under `artifacts/conscience_curriculum_v1/` |
| Articulator LM smoke training | Release gates **not** passed (repetition on micro vocab; fluency score 0 at last smoke run) |
| Production in Scatterbrain | **Off** unless `SCATTERBRAIN_INFLUENCED_ARTICULATOR=1`; default speech remains frame/pathway stack |
| GGUF / local LLM hook | Documented bias mapping + NumPy demo ([`docs/GGUF_LOGIT_BIAS_DEMO.md`](docs/GGUF_LOGIT_BIAS_DEMO.md)) |

Full gate definitions and latest numbers: [`SPEECH_STATUS.md`](SPEECH_STATUS.md). Architecture: [`docs/STEERING.md`](docs/STEERING.md).

## Tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

## Data and regeneration

`artifacts/conscience_curriculum_v1/` — curriculum JSONL, trajectories, teacher cache. From the Scatterbrain monorepo: `brain_regions/articulator/run_conscience_curriculum.py` (local Ollama teacher when not using test doubles).

## Upstream

Full mesh, dilemma harness, default speech: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain). Refresh this extract: `utero_ai/scripts/export_scatterbrain_steering.sh`.
